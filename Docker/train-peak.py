import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
import os

# --- 1. データセット定義 ---
class PeakDataset(Dataset):
    def __init__(self, data_dir):
        self.samples = []
        json_files = glob.glob(os.path.join(data_dir, "*.json"))
        
        if not json_files:
            print(f"Warning: No JSON files found in {data_dir}")
            return

        print(f"Loading {len(json_files)} files from {data_dir}...")
        for file_path in json_files:
            with open(file_path, 'r') as f:
                raw_data = json.load(f)
                self.samples.extend(raw_data['trainingDatas'])
        
        print(f"Total training samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        
        # --- 入力層 (Input): 19次元の正規化 ---
        # 1-7: 正面レイ (max 5000)
        rays = np.array(s['distances']) / 5000.0
        # 8-11: 壁距離 45/90度 (max 500)
        wall45 = np.clip(np.array(s['distToWall45']), 0, 500) / 500.0
        wall90 = np.clip(np.array(s['distToWall90']), 0, 500) / 500.0
        # 12-13: 視認フラグ & ターゲット距離
        vis = [1.0 if s.get('isTargetVisible', False) else 0.0]
        t_dist = [min(s.get('targetDistance', 5000.0) / 5000.0, 1.0)]
        # 14-15: 自己速度 (max 600)
        my_vel = np.array([s['myVelocity']['x'], s['myVelocity']['y']]) / 600.0 
        # 16: 視認継続時間 (max 2000ms)
        t_vis = [min(s.get('timeTargetVisible', 0) / 2000.0, 1.0)]
        # 17-18: 現在のエイム誤差 (分母100)
        aim_err = np.array([s['currentAimError']['x'], s['currentAimError']['y']]) / 100.0
        # 19: ピッチ角 (-90~90 -> -1~1)
        pitch = [s.get('myPitch', 0.0) / 90.0]
        
        inputs = np.concatenate([
            rays, wall45, wall90, vis, t_dist, my_vel, t_vis, aim_err, pitch
        ]).astype(np.float32)
        
        # --- 出力層 (Labels) ---
        move_r = int(s.get('moveRight', 0)) + 1   # -1,0,1 -> 0,1,2
        move_f = int(s.get('moveForward', 0)) + 1 # -1,0,1 -> 0,1,2
        fire = 1.0 if s.get('isFire', False) else 0.0
        # エイム操作量はそのまま（回帰）
        turn = float(s.get('myTurn', 0.0))
        lockup = float(s.get('myLookup', 0.0))

        return (torch.tensor(inputs), 
                torch.tensor(move_r, dtype=torch.long), 
                torch.tensor(move_f, dtype=torch.long), 
                torch.tensor(fire, dtype=torch.float32),
                torch.tensor([turn, lockup], dtype=torch.float32))

# --- 2. ネットワークモデル ---
class PeakAI(nn.Module):
    def __init__(self):
        super(PeakAI, self).__init__()
        self.backbone = nn.Sequential(
            nn.Linear(19, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        # 各種出力ヘッド
        self.move_r_head = nn.Linear(64, 3)
        self.move_f_head = nn.Linear(64, 3)
        self.fire_head = nn.Linear(64, 1)
        self.aim_head = nn.Linear(64, 2)

    def forward(self, x):
        feat = self.backbone(x)
        return (self.move_r_head(feat), 
                self.move_f_head(feat), 
                torch.sigmoid(self.fire_head(feat)),
                self.aim_head(feat))

# --- 3. 学習メイン ---
def train():
    DATA_PATH = "./data-peak/"
    dataset = PeakDataset(DATA_PATH)
    if len(dataset) == 0: return

    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    model = PeakAI()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    criterion_bin = nn.BCELoss()

    print("Starting training...")
    model.train()
    for epoch in range(100):
        t_loss = 0
        for inputs, mr, mf, fr, aim in loader:
            optimizer.zero_grad()
            p_mr, p_mf, p_fr, p_aim = model(inputs)
            
            # 各損失の合算（エイムの重みを10倍にして精度を優先）
            loss = (criterion_cls(p_mr, mr) + 
                    criterion_cls(p_mf, mf) + 
                    criterion_bin(p_fr.squeeze(), fr) + 
                    criterion_reg(p_aim, aim) * 10.0)
            
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Avg Loss: {t_loss/len(loader):.4f}")

    # ONNXエクスポート
    model.eval()
    dummy_input = torch.randn(1, 19)
    onnx_path = "peak_model_v1.onnx"
    torch.onnx.export(model, dummy_input, onnx_path, 
                      input_names=['input'], 
                      output_names=['moveR', 'moveF', 'fire', 'aim'],
                      opset_version=15)
    print(f"Success: Exported {onnx_path}")

if __name__ == "__main__":
    train()