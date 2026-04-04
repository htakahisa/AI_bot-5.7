import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
import os

class PeakDataset(Dataset):
    def __init__(self, data_dir):
        self.samples = []
        json_files = glob.glob(os.path.join(data_dir, "*.json"))
        for file_path in json_files:
            with open(file_path, 'r') as f:
                raw_data = json.load(f)
                self.samples.extend(raw_data['trainingDatas'])
        print(f"Total training samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        
        # --- 入力層 (Input): 22次元の正規化 ---
        rays = np.array(s['distances']) / 5000.0                       # 0-6 (7次元)
        wall45 = np.clip(np.array(s['distToWall45']), 0, 500) / 500.0  # 7-8 (2次元)
        wall90 = np.clip(np.array(s['distToWall90']), 0, 500) / 500.0  # 9-10 (2次元)
        vis = [1.0 if s.get('isTargetVisible', False) else 0.0]        # 11 (1次元)
        t_dist = [min(s.get('targetDistance', 5000.0) / 5000.0, 1.0)]  # 12 (1次元)
        
        # 自己速度 (13-14: 2次元)
        my_vel = np.array([s['myVelocity']['x'], s['myVelocity']['y']]) / 600.0 
        
        # 敵速度 (15-17: 3次元) - 偏差射撃に不可欠
        t_vel = np.array([
            s['targetVelocity'].get('x', 0), 
            s['targetVelocity'].get('y', 0), 
            s['targetVelocity'].get('z', 0)
        ]) / 600.0
        
        t_vis = [min(s.get('timeTargetVisible', 0) / 2000.0, 1.0)]     # 18 (1次元)
        aim_err = np.array([s['currentAimError']['x'], s['currentAimError']['y']]) / 100.0 # 19-20 (2次元)
        pitch = [s.get('myPitch', 0.0) / 90.0]                         # 21 (1次元)
        
        # 合計: 7+2+2+1+1+2+3+1+2+1 = 22次元
        inputs = np.concatenate([
            rays, wall45, wall90, vis, t_dist, my_vel, t_vel, t_vis, aim_err, pitch
        ]).astype(np.float32)
        
        # --- 出力層 (Labels) ---
        move_r = int(s.get('moveRight', 0)) + 1
        move_f = int(s.get('moveForward', 0)) + 1
        fire = 1.0 if s.get('isFire', False) else 0.0
        turn = float(s.get('myTurn', 0.0))
        lockup = float(s.get('myLockup', 0.0))

        return (torch.tensor(inputs), 
                torch.tensor(move_r, dtype=torch.long), 
                torch.tensor(move_f, dtype=torch.long), 
                torch.tensor(fire, dtype=torch.float32),
                torch.tensor([turn, lockup], dtype=torch.float32))

class PeakAI(nn.Module):
    def __init__(self):
        super(PeakAI, self).__init__()
        self.backbone = nn.Sequential(
            nn.Linear(22, 128), # 22次元入力に修正
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
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

def train():
    dataset = PeakDataset("./data-peak/")
    if len(dataset) == 0: return
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    model = PeakAI()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    criterion_bin = nn.BCELoss()

    model.train()
    for epoch in range(100):
        t_loss = 0
        for inputs, mr, mf, fr, aim in loader:
            optimizer.zero_grad()
            p_mr, p_mf, p_fr, p_aim = model(inputs)
            loss = (criterion_cls(p_mr, mr) + criterion_cls(p_mf, mf) + 
                    criterion_bin(p_fr.squeeze(), fr) + criterion_reg(p_aim, aim) * 10.0)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss: {t_loss/len(loader):.4f}")

    # ONNX出力
    model.eval()
    dummy_input = torch.randn(1, 22)
    torch.onnx.export(model, dummy_input, "peak_model_v2.onnx", 
                      input_names=['input'], 
                      output_names=['moveR', 'moveF', 'fire', 'aim'],
                      opset_version=15)
    print("Exported peak_model_v2.onnx (22 dims)")

if __name__ == "__main__":
    train()