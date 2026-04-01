import json
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# GPU(RTX 4070 Ti SUPER)が使用可能か確認
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class UEPeakDataset(Dataset):
    def __init__(self, json_dir):
        self.inputs = []
        self.outputs = []
        
        json_files = glob.glob(os.path.join(json_dir, "*.json"))
        if not json_files:
            print(f"Error: No JSON files found in {json_dir}")
            return

        for filepath in json_files:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = json.load(f)
                data_list = content.get('trainingDatas', [])
                
            for d in data_list:
                # LIDARデータが揃っていない場合はスキップ
                if len(d.get('distances', [])) != 7: continue
                
                # --- 21次元 Input (UE5.7 最新データ構造準拠) ---
                inp = [
                    *[v / 5000.0 for v in d['distances']],      # 1-7: LIDAR
                    d['distToWall'] / 5000.0,                   # 8: 壁距離
                    1.0 if d['isTargetVisible'] else 0.0,       # 9: 視認フラグ
                    min(d['timeTargetVisible'] / 2.0, 1.0),     # 10: 視認時間
                    d['targetDistance'] / 5000.0,               # 11: 敵距離
                    np.linalg.norm([d['targetVelocity']['x'], d['targetVelocity']['y']]) / 600.0, # 12: 敵速度
                    
                    # 角度データ：Normalize Axis済みの値を -1.0 ~ 1.0 にスケール
                    d['currentAimError']['x'] / 180.0,          # 13: Yaw Error (左右ズレ)
                    d['currentAimError']['y'] / 90.0,           # 14: Pitch Error (上下ズレ)
                    
                    np.linalg.norm([d['myVelocity']['x'], d['myVelocity']['y']]) / 600.0, # 15: 自機速度
                    1.0 if d['isReloading'] else 0.0,           # 16: リロード中
                    float(d['moveRight']),                      # 17: 入力右
                    float(d['moveForward']),                    # 18: 入力前
                    1.0 if d['isStoppingTrigger'] else 0.0,     # 19: 停止トリガー
                    1.0 if d['isFire'] else 0.0,                # 20: 射撃フラグ
                    
                    # 代表が修正した正規化済み Pitch (0度基準)
                    d['myPitch'] / 90.0                         # 21: 自機Pitch
                ]

                # --- 5次元 Output (代表の操作を再現する教師データ) ---
                # out の定義を以下に変更
                out = [
                    float(d['moveRight']),
                    float(d['moveForward']),
                    1.0 if d['isFire'] else 0.0, # ここを 0.0 に！
                    float(d['myTurn']),
                    float(d['myLockup'])
                ]
                
                self.inputs.append(inp)
                self.outputs.append(out)

        self.inputs = torch.tensor(self.inputs, dtype=torch.float32)
        self.outputs = torch.tensor(self.outputs, dtype=torch.float32)
        print(f"Total Loaded: {len(self.inputs)} frames.")

    def __len__(self): return len(self.inputs)
    def __getitem__(self, idx): return self.inputs[idx], self.outputs[idx]

class PeakAI(nn.Module):
    def __init__(self):
        super(PeakAI, self).__init__()
        # 共通の戦況認識層
        self.shared = nn.Sequential(
            nn.Linear(21, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 1024), nn.BatchNorm1d(1024), nn.ReLU()
        )
        
        # エイム・移動用（4出力）
        self.aim_head = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 4), nn.Tanh()
        )
        
        # 射撃用（1出力）: Sigmoidで 0.0 ～ 1.0 に強制
        self.fire_head = nn.Sequential(
            nn.Linear(1024, 256), nn.ReLU(),
            nn.Linear(256, 1), nn.Sigmoid()
        )

    def forward(self, x):
        features = self.shared(x)
        aim_out = self.aim_head(features)   # index 0,1,2,3 (Right, Forward, Turn, Lockup)
        fire_out = self.fire_head(features) # index 0 (Fire)
        
        # UE側の受け取り順序 [moveRight, moveForward, isFire, myTurn, myLockup] に戻す
        return torch.cat([aim_out[:, 0:2], fire_out, aim_out[:, 2:4]], dim=1)

def train():
    dataset = UEPeakDataset('data-peak')
    if len(dataset) == 0: return
    loader = DataLoader(dataset, batch_size=2048, shuffle=True)
    
    model = PeakAI().to(device)
    # 項目ごとの詳細な制御のため、Reduction='none' を使用
    criterion = nn.MSELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    print("Training with High-Priority Fire Logic...")
    for epoch in range(150):
        model.train()
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            outputs = model(x)
            loss_matrix = criterion(outputs, y) # [batch_size, 5]
            
            # --- 射撃(Index 2)の重みを強化 ---
            # 教師データが 1.0 (射撃中) の箇所の重みを 20倍にする
            fire_weights = torch.ones_like(y[:, 2])
            fire_weights[y[:, 2] > 0.5] = 5.0 
            
            # 射撃項のみ重みを適用して平均を取る
            loss_fire = (loss_matrix[:, 2] * fire_weights).mean()
            # 他の項（移動、エイム）はそのまま
            loss_others = loss_matrix[:, [0, 1, 3, 4]].mean()
            
            loss = loss_fire + loss_others
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/150, Loss: {total_loss/len(loader):.8f}")

    # ONNXエクスポート (UE5.7 NNE/NNI用)
    model.eval().cpu()
    dummy_input = torch.randn(1, 21)
    torch.onnx.export(model, dummy_input, "peak_v21_10.onnx", opset_version=15, 
                      input_names=['input'], output_names=['output'])
    print("Export Complete: .onnx")

if __name__ == "__main__":
    train()