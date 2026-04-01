import json
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

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
                if len(d.get('distances', [])) != 7: continue
                
                # --- 21次元 Input (不変) ---
                inp = [
                    *[v / 5000.0 for v in d['distances']],      # 1-7
                    d['distToWall'] / 5000.0,                   # 8
                    1.0 if d['isTargetVisible'] else 0.0,       # 9
                    min(d['timeTargetVisible'] / 2.0, 1.0),     # 10
                    d['targetDistance'] / 5000.0,               # 11
                    np.linalg.norm([d['targetVelocity']['x'], d['targetVelocity']['y']]) / 600.0, # 12
                    d['currentAimError']['x'] / 180.0,          # 13
                    np.linalg.norm([d['myVelocity']['x'], d['myVelocity']['y']]) / 600.0,        # 14
                    1.0 if d['isReloading'] else 0.0,           # 15
                    float(d['moveRight']),                      # 16
                    float(d['moveForward']),                    # 17
                    1.0 if d['isStoppingTrigger'] else 0.0,     # 18
                    1.0 if d['isFire'] else 0.0,                # 19
                    1.0 if d['isCrouching'] else 0.0,           # 20
                    ((d['myPitch'] - 360.0) if d['myPitch'] > 180.0 else d['myPitch']) / 90.0 # 21 (正規化)                         # 21
                ]
                self.inputs.append(inp)
                
                # --- 5次元 Output (新設項目 myTurn, myLockup を反映) ---
                out = [
                    float(d['moveRight']),
                    float(d['moveForward']),
                    1.0 if d['isFire'] else -1.0,
                    d.get('myTurn', 0.0),    # 左右エイム操作量
                    d.get('myLockup', 0.0)   # 上下エイム操作量
                ]
                self.outputs.append(out)

        self.inputs = torch.tensor(self.inputs, dtype=torch.float32)
        self.outputs = torch.tensor(self.outputs, dtype=torch.float32)
        print(f"Total Loaded: {len(self.inputs)} frames.")

    def __len__(self): return len(self.inputs)
    def __getitem__(self, idx): return self.inputs[idx], self.outputs[idx]

class PeakAI(nn.Module):
    def __init__(self):
        super(PeakAI, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(21, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 5), nn.Tanh() # エイム操作も -1.0 ~ 1.0 に収まるため Tanh
        )
    def forward(self, x): return self.net(x)

def train():
    dataset = UEPeakDataset('data-peak')
    if len(dataset) == 0: return
    loader = DataLoader(dataset, batch_size=2048, shuffle=True)
    
    model = PeakAI().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    print("Training representing the Director's 0.2x sensitivity...")
    for epoch in range(150):
        model.train()
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/150, Loss: {total_loss/len(loader):.8f}")

    # ONNXエクスポート
    model.eval().cpu()
    torch.onnx.export(model, torch.randn(1, 21), "peak_v21_5_final.onnx", opset_version=15, 
                      input_names=['input'], output_names=['output'])
    print("Export Complete: peak_v21_5_final.onnx")

if __name__ == "__main__":
    train()