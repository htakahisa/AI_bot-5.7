import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class UEShootingDataset(Dataset):
    def __init__(self, json_dir):
        self.data = []
        for file_name in os.listdir(json_dir):
            if file_name.endswith('.json'):
                with open(os.path.join(json_dir, file_name), 'r', encoding='utf-8') as f:
                    content = json.load(f)
                    if 'trainingDatas' in content:
                        self.data.extend(content['trainingDatas'])
        print(f"Loaded {len(self.data)} frames.")

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 角度の最短経路を計算 (-180 ~ 180)
        ry = item['relYaw']
        while ry > 180: ry -= 360
        while ry < -180: ry += 360
        
        rp = item['relPitch']
        rp = max(min(rp, 89.0), -89.0)

        # 入力正規化
        inputs = torch.tensor([rp / 45.0, ry / 180.0, item['distance'] / 3000.0], dtype=torch.float32)

        # 射撃ラベルを -1(False) か 1(True) に変換
        fire_label = 1.0 if item.get('isFire', False) else -1.0
        
        labels = torch.tensor([item['turn'] / 12.0, item['lockup'] / 3.0, fire_label], dtype=torch.float32)
        return inputs, labels

class ShootingAI(nn.Module):
    def __init__(self):
        super(ShootingAI, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 3), nn.Tanh()
        )
    def forward(self, x): return self.net(x)

def train():
    dataset = UEShootingDataset('./data-aim')
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True)
    model = ShootingAI().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    for epoch in range(150):
        model.train()
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/150], Loss: {loss.item():.8f}")

    model.eval().cpu()
    torch.onnx.export(model, torch.randn(1, 3), "shooting_ai_v5.onnx", opset_version=15)
    print("Done. Exported to shooting_ai_vx.onnx")

if __name__ == "__main__": train()