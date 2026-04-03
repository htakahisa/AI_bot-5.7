import json
import os

data = []
for file_name in os.listdir('./data-aim'):
    if file_name.endswith('.json'):
        with open(os.path.join('./data-aim', file_name), 'r', encoding='utf-8') as f:
            content = json.load(f)
            if 'trainingDatas' in content:
                data.extend(content['trainingDatas'])

print(f"総データ数: {len(data)}")

# 問題の組み合わせ: relPitch > 5 かつ relYaw が -10 〜 -35
target = [d for d in data if d['relPitch'] > 5 and -35 < d['relYaw'] < -10]
print(f"relPitch>5 かつ relYaw -10〜-35: {len(target)}件")

# 比較用: 同じrelYaw範囲でrelPitchがマイナス
normal = [d for d in data if d['relPitch'] < -5 and -35 < d['relYaw'] < -10]
print(f"relPitch<-5 かつ relYaw -10〜-35: {len(normal)}件")

# さらに細かく分布確認
print("\n--- relPitch x relYaw の組み合わせ分布 ---")
bins = [
    ("relPitch>5,  relYaw:-10〜-35", lambda d: d['relPitch'] >  5 and -35 < d['relYaw'] < -10),
    ("relPitch>5,  relYaw:-35〜-60", lambda d: d['relPitch'] >  5 and -60 < d['relYaw'] < -35),
    ("relPitch>5,  relYaw: 10〜 35", lambda d: d['relPitch'] >  5 and  10 < d['relYaw'] <  35),
    ("relPitch<-5, relYaw:-10〜-35", lambda d: d['relPitch'] < -5 and -35 < d['relYaw'] < -10),
    ("relPitch<-5, relYaw: 10〜 35", lambda d: d['relPitch'] < -5 and  10 < d['relYaw'] <  35),
    ("relPitch~0,  relYaw:-10〜-35", lambda d: -5 < d['relPitch'] < 5 and -35 < d['relYaw'] < -10),
]
for label, cond in bins:
    count = sum(1 for d in data if cond(d))
    print(f"  {label}: {count}件")