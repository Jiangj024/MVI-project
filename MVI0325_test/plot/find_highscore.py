# -*- coding: utf-8 -*-
"""
扫描高置信度 MVI 阳性样本 (Prob > 0.85)
"""

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torchvision import transforms, models
from PIL import Image

# =========================
# 1. 核心配置 (需与你的 Grad-CAM 脚本完全一致)
# =========================
WEIGHT_PATH = "/home/fuxiangyu/jlx/MVI/MVI0325_test/pth/pth_vit2_5d/best_multimodal_fold5.pth" 
EXCEL_POS = "/home/fuxiangyu/jlx/MVI/MVI_ceus0325/0325data/MVI1.xlsx"
EXCEL_NEG = "/home/fuxiangyu/jlx/MVI/MVI_ceus0325/0325data/MVI0.xlsx"
POS_IMG_DIR = "/home/fuxiangyu/jlx/MVI/MVI_ceus0325/0325processed/1_MVI_Positive"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLINICAL_FEATURES = 5 # 【注意】: 5维填5，21维填21

# =========================
# 2. 网络与数据加载复用
# =========================
class MultiModalViT(nn.Module):
    def __init__(self, num_clinical_features, num_classes=2):
        super(MultiModalViT, self).__init__()
        self.vit = models.vit_b_16(weights=None)
        self.vit.heads = nn.Identity()
        self.clinical_mlp = nn.Sequential(
            nn.Linear(num_clinical_features, 32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(32, 64), nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(768 + 64, 128), nn.BatchNorm1d(128),
            nn.ReLU(), nn.Dropout(p=0.3), nn.Linear(128, num_classes)
        )
    def forward(self, image, clinical_data):
        return self.classifier(torch.cat((self.vit(image), self.clinical_mlp(clinical_data)), dim=1))

def get_all_clinical_dict():
    df_pos = pd.read_excel(EXCEL_POS).iloc[:, :22]
    df_neg = pd.read_excel(EXCEL_NEG).iloc[:, :22]
    standard_columns = [
        '超声号', '性别', '年龄', 'HBV', 'HCV', '总胆红素', '直接胆红素', 
        '总蛋白', '白蛋白', 'ALT', 'AST', '碱性磷酸酶', '谷氨酰转移酶', 
        '总胆酸', '乳酸脱氢酶', '甲胎蛋白', '癌胚抗原', 'CA199', 'CA125', 
        '甲胎异质体', '异常凝血酶原', '凝血酶原时间'
    ]
    df_pos.columns, df_neg.columns = standard_columns, standard_columns
    df_clinical = pd.concat([df_pos, df_neg], ignore_index=True)
    df_clinical.fillna(0, inplace=True)
    df_clinical['性别'] = df_clinical['性别'].map({'男': 1, '女': 0})
    df_clinical['年龄'] = (df_clinical['年龄'] - df_clinical['年龄'].mean()) / df_clinical['年龄'].std()
    
    feature_cols = ['甲胎蛋白', '异常凝血酶原', 'HBV', '年龄', '性别'] if NUM_CLINICAL_FEATURES == 5 else [col for col in standard_columns if col != '超声号']
    
    clinical_dict = {}
    for _, row in df_clinical.iterrows():
        uid = str(row['超声号']).strip()
        clinical_dict[uid] = torch.tensor(row[feature_cols].values.astype(np.float32)).to(DEVICE)
    return clinical_dict

# =========================
# 3. 扫描执行
# =========================
def scan_high_confidence_samples():
    print(f"▶️ 加载模型: {WEIGHT_PATH}")
    model = MultiModalViT(num_clinical_features=NUM_CLINICAL_FEATURES).to(DEVICE)
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location=DEVICE))
    model.eval()
    
    clinical_dict = get_all_clinical_dict()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("\n🔍 开始扫描阳性文件夹...")
    found_count = 0
    results = []

    with torch.no_grad():
        for img_name in os.listdir(POS_IMG_DIR):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')): continue
            
            uid = img_name.split('_')[0]
            if uid not in clinical_dict: continue
            
            img_path = os.path.join(POS_IMG_DIR, img_name)
            image_tensor = transform(Image.open(img_path).convert('RGB')).unsqueeze(0).to(DEVICE)
            clinical_tensor = clinical_dict[uid].unsqueeze(0) # 加上 batch 维度
            
            out = model(image_tensor, clinical_tensor)
            prob = torch.softmax(out, dim=1)[0, 1].item()
            
            if prob > 0.85:
                results.append((prob, uid, img_path))
                found_count += 1

    # 按概率从高到低排序输出
    results.sort(key=lambda x: x[0], reverse=True)
    print(f"\n✅ 扫描完毕，共找到 {found_count} 个高置信度样本 (Prob > 0.85)：")
    print("-" * 60)
    for prob, uid, path in results[:10]: # 只打印前10个最笃定的
        print(f"Prob: {prob:.4f} | UID: {uid:<12} | 文件名: {os.path.basename(path)}")
    print("-" * 60)

if __name__ == "__main__":
    scan_high_confidence_samples()