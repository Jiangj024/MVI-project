#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单张图片推理脚本 - MultiModal ResNet18
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torchvision import transforms, models
from PIL import Image

# ==========================================
# 1. 配置
# ==========================================
IMAGE_PATH = "/home/fuxiangyu/jlx/MVI/MVI_ceus0325/0325processed/1_MVI_Positive/ZS10168702_pp_sonovue_crop.jpg"
WEIGHT_PATH = "/home/fuxiangyu/jlx/MVI/MVI0325_test/pth/pth_resnet18_5d/best_multimodal_resnet_fold4.pth"
POS_EXCEL_PATH = "/home/fuxiangyu/jlx/MVI/MVI_ceus0325/0325data/MVI1.xlsx"
NEG_EXCEL_PATH = "/home/fuxiangyu/jlx/MVI/MVI_ceus0325/0325data/MVI0.xlsx"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 2. 加载临床数据
# ==========================================
def load_clinical_data(pos_path, neg_path):
    """加载并处理临床数据"""
    print("▶️ 加载临床数据...")
    df_pos = pd.read_excel(pos_path).iloc[:, :22]
    df_neg = pd.read_excel(neg_path).iloc[:, :22]

    standard_columns = [
        '超声号', '性别', '年龄', 'HBV', 'HCV', '总胆红素', '直接胆红素',
        '总蛋白', '白蛋白', 'ALT', 'AST', '碱性磷酸酶', '谷氨酰转移酶',
        '总胆酸', '乳酸脱氢酶', '甲胎蛋白', '癌胚抗原', 'CA199', 'CA125',
        '甲胎异质体', '异常凝血酶原', '凝血酶原时间'
    ]
    df_pos.columns = standard_columns
    df_neg.columns = standard_columns

    df_clinical = pd.concat([df_pos, df_neg], ignore_index=True)
    df_clinical.fillna(0, inplace=True)
    df_clinical['性别'] = df_clinical['性别'].map({'男': 1, '女': 0})
    df_clinical['年龄'] = (df_clinical['年龄'] - df_clinical['年龄'].mean()) / df_clinical['年龄'].std()

    # 提取5维核心特征（与训练时一致）
    feature_cols = ['甲胎蛋白', '异常凝血酶原', 'HBV', '年龄', '性别']

    clinical_dict = {}
    for _, row in df_clinical.iterrows():
        uid = str(row['超声号']).strip()
        features = row[feature_cols].values.astype(np.float32)
        clinical_dict[uid] = features

    return clinical_dict

# ==========================================
# 3. 模型定义（与训练脚本完全一致）
# ==========================================
class MultiModalResNet18(nn.Module):
    def __init__(self, num_clinical_features, num_classes=2):
        super(MultiModalResNet18, self).__init__()

        # 1. 图像分支: ResNet18 (提取 512 维特征)
        self.resnet = models.resnet18(pretrained=False)
        self.resnet_hidden_dim = self.resnet.fc.in_features  # 512
        self.resnet.fc = nn.Identity() # 剥离原始分类头

        # 2. 临床分支: 轻量级 MLP (5 维 -> 16 维 -> 32 维)
        self.clinical_mlp = nn.Sequential(
            nn.Linear(num_clinical_features, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 32),
            nn.ReLU()
        )

        # 3. 融合分类头: 512 + 32 = 544 维
        fused_dim = self.resnet_hidden_dim + 32

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(fused_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, image, clinical_data):
        img_features = self.resnet(image)            # [B, 512]
        clin_features = self.clinical_mlp(clinical_data) # [B, 32]
        fused_features = torch.cat((img_features, clin_features), dim=1) # [B, 544]
        return self.classifier(fused_features)

# ==========================================
# 4. 推理
# ==========================================
def inference():
    # 加载临床数据
    clinical_dict = load_clinical_data(POS_EXCEL_PATH, NEG_EXCEL_PATH)

    # 从图片名提取患者ID
    img_name = os.path.basename(IMAGE_PATH)
    patient_id = img_name.split('_')[0]  # ZS10168702

    print(f"▶️ 患者ID: {patient_id}")

    if patient_id not in clinical_dict:
        print(f"❌ 错误：未找到患者 {patient_id} 的临床数据")
        print(f"可用的患者ID示例: {list(clinical_dict.keys())[:5]}")
        return

    clinical_features = clinical_dict[patient_id]
    clinical_tensor = torch.tensor(clinical_features).unsqueeze(0).to(DEVICE)

    # 加载图像
    print(f"▶️ 加载图像...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(IMAGE_PATH).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    # 加载模型
    print(f"▶️ 加载模型...")
    model = MultiModalResNet18(num_clinical_features=5, num_classes=2)
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # 推理
    print(f"▶️ 执行推理...")
    with torch.no_grad():
        output = model(img_tensor, clinical_tensor)
        probs = torch.softmax(output, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class].item()

    # 输出结果
    class_names = {0: 'MVI Negative (MVI-)', 1: 'MVI Positive (MVI+)'}
    print("\n" + "="*50)
    print("🎯 推理结果")
    print("="*50)
    print(f"预测类别: {class_names[pred_class]}")
    print(f"置信度: {confidence:.4f} ({confidence*100:.2f}%)")
    print(f"各类别概率:")
    print(f"  - MVI-: {probs[0, 0].item():.4f} ({probs[0, 0].item()*100:.2f}%)")
    print(f"  - MVI+: {probs[0, 1].item():.4f} ({probs[0, 1].item()*100:.2f}%)")
    print("="*50)

    return pred_class, confidence, probs.cpu().numpy()

if __name__ == "__main__":
    inference()
