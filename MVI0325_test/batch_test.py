#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量测试脚本 - 测试多张图片
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torchvision import transforms, models
from PIL import Image

# ==========================================
# 配置
# ==========================================
WEIGHT_PATH = "/home/fuxiangyu/jlx/MVI/MVI0325_test/pth/pth_resnet18_5d/best_multimodal_resnet_fold4.pth"
POS_EXCEL_PATH = "/home/fuxiangyu/jlx/MVI/MVI_ceus0325/0325data/MVI1.xlsx"
NEG_EXCEL_PATH = "/home/fuxiangyu/jlx/MVI/MVI_ceus0325/0325data/MVI0.xlsx"
IMAGE_ROOT = "/home/fuxiangyu/jlx/MVI/MVI_ceus0325/0325processed"

# 测试图片列表
TEST_IMAGES = [
    ("1_MVI_Positive", "ZS24128374_ap_sonovue_crop.jpg", "MVI+"),
    ("1_MVI_Positive", "ZS23440251_pp_sonovue_crop.jpg", "MVI+"),
    ("0_MVI_Negative", "ZS24228418_grey_crop.jpg", "MVI-"),
    ("0_MVI_Negative", "ZS24044132_pp_sonovue_crop.jpg", "MVI-"),
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 模型定义
# ==========================================
class MultiModalResNet18(nn.Module):
    def __init__(self, num_clinical_features, num_classes=2):
        super(MultiModalResNet18, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        self.resnet_hidden_dim = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        self.clinical_mlp = nn.Sequential(
            nn.Linear(num_clinical_features, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 32),
            nn.ReLU()
        )

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
        img_features = self.resnet(image)
        clin_features = self.clinical_mlp(clinical_data)
        fused_features = torch.cat((img_features, clin_features), dim=1)
        return self.classifier(fused_features)

# ==========================================
# 加载临床数据
# ==========================================
def load_clinical_data():
    df_pos = pd.read_excel(POS_EXCEL_PATH).iloc[:, :22]
    df_neg = pd.read_excel(NEG_EXCEL_PATH).iloc[:, :22]

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

    feature_cols = ['甲胎蛋白', '异常凝血酶原', 'HBV', '年龄', '性别']

    clinical_dict = {}
    for _, row in df_clinical.iterrows():
        uid = str(row['超声号']).strip()
        features = row[feature_cols].values.astype(np.float32)
        clinical_dict[uid] = features

    return clinical_dict

# ==========================================
# 批量推理
# ==========================================
def batch_inference():
    print("▶️ 加载临床数据...")
    clinical_dict = load_clinical_data()

    # 加载模型
    print(f"▶️ 加载模型...")
    model = MultiModalResNet18(num_clinical_features=5, num_classes=2)
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    class_names = {0: 'MVI-', 1: 'MVI+'}

    print("\n" + "="*80)
    print("🎯 批量测试结果")
    print("="*80)

    correct = 0
    total = 0

    for folder, img_name, true_label in TEST_IMAGES:
        img_path = os.path.join(IMAGE_ROOT, folder, img_name)
        patient_id = img_name.split('_')[0]

        # 检查临床数据
        if patient_id not in clinical_dict:
            print(f"\n❌ {img_name}: 未找到患者 {patient_id} 的临床数据")
            continue

        # 加载图像
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)

        # 加载临床特征
        clinical_features = clinical_dict[patient_id]
        clinical_tensor = torch.tensor(clinical_features).unsqueeze(0).to(DEVICE)

        # 推理
        with torch.no_grad():
            output = model(img_tensor, clinical_tensor)
            probs = torch.softmax(output, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item()

        pred_label = class_names[pred_class]
        is_correct = (pred_label == true_label)

        if is_correct:
            correct += 1
        total += 1

        # 输出结果
        status = "✓" if is_correct else "✗"
        print(f"\n{status} 图片: {img_name}")
        print(f"   真实标签: {true_label}")
        print(f"   预测结果: {pred_label} (置信度: {confidence:.2%})")
        print(f"   概率分布: MVI-={probs[0, 0].item():.2%}, MVI+={probs[0, 1].item():.2%}")

    # 统计
    print("\n" + "="*80)
    print(f"📊 测试统计")
    print("="*80)
    print(f"总测试数: {total}")
    print(f"正确数: {correct}")
    print(f"错误数: {total - correct}")
    print(f"准确率: {correct/total*100:.2f}%")
    print("="*80)

if __name__ == "__main__":
    batch_inference()
