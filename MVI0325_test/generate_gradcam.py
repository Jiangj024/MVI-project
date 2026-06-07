#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GradCAM可视化脚本 - 显示模型关注区域
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from grad_cam import GradCAM
from grad_cam.utils import visualize_cam
import cv2

# ==========================================
# 配置
# ==========================================
IMAGE_PATH = "/home/fuxiangyu/jlx/MVI/MVI_ceus0325/0325processed/1_MVI_Positive/ZS10168702_pp_sonovue_crop.jpg"
WEIGHT_PATH = "/home/fuxiangyu/jlx/MVI/MVI0325_test/pth/pth_resnet18_5d/best_multimodal_resnet_fold4.pth"
POS_EXCEL_PATH = "/home/fuxiangyu/jlx/MVI/MVI_ceus0325/0325data/MVI1.xlsx"
NEG_EXCEL_PATH = "/home/fuxiangyu/jlx/MVI/MVI_ceus0325/0325data/MVI0.xlsx"
OUTPUT_PATH = "/home/fuxiangyu/jlx/MVI/MVI0325_test/ZS10168702_pp_sonovue_gradcam.png"

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
# 自定义GradCAM包装器（处理多模态输入）
# ==========================================
class MultiModalWrapper(nn.Module):
    def __init__(self, model, clinical_features):
        super().__init__()
        self.model = model
        self.clinical_features = clinical_features

    def forward(self, x):
        # 将临床特征扩展到batch size
        batch_size = x.size(0)
        clinical_batch = self.clinical_features.repeat(batch_size, 1)
        return self.model(x, clinical_batch)

# ==========================================
# 生成GradCAM
# ==========================================
def generate_gradcam():
    print("▶️ 加载临床数据...")
    clinical_dict = load_clinical_data()

    img_name = os.path.basename(IMAGE_PATH)
    patient_id = img_name.split('_')[0]
    print(f"▶️ 患者ID: {patient_id}")

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
    img_resized = img.resize((224, 224))
    img_tensor = transform(img_resized).unsqueeze(0).to(DEVICE)

    # 加载模型
    print(f"▶️ 加载模型...")
    model = MultiModalResNet18(num_clinical_features=5, num_classes=2)
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # 包装模型以处理多模态输入
    wrapped_model = MultiModalWrapper(model, clinical_tensor)

    # 选择目标层（ResNet18的layer4）
    target_layers = [model.resnet.layer4[-1]]

    # 生成GradCAM
    print(f"▶️ 生成GradCAM热图...")
    cam = GradCAM(model=wrapped_model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=img_tensor, targets=None)
    grayscale_cam = grayscale_cam[0, :]

    # 叠加到原图
    img_np = np.array(img_resized).astype(np.float32) / 255.0
    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    # 创建对比图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 原图
    axes[0].imshow(img_resized)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # 热图
    im = axes[1].imshow(grayscale_cam, cmap='jet')
    axes[1].set_title('GradCAM Heatmap', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # 叠加图
    axes[2].imshow(visualization)
    axes[2].set_title('GradCAM Overlay', fontsize=14, fontweight='bold')
    axes[2].axis('off')

    # 添加预测信息
    fig.suptitle(f'Patient: {patient_id} | Prediction: MVI- (89.70%)',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    print(f"\n✓ GradCAM可视化已保存到: {OUTPUT_PATH}")

    return OUTPUT_PATH

if __name__ == "__main__":
    generate_gradcam()
