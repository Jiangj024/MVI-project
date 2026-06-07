# -*- coding: utf-8 -*-
"""
MultiModal-ViT Grad-CAM 关键区域可视化
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms, models
from PIL import Image

# =========================
# 1. 核心路径配置 (自行核对)
# =========================
# 【修改处 1】: 你刚刚跑出的 5维 或 21维 最优权重的具体路径
WEIGHT_PATH = "/home/fuxiangyu/jlx/MVI/MVI0325_test/pth/pth_vit2_5d/best_multimodal_fold5.pth" 
EXCEL_POS = "/home/fuxiangyu/jlx/MVI/MVI_ceus0325/0325data/MVI1.xlsx"
EXCEL_NEG = "/home/fuxiangyu/jlx/MVI/MVI_ceus0325/0325data/MVI0.xlsx"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLINICAL_FEATURES = 5 # 【修改处 2】: 如果用5维模型就填5，21维模型填21

# =========================
# 2. 网络架构 (严格对齐训练阶段)
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
        img_features = self.vit(image)
        clin_features = self.clinical_mlp(clinical_data)
        fused_features = torch.cat((img_features, clin_features), dim=1)
        return self.classifier(fused_features)

# =========================
# 3. 临床数据加载
# =========================
def load_single_clinical_data(target_uid):
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
    
    patient_row = df_clinical[df_clinical['超声号'].astype(str).str.strip() == str(target_uid).strip()]
    if patient_row.empty:
        raise ValueError(f"严重错误: 找不到超声号 {target_uid} 的临床数据！")
    return torch.tensor(patient_row[feature_cols].values.astype(np.float32)).to(DEVICE)

# =========================
# 4. ViT Grad-CAM 核心逻辑
# =========================
class ViT_GradCAM:
    def __init__(self, model):
        self.model = model
        self.target_layer = model.vit.encoder.layers[-2]
        self.feature_maps = None
        self.gradients = None
        
        self.target_layer.register_forward_hook(self.save_feature_maps)
        self.target_layer.register_full_backward_hook(self.save_gradients)

    def save_feature_maps(self, module, input, output):
        self.feature_maps = output.detach() # [B, 197, 768]

    def save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach() # [B, 197, 768]

    def generate_heatmap(self, image_tensor, clinical_tensor, target_class=1):
        self.model.zero_grad()
        output = self.model(image_tensor, clinical_tensor)
        score = output[0, target_class]
        score.backward()
        
        # 抛弃首个 Class Token，提取空间 Token (14x14)
        spatial_features = self.feature_maps[:, 1:, :].permute(0, 2, 1).reshape(1, 768, 14, 14)
        spatial_grads = self.gradients[:, 1:, :].permute(0, 2, 1).reshape(1, 768, 14, 14)
        
        # 梯度全局平均池化得到通道权重
        weights = torch.mean(spatial_grads, dim=(2, 3), keepdim=True)
        
        # 权重乘加，并 ReLU 去除负相关特征
        cam = torch.sum(weights * spatial_features, dim=1).squeeze()
        cam = F.relu(cam)
        
        # 归一化并放大到 224x224
        cam = cam.cpu().numpy()
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        cam = cv2.resize(cam, (224, 224))
        return cam, torch.softmax(output, dim=1)[0, target_class].item()

# =========================
# 5. 主程序
# =========================
def main():
    # 【修改处 3】: 填入你要测试的具体超声图像路径和患者编号
    IMAGE_PATH = "/home/fuxiangyu/jlx/MVI/MVI_ceus0325/0325processed/1_MVI_Positive/ZS24131864_ap_sonovue_crop.jpg"
    TARGET_UID = "ZS24131864"
    
    print(f"▶️ 加载权重: {WEIGHT_PATH}")
    model = MultiModalViT(num_clinical_features=NUM_CLINICAL_FEATURES).to(DEVICE)
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location=DEVICE))
    model.eval()
    
    cam_extractor = ViT_GradCAM(model)
    clinical_tensor = load_single_clinical_data(TARGET_UID)
    
    raw_image = Image.open(IMAGE_PATH).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(raw_image).unsqueeze(0).to(DEVICE)
    
    # 强制让模型解释“为什么判定为阳性 (MVI=1)”
    heatmap, prob = cam_extractor.generate_heatmap(image_tensor, clinical_tensor, target_class=1)
    
    # 合成可视化图
    raw_image_resized = np.array(raw_image.resize((224, 224))) / 255.0
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_color = np.float32(heatmap_color[:, :, ::-1]) / 255.0 # BGR 转 RGB
    
    superimposed_img = np.clip(heatmap_color * 0.5 + raw_image_resized * 0.5, 0, 1)
    
    # 画图
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(raw_image_resized)
    plt.title(f"Original Ultrasound (UID: {TARGET_UID})", fontweight='bold')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(superimposed_img)
    plt.title(f"ViT Grad-CAM (Pred Positive Prob: {prob:.3f})", fontweight='bold')
    plt.axis('off')
    
    save_path = f"/home/fuxiangyu/jlx/MVI/MVI0325_test/plot/gradcam/GradCAM_{TARGET_UID}.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✅ 热力图已保存: {save_path} | 预测阳性概率: {prob:.3f}")

if __name__ == "__main__":
    main()