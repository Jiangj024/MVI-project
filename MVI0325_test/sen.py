# -*- coding: utf-8 -*-
"""
多模态融合模型离线阈值调优脚本
用于寻找 ACC、SEN 和 SPE 的最佳临床平衡点
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from PIL import Image

# ==========================================
# 1. 路径与全局配置 (必须与训练时完全一致)
# ==========================================
POS_EXCEL_PATH = "/home/fuxiangyu/jlx/MVI/MVI_ceus0325/0325data/MVI1.xlsx"
NEG_EXCEL_PATH = "/home/fuxiangyu/jlx/MVI/MVI_ceus0325/0325data/MVI0.xlsx"
IMAGE_ROOT_DIR = "/home/fuxiangyu/jlx/MVI/MVI_ceus0325/0325processed"
WEIGHT_DIR = "/home/fuxiangyu/jlx/MVI/MVI0325_test/pth_vit2_5d" # 你的权重保存路径

FOLDER_MAPPING = {'0_MVI_Negative': 0, '1_MVI_Positive': 1}
BATCH_SIZE = 16
N_SPLITS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 2. 数据处理与模型定义 (直接复用)
# ==========================================
def clean_clinical_data(pos_path, neg_path):
    df_pos = pd.read_excel(pos_path)
    df_neg = pd.read_excel(neg_path)
    standard_columns = [
        '超声号', '性别', '年龄', 'HBV', 'HCV', '总胆红素', '直接胆红素', 
        '总蛋白', '白蛋白', 'ALT', 'AST', '碱性磷酸酶', '谷氨酰转移酶', 
        '总胆酸', '乳酸脱氢酶', '甲胎蛋白', '癌胚抗原', 'CA199', 'CA125', 
        '甲胎异质体', '异常凝血酶原', '凝血酶原时间'
    ]
    df_pos = df_pos.iloc[:, :22]
    df_pos.columns = standard_columns
    df_neg = df_neg.iloc[:, :22]
    df_neg.columns = standard_columns
    
    df_clinical = pd.concat([df_pos, df_neg], ignore_index=True)
    df_clinical.fillna(0, inplace=True)
    df_clinical['性别'] = df_clinical['性别'].map({'男': 1, '女': 0})
    df_clinical['年龄'] = (df_clinical['年龄'] - df_clinical['年龄'].mean()) / df_clinical['年龄'].std()
    
    feature_cols = ['甲胎蛋白', '异常凝血酶原', 'HBV', '年龄', '性别']
    clinical_dict = {}
    for _, row in df_clinical.iterrows():
        uid = str(row['超声号']).strip()
        clinical_dict[uid] = row[feature_cols].values.astype(np.float32)
    return clinical_dict, len(feature_cols)

class MultimodalDataset(Dataset):
    def __init__(self, root_dir, clinical_dict, transform=None):
        self.image_paths, self.labels, self.clinical_data = [], [], []
        for folder_name, label in FOLDER_MAPPING.items():
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path): continue
            for img_name in os.listdir(folder_path):
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')): continue
                patient_id = img_name.split('_')[0]
                if patient_id in clinical_dict:
                    self.image_paths.append(os.path.join(folder_path, img_name))
                    self.labels.append(label)
                    self.clinical_data.append(torch.tensor(clinical_dict[patient_id]))

    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])(image), self.clinical_data[idx], self.labels[idx]

class KFoldDatasetWrapper(Dataset):
    def __init__(self, subset): self.subset = subset
    def __getitem__(self, index): return self.subset[index]
    def __len__(self): return len(self.subset)

class MultiModalViT(nn.Module):
    def __init__(self, num_clinical_features, num_classes=2):
        super(MultiModalViT, self).__init__()
        self.vit = models.vit_b_16(weights=None) # 推理时不加载预训练，直接加载保存的权重
        self.vit.heads = nn.Identity()
        self.clinical_mlp = nn.Sequential(
            nn.Linear(num_clinical_features, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(self.vit.hidden_dim + 64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, image, clinical_data):
        return self.classifier(torch.cat((self.vit(image), self.clinical_mlp(clinical_data)), dim=1))

# ==========================================
# 3. 扫描并测试不同阈值
# ==========================================
def run_threshold_scan():
    clinical_dict, num_features = clean_clinical_data(POS_EXCEL_PATH, NEG_EXCEL_PATH)
    base_dataset = MultimodalDataset(root_dir=IMAGE_ROOT_DIR, clinical_dict=clinical_dict)
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    
    all_y_true, all_y_prob = [], []
    print("========== 开始提取 5-Fold 验证集的预测概率 ==========")
    
    for fold, (_, test_idx) in enumerate(kf.split(base_dataset)):
        test_loader = DataLoader(KFoldDatasetWrapper(Subset(base_dataset, test_idx)), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        model = MultiModalViT(num_clinical_features=num_features).to(DEVICE)
        model.load_state_dict(torch.load(os.path.join(WEIGHT_DIR, f"best_multimodal_fold{fold+1}.pth"), map_location=DEVICE))
        model.eval()
        
        with torch.no_grad():
            for x, c, y in test_loader:
                probs = torch.softmax(model(x.to(DEVICE), c.to(DEVICE)), dim=1)[:, 1].cpu().numpy()
                all_y_prob.extend(probs)
                all_y_true.extend(y.numpy())

    y_true, y_prob = np.array(all_y_true), np.array(all_y_prob)
    
    print("\n========== 多模态模型阈值扫描结果 ==========")
    print(f"{'阈值 (Threshold)':<16} | {'ACC':<8} | {'SEN (阳性准确率)':<18} | {'SPE (阴性准确率)':<18}")
    print("-" * 65)
    for thresh in np.arange(0.20, 0.65, 0.05):
        y_pred = (y_prob >= thresh).astype(int)
        acc = accuracy_score(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sen, spe = tp / (tp + fn + 1e-6), tn / (tn + fp + 1e-6)
        marker = " <--(Default)" if np.isclose(thresh, 0.5) else ""
        print(f"{thresh:.2f}{marker:<14} | {acc:.3f}    | {sen:.3f}                | {spe:.3f}")

    print("\n" + "="*65)
    print(f"总体综合 AUC (衡量模型排序硬实力): {roc_auc_score(y_true, y_prob):.4f}")

if __name__ == "__main__":
    run_threshold_scan()