# -*- coding: utf-8 -*-
"""
医学顶刊级 ROC 曲线与混淆矩阵绘制脚本
自动读取 5-Fold 权重并生成带标准差阴影的高清 ROC 图
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, confusion_matrix

# ==========================================
# 1. 路径与全局配置 (请核对你的权重保存路径)
# ==========================================
POS_EXCEL_PATH = "/home/fuxiangyu/jlx/MVI/MVI_ceus0325/0325data/MVI1.xlsx"
NEG_EXCEL_PATH = "/home/fuxiangyu/jlx/MVI/MVI_ceus0325/0325data/MVI0.xlsx"
IMAGE_ROOT_DIR = "/home/fuxiangyu/jlx/MVI/MVI_ceus0325/0325processed"

# 【重要】：请确认你的多模态 ViT 权重是保存在这个文件夹里
WEIGHT_DIR = "/home/fuxiangyu/jlx/MVI/MVI0325_test/pth/pth_vit2_5d" 
WEIGHT_PREFIX = "best_multimodal_fold" # 权重文件的前缀

FOLDER_MAPPING = {'0_MVI_Negative': 0, '1_MVI_Positive': 1}
BATCH_SIZE = 16
N_SPLITS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 2. 数据处理与模型定义 (严格复用训练结构)
# ==========================================
def clean_clinical_data(pos_path, neg_path):
    df_pos = pd.read_excel(pos_path).iloc[:, :22]
    df_neg = pd.read_excel(neg_path).iloc[:, :22]
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
    
    # 5 维核心特征
    feature_cols = ['甲胎蛋白', '异常凝血酶原', 'HBV', '年龄', '性别']
    clinical_dict = {str(row['超声号']).strip(): row[feature_cols].values.astype(np.float32) for _, row in df_clinical.iterrows()}
    return clinical_dict, len(feature_cols)

class MultimodalDataset(Dataset):
    def __init__(self, root_dir, clinical_dict):
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
        image = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])(Image.open(self.image_paths[idx]).convert('RGB'))
        return image, self.clinical_data[idx], self.labels[idx]

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
            nn.Dropout(p=0.5), nn.Linear(self.vit.hidden_dim + 64, 128), nn.BatchNorm1d(128),
            nn.ReLU(), nn.Dropout(p=0.3), nn.Linear(128, num_classes)
        )
    def forward(self, image, clinical_data):
        return self.classifier(torch.cat((self.vit(image), self.clinical_mlp(clinical_data)), dim=1))

class KFoldDatasetWrapper(Dataset):
    def __init__(self, subset): self.subset = subset
    def __getitem__(self, index): return self.subset[index]
    def __len__(self): return len(self.subset)

# ==========================================
# 3. 核心绘图逻辑
# ==========================================
def generate_paper_figures():
    clinical_dict, num_features = clean_clinical_data(POS_EXCEL_PATH, NEG_EXCEL_PATH)
    base_dataset = MultimodalDataset(root_dir=IMAGE_ROOT_DIR, clinical_dict=clinical_dict)
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    
    # 画图所需变量
    tprs, aucs = [], []
    mean_fpr = np.linspace(0, 1, 100)
    all_y_true, all_y_pred = [], []
    
    # 设置画图风格 (学术风)
    plt.style.use('default')
    fig_roc, ax_roc = plt.subplots(figsize=(8, 8))
    
    print("========== 开始推理并绘制图表 ==========")
    for fold, (_, test_idx) in enumerate(kf.split(base_dataset)):
        print(f"处理 Fold {fold+1}...")
        test_loader = DataLoader(KFoldDatasetWrapper(Subset(base_dataset, test_idx)), batch_size=BATCH_SIZE, shuffle=False)
        
        model = MultiModalViT(num_clinical_features=num_features).to(DEVICE)
        weight_path = os.path.join(WEIGHT_DIR, f"{WEIGHT_PREFIX}{fold+1}.pth")
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
        model.eval()
        
        y_true, y_prob = [], []
        with torch.no_grad():
            for x, c, y in test_loader:
                out = model(x.to(DEVICE), c.to(DEVICE))
                probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                preds = torch.argmax(out, dim=1).cpu().numpy()
                y_true.extend(y.numpy())
                y_prob.extend(probs)
                
                all_y_true.extend(y.numpy())
                all_y_pred.extend(preds)
        
        # 计算当前 Fold 的 ROC 和 AUC
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        
        # 插值以便后续计算均值
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        
        # 画出单折的浅色细线
        ax_roc.plot(fpr, tpr, lw=1.5, alpha=0.3, label=f'Fold {fold+1} ROC (AUC = {roc_auc:.3f})')

    # --- 绘制 Mean ROC 曲线与阴影带 ---
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    
    ax_roc.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_auc:.3f} $\pm$ {std_auc:.3f})', lw=2.5, alpha=0.9)
    
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax_roc.fill_between(mean_fpr, tprs_lower, tprs_upper, color='blue', alpha=0.15, label=r'$\pm$ 1 std. dev.')

    # ROC 对角线和美化
    ax_roc.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=0.8)
    ax_roc.set_xlim([-0.05, 1.05])
    ax_roc.set_ylim([-0.05, 1.05])
    ax_roc.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=14, fontweight='bold')
    ax_roc.set_ylabel('True Positive Rate (Sensitivity)', fontsize=14, fontweight='bold')
    ax_roc.set_title('Cross-Validation ROC Curve (MultiModal-ViT)', fontsize=16, fontweight='bold')
    ax_roc.legend(loc="lower right", fontsize=11)
    ax_roc.grid(True, linestyle='--', alpha=0.6)
    
    fig_roc.tight_layout()
    roc_save_path = "ROC_Curve_MultiModal.png"
    fig_roc.savefig(roc_save_path, dpi=300, bbox_inches='tight')
    print(f"✅ ROC 曲线已保存为: {roc_save_path}")
    
    # --- 绘制全局混淆矩阵热力图 ---
    cm = confusion_matrix(all_y_true, all_y_pred)
    
    fig_cm, ax_cm = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, annot_kws={"size": 16, "weight": "bold"})
    
    ax_cm.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    ax_cm.set_ylabel('True Label', fontsize=14, fontweight='bold')
    ax_cm.set_xticklabels(['Negative (0)', 'Positive (1)'], fontsize=12)
    ax_cm.set_yticklabels(['Negative (0)', 'Positive (1)'], fontsize=12, va='center')
    ax_cm.set_title('Overall Confusion Matrix (5 Folds)', fontsize=16, fontweight='bold')
    
    fig_cm.tight_layout()
    cm_save_path = "/home/fuxiangyu/jlx/MVI/MVI0325_test/plot/fig_vit2/Confusion_Matrix_MultiModal.png"
    fig_cm.savefig(cm_save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 混淆矩阵已保存为: {cm_save_path}")

if __name__ == "__main__":
    generate_paper_figures()