# -*- coding: utf-8 -*-
"""
Dual MultiModal Ensemble Evaluator: MultiModal-ViT + MultiModal-ResNet18
多模态决策级软投票融合 (Soft Voting)
通过“双专家会诊”彻底榨干小样本数据集的终极潜力
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
# 1. 全局配置与路径 (请务必核对这两个权重文件夹的名字！)
# ==========================================
POS_EXCEL_PATH = "/home/fuxiangyu/jlx/MVI/MVI_ceus0325/0325data/MVI1.xlsx"
NEG_EXCEL_PATH = "/home/fuxiangyu/jlx/MVI/MVI_ceus0325/0325data/MVI0.xlsx"
IMAGE_ROOT_DIR = "/home/fuxiangyu/jlx/MVI/MVI_ceus0325/0325processed"

# 权重路径：指向你之前分别跑出来的两个多模态模型的保存文件夹
WEIGHT_DIR_VIT = "/home/fuxiangyu/jlx/MVI/MVI0325_test/pth/pth_vit2_5d"             
WEIGHT_DIR_RESNET = "/home/fuxiangyu/jlx/MVI/MVI0325_test/pth/pth_resnet18"   

FOLDER_MAPPING = {'0_MVI_Negative': 0, '1_MVI_Positive': 1}
BATCH_SIZE = 16
N_SPLITS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 2. 数据处理与划分 (严格保持一致)
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
    
    feature_cols = ['甲胎蛋白', '异常凝血酶原', 'HBV', '年龄', '性别'] # 5维核心特征
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

class KFoldDatasetWrapper(Dataset):
    def __init__(self, subset): self.subset = subset
    def __getitem__(self, index): return self.subset[index]
    def __len__(self): return len(self.subset)

# ==========================================
# 3. 网络架构重构 (需与训练时严格一致)
# ==========================================
class MultiModalViT(nn.Module):
    def __init__(self, num_clinical_features=5, num_classes=2):
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

class MultiModalResNet18(nn.Module):
    def __init__(self, num_clinical_features=5, num_classes=2):
        super(MultiModalResNet18, self).__init__()
        self.resnet = models.resnet18(weights=None)
        self.resnet.fc = nn.Identity()
        self.clinical_mlp = nn.Sequential(
            nn.Linear(num_clinical_features, 16), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(16, 32), nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(512 + 32, 128), nn.BatchNorm1d(128),
            nn.ReLU(), nn.Dropout(p=0.3), nn.Linear(128, num_classes)
        )
    def forward(self, image, clinical_data):
        return self.classifier(torch.cat((self.resnet(image), self.clinical_mlp(clinical_data)), dim=1))

# ==========================================
# 4. 集成推理主循环
# ==========================================
def run_ensemble():
    clinical_dict, num_features = clean_clinical_data(POS_EXCEL_PATH, NEG_EXCEL_PATH)
    base_dataset = MultimodalDataset(root_dir=IMAGE_ROOT_DIR, clinical_dict=clinical_dict)
    
    # 极度重要：必须保持 random_state=42，保证每次切分的测试集和训练时一模一样！
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    
    auc_list, acc_list, sen_list, spe_list = [], [], [], []
    print("\n========== 开始双模态专家集成 (ViT + ResNet18) ==========")

    for fold, (_, test_idx) in enumerate(kf.split(base_dataset)):
        print(f"\n正在处理 Fold {fold+1}/{N_SPLITS}...")
        test_loader = DataLoader(KFoldDatasetWrapper(Subset(base_dataset, test_idx)), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

        # 检查权重文件
        vit_path = os.path.join(WEIGHT_DIR_VIT, f"best_multimodal_fold{fold+1}.pth")
        resnet_path = os.path.join(WEIGHT_DIR_RESNET, f"best_multimodal_resnet_fold{fold+1}.pth")
        
        if not os.path.exists(vit_path) or not os.path.exists(resnet_path):
            print(f"  ❌ 严重警告: 找不到 Fold {fold+1} 的权重文件！")
            continue

        # 加载模型
        vit = MultiModalViT(num_features).to(DEVICE)
        vit.load_state_dict(torch.load(vit_path, map_location=DEVICE))
        vit.eval()
        
        resnet = MultiModalResNet18(num_features).to(DEVICE)
        resnet.load_state_dict(torch.load(resnet_path, map_location=DEVICE))
        resnet.eval()

        y_true, y_prob_ensemble, y_pred_ensemble = [], [], []

        with torch.no_grad():
            for x, c, y in test_loader:
                x, c, y = x.to(DEVICE), c.to(DEVICE), y.to(DEVICE)
                
                # 两位专家分别提取阳性概率
                prob_vit = torch.softmax(vit(x, c), dim=1)[:, 1]
                prob_resnet = torch.softmax(resnet(x, c), dim=1)[:, 1]
                
                # 【核心】：软投票平均
                prob_avg = (prob_vit + prob_resnet) / 2.0
                pred_avg = (prob_avg > 0.5).long()
                
                y_true.extend(y.cpu().numpy())
                y_prob_ensemble.extend(prob_avg.cpu().numpy())
                y_pred_ensemble.extend(pred_avg.cpu().numpy())

        # 计算集成指标
        auc = roc_auc_score(y_true, y_prob_ensemble)
        acc = accuracy_score(y_true, y_pred_ensemble)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_ensemble, labels=[0, 1]).ravel()
        sen = tp / (tp + fn + 1e-6)
        spe = tn / (tn + fp + 1e-6)

        print(f"  ✅ 集成诊断完毕 -> AUC: {auc:.4f} | ACC: {acc:.4f} | SEN: {sen:.4f} | SPE: {spe:.4f}")
        
        auc_list.append(auc)
        acc_list.append(acc)
        sen_list.append(sen)
        spe_list.append(spe)

    print("\n" + "="*50)
    print("🏆 双专家会诊 (Ensemble) 最终成绩单")
    print(f"Test AUC         : {np.mean(auc_list):.4f} ± {np.std(auc_list):.4f}")
    print(f"Test Accuracy    : {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}")
    print(f"Test Sensitivity : {np.mean(sen_list):.4f} ± {np.std(sen_list):.4f}")
    print(f"Test Specificity : {np.mean(spe_list):.4f} ± {np.std(spe_list):.4f}")
    print("="*50)

if __name__ == "__main__":
    run_ensemble()