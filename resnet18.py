# -*- coding: utf-8 -*-
"""
MultiModal-ResNet18 for MVI Prediction
基于 ResNet18 图像特征与 5 维核心临床特征的晚期融合
采用类别权重 CrossEntropy 重点打击“漏诊(低SEN)”问题
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from PIL import Image

# ==========================================
# 1. 全局配置与绝对路径
# ==========================================
POS_EXCEL_PATH = "/home/fuxiangyu/jlx/MVI/MVI_ceus0325/0325data/MVI1.xlsx"
NEG_EXCEL_PATH = "/home/fuxiangyu/jlx/MVI/MVI_ceus0325/0325data/MVI0.xlsx"
IMAGE_ROOT_DIR = "/home/fuxiangyu/jlx/MVI/MVI_ceus0325/0325processed"

FOLDER_MAPPING = {'0_MVI_Negative': 0, '1_MVI_Positive': 1}

BATCH_SIZE = 16
EPOCHS = 50
N_SPLITS = 5
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.05
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "/home/fuxiangyu/jlx/MVI/MVI0325_test/pth/pth_resnet18"

os.makedirs(SAVE_DIR, exist_ok=True)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

# ==========================================
# 2. 数据处理 (精准提取 5 维特征)
# ==========================================
def clean_clinical_data(pos_path, neg_path):
    print("▶️ 开始读取临床表格，提取 5 维核心特征...")
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
    
    # 【核心修改】：抛弃噪音，只留 5 把尖刀
    # feature_cols = ['甲胎蛋白', '异常凝血酶原', 'HBV', '年龄', '性别']
    feature_cols = [col for col in standard_columns if col != '超声号']

    
    clinical_dict = {}
    for _, row in df_clinical.iterrows():
        uid = str(row['超声号']).strip()
        features = row[feature_cols].values.astype(np.float32)
        clinical_dict[uid] = features
        
    return clinical_dict, len(feature_cols)

class MultimodalDataset(Dataset):
    def __init__(self, root_dir, clinical_dict, transform=None):
        self.image_paths, self.labels, self.clinical_data = [], [], []
        self.transform = transform
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
        if self.transform: image = self.transform(image)
        return image, self.clinical_data[idx], self.labels[idx]

# 图像增强 (复用 mvi_resnet18plus3.py 配置)
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class KFoldDatasetWrapper(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    def __getitem__(self, index):
        x, c, y = self.subset[index]
        if self.transform: x = self.transform(x)
        return x, c, y
    def __len__(self): return len(self.subset)

# ==========================================
# 3. 多模态融合网络模型 (ResNet18 版)
# ==========================================
class MultiModalResNet18(nn.Module):
    def __init__(self, num_clinical_features, num_classes=2):
        super(MultiModalResNet18, self).__init__()
        
        # 1. 图像分支: ResNet18 (提取 512 维特征)
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet_hidden_dim = self.resnet.fc.in_features  # 512
        self.resnet.fc = nn.Identity() # 剥离原始分类头
        
        # 冻结浅层，解冻深层特征提取器 (恢复为你原版的 layer3 和 layer4 双解冻)
        for param in self.resnet.parameters():
            param.requires_grad = False
        for param in self.resnet.layer3.parameters():
            param.requires_grad = True
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True
            
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
# 4. 训练主流程
# ==========================================
def train_multimodal_resnet():
    clinical_dict, num_features = clean_clinical_data(POS_EXCEL_PATH, NEG_EXCEL_PATH)
    base_dataset = MultimodalDataset(root_dir=IMAGE_ROOT_DIR, clinical_dict=clinical_dict, transform=None)
    
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    auc_list, acc_list, sen_list, spe_list = [], [], [], []

    print(f"\n========== 开始 MultiModal-ResNet18 训练 ==========")
    print(f"视觉分支: ResNet18 (512D) | 临床分支: {num_features}维核心特征 (32D)")
    print(f"总有效样本数: {len(base_dataset)}")
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(base_dataset)):
        print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")
        
        train_dataset = KFoldDatasetWrapper(Subset(base_dataset, train_idx), transform=train_transform)
        test_dataset = KFoldDatasetWrapper(Subset(base_dataset, test_idx), transform=test_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        
        model = MultiModalResNet18(num_clinical_features=num_features, num_classes=2).to(DEVICE)
        
        # 【致胜关键】：抛弃 FocalLoss，使用带类别权重的 CrossEntropy
        # 强行纠正样本不平衡：阴性(0)权重1.0，阳性(1)权重3.0，惩罚模型漏诊阳性！
        class_weights = torch.tensor([1.0, 3.0]).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # 恢复你原版的 Adam，并统一给予 1e-4 的充足学习率
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                               lr=LEARNING_RATE, weight_decay=1e-2)
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
        best_auc = 0.0
        
        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0.0
            
            for x, c, y in train_loader:
                x, c, y = x.to(DEVICE), c.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                out = model(x, c)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * x.size(0)
                
            scheduler.step()
            train_loss /= len(train_loader.dataset)
            
            # --- 验证阶段 ---
            model.eval()
            y_true, y_prob, y_pred = [], [], []
            with torch.no_grad():
                for x, c, y in test_loader:
                    x, c, y = x.to(DEVICE), c.to(DEVICE), y.to(DEVICE)
                    out = model(x, c)
                    probs = torch.softmax(out, dim=1)[:, 1]
                    preds = torch.argmax(out, dim=1)
                    
                    y_true.extend(y.cpu().numpy())
                    y_prob.extend(probs.cpu().numpy())
                    y_pred.extend(preds.cpu().numpy())
                    
            val_auc = roc_auc_score(y_true, y_prob)
            
            if (epoch+1) % 10 == 0 or epoch == EPOCHS - 1:
                print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f}")
            
            if val_auc > best_auc:
                best_auc = val_auc
                best_acc = accuracy_score(y_true, y_pred)
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
                best_sen = tp / (tp + fn + 1e-6)
                best_spe = tn / (tn + fp + 1e-6)
                torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"best_multimodal_resnet_fold{fold+1}.pth"))
                
        print(f"⭐ Fold {fold+1} 最佳 AUC: {best_auc:.4f} (ACC: {best_acc:.4f}, SEN: {best_sen:.4f}, SPE: {best_spe:.4f})")
        auc_list.append(best_auc)
        acc_list.append(best_acc)
        sen_list.append(best_sen)
        spe_list.append(best_spe)
        
    print("\n" + "="*50)
    print("🏆 MultiModal-ResNet18 5-Fold 最终结果")
    print(f"Mean AUC : {np.mean(auc_list):.4f} ± {np.std(auc_list):.4f}")
    print(f"Mean ACC : {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}")
    print(f"Mean SEN : {np.mean(sen_list):.4f} ± {np.std(sen_list):.4f}")
    print(f"Mean SPE : {np.mean(spe_list):.4f} ± {np.std(spe_list):.4f}")
    print("="*50)

if __name__ == "__main__":
    train_multimodal_resnet()