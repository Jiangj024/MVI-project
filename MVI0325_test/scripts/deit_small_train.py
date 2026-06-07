# -*- coding: utf-8 -*-
"""
DeiT-Small for MVI Prediction
使用 DeiT-Small (Data-efficient Image Transformers) 专门为小数据集设计
基于 DeiT-Small 图像特征与 21 维临床表格数据的晚期融合
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from PIL import Image
from collections import defaultdict
import timm

# 全局配置
POS_EXCEL_PATH = "/home/fuxiangyu/jlx/MVI/MVI_ceus0325/0325data/MVI阳性临床资料-原始数值.xlsx"
NEG_EXCEL_PATH = "/home/fuxiangyu/jlx/MVI/MVI_ceus0325/0325data/MVI阴性临床资料-原始数值.xlsx"
IMAGE_ROOT_DIR = "/home/fuxiangyu/jlx/MVI/MVI_ceus0325/0325processed"

FOLDER_MAPPING = {'0_MVI_Negative': 0, '1_MVI_Positive': 1}
BATCH_SIZE = 16
EPOCHS = 50
N_SPLITS = 5
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.05
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "/home/fuxiangyu/jlx/MVI/MVI0325_test/pth/pth_deit_small"

os.makedirs(SAVE_DIR, exist_ok=True)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# 数据处理
def clean_clinical_data(pos_path, neg_path):
    print("▶️ 开始读取并清洗临床 Excel 表格数据...")
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
    df_clinical['性别'] = df_clinical['性别'].map({'男': 1, '女': 0})
    
    binary_features = ['性别', 'HBV', 'HCV']
    continuous_features = [
        '年龄', '总胆红素', '直接胆红素', '总蛋白', '白蛋白',
        'ALT', 'AST', '碱性磷酸酶', '谷氨酰转移酶', '总胆酸',
        '乳酸脱氢酶', '甲胎蛋白', '癌胚抗原', 'CA199', 'CA125',
        '甲胎异质体', '异常凝血酶原', '凝血酶原时间'
    ]
    
    for col in binary_features:
        df_clinical[col].fillna(0, inplace=True)
    for col in continuous_features:
        df_clinical[col].fillna(df_clinical[col].median(), inplace=True)
    
    for col in continuous_features:
        mean_val = df_clinical[col].mean()
        std_val = df_clinical[col].std()
        if std_val > 0:
            df_clinical[col] = (df_clinical[col] - mean_val) / std_val
        else:
            df_clinical[col] = 0.0
    
    feature_cols = [col for col in standard_columns if col != '超声号']
    clinical_dict = {}
    for _, row in df_clinical.iterrows():
        uid = str(row['超声号']).strip()
        features = row[feature_cols].values.astype(np.float32)
        clinical_dict[uid] = features
    
    return clinical_dict, len(feature_cols)

class MultimodalDataset(Dataset):
    def __init__(self, root_dir, clinical_dict, transform=None):
        self.image_paths = []
        self.labels = []
        self.clinical_data = []
        self.transform = transform
        
        for folder_name, label in FOLDER_MAPPING.items():
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
            for img_name in os.listdir(folder_path):
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                patient_id = img_name.split('_')[0]
                if patient_id in clinical_dict:
                    self.image_paths.append(os.path.join(folder_path, img_name))
                    self.labels.append(label)
                    self.clinical_data.append(clinical_dict[patient_id])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        clinical = self.clinical_data[idx]
        label = self.labels[idx]
        return image, clinical, label

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
        if self.transform:
            x = self.transform(x)
        return x, c, y
    
    def __len__(self):
        return len(self.subset)

# 模型定义
class MultiModalDeiT(nn.Module):
    def __init__(self, num_clinical_features, num_classes=2):
        super(MultiModalDeiT, self).__init__()
        
        print("▶️ 加载 DeiT-Small 预训练权重（ImageNet）...")
        
        # 使用 timm 加载 DeiT-Small
        self.deit = timm.create_model('deit_small_patch16_224', pretrained=True, num_classes=0)
        self.deit_hidden_dim = 384  # DeiT-Small 输出 384 维特征
        
        print(f"  ✅ 成功加载 DeiT-Small 预训练权重")
        print(f"  📊 DeiT-Small 参数: ~22M, 特征维度: {self.deit_hidden_dim}")
        
        # 全冻结
        for param in self.deit.parameters():
            param.requires_grad = False
        
        print(f"✅ DeiT-Small 全部冻结，可训练参数: ~0.05M (仅分类头)")
        
        # 临床分支
        self.clinical_mlp = nn.Sequential(
            nn.Linear(num_clinical_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # 融合分类头
        fused_dim = self.deit_hidden_dim + 64  # 384 + 64 = 448
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(fused_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, image, clinical_data):
        img_features = self.deit(image)
        clin_features = self.clinical_mlp(clinical_data)
        fused_features = torch.cat((img_features, clin_features), dim=1)
        return self.classifier(fused_features)

# 训练主流程
def train_multimodal_deit():
    clinical_dict, num_features = clean_clinical_data(POS_EXCEL_PATH, NEG_EXCEL_PATH)
    base_dataset = MultimodalDataset(root_dir=IMAGE_ROOT_DIR, clinical_dict=clinical_dict, transform=None)
    
    # 按患者切分
    all_patient_ids = [os.path.basename(p).split('_')[0] for p in base_dataset.image_paths]
    unique_patients = sorted(set(all_patient_ids))

    # 获取每个患者的标签（用于 StratifiedKFold）
    patient_labels = {}
    for pid in unique_patients:
        # 找到该患者的第一张图像的标签
        for i, img_pid in enumerate(all_patient_ids):
            if img_pid == pid:
                patient_labels[pid] = base_dataset.labels[i]
                break
    patient_label_list = [patient_labels[pid] for pid in unique_patients]

    print(f"\n========== DeiT-Small 训练 ==========")
    print(f"共 {len(unique_patients)} 个独立患者, {len(base_dataset)} 张图像")
    print(f"临床特征维度: {num_features}")
    print(f"标签分布: 阴性={patient_label_list.count(0)}, 阳性={patient_label_list.count(1)}")

    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    auc_list, acc_list, sen_list, spe_list = [], [], [], []

    for fold, (train_p_idx, test_p_idx) in enumerate(kf.split(unique_patients, patient_label_list)):
        train_pids = set([unique_patients[i] for i in train_p_idx])
        test_pids = set([unique_patients[i] for i in test_p_idx])
        
        train_idx = [i for i, pid in enumerate(all_patient_ids) if pid in train_pids]
        test_idx = [i for i, pid in enumerate(all_patient_ids) if pid in test_pids]
        
        print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")
        print(f"训练: {len(train_pids)}患者/{len(train_idx)}图像, 测试: {len(test_pids)}患者/{len(test_idx)}图像")
        
        train_dataset = KFoldDatasetWrapper(Subset(base_dataset, train_idx), transform=train_transform)
        test_dataset = KFoldDatasetWrapper(Subset(base_dataset, test_idx), transform=test_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        
        model = MultiModalDeiT(num_clinical_features=num_features, num_classes=2).to(DEVICE)
        
        class_weights = torch.tensor([1.0, 3.0]).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                               lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
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
            
            # 验证
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
            
            # 患者级别评估
            test_patient_ids = [os.path.basename(base_dataset.image_paths[i]).split('_')[0] for i in test_idx]
            patient_probs = defaultdict(list)
            patient_labels = {}
            for pid, prob, true_label in zip(test_patient_ids, y_prob, y_true):
                patient_probs[pid].append(prob)
                patient_labels[pid] = true_label
            
            patient_y_true = []
            patient_y_prob = []
            for pid in patient_probs:
                patient_y_true.append(patient_labels[pid])
                patient_y_prob.append(np.mean(patient_probs[pid]))
            
            patient_y_pred = [1 if p > 0.5 else 0 for p in patient_y_prob]
            val_auc_patient = roc_auc_score(patient_y_true, patient_y_prob)
            
            if (epoch+1) % 10 == 0 or epoch == EPOCHS - 1:
                print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {train_loss:.4f}")
                print(f"  图像级别 AUC: {val_auc:.4f} | 患者级别 AUC: {val_auc_patient:.4f}")
            
            if val_auc_patient > best_auc:
                best_auc = val_auc_patient
                best_acc = accuracy_score(patient_y_true, patient_y_pred)
                tn, fp, fn, tp = confusion_matrix(patient_y_true, patient_y_pred, labels=[0, 1]).ravel()
                best_sen = tp / (tp + fn + 1e-6)
                best_spe = tn / (tn + fp + 1e-6)
                torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"best_deit_small_fold{fold+1}.pth"))
        
        print(f"⭐ Fold {fold+1} 最佳 AUC: {best_auc:.4f} (ACC: {best_acc:.4f}, SEN: {best_sen:.4f}, SPE: {best_spe:.4f})")
        auc_list.append(best_auc)
        acc_list.append(best_acc)
        sen_list.append(best_sen)
        spe_list.append(best_spe)
    
    print("\n" + "="*50)
    print("🏆 DeiT-Small 5-Fold 交叉验证最终结果")
    print(f"Mean AUC : {np.mean(auc_list):.4f} ± {np.std(auc_list):.4f}")
    print(f"Mean ACC : {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}")
    print(f"Mean SEN : {np.mean(sen_list):.4f} ± {np.std(sen_list):.4f}")
    print(f"Mean SPE : {np.mean(spe_list):.4f} ± {np.std(spe_list):.4f}")
    print("="*50)

if __name__ == "__main__":
    train_multimodal_deit()
