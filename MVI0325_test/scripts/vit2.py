# -*- coding: utf-8 -*-
"""
MultiModal-ViT for MVI Prediction
基于 ViT 图像特征与 21 维临床表格数据的晚期融合 (Late Fusion)
严格对齐 2022/2025 顶刊多模态融合架构
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from PIL import Image

# ==========================================
# 1. 全局配置与绝对路径 - 使用原始数值文件
# ==========================================
POS_EXCEL_PATH = "/home/fuxiangyu/jlx/MVI/MVI_ceus0325/0325data/MVI阳性临床资料-原始数值.xlsx"
NEG_EXCEL_PATH = "/home/fuxiangyu/jlx/MVI/MVI_ceus0325/0325data/MVI阴性临床资料-原始数值.xlsx"
IMAGE_ROOT_DIR = "/home/fuxiangyu/jlx/MVI/MVI_ceus0325/0325processed"


FOLDER_MAPPING = {
    '0_MVI_Negative': 0,
    '1_MVI_Positive': 1
}

BATCH_SIZE = 16
EPOCHS = 50
N_SPLITS = 5
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.05
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "/home/fuxiangyu/jlx/MVI/MVI0325_test/pth/pth_vit2"

os.makedirs(SAVE_DIR, exist_ok=True)

# 设定随机种子以保证复现性
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

# ==========================================
# 2. 数据处理与 Dataset 定义 - 正确的归一化策略
# ==========================================
def clean_clinical_data(pos_path, neg_path):
    print("▶️ 开始读取并清洗临床 Excel 表格数据（使用原始数值）...")
    df_pos = pd.read_excel(pos_path)
    df_neg = pd.read_excel(neg_path)

    # 标准化列名（匹配原始数值文件的列名）
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

    # 性别编码
    df_clinical['性别'] = df_clinical['性别'].map({'男': 1, '女': 0})

    # 【关键修复】区分二分类特征和连续型特征
    # 二分类特征：不需要归一化
    binary_features = ['性别', 'HBV', 'HCV']

    # 连续型特征：需要 Z-score 标准化
    continuous_features = [
        '年龄', '总胆红素', '直接胆红素', '总蛋白', '白蛋白',
        'ALT', 'AST', '碱性磷酸酶', '谷氨酰转移酶', '总胆酸',
        '乳酸脱氢酶', '甲胎蛋白', '癌胚抗原', 'CA199', 'CA125',
        '甲胎异质体', '异常凝血酶原', '凝血酶原时间'
    ]

    # 【关键修复】用中位数填充缺失值，而非0
    for col in continuous_features:
        median_value = df_clinical[col].median()
        df_clinical[col].fillna(median_value, inplace=True)
        print(f"  {col}: 中位数={median_value:.2f}, 用于填充缺失值")

    # 二分类特征用0填充（0表示"无"）
    for col in binary_features:
        df_clinical[col].fillna(0, inplace=True)

    # 【关键修复】对所有连续型特征进行 Z-score 标准化
    print("\n▶️ 对连续型特征进行 Z-score 标准化...")
    for col in continuous_features:
        mean_val = df_clinical[col].mean()
        std_val = df_clinical[col].std()
        df_clinical[col] = (df_clinical[col] - mean_val) / (std_val + 1e-8)  # 加小常数避免除0
        print(f"  {col}: 均值={mean_val:.2f}, 标准差={std_val:.2f}")

    # 提取除了"超声号"之外的所有 21 维特征
    feature_cols = [col for col in standard_columns if col != '超声号']
    clinical_dict = {}
    for _, row in df_clinical.iterrows():
        uid = str(row['超声号']).strip()
        features = row[feature_cols].values.astype(np.float32)
        clinical_dict[uid] = features

    print(f"\n✅ 成功加载 {len(clinical_dict)} 个患者的临床数据")
    print(f"   - 二分类特征 ({len(binary_features)}): {', '.join(binary_features)}")
    print(f"   - 连续型特征 ({len(continuous_features)}): 已全部标准化")

    return clinical_dict, len(feature_cols)

class MultimodalDataset(Dataset):
    def __init__(self, root_dir, clinical_dict, transform=None):
        self.image_paths = []
        self.labels = []
        self.clinical_data = []
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

# 图像增强 (与原 mvi_vit2.py 保持一致)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5), # 左右翻转是可以的
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class KFoldDatasetWrapper(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        items = self.subset[index]
        if len(items) == 3:
            # 旧版: (image, clinical, label)
            x, c, y = items
            if self.transform:
                x = self.transform(x)
            return x, c, y
        elif len(items) == 4:
            # 新版: (images, clinical, label, phase_mask)
            imgs, c, y, mask = items
            if self.transform:
                if isinstance(imgs, torch.Tensor):
                    # 已经是tensor，不需要transform
                    pass
                else:
                    # imgs 是 list of PIL Images
                    imgs = torch.stack([self.transform(img) for img in imgs], dim=0)
            return imgs, c, y, mask

    def __len__(self):
        return len(self.subset)

class MultiPhaseDataset(Dataset):
    """
    患者级别的多时相数据集
    每条数据 = 一个患者的所有时相图像 + 临床特征 + 标签

    输出:
        images: [num_phases, 3, 224, 224]  (num_phases=4: grey/ap/pp/lp)
        clinical: [num_features]
        label: int (0 or 1)
        phase_mask: [num_phases] (1=该时相存在, 0=缺失)

    处理逻辑:
        - 同一患者同一时相如果有多张图（不同造影剂），随机选一张
        - 缺失时相用零张量填充，同时返回 phase_mask 标记哪些时相存在
    """
    PHASE_ORDER = ['grey', 'ap', 'pp', 'lp']

    def __init__(self, root_dir, clinical_dict, transform=None):
        self.transform = transform
        self.samples = []

        # 按患者ID分组收集图像路径
        patient_data = {}  # {pid: {'label': int, 'phases': {phase: [path1, path2, ...]}}}

        for folder_name, label in FOLDER_MAPPING.items():
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
            for img_name in os.listdir(folder_path):
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue

                parts = img_name.split('_')
                patient_id = parts[0]
                phase = parts[1].lower()  # grey / ap / pp / lp

                if patient_id not in clinical_dict:
                    continue
                if phase not in self.PHASE_ORDER:
                    print(f"警告: 未知时相 '{phase}' in {img_name}, 跳过")
                    continue

                if patient_id not in patient_data:
                    patient_data[patient_id] = {
                        'label': label,
                        'phases': {p: [] for p in self.PHASE_ORDER}
                    }
                patient_data[patient_id]['phases'][phase].append(
                    os.path.join(folder_path, img_name)
                )

        # 构建样本列表
        for pid, data in patient_data.items():
            self.samples.append({
                'patient_id': pid,
                'label': data['label'],
                'phase_paths': data['phases'],  # {phase: [path_list]}
                'clinical': torch.tensor(clinical_dict[pid], dtype=torch.float32)
            })

        # 统计
        phase_counts = {p: 0 for p in self.PHASE_ORDER}
        for s in self.samples:
            for p in self.PHASE_ORDER:
                if len(s['phase_paths'][p]) > 0:
                    phase_counts[p] += 1
        print(f"MultiPhaseDataset: {len(self.samples)} 患者")
        for p in self.PHASE_ORDER:
            print(f"  {p}: {phase_counts[p]}/{len(self.samples)} 患者有该时相")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        images = []
        phase_mask = []  # 1=该时相存在, 0=缺失

        for phase in self.PHASE_ORDER:
            paths = sample['phase_paths'][phase]
            if len(paths) > 0:
                # 如果同一时相有多张图（不同造影剂），训练时随机选一张，测试时选第一张
                chosen_path = random.choice(paths) if self.transform else paths[0]
                img = Image.open(chosen_path).convert('RGB')
                phase_mask.append(1)
            else:
                # 缺失时相: 创建占位图像（后续会被 mask 掉）
                img = Image.new('RGB', (224, 224), (0, 0, 0))
                phase_mask.append(0)

            if self.transform:
                img = self.transform(img)
            images.append(img)

        if isinstance(images[0], torch.Tensor):
            images = torch.stack(images, dim=0)  # [4, 3, 224, 224]

        phase_mask = torch.tensor(phase_mask, dtype=torch.float32)  # [4]

        return images, sample['clinical'], sample['label'], phase_mask

# ==========================================
# 3. 多模态融合网络模型 (核心！)
# ==========================================
class MultiModalViT(nn.Module):
    def __init__(self, num_clinical_features, num_classes=2):
        super(MultiModalViT, self).__init__()

        # 1. 图像分支: ViT-B/16 with ImageNet 预训练权重（全冻结策略）
        # 根据 XGBoost Sanity Check 结果：fine-tune 只提升 4%，但带来过拟合风险
        # 采用全冻结策略：可训练参数从 7M 降到 0.13M，更适合小数据集
        print("▶️ 加载 ViT-B/16 预训练权重（ImageNet，全冻结策略）...")
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        self.vit_hidden_dim = self.vit.hidden_dim  # 768

        # 剥离原有的分类头，使其仅输出特征向量
        self.vit.heads = nn.Identity()

        # 【关键修改】全部冻结，ViT 纯当特征提取器
        # 可训练参数从 7M 降到 0.13M，参数/患者比例从 35000:1 降到 650:1
        for param in self.vit.parameters():
            param.requires_grad = False

        print("✅ ViT 全部冻结，可训练参数: ~0.13M (仅分类头)")
            
        # 2. 临床分支: MLP（修改6：扩大中间维度，避免信息瓶颈）
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
        
        # 3. 融合分类头 (Late Fusion)
        fused_dim = self.vit_hidden_dim + 64 # 768 + 64 = 832
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(fused_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, image, clinical_data):
        img_features = self.vit(image)           # [B, 768]
        clin_features = self.clinical_mlp(clinical_data) # [B, 64]
        
        # 沿着特征维度进行拼接
        fused_features = torch.cat((img_features, clin_features), dim=1) # [B, 832]
        out = self.classifier(fused_features)
        return out

# ==========================================
# 4. Focal Loss (对抗不平衡数据)
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

# ==========================================
# 5. 训练主流程 (5-Fold) - 按患者切分避免数据泄露
# ==========================================
def train_multimodal():
    clinical_dict, num_features = clean_clinical_data(POS_EXCEL_PATH, NEG_EXCEL_PATH)
    base_dataset = MultimodalDataset(root_dir=IMAGE_ROOT_DIR, clinical_dict=clinical_dict, transform=None)

    # 【关键修复】按患者切分而非按图像切分，避免数据泄露
    # 1. 提取所有患者ID及其对应的图像索引
    patient_to_indices = {}
    for idx, img_path in enumerate(base_dataset.image_paths):
        patient_id = os.path.basename(img_path).split('_')[0]
        if patient_id not in patient_to_indices:
            patient_to_indices[patient_id] = []
        patient_to_indices[patient_id].append(idx)

    # 2. 获取唯一患者列表
    unique_patients = list(patient_to_indices.keys())
    print(f"\n========== 开始多模态融合模型训练 (ViT + {num_features}维临床数据) ==========")
    print(f"总有效样本数: {len(base_dataset)} (来自 {len(unique_patients)} 个患者)")
    print(f"平均每个患者图像数: {len(base_dataset) / len(unique_patients):.2f}")

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    auc_list, acc_list, sen_list, spe_list = [], [], [], []

    # 3. 对患者进行KFold切分
    for fold, (train_patient_idx, test_patient_idx) in enumerate(kf.split(unique_patients)):
        # 4. 将患者索引转换为图像索引
        train_patients = [unique_patients[i] for i in train_patient_idx]
        test_patients = [unique_patients[i] for i in test_patient_idx]

        train_idx = []
        for patient in train_patients:
            train_idx.extend(patient_to_indices[patient])

        test_idx = []
        for patient in test_patients:
            test_idx.extend(patient_to_indices[patient])

        print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")
        print(f"训练集: {len(train_patients)} 个患者, {len(train_idx)} 张图像")
        print(f"测试集: {len(test_patients)} 个患者, {len(test_idx)} 张图像")

        train_dataset = KFoldDatasetWrapper(Subset(base_dataset, train_idx), transform=train_transform)
        test_dataset = KFoldDatasetWrapper(Subset(base_dataset, test_idx), transform=test_transform)

        # 【修改4】去掉 WeightedRandomSampler，改用 class_weights 处理类别不平衡
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

        model = MultiModalViT(num_clinical_features=num_features, num_classes=2).to(DEVICE)

        # 【修改4】使用 class_weights=[1.0, 3.0] 处理类别不平衡（文档最优配置）
        class_weights = torch.tensor([1.0, 3.0]).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        # 针对不同层设置不同的学习率
        optimizer = optim.AdamW([
            {'params': model.vit.encoder.layers[-1:].parameters(), 'lr': LEARNING_RATE * 0.1},
            {'params': model.clinical_mlp.parameters(), 'lr': LEARNING_RATE},
            {'params': model.classifier.parameters(), 'lr': LEARNING_RATE}
        ], weight_decay=WEIGHT_DECAY)
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
        
        best_auc = 0.0
        
        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0.0
            
            # 【变化点】：解包出临床特征 c
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

            # 图像级别评估
            val_auc = roc_auc_score(y_true, y_prob)

            # 【Step 6】患者级别评估：同一患者的多张图像概率取平均
            from collections import defaultdict
            test_patient_ids = [os.path.basename(base_dataset.image_paths[i]).split('_')[0] for i in test_idx]

            patient_probs = defaultdict(list)
            patient_labels = {}
            for pid, prob, true_label in zip(test_patient_ids, y_prob, y_true):
                patient_probs[pid].append(prob)
                patient_labels[pid] = true_label

            # 患者级别：同一患者所有图像概率取平均
            patient_y_true = []
            patient_y_prob = []
            for pid in patient_probs:
                patient_y_true.append(patient_labels[pid])
                patient_y_prob.append(np.mean(patient_probs[pid]))

            patient_y_pred = [1 if p > 0.5 else 0 for p in patient_y_prob]

            # 患者级别指标
            val_auc_patient = roc_auc_score(patient_y_true, patient_y_prob)

            if (epoch+1) % 10 == 0 or epoch == EPOCHS - 1:
                print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {train_loss:.4f}")
                print(f"  图像级别 AUC: {val_auc:.4f} | 患者级别 AUC: {val_auc_patient:.4f}")

            # 使用患者级别AUC作为模型选择标准
            if val_auc_patient > best_auc:
                best_auc = val_auc_patient
                best_acc = accuracy_score(patient_y_true, patient_y_pred)
                tn, fp, fn, tp = confusion_matrix(patient_y_true, patient_y_pred, labels=[0, 1]).ravel()
                best_sen = tp / (tp + fn + 1e-6)
                best_spe = tn / (tn + fp + 1e-6)
                
                torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"best_multimodal_fold{fold+1}.pth"))
                
        print(f"⭐ Fold {fold+1} 最佳 AUC: {best_auc:.4f} (ACC: {best_acc:.4f}, SEN: {best_sen:.4f}, SPE: {best_spe:.4f})")
        auc_list.append(best_auc)
        acc_list.append(best_acc)
        sen_list.append(best_sen)
        spe_list.append(best_spe)
        
    print("\n" + "="*50)
    print("🏆 多模态融合 5-Fold 交叉验证最终结果")
    print(f"Mean AUC : {np.mean(auc_list):.4f} ± {np.std(auc_list):.4f}")
    print(f"Mean ACC : {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}")
    print(f"Mean SEN : {np.mean(sen_list):.4f} ± {np.std(sen_list):.4f}")
    print(f"Mean SPE : {np.mean(spe_list):.4f} ± {np.std(spe_list):.4f}")
    print("="*50)

if __name__ == "__main__":
    train_multimodal()