# -*- coding: utf-8 -*-
"""
XGBoost Sanity Check: 预训练特征 + 传统分类器
目的：搞清楚到底是特征不够好，还是 fine-tune 过程搞坏了特征
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from xgboost import XGBClassifier
from PIL import Image

# ==========================================
# 1. 全局配置
# ==========================================
POS_EXCEL_PATH = "/home/fuxiangyu/jlx/MVI/MVI_ceus0325/0325data/MVI阳性临床资料-原始数值.xlsx"
NEG_EXCEL_PATH = "/home/fuxiangyu/jlx/MVI/MVI_ceus0325/0325data/MVI阴性临床资料-原始数值.xlsx"
IMAGE_ROOT_DIR = "/home/fuxiangyu/jlx/MVI/MVI_ceus0325/0325processed"

FOLDER_MAPPING = {'0_MVI_Negative': 0, '1_MVI_Positive': 1}
BATCH_SIZE = 32  # 只做推理，可以用更大的batch size
N_SPLITS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 2. 数据处理（复用 vit2.py 的逻辑）
# ==========================================
def clean_clinical_data(pos_path, neg_path):
    print("▶️ 开始读取并清洗临床 Excel 表格数据（使用原始数值）...")
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
    df_clinical['性别'] = df_clinical['性别'].map({'男': 1, '女': 0})

    binary_features = ['性别', 'HBV', 'HCV']
    continuous_features = [
        '年龄', '总胆红素', '直接胆红素', '总蛋白', '白蛋白',
        'ALT', 'AST', '碱性磷酸酶', '谷氨酰转移酶', '总胆酸',
        '乳酸脱氢酶', '甲胎蛋白', '癌胚抗原', 'CA199', 'CA125',
        '甲胎异质体', '异常凝血酶原', '凝血酶原时间'
    ]

    # 缺失值处理
    for col in binary_features:
        df_clinical[col].fillna(0, inplace=True)
    for col in continuous_features:
        df_clinical[col].fillna(df_clinical[col].median(), inplace=True)

    # Z-score 标准化
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

# 图像预处理（只需要基本的resize和normalize，不需要数据增强）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 3. 特征提取器（全冻结的 ViT）
# ==========================================
def create_feature_extractor():
    """创建全冻结的 ViT 特征提取器"""
    print("▶️ 加载预训练 ViT-B/16 模型（ImageNet权重）...")
    vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    vit.heads = nn.Identity()  # 移除分类头，只保留特征提取

    # 全部冻结
    for param in vit.parameters():
        param.requires_grad = False

    vit.eval()  # 设置为评估模式
    return vit.to(DEVICE)

def extract_features(model, dataloader):
    """提取所有样本的特征"""
    print("▶️ 开始提取特征...")
    all_features = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (images, clinical, labels) in enumerate(dataloader):
            images = images.to(DEVICE)

            # 提取图像特征 (768维)
            img_features = model(images).cpu().numpy()

            # 拼接临床特征 (21维)
            clinical_np = np.array([c.numpy() for c in clinical])
            fused_features = np.concatenate([img_features, clinical_np], axis=1)

            all_features.append(fused_features)
            all_labels.extend(labels.numpy())

            if (batch_idx + 1) % 10 == 0:
                print(f"  已处理 {(batch_idx + 1) * BATCH_SIZE} 张图像...")

    all_features = np.vstack(all_features)
    all_labels = np.array(all_labels)

    print(f"✅ 特征提取完成: {all_features.shape}")
    return all_features, all_labels

# ==========================================
# 4. XGBoost 分类器训练与评估
# ==========================================
def train_xgboost(X_train, y_train, X_test, y_test):
    """训练 XGBoost 分类器"""
    # 计算类别权重
    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    scale_pos_weight = n_neg / n_pos

    clf = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss'
    )

    clf.fit(X_train, y_train, verbose=False)

    # 预测
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)

    # 计算指标
    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    sen = tp / (tp + fn + 1e-6)
    spe = tn / (tn + fp + 1e-6)

    return auc, acc, sen, spe, clf

# ==========================================
# 5. 主流程：按患者切分的5-Fold交叉验证
# ==========================================
def main():
    print("\n" + "="*60)
    print("XGBoost Sanity Check: 预训练特征 + 传统分类器")
    print("="*60)

    # 加载数据
    clinical_dict, num_features = clean_clinical_data(POS_EXCEL_PATH, NEG_EXCEL_PATH)
    dataset = MultimodalDataset(root_dir=IMAGE_ROOT_DIR, clinical_dict=clinical_dict, transform=transform)

    print(f"\n总样本数: {len(dataset)}")
    print(f"临床特征维度: {num_features}")

    # 按患者切分
    all_patient_ids = [os.path.basename(p).split('_')[0] for p in dataset.image_paths]
    unique_patients = sorted(set(all_patient_ids))
    print(f"独立患者数: {len(unique_patients)}")

    # 获取每个患者的标签（用于 StratifiedKFold）
    patient_labels = {}
    for pid in unique_patients:
        # 找到该患者的第一张图像的标签
        for i, img_pid in enumerate(all_patient_ids):
            if img_pid == pid:
                patient_labels[pid] = dataset.labels[i]
                break
    patient_label_list = [patient_labels[pid] for pid in unique_patients]
    print(f"标签分布: 阴性={patient_label_list.count(0)}, 阳性={patient_label_list.count(1)}")

    # 创建特征提取器
    feature_extractor = create_feature_extractor()

    # 提取所有特征（只需要做一次）
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    all_features, all_labels = extract_features(feature_extractor, dataloader)

    print(f"\n特征维度: {all_features.shape[1]} (768 图像特征 + {num_features} 临床特征)")

    # 5-Fold 交叉验证
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    auc_list, acc_list, sen_list, spe_list = [], [], [], []

    print(f"\n{'='*60}")
    print("开始 5-Fold 交叉验证")
    print('='*60)

    for fold, (train_p_idx, test_p_idx) in enumerate(kf.split(unique_patients, patient_label_list)):
        train_pids = set([unique_patients[i] for i in train_p_idx])
        test_pids = set([unique_patients[i] for i in test_p_idx])

        # 根据患者ID映射回图像索引
        train_idx = [i for i, pid in enumerate(all_patient_ids) if pid in train_pids]
        test_idx = [i for i, pid in enumerate(all_patient_ids) if pid in test_pids]

        print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")
        print(f"训练: {len(train_pids)}患者/{len(train_idx)}图像")
        print(f"测试: {len(test_pids)}患者/{len(test_idx)}图像")

        # 准备训练和测试数据
        X_train = all_features[train_idx]
        y_train = all_labels[train_idx]
        X_test = all_features[test_idx]
        y_test = all_labels[test_idx]

        # 训练 XGBoost
        auc, acc, sen, spe, clf = train_xgboost(X_train, y_train, X_test, y_test)

        print(f"结果: AUC={auc:.4f}, ACC={acc:.4f}, SEN={sen:.4f}, SPE={spe:.4f}")

        auc_list.append(auc)
        acc_list.append(acc)
        sen_list.append(sen)
        spe_list.append(spe)

    # 输出最终结果
    print(f"\n{'='*60}")
    print("🏆 XGBoost 5-Fold 交叉验证最终结果")
    print('='*60)
    print(f"Mean AUC : {np.mean(auc_list):.4f} ± {np.std(auc_list):.4f}")
    print(f"Mean ACC : {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}")
    print(f"Mean SEN : {np.mean(sen_list):.4f} ± {np.std(sen_list):.4f}")
    print(f"Mean SPE : {np.mean(spe_list):.4f} ± {np.std(spe_list):.4f}")
    print('='*60)

    # 结果解读
    print("\n📊 结果解读:")
    mean_auc = np.mean(auc_list)
    if mean_auc >= 0.75:
        print("✅ XGBoost AUC ≥ 0.75")
        print("   → 说明预训练特征质量很好")
        print("   → 问题在于 fine-tune 过程可能在破坏特征")
        print("   → 建议：全冻结 ViT 或使用更保守的训练策略")
    elif mean_auc >= 0.70:
        print("⚠️  XGBoost AUC 在 0.70-0.75 之间")
        print("   → 预训练特征有一定表达力，但不够强")
        print("   → 建议：尝试换预训练权重（DINOv2/医学预训练）")
        print("   → 同时考虑全冻结策略")
    else:
        print("❌ XGBoost AUC < 0.70")
        print("   → ImageNet 特征对超声图像表达力有限")
        print("   → 强烈建议：换预训练权重（DINOv2/医学预训练）")
        print("   → 或者等待更多数据")

    print(f"\n对比参考:")
    print(f"  - 当前 ViT fine-tune 结果: AUC ≈ 0.71")
    print(f"  - XGBoost (冻结特征):    AUC = {mean_auc:.4f}")

if __name__ == "__main__":
    main()
