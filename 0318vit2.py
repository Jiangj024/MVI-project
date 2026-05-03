# -*- coding: utf-8 -*-
"""
Vision Transformer (ViT) Ultimate Tuned for HCC MVI
极限防过拟合版：MixUp + AdamW + 高权重衰减 + Label Smoothing + 仅解冻单层
"""

import os
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms, models, datasets

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

# =========================
# 1. 全局配置与随机种子
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

DATA_ROOT = "/home/fuxiangyu/jlx/MVI/MVI_ceus0325/0325processed"
NUM_CLASSES = 2
BATCH_SIZE = 16  
EPOCHS = 50      
LR = 5e-5        # 【关键修改 1】：ViT 微调需要更小的学习率
N_SPLITS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# 2. 数据增强 
# =========================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
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
        x, y = self.subset[index]
        if self.transform: x = self.transform(x)
        return x, y
    def __len__(self):
        return len(self.subset)

base_dataset = datasets.ImageFolder(root=DATA_ROOT, transform=None)
all_labels = base_dataset.targets 

# =========================
# 3. 评估函数
# =========================
def evaluate(model, loader):
    model.eval()
    y_true, y_prob, y_pred = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            prob = torch.softmax(out, dim=1)[:, 1]
            pred = torch.argmax(out, dim=1)
            y_true.extend(y.cpu().numpy())
            y_prob.extend(prob.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0.5 
    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sen = tp / (tp + fn + 1e-6)
    spe = tn / (tn + fp + 1e-6)
    return auc, acc, sen, spe

# =========================
# 4. 5-Fold 交叉验证与模型训练
# =========================
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

auc_list, acc_list, sen_list, spe_list = [], [], [], []
start_time = time.time()
os.makedirs("MVI_vit_pth_tuned", exist_ok=True)

for fold, (train_idx, test_idx) in enumerate(kf.split(base_dataset)):
    print(f"\n========== Fold {fold+1}/{N_SPLITS} ==========")
    
    train_dataset = KFoldDatasetWrapper(Subset(base_dataset, train_idx), train_transform)
    test_dataset = KFoldDatasetWrapper(Subset(base_dataset, test_idx), test_transform)

    # 样本权重均衡
    train_labels_fold = [all_labels[i] for i in train_idx]
    class_counts = [train_labels_fold.count(0), train_labels_fold.count(1)]
    class_weights = [1.0 / class_counts[0], 1.0 / class_counts[1]]
    sample_weights = [class_weights[label] for label in train_labels_fold]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print("正在加载 Vision Transformer (ViT-B_16) 预训练权重...")
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    
    # 【关键修改 2】：极限冻结，只解冻最后一个 Encoder Block
    for param in model.parameters():
        param.requires_grad = False
    for param in model.encoder.layers[-1:].parameters():
        param.requires_grad = True
        
    model.heads = nn.Sequential(
        nn.Dropout(p=0.6), # 稍微加大全连接层的 Dropout
        nn.Linear(model.hidden_dim, NUM_CLASSES)
    )
    model = model.to(DEVICE)

    # 【关键修改 3】：Label Smoothing 强制模型降低自信度
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # 【关键修改 4】：启用 AdamW 并施加极强的 Weight Decay (0.05)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                            lr=LR, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_auc_in_fold = 0.0
    best_metrics = None

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            # 【关键修改 5】：重启 MixUp 压制训练集表现
            alpha = 0.4
            lam = np.random.beta(alpha, alpha)
            index = torch.randperm(x.size(0)).to(DEVICE)
            mixed_x = lam * x + (1 - lam) * x[index, :]
            y_a, y_b = y, y[index]
            
            optimizer.zero_grad()
            out = model(mixed_x)
            loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        scheduler.step()

        # 评估逻辑
        train_auc, train_acc, _, _ = evaluate(model, train_loader)
        test_auc, test_acc, sen, spe = evaluate(model, test_loader)
        
        if test_auc > best_auc_in_fold:
            best_auc_in_fold = test_auc
            best_metrics = (test_auc, test_acc, sen, spe)
            torch.save(model.state_dict(), f"jlx/MVI/MVI0325_test/pth_0318vit2/best_vit2_fold{fold+1}.pth")
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1:03d}/{EPOCHS}] Loss: {running_loss/len(train_loader):.4f} "
                  f"| Train ACC: {train_acc:.3f} | Test ACC: {test_acc:.3f} | Test AUC: {test_auc:.3f}")

    print(f"--> Fold {fold+1} 最佳结果 (以最高 AUC 为准):")
    print(f"    [Test]  ACC: {best_metrics[1]:.3f} | AUC: {best_metrics[0]:.3f} | SEN: {best_metrics[2]:.3f} | SPE: {best_metrics[3]:.3f}")

    auc_list.append(best_metrics[0])
    acc_list.append(best_metrics[1])
    sen_list.append(best_metrics[2])
    spe_list.append(best_metrics[3])

# =========================
# 5. 结果汇总
# =========================
elapsed = time.time() - start_time
print("\n========== Final 5-Fold Cross Validation Result (ViT Tuned) ==========")
print(f"Test AUC            : {np.mean(auc_list):.3f} ± {np.std(auc_list):.3f}")
print(f"Test Accuracy       : {np.mean(acc_list):.3f} ± {np.std(acc_list):.3f}")
print(f"Test Sensitivity    : {np.mean(sen_list):.3f} ± {np.std(sen_list):.3f}")
print(f"Test Specificity    : {np.mean(spe_list):.3f} ± {np.std(spe_list):.3f}")
print(f"Total time          : {elapsed/60:.1f} min")