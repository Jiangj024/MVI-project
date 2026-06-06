# -*- coding: utf-8 -*-
"""
纯数据处理与匹配校验脚本 (读取 xlsx 版)
用于验证临床表格数据与图像数据是否能够完美一一对应
"""

import os
import pandas as pd
import numpy as np

# ==========================================
# 1. 核心路径配置 (必须用从 /home 开始的绝对路径！)
# ==========================================
# 换成了你服务器上真实的 .xlsx 文件名
POS_EXCEL_PATH = "/home/fuxiangyu/jlx/MVI/MVI_ceus0325/0325data/MVI1.xlsx"
# 假设你的阴性文件叫 MVI0.xlsx，如果不是，请务必在这里改成真实名字！
NEG_EXCEL_PATH = "/home/fuxiangyu/jlx/MVI/MVI_ceus0325/0325data/MVI0.xlsx" 
IMAGE_ROOT_DIR = "/home/fuxiangyu/jlx/MVI/MVI_ceus0325/0325processed"

# 真实的文件夹名称映射
FOLDER_MAPPING = {
    '0_MVI_Negative': 0,
    '1_MVI_Positive': 1
}

# ==========================================
# 2. 临床表格清洗函数
# ==========================================
def clean_clinical_data(pos_path, neg_path):
    print("▶️ 开始读取并清洗临床 Excel 表格数据...")
    
    # 【核心修改点】：使用 read_excel 读取 .xlsx 文件
    df_pos = pd.read_excel(pos_path)
    df_neg = pd.read_excel(neg_path)
    
    # 强制统一列名（解决两张表表头不一致的问题，假设前22列顺序一致）
    standard_columns = [
        '超声号', '性别', '年龄', 'HBV', 'HCV', '总胆红素', '直接胆红素', 
        '总蛋白', '白蛋白', 'ALT', 'AST', '碱性磷酸酶', '谷氨酰转移酶', 
        '总胆酸', '乳酸脱氢酶', '甲胎蛋白', '癌胚抗原', 'CA199', 'CA125', 
        '甲胎异质体', '异常凝血酶原', '凝血酶原时间'
    ]
    
    # 截取前22列并重命名
    df_pos = df_pos.iloc[:, :22]
    df_pos.columns = standard_columns
    
    df_neg = df_neg.iloc[:, :22]
    df_neg.columns = standard_columns
    
    # 合并表格
    df_clinical = pd.concat([df_pos, df_neg], ignore_index=True)
    
    # 数据清洗
    df_clinical.fillna(0, inplace=True) # 缺失值补0
    df_clinical['性别'] = df_clinical['性别'].map({'男': 1, '女': 0}) # 性别数字化
    
    # 年龄标准化 (Z-score)
    mean_age = df_clinical['年龄'].mean()
    std_age = df_clinical['年龄'].std()
    df_clinical['年龄'] = (df_clinical['年龄'] - mean_age) / std_age
    
    # 转换为快速查询字典: { 'ZS12345678': [特征1, 特征2, ...] }
    feature_cols = [col for col in standard_columns if col != '超声号']
    clinical_dict = {}
    for _, row in df_clinical.iterrows():
        uid = str(row['超声号']).strip()
        features = row[feature_cols].values.astype(np.float32)
        clinical_dict[uid] = features
        
    print(f"✅ 表格处理完毕！共提取了 {len(clinical_dict)} 位患者的临床资料。")
    print(f"   提取的临床特征维度为: {len(feature_cols)} 维\n")
    return clinical_dict, feature_cols

# ==========================================
# 3. 图像与表格匹配函数
# ==========================================
def match_images_and_clinical(clinical_dict, feature_cols):
    print("▶️ 开始扫描图像文件夹并进行数据匹配...")
    
    matched_samples = []
    missing_patients = set()
    total_images_scanned = 0
    
    for folder_name, label in FOLDER_MAPPING.items():
        folder_path = os.path.join(IMAGE_ROOT_DIR, folder_name)
        
        if not os.path.isdir(folder_path):
            print(f"❌ 严重错误: 找不到图像文件夹 {folder_path}")
            continue
            
        for img_name in os.listdir(folder_path):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            total_images_scanned += 1
            
            # 提取超声号 ZSxxxxxx
            patient_id = img_name.split('_')[0]
            
            # 尝试连线
            if patient_id in clinical_dict:
                matched_samples.append({
                    'image_name': img_name,
                    'patient_id': patient_id,
                    'label': label,
                    'clinical_features': clinical_dict[patient_id]
                })
            else:
                missing_patients.add(patient_id)
                
    # ==========================================
    # 4. 打印匹配诊断报告
    # ==========================================
    print("=" * 50)
    print("📊 数据匹配最终诊断报告 📊")
    print("=" * 50)
    print(f"📁 扫描到的有效图像总数: {total_images_scanned} 张")
    print(f"🔗 成功匹配到化验单的图像: {len(matched_samples)} 张")
    
    if len(missing_patients) > 0:
        print(f"⚠️  警告: 有 {len(missing_patients)} 位患者在 Excel 表格中找不到对应数据！")
        print(f"   缺失的超声号示例: {list(missing_patients)[:5]}")
    else:
        print("🎉 完美匹配！所有的图像都找到了对应的临床表格数据！")
        
    if len(matched_samples) > 0:
        print("\n🔍 抽查第一个成功匹配的样本：")
        sample = matched_samples[0]
        print(f"   - 图片文件名: {sample['image_name']}")
        print(f"   - 提取的患者 ID: {sample['patient_id']}")
        print(f"   - MVI 真实标签: {'阳性(1)' if sample['label'] == 1 else '阴性(0)'}")
        print(f"   - 前 5 个临床特征 (性别,标准化年龄,HBV,HCV,胆红素...): {sample['clinical_features'][:5]}")
    print("=" * 50)

if __name__ == "__main__":
    c_dict, f_cols = clean_clinical_data(POS_EXCEL_PATH, NEG_EXCEL_PATH)
    match_images_and_clinical(c_dict, f_cols)