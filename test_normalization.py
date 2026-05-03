# -*- coding: utf-8 -*-
"""
测试归一化修复是否正确
"""
import pandas as pd
import numpy as np

POS_EXCEL_PATH = "/home/fuxiangyu/jlx/MVI/MVI_ceus0325/0325data/MVI阳性临床资料-原始数值.xlsx"
NEG_EXCEL_PATH = "/home/fuxiangyu/jlx/MVI/MVI_ceus0325/0325data/MVI阴性临床资料-原始数值.xlsx"

def test_normalization():
    print("=" * 60)
    print("测试临床特征归一化")
    print("=" * 60)

    df_pos = pd.read_excel(POS_EXCEL_PATH)
    df_neg = pd.read_excel(NEG_EXCEL_PATH)

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

    print(f"\n总样本数: {len(df_clinical)}")
    print(f"阳性样本: {len(df_pos)}, 阴性样本: {len(df_neg)}")

    # 性别编码
    df_clinical['性别'] = df_clinical['性别'].map({'男': 1, '女': 0})

    binary_features = ['性别', 'HBV', 'HCV']
    continuous_features = [
        '年龄', '总胆红素', '直接胆红素', '总蛋白', '白蛋白',
        'ALT', 'AST', '碱性磷酸酶', '谷氨酰转移酶', '总胆酸',
        '乳酸脱氢酶', '甲胎蛋白', '癌胚抗原', 'CA199', 'CA125',
        '甲胎异质体', '异常凝血酶原', '凝血酶原时间'
    ]

    print("\n" + "=" * 60)
    print("归一化前的数据统计")
    print("=" * 60)

    # 显示归一化前的统计信息
    print("\n连续型特征（归一化前）:")
    for col in continuous_features[:5]:  # 只显示前5个
        print(f"  {col}: 均值={df_clinical[col].mean():.2f}, 标准差={df_clinical[col].std():.2f}, 范围=[{df_clinical[col].min():.2f}, {df_clinical[col].max():.2f}]")

    # 用中位数填充缺失值
    for col in continuous_features:
        median_value = df_clinical[col].median()
        df_clinical[col].fillna(median_value, inplace=True)

    for col in binary_features:
        df_clinical[col].fillna(0, inplace=True)

    # Z-score 标准化
    for col in continuous_features:
        mean_val = df_clinical[col].mean()
        std_val = df_clinical[col].std()
        df_clinical[col] = (df_clinical[col] - mean_val) / (std_val + 1e-8)

    print("\n" + "=" * 60)
    print("归一化后的数据统计")
    print("=" * 60)

    print("\n连续型特征（归一化后）:")
    for col in continuous_features[:5]:
        print(f"  {col}: 均值={df_clinical[col].mean():.6f}, 标准差={df_clinical[col].std():.6f}, 范围=[{df_clinical[col].min():.2f}, {df_clinical[col].max():.2f}]")

    print("\n二分类特征（保持不变）:")
    for col in binary_features:
        print(f"  {col}: 唯一值={sorted(df_clinical[col].unique())}")

    print("\n" + "=" * 60)
    print("✅ 归一化测试完成")
    print("=" * 60)

if __name__ == "__main__":
    test_normalization()
