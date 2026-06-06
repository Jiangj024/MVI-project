import os
from PIL import Image
from collections import Counter

# ==========================================
# 路径配置
# ==========================================
DATA_ROOT = "MVI_ceus/MVI_processed"

def analyze_dataset(data_dir):
    if not os.path.exists(data_dir):
        print(f"错误: 找不到目录 {data_dir}")
        return

    classes = sorted(os.listdir(data_dir))
    total_images = 0
    class_counts = {}
    size_distribution = Counter()
    
    print("\n========== 数据集分布统计 ==========")
    
    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
            
        images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        count = len(images)
        class_counts[cls] = count
        total_images += count
        
        # 抽样检查一下图片尺寸（检查前 50 张即可，加快速度）
        for img_name in images[:50]:
            try:
                img_path = os.path.join(cls_dir, img_name)
                with Image.open(img_path) as img:
                    size_distribution[img.size] += 1
            except Exception as e:
                print(f"警告: 无法读取图片 {img_name} - {e}")
                
        print(f"类别 [{cls}]: {count} 张")

    print("------------------------------------")
    print(f"总图片数: {total_images} 张")
    
    if len(class_counts) == 2:
        cls_names = list(class_counts.keys())
        ratio = class_counts[cls_names[0]] / (class_counts[cls_names[1]] + 1e-6)
        print(f"正负样本比例 ({cls_names[0]} : {cls_names[1]}): {ratio:.2f}")
        
        if ratio > 2.0 or ratio < 0.5:
            print("⚠️ 警告: 数据集存在较为严重的类别不平衡！")
    
    print("\n[抽样尺寸分布 (Top 3)]:")
    for size, freq in size_distribution.most_common(3):
        print(f" - 分辨率 {size[0]}x{size[1]}: 约占比")

if __name__ == "__main__":
    analyze_dataset(DATA_ROOT)