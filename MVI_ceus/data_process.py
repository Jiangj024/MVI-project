import os
import json
import glob
from PIL import Image

# ==========================================
# 1. 路径配置（请根据你的实际路径修改）
# ==========================================
RAW_DATA_DIR = "/home/fuxiangyu/jlx/MVI/MVI_ceus"  # 包含 MVI+ 和 MVI- 的原始文件夹
OUTPUT_DIR = "/home/fuxiangyu/jlx/MVI/MVI_ceus/MVI_processed" # 裁剪后的输出文件夹

# 定义类别映射
CLASS_MAP = {
    "MVI-": "0_MVI_Negative",
    "MVI+": "1_MVI_Positive"
}

# 创建输出目录
for out_folder in CLASS_MAP.values():
    os.makedirs(os.path.join(OUTPUT_DIR, out_folder), exist_ok=True)

# ==========================================
# 2. 遍历并裁剪数据
# ==========================================
processed_count = 0
error_count = 0

# 支持的图片后缀名大全（包含大小写）
VALID_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']

for raw_class_folder, out_class_folder in CLASS_MAP.items():
    current_dir = os.path.join(RAW_DATA_DIR, raw_class_folder)
    
    # 找到当前目录下所有的 json 标注文件
    json_files = glob.glob(os.path.join(current_dir, "*.json"))
    
    for json_path in json_files:
        try:
            # 1. 解析 JSON 找到对应的坐标
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            points = data['shapes'][0]['points']
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            
            x1, y1 = min(x_coords), min(y_coords)
            x2, y2 = max(x_coords), max(y_coords)
            
            # 2. 获取没有后缀的文件基名，例如 ZS10168702_lp_sonovue
            base_name = os.path.splitext(os.path.basename(json_path))[0]
            
            # 3. 自动匹配图片实际的后缀名
            img_path = None
            for ext in VALID_EXTENSIONS:
                temp_path = os.path.join(current_dir, base_name + ext)
                if os.path.exists(temp_path):
                    img_path = temp_path
                    break
            
            # 如果遍历了所有常见后缀都没找到对应的图片，则报错跳过
            if img_path is None:
                print(f"警告: 找不到对应的图片 {base_name}.[jpg/JPG/png 等]")
                error_count += 1
                continue
                
            # 4. 读取、裁剪并保存
            img = Image.open(img_path).convert('RGB')
            roi_img = img.crop((x1, y1, x2, y2))
            
            # 统一构造新的文件名为小写的 _crop.jpg
            new_img_name = base_name + '_crop.jpg'
            save_path = os.path.join(OUTPUT_DIR, out_class_folder, new_img_name)
            
            roi_img.save(save_path)
            processed_count += 1
            
        except Exception as e:
            print(f"处理文件 {json_path} 时出错: {e}")
            error_count += 1

print("\n========== 数据预处理完成 ==========")
print(f"成功裁剪并保存: {processed_count} 张图片")
print(f"处理失败/跳过: {error_count} 张图片")
print(f"处理后的数据集已保存在: {OUTPUT_DIR}")