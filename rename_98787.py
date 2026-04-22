import pandas as pd
import re

# ================= 配置区域 =================
INPUT_FILE = 'df_IAV_8ORFs_deduplicated_labels_98787.csv'
OUTPUT_FILE = 'df_IAV_8ORFs_clean_names_98787.csv' # 输出文件名
TARGET_COL = 'strain_name' # 目标列名
# ===========================================

def clean_strain_name(name):
    """
    方案一清洗逻辑：
    1. 去除首尾空格
    2. 空格 -> 下划线
    3. 删除 括号, 单引号, 逗号
    """
    if pd.isna(name):
        return "Unknown"
    
    # 强转为字符串
    name = str(name)
    
    # 1. 去除首尾空白符
    new_name = name.strip()
    
    # 2. 替换中间空格为下划线
    new_name = new_name.replace(' ', '_')
    
    # 3. 删除特殊字符：括号() 单引号' 逗号,
    # 注意：保留了斜杠/ 和 连字符-
    new_name = re.sub(r"[()',]", "", new_name)
    
    # 4. (可选) 处理连续的下划线，避免 A__B 这种情况
    new_name = re.sub(r"_+", "_", new_name)
    
    return new_name

def make_unique(names):
    """
    对列表中的名字进行去重。
    如果发现重复，添加后缀 _1, _2 等。
    """
    seen = {}
    result = []
    
    for name in names:
        if name in seen:
            seen[name] += 1
            # 生成新名字，例如 Name_1
            new_name = f"{name}_{seen[name]}"
            result.append(new_name)
        else:
            seen[name] = 0
            result.append(name)
            
    return result, seen

# 1. 读取数据
print(f"正在读取文件: {INPUT_FILE} ...")
try:
    df = pd.read_csv(INPUT_FILE)
except FileNotFoundError:
    print(f"❌ 错误：在当前目录下找不到文件 {INPUT_FILE}")
    print("请确认文件名或路径是否正确。")
    exit()

# 检查列是否存在
if TARGET_COL not in df.columns:
    # 尝试按索引访问，如果是第二列
    print(f"⚠️ 警告：列名 '{TARGET_COL}' 未找到。正在尝试使用第二列作为目标列...")
    target_col_name = df.columns[1] # 索引1是第二列
else:
    target_col_name = TARGET_COL

# 2. 执行清洗
print(f"正在清洗列: {target_col_name} ...")
# 备份原始名字（可选，方便核对）
df['original_strain_name'] = df[target_col_name]

# 应用清洗函数
df['clean_name_temp'] = df[target_col_name].apply(clean_strain_name)

# 3. 执行查重与重命名
print("正在检查重名并进行唯一化处理...")
unique_names, seen_dict = make_unique(df['clean_name_temp'].tolist())

# 统计有多少名字被修改了后缀
duplicates_count = sum(1 for k, v in seen_dict.items() if v > 0)
if duplicates_count > 0:
    print(f"ℹ️ 发现 {duplicates_count} 个清洗后重名的条目，已自动添加后缀区分（如 _1, _2）。")
else:
    print("✅ 完美！清洗后无重名冲突。")

# 将最终的唯一名字赋值回目标列（或者新列，取决于您的需求，这里建议覆盖或存新列）
# 为了建树方便，通常需要将其作为主要的ID列
df['strain_name_clean'] = unique_names

# 如果您想直接覆盖原列，取消下面这行的注释：
# df[target_col_name] = unique_names

# 4. 保存结果
print(f"正在保存结果到: {OUTPUT_FILE} ...")
df.to_csv(OUTPUT_FILE, index=False)

print("-" * 30)
print("处理完成！预览前5个清洗结果：")
print("-" * 30)
# 展示对比
print(df[['original_strain_name', 'strain_name_clean']].head())
print("-" * 30)
print(f"✅ 文件已保存，请使用 '{OUTPUT_FILE}' 的 'strain_name_clean' 列进行建树。")