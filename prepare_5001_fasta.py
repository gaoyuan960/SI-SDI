# -*- coding: utf-8 -*-
import pandas as pd
from Bio import SeqIO
import time
import os

print("========== 开始数据抽样与拼接 (5000 + 1 标准株) ==========")
start_time = time.time()

# ==================== 路径与配置 ====================
# 全部使用相对路径，确保在当前目录下运行
csv_file = './df_IAV_8ORFs_clean_names_97736.csv'
std_fasta_file = './A_Puerto_Rico_8_1934_H1N1.fasta'
output_fasta = './sampled_5001_strains.fasta'

# 规定的8个片段顺序
segments_order = ['PB2', 'PB1', 'PA', 'HA', 'NP', 'NA', 'M1', 'NS1']

# ==================== Step 1: 处理 97736 条大库数据 ====================
print(f"1. 正在读取大库 CSV 文件: {csv_file}")
# 核心关键：keep_default_na=False 防止 Pandas 将 "NA" 识别为空值
df = pd.read_csv(csv_file, keep_default_na=False, dtype=str)

print(f"   - 总数据量: {len(df)} 条")
print("   - 正在进行随机抽样 (固定种子: 42, 抽取 5000 条)...")
df_sampled = df.sample(n=5000, random_state=42)

fasta_records = []
print("   - 正在按规定顺序拼接 8 个片段...")

# 遍历抽样的5000条数据并拼接
for index, row in df_sampled.iterrows():
    seq_id = str(row['strain_name']).strip()
    # 按照 segments_order 的顺序提取序列并拼接
    seq_concat = "".join([str(row[seg]).strip() for seg in segments_order])
    fasta_records.append((seq_id, seq_concat))

print(f"   - 成功处理 {len(fasta_records)} 条随机毒株序列。")

# ==================== Step 2: 处理 1 条标准株数据 ====================
print(f"\n2. 正在读取标准株 FASTA 文件: {std_fasta_file}")
std_segments_dict = {}

# 解析标准株的8条散装序列
if os.path.exists(std_fasta_file):
    for record in SeqIO.parse(std_fasta_file, "fasta"):
        header = record.description.upper()  # 转大写方便匹配
        # 智能匹配片段名称
        for seg in segments_order:
            if seg in header:
                std_segments_dict[seg] = str(record.seq).strip()
                break
    
    # 检查是否找齐了8个片段
    missing = [seg for seg in segments_order if seg not in std_segments_dict]
    if missing:
        print(f"   [警告] 标准株中未能找到以下片段: {missing}")
    
    # 按严格顺序拼接标准株
    std_seq_concat = "".join([std_segments_dict.get(seg, "") for seg in segments_order])
    std_id = "A_Puerto_Rico_8_1934_H1N1_Standard"
    
    fasta_records.append((std_id, std_seq_concat))
    print(f"   - 成功拼接标准株，长度为 {len(std_seq_concat)} bp。")
else:
    print(f"   [错误] 找不到标准株文件: {std_fasta_file}，请检查路径。")

# ==================== Step 3: 写入最终 FASTA 文件 ====================
print(f"\n3. 正在生成最终文件: {output_fasta}")
with open(output_fasta, 'w') as f:
    for seq_id, seq in fasta_records:
        # FASTA 格式标准写入
        f.write(f">{seq_id}\n{seq}\n")

end_time = time.time()
print(f"   - 最终文件包含 {len(fasta_records)} 条序列。")
print(f"========== 任务完成！总耗时: {end_time - start_time:.2f} 秒 ==========")