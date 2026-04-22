# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
import gc
from multiprocessing import Pool, cpu_count
import warnings
import shutil 

warnings.filterwarnings('ignore')

# ==================== 用户配置区域 (修改这里) ====================
# 在这里修改片段名称，例如: 'PB2', 'PB1', 'PA', 'HA', 'NP', 'NA', 'M', 'NS'
# 每次修改后保存为一个新的 .py 文件即可
TARGET_SEGMENT = 'PB2' 

# ==================== Configuration ====================
nt_table = ['t','c', 'a', 'g']
dnt_table = [nt1+nt2 for nt1 in nt_table for nt2 in nt_table]
dnts_table = [dnt1+dnt2 for dnt1 in dnt_table for dnt2 in dnt_table]
codon_table = [nt1+nt2+nt3 for nt1 in nt_table for nt2 in nt_table for nt3 in nt_table]

stop_codon = ['taa', 'tag', 'tga']
stop_codonpair = [i+j for i in stop_codon for j in stop_codon]

codonpair_table0 = [codon0 + codon1 for codon0 in codon_table for codon1 in codon_table]
codonpair_table = [cp for cp in codonpair_table0 if cp not in stop_codonpair]

dnt_category = ['n12', 'n23','n31']
dntpair_category = ['n12m12', 'n23m23','n31m31','n12n31','n23m12','n31m23']
dntpair_cols_list = ['Freq_'+ dnts + '_' + dnts_cat 
                      for dnts_cat in dntpair_category for dnts in dnts_table]

# ==================== Distance calculation ====================
def distance(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    cos_similarity = dot_product / (norm_vec1 * norm_vec2 + 1e-10)
    euc_distance = np.sqrt(np.sum((vec1-vec2) ** 2))
    return dot_product, cos_similarity, euc_distance

# ==================== Parallel worker ====================
def calculate_batch(args):
    """
    工作进程：计算并直接保存到该片段专属的临时文件夹
    """
    reference_array, batch_row_indices, batch_id, temp_dir = args
    
    num_reference = reference_array.shape[0]
    num_rows = len(batch_row_indices)
    
    # 预分配内存 (使用 float32 节省空间)
    dp_batch = np.zeros((num_rows, num_reference), dtype=np.float32)
    cs_batch = np.zeros((num_rows, num_reference), dtype=np.float32)
    ed_batch = np.zeros((num_rows, num_reference), dtype=np.float32)
    
    # 计算逻辑
    for local_idx, global_row_idx in enumerate(batch_row_indices):
        vec_i = reference_array[global_row_idx]
        for j in range(num_reference):
            vec_j = reference_array[j]
            dp, cs, ed = distance(vec_i, vec_j)
            dp_batch[local_idx, j] = dp
            cs_batch[local_idx, j] = cs
            ed_batch[local_idx, j] = ed
        
        # 每几行打印一次
        if local_idx % 1 == 0: 
            print(f"[{TARGET_SEGMENT}] Batch {batch_id+1}: Row {local_idx+1}/{num_rows}", flush=True)
    
    # 保存到专属临时文件夹
    dp_path = os.path.join(temp_dir, f'dp_batch_{batch_id}.npy')
    cs_path = os.path.join(temp_dir, f'cs_batch_{batch_id}.npy')
    ed_path = os.path.join(temp_dir, f'ed_batch_{batch_id}.npy')
    
    np.save(dp_path, dp_batch)
    np.save(cs_path, cs_batch)
    np.save(ed_path, ed_batch)
    
    del dp_batch, cs_batch, ed_batch
    gc.collect()

    return batch_id, batch_row_indices, dp_path, cs_path, ed_path

# ==================== Main function ====================
def main():
    seg = TARGET_SEGMENT
    print(f"========== Starting Processing for Segment: {seg} ==========")

    # 路径配置
    path = '../data/counting/'
    
    # 关键修改：临时文件夹名称包含片段名，避免冲突
    temp_dir = f'./temp_batches_{seg}/'
    
    # 输出路径
    output_path = './results_full_matrix/'
    
    # 初始化文件夹
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
        except OSError as e:
            print(f"Error cleaning temp dir {temp_dir}: {e}")
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    # 读取数据
    file = f'df_dcrcp_counting_{seg}_df_IAV_8ORFs_deduplicated_labels_98787.csv'
    full_file_path = os.path.join(path, file)
    
    if not os.path.exists(full_file_path):
        print(f"Error: File not found {full_file_path}")
        return

    df = pd.read_csv(full_file_path, index_col='seqID')
    
    # 提取特征
    df_reference = df[dntpair_cols_list]
    reference_ids = df_reference.index.tolist()
    reference_array = df_reference.values.astype(np.float32)
    
    total_rows = len(reference_ids)
    print(f"[{seg}] Total sequences: {total_rows}")
    print(f"[{seg}] Feature dimension: {reference_array.shape[1]}")
    
    # 分批配置
    NUM_BATCHES = 10
    batch_size = total_rows // NUM_BATCHES
    remainder = total_rows % NUM_BATCHES
    
    batches = []
    start = 0
    for i in range(NUM_BATCHES):
        end = start + batch_size + (1 if i < remainder else 0)
        batches.append(range(start, end))
        start = end
    
    # 准备任务
    tasks = [(reference_array, batch_indices, i, temp_dir) 
             for i, batch_indices in enumerate(batches)]
    
    # 并行计算
    # 注意：如果你同时运行8个脚本，建议限制每个脚本使用的CPU核数，
    max_workers_per_script = 15 
    num_workers = min(NUM_BATCHES, cpu_count(), max_workers_per_script)
    
    print(f"\n[{seg}] Starting {num_workers} workers...")
    
    with Pool(processes=num_workers) as pool:
        results_metadata = pool.map(calculate_batch, tasks)
    
    results_metadata.sort(key=lambda x: x[0])
    
    print(f"\n[{seg}] Calculation done. Merging files...")
    
    # 合并结果函数
    def merge_and_save(metric_idx, out_file):
        print(f"[{seg}] Merging into {os.path.basename(out_file)}...")
        first_batch = True
        
        for batch_info in results_metadata:
            batch_id = batch_info[0]
            row_indices = batch_info[1]
            npy_path = batch_info[2 + metric_idx]
            
            data_chunk = np.load(npy_path)
            chunk_index = [reference_ids[i] for i in row_indices]
            df_chunk = pd.DataFrame(data=data_chunk, columns=reference_ids, index=chunk_index)
            
            if first_batch:
                df_chunk.to_csv(out_file, mode='w', header=True)
                first_batch = False
            else:
                df_chunk.to_csv(out_file, mode='a', header=False)
            
            del data_chunk, df_chunk
            gc.collect()

    # 执行合并
    out_dp = os.path.join(output_path, f'df_dot_product_{seg}_full.csv')
    out_cs = os.path.join(output_path, f'df_cos_similarity_{seg}_full.csv')
    out_ed = os.path.join(output_path, f'df_euc_distance_{seg}_full.csv')
    
    merge_and_save(0, out_dp)
    merge_and_save(1, out_cs)
    merge_and_save(2, out_ed)
    
    # 清理临时文件夹
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    print(f"========== Completed: {seg} ==========")

if __name__ == '__main__':
    main()