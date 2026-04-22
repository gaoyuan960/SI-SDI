import h5py
import numpy as np
import pandas as pd
import os
import re
import warnings

# 忽略 Pandas 的 FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

# ==================== 配置区域 ====================
BASE_DIR = r"./" 

INPUT_H5 = os.path.join(BASE_DIR, "similarity", "IAV_similarity_db.h5")
# 为了不影响你后续的代码路径，输出文件名保持不变，但内核已经升级
OUTPUT_MINI_H5 = os.path.join(BASE_DIR, "data", "IAV_mini_stratified_capped.h5")
CLEAN_META_CSV = os.path.join(BASE_DIR, "data", "df_IAV_8ORFs_deduplicated_labels_98787_cleaned.csv")

TARGET_SIZE = 10000 
RANDOM_SEED = 42

# ==================== 工具函数 ====================

def normalize_name(name):
    if pd.isna(name): return ""
    return re.sub(r'[^a-z0-9]', '', str(name).lower())

def get_capped_indices(years_array, sero_array, target_size):
    """
    顶级演化算法：年份-亚型联合封顶抽样 (Capped Spatiotemporal Sampling)
    """
    print("   📊 Calculating CAPPED proportions (Year + Serotype)...")
    
    # 清洗年份
    years_clean = np.nan_to_num(years_array, nan=-1)
    years_int = np.floor(years_clean).astype(int)
    
    # 构建抽样池
    df_pool = pd.DataFrame({
        'original_index': np.arange(len(years_int)),
        'year': years_int,
        'serotype': sero_array
    })
    
    # 按照 年份 和 亚型 联合分组
    groups = df_pool.groupby(['year', 'serotype'])
    counts = groups.size()
    
    # 智能寻找封顶阈值 k：使得 sum(min(count, k)) 逼近 target_size
    k = 1
    while sum(np.minimum(counts, k)) < target_size:
        k += 1
        if k > counts.max(): 
            break
            
    print(f"      🎯 Calculated Cap (k): Maximum {k} sequences per Year-Serotype combination.")
    
    # 执行封顶抽样
    try:
        sampled_df = groups.apply(
            lambda x: x.sample(n=min(len(x), k), random_state=RANDOM_SEED), 
            include_groups=False
        ).reset_index(drop=True)
    except TypeError:
        sampled_df = groups.apply(
            lambda x: x.sample(n=min(len(x), k), random_state=RANDOM_SEED)
        ).reset_index(drop=True)
    
    # 精度微调：确保恰好是 10000 条
    current_size = len(sampled_df)
    if current_size > target_size:
        sampled_df = sampled_df.sample(n=target_size, random_state=RANDOM_SEED)
    elif current_size < target_size:
        # 如果不够，从没被抽中的池子里再随机补齐
        remaining = df_pool[~df_pool['original_index'].isin(sampled_df['original_index'])]
        needed = target_size - current_size
        if len(remaining) > 0:
            add_on = remaining.sample(n=min(needed, len(remaining)), random_state=RANDOM_SEED)
            sampled_df = pd.concat([sampled_df, add_on])
            
    print(f"      Final Sample Count: {len(sampled_df)}")
    return np.sort(sampled_df['original_index'].values)

# ==================== 主逻辑 ====================

def main():
    print(f"📉 Starting CAPPED Downsampling (Evolutionary Topology Mode)...")
    
    if not os.path.exists(CLEAN_META_CSV):
        print(f"❌ Error: Cleaned metadata not found at {CLEAN_META_CSV}")
        return
    if not os.path.exists(INPUT_H5):
        print(f"❌ Error: Input H5 not found at {INPUT_H5}")
        return

    print("   Reading correct metadata (Dates & Serotypes)...")
    df_meta = pd.read_csv(CLEAN_META_CSV, low_memory=False)
    df_meta['match_key'] = df_meta['strain_name'].apply(normalize_name)
    df_meta['Year'] = pd.to_numeric(df_meta['Year'], errors='coerce')
    
    # 构建映射字典
    date_map = dict(zip(df_meta['match_key'], df_meta['Year']))
    # 注意：根据你之前的数据，你的亚型列叫 Serotype
    sero_map = dict(zip(df_meta['match_key'], df_meta['Serotype']))

    with h5py.File(INPUT_H5, 'r') as f_in:
        raw_names = f_in['strain_names'][:]
        full_names = [n.decode('utf-8') if isinstance(n, bytes) else str(n) for n in raw_names]
        
        print("   Aligning parameters to H5 backbone...")
        aligned_years = np.array([date_map.get(normalize_name(name), np.nan) for name in full_names])
        aligned_seros = np.array([sero_map.get(normalize_name(name), 'Unknown') for name in full_names])
        
        # 传入年份和亚型，执行联合封顶抽样
        sample_indices = get_capped_indices(aligned_years, aligned_seros, TARGET_SIZE)
        
        if os.path.exists(OUTPUT_MINI_H5): 
            os.remove(OUTPUT_MINI_H5)
            
        with h5py.File(OUTPUT_MINI_H5, 'w') as f_out:
            # 解决 UTF-8 编码报错
            sampled_names = np.array(full_names)[sample_indices]
            sampled_names_encoded = [s.encode('utf-8') for s in sampled_names]
            
            f_out.create_dataset('strain_names', data=sampled_names_encoded)
            f_out.create_dataset('years', data=aligned_years[sample_indices])
            
            segments = ['PB2', 'PB1', 'PA', 'HA', 'NP', 'NA', 'M1', 'NS1']
            for seg in segments:
                print(f"   ✂️ Slicing matrix for {seg}...")
                full_matrix = f_in[f'{seg}/data'][:]
                mini_matrix = full_matrix[sample_indices, :][:, sample_indices]
                
                grp = f_out.create_group(seg)
                grp.create_dataset('data', data=mini_matrix, compression="gzip", compression_opts=4)
                for k, v in f_in[f'{seg}/data'].attrs.items(): 
                    grp['data'].attrs[k] = v
                    
                del full_matrix, mini_matrix

    print(f"\n🎉 Capped Matrix Generation Complete! Output file: {OUTPUT_MINI_H5}")

if __name__ == '__main__':
    main()