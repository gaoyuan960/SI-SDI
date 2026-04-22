import pandas as pd
import numpy as np
import os
import h5py
import gc
from sklearn.decomposition import PCA
from datetime import datetime, timedelta
import warnings
import multiprocessing
import time
import shutil

# 忽略警告
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================

# 服务器路径
BASE_DIR = "/media/hw/Getea/gy_98787"
INPUT_DIR = os.path.join(BASE_DIR, "distance", "results_full_matrix")
METADATA_FILE = os.path.join(BASE_DIR, "data", "df_IAV_8ORFs_deduplicated_labels_98787.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "similarity")
OUTPUT_H5 = os.path.join(OUTPUT_DIR, "IAV_similarity_db.h5")
TEMP_DIR = os.path.join(OUTPUT_DIR, "temp_segments") # 临时文件目录

# 确保目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# 片段列表
SEGMENTS = ['PB2', 'PB1', 'PA', 'HA', 'NP', 'NA', 'M1', 'NS1']

# 并行设置：3 个进程（IO密集型最优）
WORKER_COUNT = 3 

# ==================== DATE PARSING UTILS ====================

def parse_dates_vectorized(date_series):
    """高性能日期转换"""
    def convert_single(val):
        if pd.isna(val) or val == '':
            return np.nan
        s_val = str(val).strip()
        
        # 1. Excel 序列号
        if s_val.replace('.', '', 1).isdigit():
            try:
                num_val = float(s_val)
                if 20000 < num_val < 70000: 
                    dt = datetime(1899, 12, 30) + timedelta(days=num_val)
                    days_in_year = 366 if (dt.year % 4 == 0 and dt.year % 100 != 0) or (dt.year % 400 == 0) else 365
                    return dt.year + (dt.timetuple().tm_yday - 1) / days_in_year
                elif 1900 <= num_val <= 2030:
                    return float(num_val)
            except:
                pass

        # 2. 标准日期
        for fmt in ('%Y-%m-%d', '%m/%d/%Y', '%Y/%m/%d', '%Y.%m.%d', '%Y'):
            try:
                dt = datetime.strptime(s_val, fmt)
                days_in_year = 366 if (dt.year % 4 == 0 and dt.year % 100 != 0) or (dt.year % 400 == 0) else 365
                return dt.year + (dt.timetuple().tm_yday - 1) / days_in_year
            except ValueError: continue
            
        return np.nan

    return date_series.apply(convert_single).astype('float32')

# ==================== WORKER PROCESS LOGIC ====================

def get_paths(seg):
    return {
        'cos': os.path.join(INPUT_DIR, f"df_cos_similarity_{seg}_full.csv"),
        'dot': os.path.join(INPUT_DIR, f"df_dot_product_{seg}_full.csv"),
        'euc': os.path.join(INPUT_DIR, f"df_euc_distance_{seg}_full.csv")
    }

def read_large_csv(path):
    print(f"    [IO] Reading {os.path.basename(path)}...")
    try:
        # 保持 low_memory=False 和 engine='c' 以利用大内存
        return pd.read_csv(path, index_col=0, engine='c', low_memory=False)
    except Exception as e:
        print(f"❌ Error reading {path}: {e}")
        return None

def process_single_segment_task(seg):
    """
    V4 核心修改：算完直接写入临时 H5 文件，不返回大数据。
    """
    temp_h5_path = os.path.join(TEMP_DIR, f"temp_{seg}.h5")
    
    try:
        print(f"\n🚀 [Process {os.getpid()}] Starting Segment: {seg}")
        start_time = time.time()
        paths = get_paths(seg)
        
        # 1. 依次读取
        df_cos = read_large_csv(paths['cos'])
        if df_cos is None: return None
        
        current_index_head = df_cos.index[:100].tolist()
        
        # 安全转换
        mat_cos = np.clip(df_cos.values, 0, 1).astype(np.float32)
        del df_cos
        
        df_dot = read_large_csv(paths['dot'])
        mat_dot = df_dot.values.astype(np.float32)
        del df_dot
        
        df_euc = read_large_csv(paths['euc'])
        mat_euc = df_euc.values.astype(np.float32)
        del df_euc
        
        gc.collect() 
        
        print(f"    [Process {os.getpid()}] Computing {seg} Statistics...")
        
        # 2. 归一化
        dot_min, dot_max = mat_dot.min(), mat_dot.max()
        dot_denom = dot_max - dot_min if (dot_max - dot_min) != 0 else 1e-10
        mat_dot = (mat_dot - dot_min) / dot_denom
        
        euc_min, euc_max = mat_euc.min(), mat_euc.max()
        euc_denom = euc_max - euc_min if (euc_max - euc_min) != 0 else 1e-10
        mat_euc = (euc_max - mat_euc) / euc_denom 
        
        # 3. PCA
        print(f"    [Process {os.getpid()}] Fitting PCA for {seg}...")
        rng = np.random.RandomState(42)
        sample_size = min(500000, mat_cos.size)
        sample_indices = rng.choice(mat_cos.size, size=sample_size, replace=False)
        
        X_sample = np.column_stack((
            mat_cos.flat[sample_indices],
            mat_dot.flat[sample_indices],
            mat_euc.flat[sample_indices]
        ))
        
        pca = PCA(n_components=1).fit(X_sample)
        weights = np.abs(pca.components_[0])
        weights /= weights.sum()
        print(f"    [Process {os.getpid()}] Weights {seg}: {weights}")
        
        # 4. 合并
        mat_cos *= weights[0]
        mat_cos += (mat_dot * weights[1])
        mat_cos += (mat_euc * weights[2])
        
        del mat_dot, mat_euc
        gc.collect()
        
        # 5. [V4 核心] 子进程直接写入临时文件，规避 4GB 传输限制
        print(f"💾 [Process {os.getpid()}] Saving temp H5 for {seg}...")
        with h5py.File(temp_h5_path, 'w') as f_temp:
            grp = f_temp.create_group(seg)
            # 压缩写入
            grp.create_dataset('data', data=mat_cos, compression="gzip", compression_opts=4)
            grp.attrs['weights'] = weights
            grp.attrs['norm_params'] = [dot_min, dot_max, euc_min, euc_max]
            
        del mat_cos
        gc.collect()
        
        elapsed = time.time() - start_time
        print(f"✅ [Process {os.getpid()}] Segment {seg} Saved to disk in {elapsed/60:.1f} mins.")
        
        # 只返回元数据和成功状态
        return {
            'seg': seg,
            'status': 'success',
            'temp_path': temp_h5_path,
            'index_head': current_index_head
        }
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR in worker {seg}: {e}")
        return {'seg': seg, 'status': 'failed', 'error': str(e)}

# ==================== MAIN PROCESS UTILS ====================

def merge_temp_to_final(final_h5, result, ref_index_head=None):
    """将临时文件合并入最终文件"""
    seg = result['seg']
    temp_path = result['temp_path']
    
    if result['status'] != 'success':
        print(f"⚠️ Skipping merge for {seg} due to failure.")
        return

    # 索引校验
    if ref_index_head and result['index_head'] != ref_index_head:
        print(f"⚠️ WARNING: Index mismatch in {seg}!")
    
    print(f"📦 Merging {seg} from temp file to final DB...")
    
    # 打开源文件和目标文件进行复制
    with h5py.File(temp_path, 'r') as f_src, h5py.File(final_h5, 'a') as f_dst:
        # 使用 h5py 的 copy 方法，高效复制 Group
        f_src.copy(seg, f_dst)
        print(f"🎉 Merged {seg} successfully.")
    
    # 删除临时文件以释放空间
    try:
        os.remove(temp_path)
    except:
        pass

def process_metadata(h5_file):
    """元数据处理"""
    print("\n[Metadata] Processing Metadata...")
    sample_csv = get_paths(SEGMENTS[0])['cos']
    print(f"  Reading reference index from {os.path.basename(sample_csv)}...")
    df_index = pd.read_csv(sample_csv, usecols=[0], index_col=0, engine='c')
    full_index = df_index.index
    
    print("  Reading Metadata CSV...")
    df_meta = pd.read_csv(METADATA_FILE, keep_default_na=False)
    
    df_meta.columns = df_meta.columns.str.strip()
    name_col = None
    if 'strain_name' in df_meta.columns:
        name_col = 'strain_name'
    elif 'Unnamed: strain_name' in df_meta.columns:
        name_col = 'Unnamed: strain_name'
    
    if name_col:
        df_meta['strain_name_clean'] = df_meta[name_col].astype(str).str.strip()
    else:
        df_meta['strain_name_clean'] = df_meta.iloc[:, 1].astype(str).str.strip()
        
    print("  Parsing dates...")
    df_meta['DecYear'] = parse_dates_vectorized(df_meta['Date'])
    
    print("  Aligning metadata...")
    date_map = dict(zip(df_meta['strain_name_clean'], df_meta['DecYear']))
    aligned_years = []
    
    for name in full_index:
        val = date_map.get(str(name).strip(), np.nan)
        aligned_years.append(val)
            
    with h5py.File(h5_file, 'a') as f:
         dt_str = h5py.special_dtype(vlen=str)
         if 'strain_names' in f: del f['strain_names']
         f.create_dataset('strain_names', data=full_index.astype(str), dtype=dt_str)
         if 'years' in f: del f['years']
         f.create_dataset('years', data=np.array(aligned_years, dtype='float32'))
    print("✅ Metadata Saved.")

# ==================== ENTRY POINT ====================

if __name__ == '__main__':
    print(f"🔥 High-Performance Parallel Processing V4 (Limit-Bypass Mode)")
    print(f"   Output File: {OUTPUT_H5}")
    print(f"   Temp Dir: {TEMP_DIR}")
    
    # 清理工作
    if os.path.exists(OUTPUT_H5): os.remove(OUTPUT_H5)
    if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # 创建主文件
    with h5py.File(OUTPUT_H5, 'w') as f:
        pass 
    
    pool = multiprocessing.Pool(processes=WORKER_COUNT)
    
    tasks = []
    for seg in SEGMENTS:
        tasks.append(pool.apply_async(process_single_segment_task, args=(seg,)))
    
    pool.close() 
    
    ref_index_head = None
    
    # 循环收集结果
    for i, task in enumerate(tasks):
        try:
            result = task.get() # 这里返回的只是状态字符串，只有几百字节，绝不会报错
            if result and result['status'] == 'success':
                if ref_index_head is None:
                    ref_index_head = result['index_head']
                # 合并临时文件
                merge_temp_to_final(OUTPUT_H5, result, ref_index_head)
            else:
                print(f"❌ A task failed: {result.get('error') if result else 'Unknown'}")
        except Exception as e:
            print(f"❌ Main process error receiving task: {e}")
            
    pool.join()
    
    # 处理元数据
    process_metadata(OUTPUT_H5)
    
    # 清理临时目录
    try:
        shutil.rmtree(TEMP_DIR)
    except:
        pass
    
    print("\n🏁 ALL SEGMENTS COMPLETE.")