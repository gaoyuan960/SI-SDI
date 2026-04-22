# -*- coding: utf-8 -*-
import h5py
import pandas as pd
import numpy as np
import os
import time
import warnings

warnings.filterwarnings('ignore')

BASE_DIR = "../../"
H5_FILE = os.path.join("E:/gy_98787/similarity/", "IAV_similarity_db.h5")
META_FILE = os.path.join(BASE_DIR, "data", "df_IAV_8ORFs_deduplicated_labels_98787_cleaned.csv")
DKL_FILE = os.path.join(BASE_DIR, "data", "df_IAV_evolution_metrics_FULL_v4.csv")
VRP_FILE = os.path.join(BASE_DIR, "data", "df_IAV_VRP_H3_Row.csv")

OUTPUT_MASTER = os.path.join(BASE_DIR, "data", "Fig7_Master_Data_FULL_v4.csv")
os.makedirs(os.path.dirname(OUTPUT_MASTER), exist_ok=True)
SEGMENTS = ['PB2', 'PB1', 'PA', 'HA', 'NP', 'NA', 'M1', 'NS1']

# ================= 极简宿主清洗逻辑 =================
def categorize_host_v3(row):
    h1 = str(row.get('Host1', '')).lower()
    if 'human' in h1: return 'Human'
    if 'swine' in h1: return 'Swine'
    if 'avian' in h1: return 'Avian'
    return 'Other'

def main():
    print("🚀 Starting Data Synthesis (Dual-Era Baseline Strategy)...")
    
    df_meta = pd.read_csv(META_FILE, low_memory=False)
    df_meta['Host_Clean'] = df_meta.apply(categorize_host_v3, axis=1)
    
    # 强制数值化年份以防脏数据报错
    df_meta['Year_Num'] = pd.to_numeric(df_meta['Year'], errors='coerce')
    
    df_master_init = df_meta[['strain_name', 'Host_Clean', 'Serotype', 'Year_Num', 'Country']].copy()

    print("📦 Loading H5 Database...")
    with h5py.File(H5_FILE, 'r') as f:
        # 你的正确绝对键名
        h5_names = [n.decode('utf-8') for n in f['strain_names'][:]]
        name_to_h5_idx = {name: i for i, name in enumerate(h5_names)}

    valid_mask = df_master_init['strain_name'].apply(lambda n: n in name_to_h5_idx)
    df_master_init = df_master_init[valid_mask].reset_index(drop=True)
    
    print(f"   => Retained {len(df_master_init)} mutually valid strains.")

    # 【严守红线4】：绝对全量计算。拆分出两套索引
    # 1. 前瞻预警索引 (Prosp: All Humans)
    idx_h1_prosp = np.where((df_master_init['Host_Clean'] == 'Human') & (df_master_init['Serotype'] == 'H1N1'))[0]
    idx_h3_prosp = np.where((df_master_init['Host_Clean'] == 'Human') & (df_master_init['Serotype'] == 'H3N2'))[0]
    
    # 2. 历史回溯索引 (Retro: Pre-2009 Humans)
    idx_h1_retro = np.where((df_master_init['Host_Clean'] == 'Human') & (df_master_init['Serotype'] == 'H1N1') & (df_master_init['Year_Num'] < 2009))[0]
    idx_h3_retro = np.where((df_master_init['Host_Clean'] == 'Human') & (df_master_init['Serotype'] == 'H3N2') & (df_master_init['Year_Num'] < 2009))[0]
    
    idx_av = np.where(df_master_init['Host_Clean'] == 'Avian')[0]

    print(f"   => Baseline Counts (Prosp/All): H1N1: {len(idx_h1_prosp)}, H3N2: {len(idx_h3_prosp)}")
    print(f"   => Baseline Counts (Retro/<2009): H1N1: {len(idx_h1_retro)}, H3N2: {len(idx_h3_retro)}")

    # 映射为绝对 H5 索引
    idx_h1_prosp_h5 = [name_to_h5_idx[df_master_init.iloc[i]['strain_name']] for i in idx_h1_prosp]
    idx_h3_prosp_h5 = [name_to_h5_idx[df_master_init.iloc[i]['strain_name']] for i in idx_h3_prosp]
    idx_h1_retro_h5 = [name_to_h5_idx[df_master_init.iloc[i]['strain_name']] for i in idx_h1_retro]
    idx_h3_retro_h5 = [name_to_h5_idx[df_master_init.iloc[i]['strain_name']] for i in idx_h3_retro]
    idx_av_h5 = [name_to_h5_idx[df_master_init.iloc[i]['strain_name']] for i in idx_av]

    idx_map = [name_to_h5_idx[n] for n in df_master_init['strain_name']]
    
    df_merged = df_master_init.copy()
    seg_cols_prosp = []
    seg_cols_retro = []

    print("📊 Calculating Dual-Era Base Similarities (Full Matrix Exact Mean)...")
    with h5py.File(H5_FILE, 'r') as f:
        for seg in SEGMENTS:
            if seg in f:
                ts = time.time()
                # 禽类基底共用
                sim_av = np.nanmean(f[seg]['data'][:, idx_av_h5][idx_map, :], axis=1)
                
                # 1. 计算前瞻预警分数 (主变量)
                sim_h1_p = np.nanmean(f[seg]['data'][:, idx_h1_prosp_h5][idx_map, :], axis=1)
                sim_h3_p = np.nanmean(f[seg]['data'][:, idx_h3_prosp_h5][idx_map, :], axis=1)
                score_prosp = np.maximum(sim_h1_p, sim_h3_p) - sim_av
                df_merged[f'{seg}_Vector_Score'] = score_prosp
                seg_cols_prosp.append(f'{seg}_Vector_Score')
                
                # 2. 计算历史回溯分数 (伴生变量)
                sim_h1_r = np.nanmean(f[seg]['data'][:, idx_h1_retro_h5][idx_map, :], axis=1)
                sim_h3_r = np.nanmean(f[seg]['data'][:, idx_h3_retro_h5][idx_map, :], axis=1)
                score_retro = np.maximum(sim_h1_r, sim_h3_r) - sim_av
                df_merged[f'{seg}_Vector_Score_Retro'] = score_retro
                seg_cols_retro.append(f'{seg}_Vector_Score_Retro')
                
                print(f"   ✓ {seg} Dual-Track calculated in {time.time()-ts:.2f}s.")

    # 聚合均值
    if seg_cols_prosp:
        df_merged['Genome_Vector_Score'] = df_merged[seg_cols_prosp].mean(axis=1)
    if seg_cols_retro:
        df_merged['Genome_Vector_Score_Retro'] = df_merged[seg_cols_retro].mean(axis=1)

    df_master = df_merged.copy()

    print("🔗 Merging DKL and Spatial Dissemination (SDI/VRP) data...")
    if os.path.exists(DKL_FILE):
        df_dkl = pd.read_csv(DKL_FILE)
        dkl_cols = ['strain_name'] + [c for c in df_dkl.columns if 'DKL' in c or 'CpG' in c or 'UpA' in c]
        df_master = pd.merge(df_master, df_dkl[dkl_cols], on='strain_name', how='left')
    
    if os.path.exists(VRP_FILE):
        df_vrp = pd.read_csv(VRP_FILE)
        # 绝不提生殖力，严格执行 SDI 空间逻辑对接
        vrp_cols = ['strain_name', 'VRP_Score', 'VRP_Norm', 'Hex_Lat', 'Hex_Lon']
        vrp_cols_exist = [c for c in vrp_cols if c in df_vrp.columns]
        df_master = pd.merge(df_master, df_vrp[vrp_cols_exist], on='strain_name', how='left')

    print("🎯 Establishing Dual-Era Quadrant Thresholds...")
    # 1. 现代预警阈值 (前瞻)
    T_si_prosp = df_master[df_master['Host_Clean'] == 'Human']['Genome_Vector_Score'].quantile(0.05)
    human_vrp_prosp = df_master[df_master['Host_Clean'] == 'Human']['VRP_Norm'].dropna()
    T_sdi_prosp = human_vrp_prosp.quantile(0.05) if len(human_vrp_prosp) > 0 else np.nan
    
    # 2. 历史回溯阈值 (<2009)
    human_retro_mask = (df_master['Host_Clean'] == 'Human') & (df_master['Year_Num'] < 2009)
    T_si_retro = df_master[human_retro_mask]['Genome_Vector_Score_Retro'].quantile(0.05)
    human_vrp_retro = df_master[human_retro_mask]['VRP_Norm'].dropna()
    T_sdi_retro = human_vrp_retro.quantile(0.05) if len(human_vrp_retro) > 0 else np.nan
        
    print(f"   => PROSPECTIVE Thresholds -> T_SI: {T_si_prosp:.4f}, T_SDI: {T_sdi_prosp:.4f}")
    print(f"   => RETROSPECTIVE Thresholds -> T_SI_Retro: {T_si_retro:.4f}, T_SDI_Retro: {T_sdi_retro:.4f}")

    df_master.to_csv(OUTPUT_MASTER, index=False)
    print(f"🎉 All Done! Fig 7 Master Data saved to: {OUTPUT_MASTER}")

if __name__ == "__main__":
    main()