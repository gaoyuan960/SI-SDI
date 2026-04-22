# -*- coding: utf-8 -*-
"""
Figure 3 Composite Generation
---------------------------------------------
Features:
1. Robust absolute pathing engine.
2. Evaluates macroscopic systemic driver logic via Cross-scale Validation.
3. Maps to the latest Fig7_Master_Data_FULL_v4.csv for Genome Vector Score.
4. Corrected for Multiple Testing via Benjamini-Hochberg (FDR).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
import os
import re
import warnings

warnings.filterwarnings('ignore')

# ================= 1. 全局配置与路径引擎 =================
current_script_path = os.path.abspath(__file__)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))

COUNTING_DIR = os.path.join(BASE_DIR, "data", "counting")
CLEAN_META_CSV = os.path.join(BASE_DIR, "data", "df_IAV_8ORFs_deduplicated_labels_98787_cleaned.csv")
SCORE_CSV = os.path.join(BASE_DIR, "data", "Fig7_Master_Data_FULL_v4.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "fig", "fig3")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLE_SIZE_PER_HOST = 5000 
SEGMENTS = ['PB2', 'PB1', 'PA', 'HA', 'NP', 'NA', 'M1', 'NS1']
MAIN_SEGS = ['PB2', 'HA', 'NP', 'NS1']
SUPP_SEGS = ['PB1', 'PA', 'NA', 'M1']

COLOR_H = '#E74C3C' 
COLOR_A = '#3498DB' 

# 严格遵循期刊发表规范
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.size'] = 7
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8

# 强制全局所有层级的文本保持正常不加粗
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['axes.titleweight'] = 'normal'
plt.rcParams['axes.labelweight'] = 'normal'

# ================= 2. 工具函数 =================
def normalize_name(name):
    if pd.isna(name): return ""
    return re.sub(r'[^a-z0-9]', '', str(name).lower())

def get_filename(seg):
    if seg == 'NA': return "df_dcrcp_counting_NA _df_IAV_8ORFs_deduplicated_labels_98787.csv"
    return f"df_dcrcp_counting_{seg}_df_IAV_8ORFs_deduplicated_labels_98787.csv"

def is_boring_feature(feature_name):
    parts = feature_name.split('_')
    if len(parts) < 2: return True
    seq = parts[1].lower()
    if 'n' in seq: return True
    if len(set(seq)) == 1: return True 
    return False

def clean_label(feature_name):
    parts = feature_name.split('_')
    if len(parts) >= 2: return parts[1]
    return feature_name

def get_sig_stars(val):
    if val < 1e-10: return "***"
    if val < 1e-5: return "**"
    if val < 0.05: return "*"
    return "ns"

# ================= 3. 数据处理 =================
def get_sample_ids():
    print("📋 Parsing Metadata & Selecting Sample IDs...")
    df = pd.read_csv(CLEAN_META_CSV, low_memory=False)
    df['Host_Group'] = df['Host'].astype(str).str.capitalize()
    df.loc[df['Host_Group'].str.contains('Human|Homo', case=False), 'Host_Group'] = 'Human'
    df.loc[df['Host_Group'].str.contains('Avian|Chicken|Duck|Bird|Poultry', case=False), 'Host_Group'] = 'Avian'
    
    df['match_key'] = df['strain_name'].apply(normalize_name)
    h = df[df['Host_Group'] == 'Human']
    a = df[df['Host_Group'] == 'Avian']
    if len(h) > SAMPLE_SIZE_PER_HOST: h = h.sample(n=SAMPLE_SIZE_PER_HOST, random_state=42)
    if len(a) > SAMPLE_SIZE_PER_HOST: a = a.sample(n=SAMPLE_SIZE_PER_HOST, random_state=42)
    
    if os.path.exists(SCORE_CSV):
        df_score = pd.read_csv(SCORE_CSV, usecols=['strain_name', 'Genome_Vector_Score'], low_memory=False)
        df_score['match_key'] = df_score['strain_name'].apply(normalize_name)
        score_dict = dict(zip(df_score['match_key'], df_score['Genome_Vector_Score']))
        print("   ✅ Latest V4 Genome Vector Score Mapped Successfully.")
    else:
        score_dict = {}
        print("   ⚠️ Warning: Score CSV not found at expected path.")

    meta_dict = pd.concat([df[df['Host_Group'] == 'Human'], df[df['Host_Group'] == 'Avian']]).set_index('match_key')['Host_Group'].to_dict()
    sample_map = pd.concat([h, a]).set_index('match_key')['Host_Group'].to_dict()
    
    return sample_map, meta_dict, score_dict

def load_aggregated_genome_data(sample_map):
    print("🧬 Building Whole Genome Average Profile...")
    first_path = os.path.join(COUNTING_DIR, get_filename('PB2'))
    if not os.path.exists(first_path):
        raise FileNotFoundError(f"Feature counting matrix not found: {first_path}")
        
    header = pd.read_csv(first_path, nrows=1)
    id_col = header.columns[0]
    feat_cols = [c for c in header.columns if c.startswith('Freq_')]
    valid_ids = list(sample_map.keys())
    
    genome_sum = pd.DataFrame(0.0, index=valid_ids, columns=feat_cols)
    count_valid_segs = pd.Series(0, index=valid_ids) 
    
    for seg in SEGMENTS:
        fpath = os.path.join(COUNTING_DIR, get_filename(seg))
        if not os.path.exists(fpath): continue
        chunk_iter = pd.read_csv(fpath, usecols=[id_col] + feat_cols, chunksize=5000)
        for chunk in chunk_iter:
            chunk['match_key'] = chunk[id_col].apply(normalize_name)
            mask = chunk['match_key'].isin(valid_ids)
            if not mask.any(): continue
            subset = chunk[mask].set_index('match_key')[feat_cols]
            if subset.index.duplicated().any(): subset = subset.groupby(level=0).mean()
            genome_sum = genome_sum.add(subset, fill_value=0)
            count_valid_segs.loc[subset.index] += 1
            
    count_valid_segs[count_valid_segs == 0] = 1 
    df_avg = genome_sum.div(count_valid_segs, axis=0)
    df_avg['Host_Group'] = df_avg.index.map(sample_map)
    return df_avg

def check_feature_systemic_score(feature):
    active_segments = 0
    total_mean = 0
    for seg in SEGMENTS:
        path = os.path.join(COUNTING_DIR, get_filename(seg))
        if not os.path.exists(path): continue
        try:
            header = pd.read_csv(path, nrows=1)
            if feature not in header.columns: continue
            df_sample = pd.read_csv(path, usecols=[feature], nrows=500)
            mean_val = df_sample[feature].mean()
            if mean_val > 0.005: 
                active_segments += 1
                total_mean += mean_val
        except: continue
    return active_segments, total_mean

def get_top_systemic_driver(df_avg):
    print("🔍 Mining True Systemic Driver (Strict Scanning with FDR)...")
    features = [c for c in df_avg.columns if c.startswith('Freq_') and not is_boring_feature(c)]
    h = df_avg[df_avg['Host_Group'] == 'Human']
    a = df_avg[df_avg['Host_Group'] == 'Avian']
    
    res = []
    for f in features:
        hv = h[f].values; av = a[f].values
        fc = np.log2((np.mean(hv)+1e-9)/(np.mean(av)+1e-9))
        _, p = stats.ttest_ind(hv, av, equal_var=False)
        res.append({'Feature': f, 'Log2FC': fc, 'P': p})
        
    stats_df = pd.DataFrame(res)
    
    # 引入 Benjamini-Hochberg FDR 校正
    valid_mask = stats_df['P'].notna()
    stats_df['FDR'] = 1.0
    if valid_mask.any():
        _, fdr, _, _ = multipletests(stats_df.loc[valid_mask, 'P'], method='fdr_bh')
        stats_df.loc[valid_mask, 'FDR'] = fdr

    stats_df['NegLogP'] = -np.log10(stats_df['P'] + 1e-300)
    # 显著性判定使用 FDR 替代原始 P 值
    stats_df['Sig'] = (stats_df['FDR'] < 1e-5) & (stats_df['Log2FC'].abs() > 0.3)
    
    candidates = stats_df[stats_df['Sig'] & (stats_df['Log2FC'] > 0)].sort_values('FDR').head(30)
    
    final_scores = []
    for idx, row in candidates.iterrows():
        feat = row['Feature']
        score, avg_mean = check_feature_systemic_score(feat)
        final_scores.append({'Feature': feat, 'Score': score, 'Log2FC': row['Log2FC']})
        
    score_df = pd.DataFrame(final_scores)
    top_picks = score_df.sort_values(by=['Score', 'Log2FC'], ascending=[False, False])
    best_driver = top_picks.iloc[0]['Feature']
    
    print(f"   🏆 Driver Selected: {clean_label(best_driver)} (Active in {top_picks.iloc[0]['Score']}/8 segments)")
    return stats_df, best_driver

def load_full_validation_data(target_feature, meta_map):
    print(f"📦 Streaming Full Data for: {clean_label(target_feature)}")
    res_dict = {}
    id_acc = {}
    
    for seg in SEGMENTS:
        path = os.path.join(COUNTING_DIR, get_filename(seg))
        if not os.path.exists(path): continue
        try:
            df = pd.read_csv(path, usecols=[pd.read_csv(path, nrows=1).columns[0], target_feature], low_memory=False)
            df['match_key'] = df.iloc[:, 0].apply(normalize_name)
            df['val'] = pd.to_numeric(df.iloc[:, 1], errors='coerce').fillna(0)
            df['Host'] = df['match_key'].map(meta_map)
            df = df.dropna(subset=['Host'])
            res_dict[seg] = {'h': df[df['Host']=='Human']['val'].values, 'a': df[df['Host']=='Avian']['val'].values}
            
            for idx, val in zip(df['match_key'], df['val']):
                if idx not in id_acc: id_acc[idx] = [0.0, 0]
                id_acc[idx][0] += val; id_acc[idx][1] += 1
        except: continue
            
    wg_h, wg_a = [], []
    for idx, (s, c) in id_acc.items():
        if c == 0: continue
        host = meta_map.get(idx)
        if host == 'Human': wg_h.append(s/c)
        elif host == 'Avian': wg_a.append(s/c)
    res_dict['WG'] = {'h': np.array(wg_h), 'a': np.array(wg_a)}
    return res_dict

# ================= 4. 绘图装配 =================
def draw_boxplot(ax, data_h, data_a, title_tag, ylabel):
    bp = ax.boxplot([data_h, data_a], labels=['Human', 'Avian'], patch_artist=True, widths=0.5, showfliers=False)
    for patch, color in zip(bp['boxes'], [COLOR_H, COLOR_A]):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
        patch.set_edgecolor('black')
        patch.set_linewidth(0.8)
    for element in ['whiskers', 'caps', 'medians']:
        plt.setp(bp[element], color='black', linewidth=0.8)
    
    try: _, p = stats.mannwhitneyu(data_h, data_a); p_str = f"P={p:.1e}" if p > 0 else "P<1e-300"
    except: p_str = "N/A"
    
    ax.text(-0.15, 1.05, title_tag, transform=ax.transAxes, fontsize=12, fontweight='normal', va='bottom')
    ax.text(0.5, 0.9, p_str, transform=ax.transAxes, ha='center', fontsize=7)
    ax.set_ylabel(ylabel, fontweight='normal')
    sns.despine(ax=ax)

def plot_master_composite(stats_df, df_avg, top_driver, val_data, score_map):
    print("🎨 Rendering Main 2x4 Composite Figure...")
    fig = plt.figure(figsize=(12, 5.5))
    gs = gridspec.GridSpec(2, 4, wspace=0.35, hspace=0.45)
    
    # --- A. Volcano ---
    ax_a = fig.add_subplot(gs[0, 0])
    stats_df['NegLogP_Clip'] = stats_df['NegLogP'].clip(upper=300)
    bg = stats_df[~stats_df['Sig']]
    ax_a.scatter(bg['Log2FC'], bg['NegLogP_Clip'], c='lightgrey', s=2, rasterized=True, lw=0)
    up = stats_df[stats_df['Sig'] & (stats_df['Log2FC'] > 0)]
    down = stats_df[stats_df['Sig'] & (stats_df['Log2FC'] < 0)]
    ax_a.scatter(up['Log2FC'], up['NegLogP_Clip'], c=COLOR_H, s=6, alpha=0.8, rasterized=True, lw=0, label='Human')
    ax_a.scatter(down['Log2FC'], down['NegLogP_Clip'], c=COLOR_A, s=6, alpha=0.8, rasterized=True, lw=0, label='Avian')
    ax_a.axhline(10, ls=':', c='k', lw=0.5); ax_a.axvline(0.5, ls=':', c='k', lw=0.5); ax_a.axvline(-0.5, ls=':', c='k', lw=0.5)
    ax_a.set_xlabel("Log2 Fold Change", fontweight='normal')
    ax_a.set_ylabel("-Log10 P-value", fontweight='normal')
    ax_a.text(-0.15, 1.05, 'A', transform=ax_a.transAxes, fontsize=12, fontweight='normal', va='bottom')
    sns.despine(ax=ax_a)

    # --- B. PCA ---
    ax_b = fig.add_subplot(gs[0, 1])
    feat_cols = [c for c in df_avg.columns if c.startswith('Freq_') and not is_boring_feature(c)]
    x = StandardScaler().fit_transform(df_avg[feat_cols])
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(x)
    pca_df = pd.DataFrame(pcs, columns=['PC1', 'PC2'])
    pca_df['Host'] = df_avg['Host_Group'].values
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Host', palette={'Human': COLOR_H, 'Avian': COLOR_A}, 
                    s=3, alpha=0.5, ax=ax_b, rasterized=True, lw=0, legend=False)
    ax_b.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontweight='normal')
    ax_b.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontweight='normal')
    ax_b.text(-0.15, 1.05, 'B', transform=ax_b.transAxes, fontsize=12, fontweight='normal', va='bottom')
    sns.despine(ax=ax_b)

    # --- C. Barplot ---
    ax_c = fig.add_subplot(gs[0, 2])
    top_up = stats_df[stats_df['Sig'] & (stats_df['Log2FC'] > 0)].nlargest(8, 'Log2FC').sort_values('Log2FC')
    top_down = stats_df[stats_df['Sig'] & (stats_df['Log2FC'] < 0)].nsmallest(8, 'Log2FC').sort_values('Log2FC', ascending=False)
    div = pd.concat([top_down, top_up])
    bars = ax_c.barh(range(len(div)), div['Log2FC'], color=[COLOR_A]*8 + [COLOR_H]*8)
    ax_c.set_yticks(range(len(div)))
    ax_c.set_yticklabels([clean_label(x) for x in div['Feature']], fontsize=6)
    for bar, fdr in zip(bars, div['FDR']):
        xval = bar.get_width()
        ax_c.text(xval + (0.2 if xval>0 else -0.2), bar.get_y()+bar.get_height()/2, get_sig_stars(fdr), ha='left' if xval>0 else 'right', va='center', fontsize=6)
    ax_c.set_xlabel("Log2 Fold Change", fontweight='normal')
    ax_c.text(-0.15, 1.05, 'C', transform=ax_c.transAxes, fontsize=12, fontweight='normal', va='bottom')
    sns.despine(ax=ax_c)

    # --- D. Corr ---
    ax_d = fig.add_subplot(gs[0, 3])
    pca_df['match_key'] = df_avg.index
    pca_df['Score'] = pca_df['match_key'].map(score_map)
    pca_corr = pca_df.dropna(subset=['Score'])
    if not pca_corr.empty:
        r_val, _ = stats.pearsonr(pca_corr['Score'], pca_corr['PC1'])
        sns.scatterplot(data=pca_corr, x='Score', y='PC1', hue='Host', palette={'Human': COLOR_H, 'Avian': COLOR_A}, 
                        s=3, alpha=0.5, ax=ax_d, rasterized=True, legend=False, lw=0)
        ax_d.text(0.95, 0.95, f"r = {r_val:.2f}", transform=ax_d.transAxes, ha='right', va='top', fontsize=7)
    ax_d.set_xlabel("Similarity Index (SI)", fontweight='normal')
    ax_d.set_ylabel("PC1", fontweight='normal')
    ax_d.text(-0.15, 1.05, 'D', transform=ax_d.transAxes, fontsize=12, fontweight='normal', va='bottom')
    sns.despine(ax=ax_d)

    # --- E, F, G, H. Validation Boxplots ---
    letters = ['E', 'F', 'G', 'H']
    for idx, seg in enumerate(MAIN_SEGS):
        ax = fig.add_subplot(gs[1, idx])
        draw_boxplot(ax, val_data[seg]['h'], val_data[seg]['a'], letters[idx], f"{seg} Freq ({clean_label(top_driver)})")

    import matplotlib.patches as mpatches
    h_patch = mpatches.Patch(color=COLOR_H, label='Human')
    a_patch = mpatches.Patch(color=COLOR_A, label='Avian')
    fig.legend(handles=[h_patch, a_patch], loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=2, frameon=False, prop={'weight':'normal'})

    plt.savefig(os.path.join(OUTPUT_DIR, "Fig3_Main_Composite.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, "Fig3_Main_Composite.svg"), format='svg', bbox_inches='tight')
    plt.close()
    print(f"   ✅ Main Composite Image Saved to {OUTPUT_DIR}")

def plot_supp_composite(val_data, top_driver):
    print("🎨 Rendering Supp Figure (Remaining Validation)...")
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(2, 3, wspace=0.4, hspace=0.45)
    
    supp_targets = [('WG', 'Whole Genome')] + [(s, s) for s in SUPP_SEGS]
    letters = ['A', 'B', 'C', 'D', 'E']
    
    for idx, (key, label) in enumerate(supp_targets):
        ax = fig.add_subplot(gs[idx//3, idx%3])
        draw_boxplot(ax, val_data[key]['h'], val_data[key]['a'], letters[idx], f"{label} Freq ({clean_label(top_driver)})")
        
    plt.savefig(os.path.join(OUTPUT_DIR, "Fig3_Supp_Boxplots.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, "Fig3_Supp_Boxplots.svg"), format='svg', bbox_inches='tight')
    plt.close()
    print(f"   ✅ Supp Boxplots Saved to {OUTPUT_DIR}")

# ================= 5. 主执行 =================
def main():
    print(f"🚀 Starting Fig 3 Generation Process...")
    sample_map, meta_dict, score_dict = get_sample_ids()
    df_avg = load_aggregated_genome_data(sample_map)
    stats_df, top_driver = get_top_systemic_driver(df_avg)
    val_data = load_full_validation_data(top_driver, meta_dict)
    
    plot_master_composite(stats_df, df_avg, top_driver, val_data, score_dict)
    plot_supp_composite(val_data, top_driver)
    print(f"\n🏁 Mission Complete! High-quality visuals stored in: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()