import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as PathEffects
import umap
import pandas as pd
import os
import re
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# ==================== 1. 配置区域 ====================
BASE_DIR = "./"

# 切换为刚刚生成的 Capped 版本 H5 数据
DATA_H5 = os.path.join(BASE_DIR, "data", "IAV_mini_stratified_capped.h5")
DATA_CSV = os.path.join(BASE_DIR, "data", "df_IAV_8ORFs_deduplicated_labels_98787_cleaned.csv")

# [不覆盖原图] 创建全新的输出文件夹
FIG_DIR = os.path.join(BASE_DIR, "fig", "Figure4_Final_Capped")
os.makedirs(FIG_DIR, exist_ok=True)

# 主辅图分离决策 (HA, PB2 抓典型)
MAIN_SEGS = ['HA', 'PB2']
SUPP_SEGS = ['PB1', 'PA', 'NP', 'NA', 'M1', 'NS1']

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['svg.fonttype'] = 'none'

# === 新增：强制全局所有层级的文本保持正常不加粗 ===
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['axes.titleweight'] = 'normal'
plt.rcParams['axes.labelweight'] = 'normal'
# ==================================================

# 视觉统一人禽配色
HOST_COLORS = {
    'Human': '#E74C3C',  # 红色
    'Avian': '#3498DB',  # 蓝色
    'Swine': '#2ECC71',  # 绿色
    'Other': '#95A5A6',  # 灰色
    'Unknown': '#ECF0F1'
}

SUBTYPE_CMAP = plt.cm.get_cmap('Set1') 
CONT_CMAP = plt.cm.get_cmap('Set2')

# 6个大流行与跨宿主里程碑节点
PANDEMIC_EVENTS = [
    (1918, 'H1N1', '1918 Spanish'),
    (1957, 'H2N2', '1957 Asian'),
    (1968, 'H3N2', '1968 HK'),
    (1997, 'H5N1', '1997 Spillover'),  
    (2009, 'H1N1', '2009 pdm'),
    (2013, 'H7N9', '2013 Emergence')   
]

# ==================== 2. 工具函数 ====================
def normalize_name(name):
    if pd.isna(name): return ""
    return re.sub(r'[^a-z0-9]', '', str(name).lower())

def simplify_host(host_raw):
    h = str(host_raw).lower().strip()
    if h in ['nan', 'unknown', '', 'none']: return 'Unknown'
    if any(x in h for x in ['human', 'homo', 'sapiens']): return 'Human'
    if any(x in h for x in ['swine', 'pig', 'porcine', 'suiformes']): return 'Swine'
    if any(x in h for x in ['avian', 'duck', 'chicken', 'bird', 'poultry', 'wild']): return 'Avian'
    return 'Other'

def add_text_with_halo(ax, x, y, text, color='black', fontsize=11, offset=(0,0), fontweight='normal'):
    txt = ax.text(x + offset[0], y + offset[1], text, 
                  fontsize=fontsize, fontweight=fontweight, color=color,
                  ha='center', va='center')
    txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])

# ==================== 3. 绘图引擎 ====================
def plot_segment_row(fig, gs, row_idx, seg, embedding, labels, start_letter_ord):
    """渲染一行的 4 个子图"""
    panels = [
        {'label': chr(start_letter_ord),     'col': 0, 'type': 'sub'},
        {'label': chr(start_letter_ord + 1), 'col': 1, 'type': 'host'},
        {'label': chr(start_letter_ord + 2), 'col': 2, 'type': 'year'},
        {'label': chr(start_letter_ord + 3), 'col': 3, 'type': 'cont'}
    ]
    
    for p in panels:
        ax = fig.add_subplot(gs[row_idx, p['col']])
        p_type = p['type']
        
        # 背景灰点 (Rasterized 防止 SVG 卡死)
        ax.scatter(embedding[:, 0], embedding[:, 1], c='#f5f5f5', s=2, alpha=0.2, rasterized=True)
        
        if p_type == 'year':
            valid = ~np.isnan(labels['year'])
            # 采用 viridis 渐变，vmin=1996 修复红海问题，展现近期顺滑渐变
            sc = ax.scatter(embedding[valid, 0], embedding[valid, 1], 
                           c=labels['year'][valid], cmap='viridis', 
                           s=4, alpha=0.7, vmin=1996, vmax=2024, rasterized=True)
            cbar = plt.colorbar(sc, ax=ax, orientation='horizontal', fraction=0.04, pad=0.04)
            cbar.set_label('Sampling Year', fontsize=10, fontweight='normal')
            cbar.ax.tick_params(labelsize=8)
            
            for yr, sub, lbl in PANDEMIC_EVENTS:
                mask_pan = (labels['year'] == yr) & (labels['sub'] == sub)
                if np.sum(mask_pan) > 0:
                    x_pan = np.median(embedding[mask_pan, 0])
                    y_pan = np.median(embedding[mask_pan, 1])
                    ax.scatter(x_pan, y_pan, s=350, c='gold', edgecolors='black', marker='*', zorder=10)
                    add_text_with_halo(ax, x_pan, y_pan + (np.ptp(embedding[:, 1])*0.05), lbl, fontsize=12, fontweight='normal')
        
        else:
            data_vec = labels[p_type]
            counts = pd.Series(data_vec).value_counts()
            top_n = counts.head(8).index.tolist()
            if 'Unknown' in top_n: top_n.remove('Unknown')
            
            if p_type == 'host': color_map = HOST_COLORS
            elif p_type == 'sub': color_map = {k: SUBTYPE_CMAP(i) for i, k in enumerate(top_n)}
            else: color_map = {k: CONT_CMAP(i) for i, k in enumerate(top_n)}
            
            for cat in top_n:
                if cat not in color_map and p_type != 'host': continue
                mask = (data_vec == cat)
                c = color_map.get(cat, '#808080')
                ax.scatter(embedding[mask, 0], embedding[mask, 1], c=[c], s=5, alpha=0.8, 
                           label=f"{cat} (n={np.sum(mask)})", rasterized=True)
                
                # 簇群文本标注 (数量足够大才标注，防重叠)
                if p_type in ['sub', 'host'] and np.sum(mask) > 100:
                    xm, ym = np.median(embedding[mask, 0]), np.median(embedding[mask, 1])
                    add_text_with_halo(ax, xm, ym, cat, fontsize=11)
                    
            ax.legend(markerscale=3, fontsize=9, loc='upper right', frameon=False)

        # 极致纯净：只保留不加粗大写字母
        ax.set_title(p['label'], loc='left', fontsize=22, fontweight='normal', pad=10)
        
        # 仅在每行最左侧的子图 Y 轴上标注这是哪个基因片段
        if p['col'] == 0:
            ax.set_ylabel(f"{seg} Segment", fontsize=16, fontweight='normal', labelpad=10)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values(): spine.set_visible(False)
        else:
            ax.axis('off')

# ==================== 4. 主程序 ====================
def main():
    print(f"🚀 Starting UMAP Generation (Capped Data Mode)...")
    
    try:
        df = pd.read_csv(DATA_CSV, low_memory=False)
        df['match_key'] = df['strain_name'].apply(normalize_name)
        # 去重，解决 DataFrame index 报错
        df = df.drop_duplicates(subset=['match_key'])
        
        df['Serotype'] = df['Serotype'].astype(str).str.strip()
        df['Continent'] = df['Continent'].astype(str).str.strip()
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        df['Host_Clean'] = df['Host'].apply(simplify_host)
        
        map_meta = df.set_index('match_key').to_dict('index')
    except Exception as e:
        print(f"❌ CSV Error: {e}"); return

    with h5py.File(DATA_H5, 'r') as f:
        h5_names_raw = f['strain_names'][:]
        h5_names = [n.decode('utf-8').strip() if isinstance(n, bytes) else str(n).strip() for n in h5_names_raw]
        
        labels = {'sub': [], 'year': [], 'cont': [], 'host': []}
        for name in h5_names:
            info = map_meta.get(normalize_name(name), {})
            labels['sub'].append(info.get('Serotype', 'Unknown'))
            labels['year'].append(info.get('Year', np.nan))
            labels['cont'].append(info.get('Continent', 'Unknown'))
            labels['host'].append(info.get('Host_Clean', 'Unknown'))
        for k in labels: labels[k] = np.array(labels[k])

        def compute_umap(seg):
            print(f"   🧮 Computing CAPPED UMAP for {seg}...")
            sim_matrix = f[seg]['data'][:]
            # 底层除脏逻辑：将所有 NaN 强制视为无相似性(0)，避免 M1/NS1 画出空图
            sim_matrix = np.nan_to_num(sim_matrix, nan=0.0)
            dist_matrix = np.clip(1.0 - sim_matrix, 0.0, 1.0)
            np.fill_diagonal(dist_matrix, 0.0)
            reducer = umap.UMAP(n_neighbors=200, min_dist=0.3, metric='precomputed', random_state=42, n_jobs=-1)
            emb = reducer.fit_transform(dist_matrix)
            
            # === 新增：将生成的二维降维坐标缓存至本地 ===
            out_coords = os.path.join(FIG_DIR, f"{seg}_umap_coords.npy")
            np.save(out_coords, emb)
            print(f"      💾 UMAP coordinates locally saved to: {out_coords}")
            # ============================================
            
            return emb

        # ---------------- 绘制 MAIN FIGURE (HA, PB2) ----------------
        print("\n🎨 Assembling Main Figure (HA & PB2)...")
        fig_main = plt.figure(figsize=(20, 18), constrained_layout=True)
        gs_main = gridspec.GridSpec(2, 4, figure=fig_main)
        
        for i, seg in enumerate(MAIN_SEGS):
            emb = compute_umap(seg)
            plot_segment_row(fig_main, gs_main, i, seg, emb, labels, 65 + i*4) # A-H
            
        out_main = os.path.join(FIG_DIR, "Fig4_Main_HA_PB2_Capped")
        plt.savefig(out_main + ".png", dpi=300, bbox_inches='tight')
        plt.savefig(out_main + ".svg", format='svg', bbox_inches='tight')
        plt.close(fig_main)
        print(f"   ✅ Main Figure Saved!")

        # ---------------- 绘制 SUPP FIGURE (Other 6) ----------------
        print("\n🎨 Assembling Supplementary Figure (6 Segments)...")
        fig_supp = plt.figure(figsize=(20, 50), constrained_layout=True)
        gs_supp = gridspec.GridSpec(6, 4, figure=fig_supp)
        
        for i, seg in enumerate(SUPP_SEGS):
            emb = compute_umap(seg)
            plot_segment_row(fig_supp, gs_supp, i, seg, emb, labels, 65 + i*4) # A-X
            
        out_supp = os.path.join(FIG_DIR, "FigS_Supp_Other6_Segments_Capped")
        plt.savefig(out_supp + ".png", dpi=300, bbox_inches='tight')
        plt.savefig(out_supp + ".svg", format='svg', bbox_inches='tight')
        plt.close(fig_supp)
        print(f"   ✅ Supplementary Figure Saved!")

    print(f"\n🏁 Mission Complete! Capped 版本高质量图表已存入: {FIG_DIR}")

if __name__ == '__main__':
    main()