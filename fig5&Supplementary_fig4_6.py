# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import warnings
from datetime import datetime
import h3

warnings.filterwarnings('ignore')

# ==================== 1. 全局配置 ====================
WORK_DIR = r"d:\gaoyuan\Desktop\beijing2\20260311_fig1_7\code\geo_vrp_fig5"
INPUT_FILE = os.path.join(WORK_DIR, "df_IAV_VRP_H3_Row.csv")
OUTPUT_DIR = os.path.join(WORK_DIR, "fig5_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 期刊规范：Arial 字体，SVG 文本化，全局不加粗
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['axes.titleweight'] = 'normal'
plt.rcParams['axes.labelweight'] = 'normal'

PALETTE = {'H1N1': '#0072B2', 'H3N2': '#E69F00', 'H5N1': '#D55E00', 'H7N9': '#CC79A7', 'H9N2': '#009E73'}
TARGET_SEROTYPES = list(PALETTE.keys())

# ==================== 2. 工具函数 ====================
def parse_dates_randomized(date_series, year_series, month_series):
    result = []
    np.random.seed(42) 
    for d_val, y_val, m_val in zip(date_series, year_series, month_series):
        res = np.nan
        m_str = str(m_val).strip().upper()
        s_val = str(d_val).strip() if pd.notna(d_val) else ""
        y_int = float(y_val) if pd.notna(y_val) else np.nan
        
        if m_str == 'TRUE' and s_val and s_val not in ('TRUE', 'FALSE', 'nan', 'NaT'):
            dt = None
            for fmt in ('%Y/%m/%d', '%Y-%m-%d', '%m/%d/%Y', '%Y.%m.%d'):
                try:
                    dt = datetime.strptime(s_val, fmt)
                    break
                except ValueError:
                    continue
            if dt:
                days_in_year = 366 if (dt.year % 4 == 0 and dt.year % 100 != 0) or (dt.year % 400 == 0) else 365
                res = dt.year + (dt.timetuple().tm_yday - 1) / days_in_year
        if pd.isna(res) and pd.notna(y_int): res = y_int + np.random.uniform(0.01, 0.99)
        result.append(res)
    return pd.Series(result, index=date_series.index)

def get_hex_boundary(hex_id):
    if hasattr(h3, 'cell_to_boundary'): return [(lon, lat) for lat, lon in h3.cell_to_boundary(hex_id)]
    else: return h3.h3_to_geo_boundary(hex_id, geo_json=True)

def get_weighted_trend(sub_df, window=1.0):
    if sub_df.empty or 'Decimal_Date' not in sub_df: return pd.DataFrame()
    time_grid = np.arange(np.floor(sub_df['Decimal_Date'].min()), np.ceil(sub_df['Decimal_Date'].max()), 0.5)
    trend_res = []
    for t in time_grid:
        window_mask = (sub_df['Decimal_Date'] >= t - window) & (sub_df['Decimal_Date'] <= t + window)
        window_data = sub_df[window_mask]
        if len(window_data) > 10:
            weights = window_data['VRP_Norm'].values + 1e-6
            w_lat = np.average(window_data['Hex_Lat'].values, weights=weights)
            trend_res.append({'Time': t, 'W_Lat': w_lat})
    return pd.DataFrame(trend_res)

def get_sampling_trend(sub_df, window=1.0):
    if sub_df.empty or 'Decimal_Date' not in sub_df: return pd.DataFrame()
    time_grid = np.arange(np.floor(sub_df['Decimal_Date'].min()), np.ceil(sub_df['Decimal_Date'].max()), 0.5)
    trend_res = []
    for t in time_grid:
        window_mask = (sub_df['Decimal_Date'] >= t - window) & (sub_df['Decimal_Date'] <= t + window)
        window_data = sub_df[window_mask]
        if len(window_data) > 10:
            m_lat = window_data['Hex_Lat'].mean()
            trend_res.append({'Time': t, 'W_Lat': m_lat})
    return pd.DataFrame(trend_res)

# ==================== 3. 主图 Fig 5 ====================
def plot_main_fig5(df):
    print("🎨 正在生成 Main Fig 5...")
    fig = plt.figure(figsize=(18, 17))
    plt.subplots_adjust(bottom=0.1)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.2, 1, 1], hspace=0.3, wspace=0.1)
    
    # --- 图 A: 全局气泡图 ---
    ax_bubble = fig.add_subplot(gs[0, :])
    agg_year = df.groupby(['Year', 'Serotype'])['VRP_Norm'].agg(P95=lambda x: np.percentile(x, 95), Count='count').reset_index()
    agg_year = agg_year[agg_year['Count'] >= 10]
    
    size_min, size_max = 30, 400
    if not agg_year.empty:
        norm_count = (np.log1p(agg_year['Count']) - np.log1p(agg_year['Count'].min())) / (np.log1p(agg_year['Count'].max()) - np.log1p(agg_year['Count'].min()))
        agg_year['Size'] = norm_count * (size_max - size_min) + size_min
        scatter_bubble = ax_bubble.scatter(agg_year['Year'], agg_year['Serotype'], c=agg_year['P95'], s=agg_year['Size'], cmap='Reds', alpha=0.85, edgecolors='#333333', linewidth=0.6, zorder=3, vmin=0, vmax=100)
        
        cbar_bubble = plt.colorbar(scatter_bubble, ax=ax_bubble, pad=0.01, aspect=20)
        cbar_bubble.set_label("Max SDI (P95)", fontsize=10, fontweight='normal')
        
        ref_counts = [10, 100, 1000] if agg_year['Count'].max() >= 1000 else [10, 50, 200]
        legend_elements_a = []
        min_c, max_c = agg_year['Count'].min(), agg_year['Count'].max()
        for val in ref_counts:
            if max_c > min_c:
                val_clipped = max(min_c, min(val, max_c))
                norm_v = (np.log1p(val_clipped) - np.log1p(min_c)) / (np.log1p(max_c) - np.log1p(min_c))
                s = norm_v * (size_max - size_min) + size_min
            else:
                s = size_max
            legend_elements_a.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#888888', markeredgecolor='#333333', 
                                                alpha=0.6, markersize=np.sqrt(s), label=f'{val:,}'))
        
        ax_bubble.legend(handles=legend_elements_a, loc='center left', bbox_to_anchor=(1.08, 0.5), 
                         title="Strain Count", title_fontproperties={'weight': 'normal'}, frameon=False, labelspacing=1.2)

    ax_bubble.grid(True, linestyle=':', color='0.8')
    ax_bubble.set_title("A", fontsize=16, fontweight='normal', loc='left', pad=15)
    #ax_bubble.set_xlabel("Year", fontsize=12, fontweight='normal')
    #ax_bubble.set_ylabel("Subtype", fontsize=12, fontweight='normal')
    sns.despine(ax=ax_bubble, left=True, bottom=True)
    
    # --- 图 B-G: 地图部分 ---
    global_agg = df.groupby(['Hex_ID']).size().reset_index(name='Count')
    global_c_min, global_c_max = global_agg['Count'].min(), global_agg['Count'].max()
    map_size_min, map_size_max = 5, 45 

    # 恢复了亚型名称作为独立配置变量
    map_configs = [
        (1, 0, 'All Subtypes', df, "B"), (1, 1, 'H1N1', df[df['Serotype']=='H1N1'], "C"),
        (1, 2, 'H3N2', df[df['Serotype']=='H3N2'], "D"), (2, 0, 'H5N1', df[df['Serotype']=='H5N1'], "E"),
        (2, 1, 'H7N9', df[df['Serotype']=='H7N9'], "F"), (2, 2, 'H9N2', df[df['Serotype']=='H9N2'], "G")
    ]
    
    for row, col, title, data, letter in map_configs:
        ax = fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree(central_longitude=150))
        ax.set_global() 
        ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree()) 
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='gray', zorder=1)
        ax.patch.set_alpha(0.0) 
        ax.spines['geo'].set_visible(False)
        ax.set_title(letter, fontsize=14, fontweight='normal', loc='left', pad=10)
        
        # 将亚型名称作为文本标注恢复至子图上方
        ax.text(0.5, 1.05, title, transform=ax.transAxes, ha='center', fontsize=12, fontweight='normal')
        
        if data.empty: continue
        agg_df = data.groupby(['Hex_ID', 'Hex_Lat', 'Hex_Lon']).agg(Mean_VRP=('VRP_Norm', 'mean'), Count=('VRP_Norm', 'count')).reset_index().sort_values('Mean_VRP')
        
        if global_c_max > global_c_min: 
            agg_df['Size'] = ((np.log1p(agg_df['Count']) - np.log1p(global_c_min)) / (np.log1p(global_c_max) - np.log1p(global_c_min))) * (map_size_max - map_size_min) + map_size_min
        else: 
            agg_df['Size'] = map_size_max

        for _, hex_row in agg_df.iterrows():
            bnd = get_hex_boundary(hex_row['Hex_ID'])
            hex_color = plt.cm.Reds(0.2 + (hex_row['Mean_VRP']/100)*0.7)
            if max(lon for lon, lat in bnd) - min(lon for lon, lat in bnd) < 180:
                ax.add_patch(Polygon(bnd, transform=ccrs.PlateCarree(), facecolor=hex_color, edgecolor='white', linewidth=0.2, alpha=0.9, zorder=2))
            
            # 使用带半透明白底的圆圈提升密集区辨识度
            ax.scatter(hex_row['Hex_Lon'], hex_row['Hex_Lat'], s=hex_row['Size'], facecolor='white', edgecolor='#333333', linewidth=0.5, alpha=0.4, transform=ccrs.PlateCarree(), zorder=3)

        ax.text(0.02, 0.02, f"Total strains: n={len(data):,}\nActive Hex: {len(agg_df):,}", transform=ax.transAxes, fontsize=8, color='#555555')

    # ==================== 统一图注系统构建 ====================
    ref_sizes = [1, 50, 500] if global_c_max < 1000 else [10, 100, 1000]
    legend_handles = []
    for val in ref_sizes:
        if global_c_max > global_c_min:
            n_val = (np.log1p(val) - np.log1p(global_c_min)) / (np.log1p(global_c_max) - np.log1p(global_c_min))
            s = n_val * (map_size_max - map_size_min) + map_size_min
        else:
            s = map_size_max
        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='white', markeredgecolor='#333333', 
                                         linewidth=0.5, alpha=0.4, markersize=np.sqrt(s)*2, label=f'{val:,}'))
    
    fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.35, 0.02), ncol=3, 
               frameon=False, title="Strain Count per Hexagon", title_fontproperties={'weight':'normal'})

    cbar_ax = fig.add_axes([0.55, 0.04, 0.15, 0.015])
    cmap_leg = mcolors.LinearSegmentedColormap.from_list("unified_vrp", ["#E5E5E5", "#FB6A4A", "#67000D"])
    norm = mcolors.Normalize(vmin=0, vmax=100)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap_leg), cax=cbar_ax, orientation='horizontal')
    cb.set_label("Mean SDI (Color Intensity)", fontsize=10, fontweight='normal')
    cb.outline.set_linewidth(0.5)

    plt.savefig(os.path.join(OUTPUT_DIR, "Fig5_Main_H3_Polygons.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, "Fig5_Main_H3_Polygons.svg"), dpi=300, bbox_inches='tight')
    print("✅ Fig 5 主图保存完毕！")
    plt.close()

# ==================== 4. 补充图 ====================
def plot_supp_figs(df):
    print("🎨 正在生成 Supplementary Figures S1 和 S2...")
    
    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(11, 14), sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0.05) 
    for i, serotype in enumerate(TARGET_SEROTYPES):
        ax = axes[i]
        subset = df[df['Serotype'] == serotype]
        ax.scatter(subset['Decimal_Date'], subset['Hex_Lat'], c=PALETTE[serotype], s=2.5, alpha=0.35, linewidths=0, rasterized=True, zorder=3)
        trend_df = get_weighted_trend(subset)
        if not trend_df.empty: ax.plot(trend_df['Time'], trend_df['W_Lat'], color='black', linewidth=1.5, linestyle='--', alpha=0.9, zorder=5)
        ax.set_ylim(-60, 85)
        if i == 2: ax.set_ylabel("Latitude (°)", fontsize=14, fontweight='normal')
        sns.despine(ax=ax, top=True, right=True, bottom=(i != 4))
    #axes[-1].set_xlabel("Year", fontsize=14, fontweight='normal')
    axes[-1].set_xlim(1996, 2025)
    axes[0].set_title("A", loc='left', fontsize=16, fontweight='normal', pad=15) 
    plt.savefig(os.path.join(OUTPUT_DIR, "Supp_FigS1_LatFlow_Trend.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, "Supp_FigS1_LatFlow_Trend.svg"), dpi=300, bbox_inches='tight')
    plt.close()
    
    if 'Country' in df.columns:
        top_countries = df['Country'].value_counts().nlargest(20).index.tolist()
        df_country = df[df['Country'].isin(top_countries)].copy()
        agg_country = df_country.groupby(['Country', 'Serotype'])['VRP_Norm'].agg(P95=lambda x: np.percentile(x, 95), Count='count').reset_index()
        agg_country = agg_country[agg_country['Count'] >= 10]
        country_order = agg_country.groupby('Country')['P95'].mean().sort_values(ascending=False).index
        agg_country['Country'] = pd.Categorical(agg_country['Country'], categories=country_order, ordered=True)
        agg_country = agg_country.sort_values('Country')
        plt.figure(figsize=(14, 6))
        size_min, size_max = 30, 400
        if not agg_country.empty:
            agg_country['Size'] = ((np.log1p(agg_country['Count']) - np.log1p(agg_country['Count'].min())) / (np.log1p(agg_country['Count'].max()) - np.log1p(agg_country['Count'].min()))) * (size_max - size_min) + size_min
            
            # 1. 捕获散点对象 (将其赋给变量 sc)
            sc = plt.scatter(x=agg_country['Country'], y=agg_country['Serotype'], c=agg_country['P95'], s=agg_country['Size'], cmap='Reds', alpha=0.85, edgecolors='#333333', linewidth=0.6, zorder=3, vmin=0, vmax=100)
            
            # 2. 挂载颜色条 (Colorbar)，严格执行全局统一缩写 SDI
            cbar = plt.colorbar(sc, pad=0.02, aspect=25)
            cbar.set_label("Max SDI (P95)", fontsize=11, fontweight='normal')
            
            # 3. 构建并挂载气泡大小图注 (Size Legend)
            # 自动提取当前数据集中的数量极值，作为基准参照
            min_c = int(agg_country['Count'].min())
            max_c = int(agg_country['Count'].max())
            med_c = int(np.median([min_c, max_c]))
            
            handles = []
            for val in sorted(list(set([min_c, med_c, max_c]))):
                if val == 0: continue
                # 依据与主图完全一致的 log1p 映射公式推导图注圆圈大小
                norm_val = ((np.log1p(val) - np.log1p(min_c)) / (np.log1p(max_c) - np.log1p(min_c))) if max_c > min_c else 1
                s_val = norm_val * (size_max - size_min) + size_min if max_c > min_c else size_max
                handles.append(plt.scatter([], [], s=s_val, c='none', edgecolors='#555555', linewidth=0.8, label=f'{val:,}'))
                
            # 将大小图注锚定在主图区域的右侧偏上位置，彻底剥离与 Colorbar 的空间冲突
            plt.legend(handles=handles, title="Unique Strains", loc='upper left', bbox_to_anchor=(1.12, 1.0), frameon=False, labelspacing=1.2, title_fontsize=11)

        plt.grid(True, linestyle=':', color='0.8')
        plt.title("A", fontsize=16, fontweight='normal', loc='left', pad=15)
        plt.xticks(rotation=45, ha='right')
        sns.despine(left=True, bottom=True)
        plt.savefig(os.path.join(OUTPUT_DIR, "Supp_FigS2_CountryRisk.png"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_DIR, "Supp_FigS2_CountryRisk.svg"), dpi=300, bbox_inches='tight')
        plt.close()

    print("🎨 正在生成 Supplementary Figures S3...")
    fig3, axes3 = plt.subplots(nrows=5, ncols=1, figsize=(11, 14), sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0.05) 
    
    for i, serotype in enumerate(TARGET_SEROTYPES):
        ax = axes3[i]
        subset = df[df['Serotype'] == serotype]
        ax.scatter(subset['Decimal_Date'], subset['Hex_Lat'], c=PALETTE[serotype], s=2.0, alpha=0.25, linewidths=0, rasterized=True, zorder=2)
        trend_base = get_sampling_trend(subset)
        if not trend_base.empty: 
            ax.plot(trend_base['Time'], trend_base['W_Lat'], color='royalblue', linewidth=1.5, linestyle='-', alpha=0.8, zorder=4, label='Baseline Sampling Trend' if i == 0 else "")
        trend_vrp = get_weighted_trend(subset)
        if not trend_vrp.empty: 
            ax.plot(trend_vrp['Time'], trend_vrp['W_Lat'], color='black', linewidth=2.0, linestyle='--', alpha=0.9, zorder=5, label='SDI Risk Trend' if i == 0 else "")
                   
        ax.set_ylim(-60, 85)
        if i == 2: ax.set_ylabel("Latitude (°)", fontsize=14, fontweight='normal')
        sns.despine(ax=ax, top=True, right=True, bottom=(i != 4))
        if i == 0:
            ax.legend(loc='upper right', frameon=True, fontsize=10, facecolor='white', edgecolor='0.8')
            ax.set_title("A", loc='left', fontsize=16, fontweight='normal', pad=15)

    #axes3[-1].set_xlabel("Year", fontsize=14, fontweight='normal')
    axes3[-1].set_xlim(1996, 2025)
    plt.savefig(os.path.join(OUTPUT_DIR, "Supp_FigS3_Trend_Comparison.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, "Supp_FigS3_Trend_Comparison.svg"), dpi=300, bbox_inches='tight')
    print("✅ 补充图表 S3 保存完毕！")
    plt.close()

if __name__ == "__main__":
    print("加载数据并重建时间特征...")
    df_raw = pd.read_csv(INPUT_FILE, low_memory=False)
    
    df_raw['Year'] = pd.to_numeric(df_raw['Year'], errors='coerce')
    df_raw = df_raw.dropna(subset=['Year']).copy()
    df_raw = df_raw[df_raw['Serotype'].isin(TARGET_SEROTYPES)].copy()
    
    if 'clean_date' in df_raw.columns and 'month' in df_raw.columns:
        df_raw['Decimal_Date'] = parse_dates_randomized(df_raw['clean_date'], df_raw['Year'], df_raw['month'])
    else:
        df_raw['Decimal_Date'] = df_raw['Year'] + np.random.uniform(0.01, 0.99, size=len(df_raw))

    df_raw = df_raw[(df_raw['Decimal_Date'] >= 1995.5) & (df_raw['Decimal_Date'] <= 2024.5)].copy()
    
    plot_main_fig5(df_raw)
    plot_supp_figs(df_raw)