# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib import cm
from matplotlib.colors import Normalize
import os
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# ==================== 1. 全局配置 ====================
current_script_path = os.path.abspath(__file__)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
data_dir = os.path.join(BASE_DIR, "data")
out_dir = os.path.join(BASE_DIR, "fig", "fig1")
os.makedirs(out_dir, exist_ok=True)

DATED_FILE = os.path.join(data_dir, "df_IAV_8ORFs_deduplicated_labels_98787_cleaned.csv")
# 替换为最新 H3 版本的特征文件
GEO_FILE = os.path.join(data_dir, "df_IAV_VRP_H3_Row.csv")

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['figure.dpi'] = 300

plt.rcParams['font.weight'] = 'normal'
plt.rcParams['axes.titleweight'] = 'normal'
plt.rcParams['axes.labelweight'] = 'normal'

COLOR_MAP = {
    'Human': '#E74C3C', 'Avian': '#3498DB', 'Swine': '#F39C12', 'Other': '#95A5A6',
    'North': '#3498DB', 'South': '#E74C3C'
}
SUBTYPE_PALETTE = {
    'H1N1': '#0072B2', 'H3N2': '#E69F00', 'H5N1': '#D55E00',
    'H7N9': '#CC79A7', 'H9N2': '#009E73', 'Other': '#95A5A6'
}
TARGET_E_SUBTYPES = ['H1N1', 'H3N2', 'H5N1', 'H7N9', 'H9N2']

# ==================== 2. 数据加载与预处理 ====================
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
        
        if pd.isna(res) and pd.notna(y_int):
            res = y_int + np.random.uniform(0.01, 0.99)
            
        result.append(res)
    return pd.Series(result, index=date_series.index)

def load_and_prep_data():
    print("📂 Loading datasets...")
    df_dated = pd.read_csv(DATED_FILE, low_memory=False)
    
    df_dated['Host_Group'] = df_dated['Host'].astype(str).str.capitalize()
    df_dated.loc[df_dated['Host_Group'].str.contains('Human|Homo', case=False), 'Host_Group'] = 'Human'
    df_dated.loc[df_dated['Host_Group'].str.contains('Avian|Chicken|Duck|Bird|Poultry|Wild_bird', case=False), 'Host_Group'] = 'Avian'
    df_dated.loc[df_dated['Host_Group'].str.contains('Swine|Pig|Suiformes', case=False), 'Host_Group'] = 'Swine'
    df_dated.loc[~df_dated['Host_Group'].isin(['Human', 'Avian', 'Swine']), 'Host_Group'] = 'Other'
    
    df_dated['Continent'] = df_dated['Continent'].fillna('Unknown').astype(str)
    df_dated['Year'] = pd.to_numeric(df_dated['Year'], errors='coerce')
    
    df_geo = pd.read_csv(GEO_FILE, low_memory=False)
    df_geo['Year'] = pd.to_numeric(df_geo['Year'], errors='coerce')
    df_geo = df_geo.dropna(subset=['Year', 'Hex_Lat']).copy()
    
    # 引入一致的时间解析逻辑，生成 Plot_Date
    if 'clean_date' in df_geo.columns and 'month' in df_geo.columns:
        df_geo['Plot_Date'] = parse_dates_randomized(df_geo['clean_date'], df_geo['Year'], df_geo['month'])
    else:
        df_geo['Plot_Date'] = df_geo['Year'] + np.random.uniform(0.01, 0.99, size=len(df_geo))

    df_geo = df_geo[(df_geo['Plot_Date'] >= 1995.5) & (df_geo['Plot_Date'] <= 2024.5)].copy()
    
    return df_dated, df_geo

def get_sampling_trend(sub_df, window=1.0):
    time_grid = np.arange(np.floor(sub_df['Plot_Date'].min()), np.ceil(sub_df['Plot_Date'].max()), 0.5)
    trend_res = []
    for t in time_grid:
        window_mask = (sub_df['Plot_Date'] >= t - window) & (sub_df['Plot_Date'] <= t + window)
        window_data = sub_df[window_mask]
        if len(window_data) > 10:
            # 修改点：统一使用 Hex_Lat 替代旧的 Lat
            m_lat = window_data['Hex_Lat'].mean()
            trend_res.append({'Time': t, 'Centroid_Lat': m_lat})
    return pd.DataFrame(trend_res)

# ==================== 3. 面板绘制函数 ====================
def add_panel_label(ax, label, x=-0.05, y=1.05):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=20, fontweight='normal', va='top', ha='right')

def plot_panel_a_polar(ax, df_dated):
    df_polar = df_dated[(df_dated['month'].astype(str).str.upper() == 'TRUE') & (df_dated['Year'] <= 2024)].copy()
    df_polar['Parsed_Month'] = pd.to_datetime(df_polar['clean_date']).dt.month
    
    north_continents = ['Asia', 'Europe', 'North America']
    south_continents = ['South America', 'Oceania', 'Africa']
    
    count_north = df_polar[df_polar['Continent'].isin(north_continents)].groupby('Parsed_Month').size().reindex(range(1,13), fill_value=0)
    count_south = df_polar[df_polar['Continent'].isin(south_continents)].groupby('Parsed_Month').size().reindex(range(1,13), fill_value=0)
    
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location('N')
    theta = np.linspace(0.0, 2*np.pi, 12, endpoint=False)
    width = 2*np.pi/12
    
    ax.bar(theta, count_north, width=width, color=COLOR_MAP['North'], alpha=0.85, label='North Hemisphere')
    ax.bar(theta, count_south, width=width, bottom=count_north, color=COLOR_MAP['South'], alpha=0.85, label='South Hemisphere')
    
    ax.set_xticks(theta)
    ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], fontsize=8) 
    ax.tick_params(axis='x', pad=10) 
    
    yticks = ax.get_yticks()
    valid_ticks = [t for t in yticks if t > 0]
    if valid_ticks:
        outer_tick = valid_ticks[-1]
        ax.set_yticks(valid_ticks)
        ax.set_yticklabels([]) 
        ax.text(np.pi/4, outer_tick, f"{int(outer_tick):,}", color='dimgray', fontsize=8, fontweight='normal', 
                ha='center', va='center', bbox=dict(facecolor='white', edgecolor='none', pad=2, alpha=0.8))
                
    ax.spines['polar'].set_visible(False)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False, fontsize=9)
    add_panel_label(ax, 'A', x=-0.1, y=1.1)

def plot_panel_b_bubble(ax, df_dated):
    df_c = df_dated[(df_dated['Year'] <= 2024) & (~df_dated['Continent'].isin(['Antarctica', 'Unknown', 'unknown']))]
    top_subtypes = df_c['Serotype'].value_counts().head(8).index.tolist()
    df_c = df_c[df_c['Serotype'].isin(top_subtypes)]
    
    continents = sorted(df_c['Continent'].unique())
    subtypes = sorted(top_subtypes, reverse=True)
    cont_map = {c: i for i, c in enumerate(continents)}
    sub_map = {s: i for i, s in enumerate(subtypes)}
    
    host_agg = df_c.groupby(['Continent', 'Serotype', 'Host_Group']).size().reset_index(name='Count')
    total_agg = df_c.groupby(['Continent', 'Serotype']).size().reset_index(name='Total')
    max_total = total_agg['Total'].max()
    
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-0.5, len(continents) - 0.5)
    ax.set_ylim(-0.5, len(subtypes) - 0.5)
    ax.set_xticks(range(len(continents)))
    ax.set_xticklabels([c.replace(' ', '\n') for c in continents], fontsize=10, fontweight='normal')
    ax.set_yticks(range(len(subtypes)))
    ax.set_yticklabels(subtypes, fontsize=10, fontweight='normal')
    ax.grid(True, linestyle=':', alpha=0.4)
    
    for _, row in total_agg.iterrows():
        c, s, total = row['Continent'], row['Serotype'], row['Total']
        subset = host_agg[(host_agg['Continent']==c) & (host_agg['Serotype']==s)]
        host_counts = [subset[subset['Host_Group']==h]['Count'].sum() for h in ['Human','Avian','Swine']]
        host_counts.append(total - sum(host_counts))
        colors = [COLOR_MAP[h] for h in ['Human','Avian','Swine','Other']]
        
        if total == 0: continue
        radius = 0.4 * (np.log1p(total) / np.log1p(max_total)) + 0.02
        start_angle = 90
        
        for i, val in enumerate(host_counts):
            if val == 0: continue
            angle = (val/total) * 360
            wedge = mpatches.Wedge((cont_map[c], sub_map[s]), radius, start_angle, start_angle+angle, 
                                   facecolor=colors[i], edgecolor='white', lw=0.3)
            ax.add_patch(wedge)
            start_angle += angle

    sns.despine(ax=ax, left=True, bottom=True)
    host_handles = [mpatches.Patch(color=COLOR_MAP[h], label=h) for h in ['Human','Avian','Swine','Other']]
    legend_color = ax.legend(handles=host_handles, loc='upper left', bbox_to_anchor=(1.05, 1), frameon=False, title="Host", title_fontproperties={'weight': 'normal'})
    ax.add_artist(legend_color)
    
    ref_sizes = [100, 1000, 10000]
    size_handles = []
    for val in ref_sizes:
        r = 0.4 * (np.log1p(val) / np.log1p(max_total)) + 0.02
        size_handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markeredgecolor='white', markersize=r*35, label=f"{val:,}"))
    ax.legend(handles=size_handles, loc='lower left', bbox_to_anchor=(1.05, 0), frameon=False, title="Sequences", title_fontproperties={'weight': 'normal'}, labelspacing=1.2)
    add_panel_label(ax, 'B')

def plot_panel_c_area(ax, df_dated):
    df_d = df_dated[(df_dated['Year'] >= 1996) & (df_dated['Year'] <= 2024)].copy()
    df_d['Lineage'] = df_d['Serotype'].apply(lambda x: x if x in TARGET_E_SUBTYPES else 'Other')
    subtype_count = df_d.groupby(['Year', 'Lineage']).size().unstack(fill_value=0)
    smoothed = subtype_count.rolling(window=3, center=True, min_periods=1).mean()
    ratio = smoothed.div(smoothed.sum(axis=1), axis=0).fillna(0)
    
    years = ratio.index.values
    plot_cols = [c for c in TARGET_E_SUBTYPES + ['Other'] if c in ratio.columns]
    
    ax.stackplot(years, [ratio[c].values for c in plot_cols], labels=plot_cols, colors=[SUBTYPE_PALETTE[c] for c in plot_cols], alpha=0.85, edgecolor='white', linewidth=0.2)
    ax.set_xlim(1995.5, 2024.5)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(['0%', '50%', '100%'], fontsize=9)
    ax.tick_params(axis='x', labelbottom=False)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=9)
    sns.despine(ax=ax, top=True, right=True, bottom=True)
    add_panel_label(ax, 'C')

def plot_panel_d_composite(axes, df_dated, df_geo):
    df_e_heat = df_dated[(df_dated['Year'] >= 1996) & (df_dated['Year'] <= 2024)]
    agg_heat = df_e_heat.groupby(['Year', 'Serotype', 'Host_Group']).size().unstack(fill_value=0)
    if 'Human' not in agg_heat.columns: agg_heat['Human'] = 0
    agg_heat['Total'] = agg_heat.sum(axis=1)
    agg_heat['Human_Ratio'] = agg_heat['Human'] / agg_heat['Total']
    
    cmap = cm.get_cmap('Reds')
    norm = Normalize(vmin=0, vmax=1)
    
    for i, subtype in enumerate(TARGET_E_SUBTYPES):
        ax = axes[i]
        if subtype in agg_heat.index.get_level_values('Serotype'):
            sub_heat = agg_heat.xs(subtype, level='Serotype')
            for year, row in sub_heat.iterrows():
                ratio = row['Human_Ratio']
                if pd.notna(ratio) and row['Total'] > 0:
                    rect = Rectangle((year - 0.5, -105), 1, 30, facecolor=cmap(norm(ratio)), edgecolor='none', zorder=1)
                    ax.add_patch(rect)
        
        ax.axhline(-65, color='black', lw=0.8, zorder=2)
        sub_geo = df_geo[df_geo['Serotype'] == subtype]
        
        ax.axhline(0, color='gray', ls='-', lw=0.6, alpha=0.3, zorder=2)
        ax.axhline(23.5, color='gray', ls=':', lw=0.5, alpha=0.2, zorder=2)
        ax.axhline(-23.5, color='gray', ls=':', lw=0.5, alpha=0.2, zorder=2)
        
        trend_df = get_sampling_trend(sub_geo)
        if not trend_df.empty:
            ax.plot(trend_df['Time'], trend_df['Centroid_Lat'], color='black', lw=1.2, ls='--', alpha=0.9, zorder=5)
            
        # 修改点：散点分布使用新的 Hex_Lat 替代原 Lat
        ax.scatter(sub_geo['Plot_Date'], sub_geo['Hex_Lat'], c=SUBTYPE_PALETTE[subtype], s=2.5, alpha=0.3, lw=0, rasterized=True, zorder=4)
        
        ax.set_xlim(1995.5, 2024.5)
        ax.set_ylim(-105, 85)
        ax.set_yticks([-40, 0, 40, 80])
        ax.set_yticklabels(['-40', '0', '40', '80'], fontsize=8)
        ax.text(1996, 75, subtype, color=SUBTYPE_PALETTE[subtype], fontweight='normal', fontsize=11, va='top')
        sns.despine(ax=ax, top=True, right=True, bottom=(i < 4))
        
        if i == 0: add_panel_label(ax, 'D')
        if i == 2: ax.set_ylabel("Latitude (°)", fontsize=11, fontweight='normal', labelpad=15)
        if i < 4: ax.tick_params(axis='x', labelbottom=False)
        else: ax.set_xlabel("Year", fontsize=11, fontweight='normal')
            
    return cmap, norm

# ==================== 4. 主装配程序 ====================
def main():
    df_dated, df_geo = load_and_prep_data()
    print("🎨 Assembling Final Figure (No Flowchart, Panels A-D)...")
    fig = plt.figure(figsize=(16, 19))
    gs_main = GridSpec(3, 1, figure=fig, height_ratios=[5, 1.8, 6.5], hspace=0.3)
    
    gs_R1 = gs_main[0].subgridspec(1, 2, width_ratios=[1, 2.5], wspace=0.1)
    ax_A = fig.add_subplot(gs_R1[0], projection='polar')
    plot_panel_a_polar(ax_A, df_dated)
    ax_B = fig.add_subplot(gs_R1[1])
    plot_panel_b_bubble(ax_B, df_dated)
    
    ax_C = fig.add_subplot(gs_main[1])
    plot_panel_c_area(ax_C, df_dated)
    
    gs_R3 = gs_main[2].subgridspec(5, 1, hspace=0.1)
    axes_D = [fig.add_subplot(gs_R3[i]) for i in range(5)]
    cmap, norm = plot_panel_d_composite(axes_D, df_dated, df_geo)
    
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.08])
    cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax)
    cb.set_label("Human Host Ratio", fontsize=9, fontweight='normal')
    cb.outline.set_linewidth(0.5)

    out_png = os.path.join(out_dir, "Figure1_Final_Layout_Panels_A_to_D_H3.png")
    out_svg = os.path.join(out_dir, "Figure1_Final_Layout_Panels_A_to_D_H3.svg")
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_svg, format='svg', bbox_inches='tight')
    print(f"✅ Success! Figure Saved to {out_dir}")

if __name__ == "__main__":
    main()