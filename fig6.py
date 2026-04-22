# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import os
import warnings

warnings.filterwarnings('ignore')

# ================= 配置区域 =================
current_script_path = os.path.abspath(__file__)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
INPUT_CSV = os.path.join(BASE_DIR, "data", "Fig7_Master_Data_FULL_v4.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "fig", "Fig6_Full_Nature_6Panel")
os.makedirs(OUTPUT_DIR, exist_ok=True)

HIGH_RISK_CSV = os.path.join(OUTPUT_DIR, "Fig6_High_Risk_Strains.csv")
POTENTIAL_RISK_CSV = os.path.join(OUTPUT_DIR, "Fig6_Potential_Risk_Strains.csv")
AVIAN_CANDIDATE_CSV = os.path.join(OUTPUT_DIR, "Fig6_Avian_Candidate_Strains.csv")

# 严格执行格式规范：Arial, 全局彻底不加粗
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.size'] = 9
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['axes.titleweight'] = 'normal'
plt.rcParams['axes.linewidth'] = 1.0

def main():
    print("🚀 Starting Fig 6 FULL DATA Plotting (Final Narrative Flow: E-Heatmap -> F-Bubble)...")
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    
    # ================= 数据预处理 (作图逻辑绝对不变) =================
    df['Year_Strict'] = pd.to_numeric(df['Year'], errors='coerce')
    
    if 'Decimal_Date' not in df.columns:
        np.random.seed(42)
        df['Decimal_Date'] = df['Year_Strict'] + np.random.uniform(0.05, 0.95, size=len(df))
    else:
        df['Decimal_Date'] = pd.to_numeric(df['Decimal_Date'], errors='coerce')
        mask = df['Decimal_Date'].isna() & df['Year_Strict'].notna()
        np.random.seed(42)
        df.loc[mask, 'Decimal_Date'] = df.loc[mask, 'Year_Strict'] + np.random.uniform(0.05, 0.95, size=mask.sum())
        
    df = df.dropna(subset=['Decimal_Date', 'Genome_Vector_Score', 'VRP_Norm']) 
    df['Year_Int'] = np.floor(df['Decimal_Date']) 

    # 计算独立的 D_KL 验证指标
    segments = ['PB2', 'PB1', 'PA', 'HA', 'NP', 'NA', 'M1', 'NS1']
    dkl_cols = []
    for seg in segments:
        col_h = f'{seg}_DKL_Human'
        col_a = f'{seg}_DKL_Avian'
        new_col = f'{seg}_DKL_Diff'
        if col_h in df.columns and col_a in df.columns:
            df[col_h] = pd.to_numeric(df[col_h], errors='coerce')
            df[col_a] = pd.to_numeric(df[col_a], errors='coerce')
            df[new_col] = df[col_h] - df[col_a]
            dkl_cols.append(new_col)

    if dkl_cols:
        df['Net_DKL_Diff'] = df[dkl_cols].mean(axis=1)

    # ================= 阈值与四象限切割 (逻辑不变) =================
    human_df = df[df['Host_Clean'] == 'Human']
    
    vec_threshold = human_df['Genome_Vector_Score'].quantile(0.05)
    vrp_threshold = human_df['VRP_Norm'].quantile(0.05) 
    
    avian_swine = df[df['Host_Clean'].isin(['Avian', 'Swine'])]
    
    high_risk = avian_swine[
        (avian_swine['Genome_Vector_Score'] >= vec_threshold) & 
        (avian_swine['VRP_Norm'] >= vrp_threshold)
    ].copy()
    
    potential_risk = avian_swine[
        (avian_swine['Genome_Vector_Score'] >= vec_threshold) & 
        (avian_swine['VRP_Norm'] < vrp_threshold)
    ].copy()
    
    high_risk.to_csv(HIGH_RISK_CSV, index=False)
    potential_risk.to_csv(POTENTIAL_RISK_CSV, index=False)

    candidates = pd.concat([high_risk, potential_risk])
    avian_cand = candidates[candidates['Host_Clean'] == 'Avian'].copy()
    avian_cand['Country'] = avian_cand['Country'].replace({'United States': 'USA'})
    avian_cand.to_csv(AVIAN_CANDIDATE_CSV, index=False)

    # ================= 画布全局布局 (3 行结构) =================
    fig = plt.figure(figsize=(16, 17))
    gs_main = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1.2], hspace=0.35)
    palette_a = {'Avian': '#1f77b4', 'Swine': '#ff7f0e'}

    # ================= 行 1: Panel A (Risk) & Panel B (DKL) =================
    gs_r1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[0], width_ratios=[1.2, 0.8], wspace=0.25)
    
    # --- Panel A: Risk Matrix ---
    ax_a = fig.add_subplot(gs_r1[0])
    sns.scatterplot(data=human_df, x='Genome_Vector_Score', y='VRP_Norm',
                    color='lightgrey', alpha=0.1, s=5, edgecolor='none', ax=ax_a, rasterized=True)
    sns.scatterplot(data=avian_swine, x='Genome_Vector_Score', y='VRP_Norm', hue='Host_Clean',
                    palette=palette_a, alpha=0.2, s=8, edgecolor='none', ax=ax_a, legend=False, rasterized=True)
    if not candidates.empty:
        sns.scatterplot(data=candidates, x='Genome_Vector_Score', y='VRP_Norm', 
                        hue='Host_Clean', palette=palette_a, s=40, edgecolor='k', linewidth=0.3, ax=ax_a, legend=False, zorder=5)
    if not high_risk.empty:
        ax_a.scatter(high_risk['Genome_Vector_Score'], high_risk['VRP_Norm'],
                     s=70, facecolors='none', edgecolors='#8b0000', linewidth=1.2, zorder=6)
    if not potential_risk.empty:
        ax_a.scatter(potential_risk['Genome_Vector_Score'], potential_risk['VRP_Norm'],
                     s=70, facecolors='none', edgecolors='#ff8c00', linewidth=1.2, zorder=6)

    ax_a.axvline(vec_threshold, color='k', ls='--', alpha=0.6)
    ax_a.axhline(vrp_threshold, color='k', ls='--', alpha=0.6)
    ax_a.text(-0.06, 1.05, "A", transform=ax_a.transAxes, fontsize=18, fontweight='normal', va='bottom')
    ax_a.set_xlabel("Similarity Index (SI)", fontweight='normal')
    ax_a.set_ylabel("Spatial Dissemination Index (SDI)", fontweight='normal')

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgrey', label='Human Background'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', label='Avian Background'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7f0e', label='Swine Background'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markeredgecolor='#8b0000', lw=1.2, label='High Risk (Emerging)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markeredgecolor='#ff8c00', lw=1.2, label='Potential Risk (Predictive)')
    ]
    ax_a.legend(handles=legend_elements, loc='upper left', frameon=False, fontsize=8)

    # --- Panel B: DKL Validation ---
    ax_b = fig.add_subplot(gs_r1[1])
    if 'Net_DKL_Diff' in df.columns:
        valid_df = df.dropna(subset=['Net_DKL_Diff']).copy()
        bg_real = valid_df[valid_df['Host_Clean'].isin(['Avian', 'Swine']) & (valid_df['Genome_Vector_Score'] < vec_threshold)].copy()
        bg_real['Type'] = 'Background'
        pot_real = valid_df[valid_df.index.isin(potential_risk.index)].copy()
        pot_real['Type'] = 'Potential Risk'
        high_real = valid_df[valid_df.index.isin(high_risk.index)].copy()
        high_real['Type'] = 'High Risk'
        hum_real = valid_df[valid_df['Host_Clean'] == 'Human'].copy()
        hum_real['Type'] = 'Human'
        
        plot_data = pd.concat([bg_real, pot_real, high_real, hum_real])
        order = ['Background', 'Potential Risk', 'High Risk', 'Human']
        
        pal_violin = {'Background': '#bdc3c7', 'Potential Risk': '#ffb347', 'High Risk': '#ff6666', 'Human': '#32cd32'}
        pal_strip = {'Background': 'dimgrey', 'Potential Risk': '#cc7a00', 'High Risk': '#8b0000', 'Human': '#006400'}
        
        sns.violinplot(data=plot_data, x='Type', y='Net_DKL_Diff', order=order, 
                       palette=pal_violin, alpha=0.6, ax=ax_b, inner='quartile', linewidth=1)
        sns.stripplot(data=plot_data, x='Type', y='Net_DKL_Diff', order=order, hue='Type',
                      palette=pal_strip, alpha=0.1, size=1.5, ax=ax_b, jitter=True, legend=False, rasterized=True)
                      
        ax_b.axhline(0, color='k', ls=':', alpha=0.3)
        ax_b.text(-0.15, 1.05, "B", transform=ax_b.transAxes, fontsize=18, fontweight='normal', va='bottom')
        ax_b.set_ylabel(r"$\Delta D_{KL}$ (Human - Avian)", fontweight='normal')
        ax_b.set_xlabel("")
        ax_b.tick_params(axis='x', labelrotation=15)
        sns.despine(ax=ax_b)

    # ================= 行 2: Panel C (Genesis) & Panel D (Global Temporal) =================
    gs_r2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[1], width_ratios=[1.2, 0.8], wspace=0.25)
    
    # --- Panel C: Pre-Pandemic Genesis ---
    gs_c = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_r2[0], width_ratios=[5, 1], wspace=0.05)
    ax_c_main = fig.add_subplot(gs_c[0])
    ax_c_marg = fig.add_subplot(gs_c[1], sharey=ax_c_main)
    
    north_america = ['United States', 'Canada', 'Mexico', 'USA']
    swine_h1n1_focal = df[(df['Host_Clean'] == 'Swine') & 
                          (df['Serotype'] == 'H1N1') & 
                          (df['Year_Int'].between(1999, 2011))].copy()
    swine_h1n1_focal['Region'] = np.where(swine_h1n1_focal['Country'].isin(north_america), 'North America', 'Other Regions')
    
    color_na = '#C0392B' 
    color_other = '#bdc3c7' 
    
    sns.scatterplot(data=swine_h1n1_focal[swine_h1n1_focal['Region'] == 'Other Regions'], 
                    x='Decimal_Date', y='Genome_Vector_Score', color=color_other, 
                    alpha=0.4, s=15, edgecolor='none', ax=ax_c_main, label='Other Regions (Background)', zorder=1)
    sns.scatterplot(data=swine_h1n1_focal[swine_h1n1_focal['Region'] == 'North America'], 
                    x='Decimal_Date', y='Genome_Vector_Score', color=color_na, 
                    alpha=0.5, s=20, edgecolor='none', ax=ax_c_main, label='North America (Focal)', zorder=2)
    
    van_na = []
    for yr in range(1999, 2012):
        yr_na = swine_h1n1_focal[(swine_h1n1_focal['Year_Int'] == yr) & (swine_h1n1_focal['Region'] == 'North America')]
        if not yr_na.empty:
            t3 = yr_na.nlargest(3, 'Genome_Vector_Score')
            van_na.append({'True_Time': t3['Decimal_Date'].mean(), 'Score': t3['Genome_Vector_Score'].mean()})
            
    df_van_na = pd.DataFrame(van_na).sort_values('True_Time')
    if not df_van_na.empty:
        ax_c_main.plot(df_van_na['True_Time'], df_van_na['Score'], color='#8b0000', lw=2.5, marker='o', markersize=6, markerfacecolor='white', label='NA Vanguard Trend', zorder=4)
    
    ax_c_main.axvspan(2008.0, 2009.25, color='#8b0000', alpha=0.08, zorder=0)
    ax_c_main.axhline(vec_threshold, color='black', ls='--', lw=1.2, zorder=2)
    ax_c_main.axvline(2009.25, color='#ff8c00', ls=':', lw=1.5, zorder=2)
    
    ax_c_main.text(2008.8, vec_threshold - 0.03, 'Reassortment\nWindow', color='black', fontsize=9, fontweight='normal', ha='right')
    ax_c_main.text(2009.35, vec_threshold + 0.005, '2009 pdmH1N1', color='#ff8c00', fontsize=9, fontweight='normal')
    
    ax_c_main.text(-0.08, 1.05, "C", transform=ax_c_main.transAxes, fontsize=18, fontweight='normal', va='bottom')
    ax_c_main.set_xlabel("", fontweight='normal')
    ax_c_main.set_ylabel("Similarity Index (SI)", fontweight='normal')
    ax_c_main.set_xlim(1999, 2012)
    
    real_y_min = swine_h1n1_focal['Genome_Vector_Score'].min()
    real_y_max = swine_h1n1_focal['Genome_Vector_Score'].max()
    ax_c_main.set_ylim(real_y_min - 0.04, real_y_max + 0.02)
    ax_c_main.legend(loc='upper left', frameon=True, fontsize=8).set_zorder(5)
    sns.despine(ax=ax_c_main)
    
    sns.kdeplot(data=swine_h1n1_focal, y='Genome_Vector_Score', hue='Region', 
                palette={'North America': color_na, 'Other Regions': color_other},
                fill=True, alpha=0.5, ax=ax_c_marg, legend=False, linewidth=1.5, common_norm=False)
    ax_c_marg.axhline(vec_threshold, color='black', ls='--', lw=1.2)
    ax_c_marg.set_xlabel("Density", fontweight='normal')
    ax_c_marg.tick_params(axis='y', left=False, labelleft=False)
    ax_c_marg.set_ylabel("")
    ax_c_marg.set_xticks([]) 
    sns.despine(ax=ax_c_marg, left=True, bottom=True)

    # --- Panel D: Global Temporal Emergence ---
    ax_d = fig.add_subplot(gs_r2[1])
    if not candidates.empty:
        swine_cand = candidates[candidates['Host_Clean'] == 'Swine']
        avian_cand_plot = candidates[candidates['Host_Clean'] == 'Avian']
        ax_d.scatter(swine_cand['Decimal_Date'], swine_cand['Genome_Vector_Score'], 
                     color='#ff7f0e', s=50, edgecolor='k', alpha=0.8, zorder=3, label='Swine')
        ax_d.scatter(avian_cand_plot['Decimal_Date'], avian_cand_plot['Genome_Vector_Score'], 
                     color='#1f77b4', s=50, edgecolor='k', alpha=0.9, zorder=4, label='Avian')
        ax_d.set_xlim(candidates['Decimal_Date'].min()-1, candidates['Decimal_Date'].max()+1)
        ax_d.legend(loc='upper left', frameon=False)
        
    ax_d.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax_d.axhline(vec_threshold, color='k', ls=':', alpha=0.5)
    ax_d.text(-0.15, 1.05, "D", transform=ax_d.transAxes, fontsize=18, fontweight='normal', va='bottom')
    ax_d.set_xlabel("", fontweight='normal')
    ax_d.set_ylabel("Similarity Index (SI)", fontweight='normal')


    # ================= 行 3: Panel E (Heatmap) & Panel F (Bubble) =================
    # 【完美对齐与避让】：左宽右窄，划分为 [宽F(2.2), 隔离(0.2), 窄E(0.8)]
    # 恢复为 [1.2, 0.8]，重建与 BD 完全一致的右侧大边框
    gs_r3 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[2], width_ratios=[1.2, 0.8], wspace=0.25)
    
    # --- New Panel E: Specific Avian Strains Parsing (原 Panel F) ---
    # 热力图放在左侧的宽幅空间 gs_r3[0]
    gs_e = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_r3[0], width_ratios=[0.5, 1.4, 0.7], wspace=0.35)
    ax_e1 = fig.add_subplot(gs_e[1])  # 放在索引1，跳过左侧的空白占位
    ax_e2 = fig.add_subplot(gs_e[2])
    
    if not avian_cand.empty:
        avian_cand = avian_cand.sort_values('Genome_Vector_Score', ascending=True)
        heatmap_cols = ['PB2_Vector_Score', 'PB1_Vector_Score', 'PA_Vector_Score', 
                        'HA_Vector_Score', 'NP_Vector_Score', 'NA_Vector_Score', 
                        'M1_Vector_Score', 'NS1_Vector_Score']
        
        heatmap_data = avian_cand[heatmap_cols]
        heatmap_data.columns = ['PB2', 'PB1', 'PA', 'HA', 'NP', 'NA', 'M1', 'NS1']
        ylabels = avian_cand['strain_name'].astype(str)
        heatmap_data.index = ylabels
        
        sns.heatmap(heatmap_data, cmap='Reds', ax=ax_e1, annot=False, linewidths=0.5, linecolor='white', cbar=False)
        
        cbar_ax = ax_e1.inset_axes([1.02, 0.1, 0.03, 0.8])
        cbar = plt.colorbar(ax_e1.collections[0], cax=cbar_ax, orientation="vertical")
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label("Segment SI", fontweight='normal', fontsize=9)
        
        ax_e1.tick_params(axis='y', labelrotation=0, labelsize=8)
        ax_e1.tick_params(axis='x', labelrotation=45, labelsize=9)
        #ax_e1.set_xlabel("Viral Gene Segments", labelpad=10, fontweight='normal')
        ax_e1.set_ylabel("") 
        
        ax_e2.barh(np.arange(len(ylabels)) + 0.5, avian_cand['VRP_Norm'], color='#5dade2', edgecolor='black', height=0.6, zorder=2)
        ax_e2.axvline(vrp_threshold, color='red', linestyle='--', linewidth=1.5, zorder=3)
        ax_e2.set_yticks([]) 
        ax_e2.set_ylim(0, len(ylabels))
        ax_e2.set_xlabel("Spatial Dissemination Index (SDI)", fontweight='normal')
        ax_e2.set_xlim(0, max(avian_cand['VRP_Norm'].max() * 1.2, vrp_threshold * 1.5))
        ax_e2.text(vrp_threshold + 0.5, len(ylabels) - 0.5, 'Threshold', color='red', fontsize=8, va='center', fontweight='normal')
        ax_e2.grid(axis='x', linestyle=':', alpha=0.5, zorder=1)
        sns.despine(ax=ax_e2, left=True, top=True, right=True)
        
    ax_e1.text(-0.7, 1.05, "E", transform=ax_e1.transAxes, fontsize=18, fontweight='normal', va='bottom')

    # --- New Panel F: High-Risk Hotspots (原 Panel E) ---
    # 气泡图放在右侧与上方 BD 对齐的窄幅空间 gs_r3[2]
    # 在右侧网格内部再切分一刀，比例设为 0.82 和 0.18，给溢出的图注留出物理空间
    gs_f_container = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_r3[1], width_ratios=[0.82, 0.18], wspace=0.0)
    ax_f = fig.add_subplot(gs_f_container[0])
    if not candidates.empty:
        clean_cand = candidates[candidates['Serotype'].str.lower() != 'mixed'].copy()
        geo_data = clean_cand.groupby(['Country', 'Serotype']).size().reset_index(name='Count')
        geo_score = clean_cand.groupby(['Country', 'Serotype'])['Genome_Vector_Score'].mean().reset_index(name='Mean_Score')
        geo_plot = pd.merge(geo_data, geo_score, on=['Country', 'Serotype'])
        
        country_order = geo_plot.groupby('Country')['Count'].sum().sort_values(ascending=True).index
        geo_plot['Country'] = pd.Categorical(geo_plot['Country'], categories=country_order, ordered=True)
        geo_plot = geo_plot.sort_values(['Country', 'Serotype'])
        
        c_min, c_max = geo_plot['Count'].min(), geo_plot['Count'].max()
        size_min, size_max = 20, 350
        if c_max > c_min:
            geo_plot['Size'] = ((np.log1p(geo_plot['Count']) - np.log1p(c_min)) / (np.log1p(c_max) - np.log1p(c_min))) * (size_max - size_min) + size_min
        else:
            geo_plot['Size'] = size_max
        
        unique_st = geo_plot['Serotype'].unique()
        st_map = {st: i for i, st in enumerate(unique_st)}
        geo_plot['x_pos'] = geo_plot['Serotype'].map(st_map)
        
        sc = ax_f.scatter(x=geo_plot['x_pos'], y=geo_plot['Country'], s=geo_plot['Size'],
                          c=geo_plot['Mean_Score'], cmap='Reds', edgecolors='k', linewidth=0.5, alpha=0.9)
        ax_f.set_xticks(range(len(unique_st)))
        ax_f.set_xticklabels(unique_st)
        ax_f.set_xlim(-0.5, len(unique_st) - 0.5) 
        
        # 1. 颜色图注：使用 inset_axes 固定在右侧【下方】 (高度占比45%)
        cbar_ax_f = ax_f.inset_axes([1.05, 0.0, 0.05, 0.45]) 
        cbar = plt.colorbar(sc, cax=cbar_ax_f, orientation='vertical')
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label('Mean Similarity Index (SI)', fontsize=9, fontweight='normal')
        
        handles = []
        breaks = [int(c_min), int(np.sqrt(c_min*c_max)), int(c_max)]
        for b in set(breaks):
            if b == 0: continue
            norm_b = ((np.log1p(b) - np.log1p(c_min)) / (np.log1p(c_max) - np.log1p(c_min))) if c_max > c_min else 1
            s = norm_b * (size_max - size_min) + size_min if c_max > c_min else size_max
            handles.append(plt.scatter([], [], s=s, c='none', edgecolors='black', linewidth=0.6, label=f'{b}'))
            
        # 2. 大小图注：修改 bbox_to_anchor 固定在右侧【上方】，彻底避免重叠
        ax_f.legend(handles=handles, title="Strain Count", loc='upper left', bbox_to_anchor=(1.05, 1.0), frameon=False, labelspacing=1.0)
        ax_f.grid(True, linestyle=':', alpha=0.5)
        
    ax_f.text(-0.15, 1.05, "F", transform=ax_f.transAxes, fontsize=18, fontweight='normal', va='bottom')
    #ax_f.set_xlabel("Subtype", fontweight='normal')
    #ax_f.set_ylabel("")

    # ================= 保存与导出 =================
    plt.savefig(os.path.join(OUTPUT_DIR, "Fig6_Full_Nature_6Panel_Final_Symmetric.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, "Fig6_Full_Nature_6Panel_Final_Symmetric.svg"), format='svg', bbox_inches='tight')
    print("✅ Epic 6-Panel Fig 6 successfully reordered (Heatmap->Bubble) with PERFECT logical flow and alignment!")

if __name__ == '__main__':
    main()