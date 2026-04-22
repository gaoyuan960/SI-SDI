# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import warnings

warnings.filterwarnings('ignore')

# ================= 1. 路径与全局规范配置 =================
WORK_DIR = r"d:\gaoyuan\Desktop\beijing2\20260311_fig1_7\code\geo_vrp_fig5"
INPUT_CSV = os.path.join(WORK_DIR, "df_Hex_Grid_Risk_Metrics.csv")

OUTPUT_PNG = os.path.join(WORK_DIR, "Supp_Fig_GVS_Regression.png")
OUTPUT_SVG = os.path.join(WORK_DIR, "Supp_Fig_GVS_Regression.svg")

# 期刊规范参数设定
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

# 强制全局所有文本保持正常粗细 (不加粗)
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['axes.titleweight'] = 'normal'
plt.rcParams['axes.labelweight'] = 'normal'

# ================= 2. 回归面板绘制函数 =================
def draw_regression_panel(ax, x_data, y_data, xlabel, ylabel):
    mask = ~np.isnan(x_data) & ~np.isnan(y_data)
    x = x_data[mask]
    y = y_data[mask]

    if len(x) > 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        r_squared = r_value ** 2
        p_str = "P < 0.001" if p_value < 0.001 else f"P = {p_value:.3f}"
        stat_text = f"R² = {r_squared:.3f}\n{p_str}"
    else:
        stat_text = "N/A"

    sns.regplot(
        x=x, y=y, ax=ax, 
        scatter_kws={'alpha': 0.3, 's': 15, 'color': '#7F8C8D', 'edgecolor': 'none', 'rasterized': True}, 
        line_kws={'color': '#C0392B', 'linewidth': 1.8}
    )

    ax.text(
        0.05, 0.95, stat_text, transform=ax.transAxes, 
        fontsize=10, fontweight='normal', va='top', ha='left', 
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#CCCCCC", lw=0.8, alpha=0.9)
    )

    ax.set_xlabel(xlabel, fontsize=12, fontweight='normal')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='normal')
    sns.despine(ax=ax)

# ================= 3. 主执行程序 =================
def main():
    if not os.path.exists(INPUT_CSV):
        print(f"找不到数据文件: {INPUT_CSV}")
        return

    df_grid = pd.read_csv(INPUT_CSV)
    
    print("🎨 正在生成单幅补充回归图...")
    
    # 画布重构为单图尺寸
    fig, ax = plt.subplots(figsize=(6, 5.5))

    # 执行底层映射 (已完成 X、Y 轴数据与标签的互换)
    draw_regression_panel(ax, df_grid['Mean_GVS'], df_grid['Mean_VRP'], 
                          "Mean Similarity Index (SI)", 
                          "Mean Spatial Dissemination Index (SDI)")

    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_SVG, format='svg', bbox_inches='tight')
    print(f"✅ 图表保存完毕:\n- {OUTPUT_PNG}\n- {OUTPUT_SVG}")
    plt.close()

if __name__ == '__main__':
    main()