import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import h3

# ==========================================
# 1. 基础配置与环境初始化
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

WORK_DIR = r"d:\gaoyuan\Desktop\beijing2\20260311_fig1_7\code\geo_vrp_fig5"
DATA_DIR = r"d:\gaoyuan\Desktop\beijing2\20260311_fig1_7\data"

INPUT_CSV = os.path.join(DATA_DIR, "df_IAV_8ORFs_deduplicated_labels_98787_cleaned.csv")
JSON_DICT_PATH = os.path.join(WORK_DIR, "geo_dict_full.json")
OUTPUT_CSV = os.path.join(WORK_DIR, "df_IAV_VRP_H3_Row.csv")

# 引入离线字典
sys.path.append(WORK_DIR)
try:
    from geo_13 import GEO_DICT_OFFLINE
    logging.info("成功加载 geo_13.py 离线字典。")
except ImportError:
    logging.error("无法导入 geo_13.py，请检查文件是否存在于工作目录中。程序终止。")
    sys.exit(1)

# 加载 JSON API 字典
try:
    with open(JSON_DICT_PATH, 'r', encoding='utf-8') as f:
        GEO_DICT_JSON = json.load(f)
    logging.info("成功加载 geo_dict_full.json。")
except Exception as e:
    logging.error(f"加载 JSON 字典失败: {e}")
    sys.exit(1)

# ==========================================
# 2. 核心处理函数定义
# ==========================================
HOSTS_TO_REMOVE = {'swine', 'duck', 'pelican', 'chicken', 'goose', 'avian', 'equine', 'canine', 'feline', 'environment', 'mallard', 'turkey', 'gull'}

def parse_location(strain_name):
    """从 strain_name 提取标准化地点并剔除无关宿主"""
    if pd.isna(strain_name):
        return np.nan
    parts = str(strain_name).split('/')
    if len(parts) < 2:
        return np.nan
    
    loc = parts[1].strip().lower()
    # 如果提取出的部分为无关宿主名且路径分段足够，则取下一层级
    if loc in HOSTS_TO_REMOVE and len(parts) > 2:
        loc = parts[2].strip().lower()
    return loc

def map_coordinates(loc):
    """双轨匹配机制获取经纬度"""
    if pd.isna(loc) or not loc:
        return np.nan, np.nan
    
    # 优先级 1: JSON API 字典
    if loc in GEO_DICT_JSON:
        coords = GEO_DICT_JSON[loc]
        if coords and len(coords) >= 2 and pd.notna(coords[0]) and pd.notna(coords[1]):
            return float(coords[0]), float(coords[1])
            
    # 优先级 2: 离线备用字典
    if loc in GEO_DICT_OFFLINE:
        coords = GEO_DICT_OFFLINE[loc]
        if coords and len(coords) >= 2 and pd.notna(coords[0]) and pd.notna(coords[1]):
            return float(coords[0]), float(coords[1])
            
    return np.nan, np.nan

def get_h3_info(lat, lon, resolution=2):
    """将经纬度转换为 H3 网格 ID 并提取网格中心坐标"""
    if pd.isna(lat) or pd.isna(lon):
        return pd.Series([np.nan, np.nan, np.nan])
    try:
        # 兼容 h3-py v4 和 v3 的 API
        hex_id = h3.latlng_to_cell(lat, lon, resolution) if hasattr(h3, 'latlng_to_cell') else h3.geo_to_h3(lat, lon, resolution)
        center_lat, center_lon = h3.cell_to_latlng(hex_id) if hasattr(h3, 'cell_to_latlng') else h3.h3_to_geo(hex_id)
        return pd.Series([hex_id, center_lat, center_lon])
    except Exception:
        return pd.Series([np.nan, np.nan, np.nan])

def haversine_vectorized(lat1, lon1, lat2, lon2):
    """向量化球面距离计算 (Haversine formula), 返回公里数"""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# ==========================================
# 3. 主执行流程
# ==========================================
def main():
    logging.info("读取原始清洗数据...")
    try:
        df = pd.read_csv(INPUT_CSV)
    except Exception as e:
        logging.error(f"读取数据失败: {e}")
        sys.exit(1)

    initial_len = len(df)
    
    # --- 1. 空间坐标映射 ---
    logging.info("解析地点并匹配地理坐标...")
    df['Extracted_Loc'] = df['strain_name'].apply(parse_location)
    
    # 运用双轨匹配获取初始经纬度
    coords = df['Extracted_Loc'].apply(map_coordinates)
    df['Raw_Lat'] = coords.apply(lambda x: x[0])
    df['Raw_Lon'] = coords.apply(lambda x: x[1])
    
    # 剔除无法匹配的空间数据（严格禁止国家级中心点填补）
    df = df.dropna(subset=['Raw_Lat', 'Raw_Lon'])
    mapped_len = len(df)
    logging.info(f"空间坐标匹配完成。初始样本: {initial_len}, 保留样本: {mapped_len} (去除缺失 {initial_len - mapped_len} 条)")

    # --- 2. H3 六边形网格化 ---
    logging.info("执行 H3 (Resolution 2) 六边形网格化...")
    df[['Hex_ID', 'Hex_Lat', 'Hex_Lon']] = df.apply(
        lambda row: get_h3_info(row['Raw_Lat'], row['Raw_Lon'], resolution=2), 
        axis=1
    )
    df = df.dropna(subset=['Hex_ID']) # 再次剔除网格化异常数据

    # --- 3. VRP 白盒算法核心 ---
    logging.info("计算局部密度 (Local_Density)...")
    density_df = df.groupby(['Year', 'Serotype', 'Hex_ID']).size().reset_index(name='Local_Density')
    df = df.merge(density_df, on=['Year', 'Serotype', 'Hex_ID'], how='left')

    logging.info("计算加权质心 (Weighted Centroids)...")
    # 提取唯一的网格以准确计算质心，防止样本重复导致密度权重二次放大
    hex_unique = df[['Year', 'Serotype', 'Hex_ID', 'Hex_Lat', 'Hex_Lon', 'Local_Density']].drop_duplicates()
    
    def calculate_weighted_centroid(group):
        w = group['Local_Density']
        # 处理可能的权重全为0的情况（虽理论上不可能）
        if w.sum() == 0:
            return pd.Series({'Center_Lat': np.nan, 'Center_Lon': np.nan})
        return pd.Series({
            'Center_Lat': np.average(group['Hex_Lat'], weights=w),
            'Center_Lon': np.average(group['Hex_Lon'], weights=w)
        })

    centers = hex_unique.groupby(['Year', 'Serotype']).apply(calculate_weighted_centroid).reset_index()
    df = df.merge(centers, on=['Year', 'Serotype'], how='left')

    logging.info("计算空间距离 (Dist_to_Center)...")
    df['Dist_to_Center'] = haversine_vectorized(df['Hex_Lat'], df['Hex_Lon'], df['Center_Lat'], df['Center_Lon'])

    logging.info("执行白盒算法公式与归一化...")
    # 公式：VRP_Score = ln(1 + Dist_to_Center) * ln(1 + Local_Density)
    df['VRP_Score'] = np.log(1 + df['Dist_to_Center']) * np.log(1 + df['Local_Density'])

    # 归一化 Min-Max 到 0-100
    min_vrp = df['VRP_Score'].min()
    max_vrp = df['VRP_Score'].max()
    if max_vrp > min_vrp:
        df['VRP_Norm'] = (df['VRP_Score'] - min_vrp) / (max_vrp - min_vrp) * 100
    else:
        df['VRP_Norm'] = 0.0
        logging.warning("VRP_Score 最大值与最小值相同，归一化均记为 0。")

    # --- 4. 导出结果 ---
    logging.info(f"保存处理结果至: {OUTPUT_CSV}")
    df.to_csv(OUTPUT_CSV, index=False)
    logging.info("处理完毕。")

if __name__ == "__main__":
    main()