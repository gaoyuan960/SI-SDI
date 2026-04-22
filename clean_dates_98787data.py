import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# 配置输入和输出路径（不改变原文件，生成新文件）
input_file = './data/df_IAV_8ORFs_deduplicated_labels_98787.csv'
output_file = './data/df_IAV_8ORFs_deduplicated_labels_98787_cleaned.csv'

def process_date_logic(row):
    year_val = row['Year']
    date_val = row['Date']
    
    # 规则 4: Year 为 Unknown (或为空)
    if pd.isna(year_val) or str(year_val).strip().lower() == 'unknown':
        return 'Unknown', False
        
    try:
        # 将 Year 转换为数字基准，处理可能存在的浮点表示 (如 2022.0)
        target_year = int(float(year_val))
    except (ValueError, TypeError):
        return 'Unknown', False

    # 如果 Date 为空，直接按 Year 的正确年份处理
    if pd.isna(date_val) or str(date_val).strip() == '':
        return str(target_year), False
        
    date_str = str(date_val).strip()
    # 清理一下某些系统下自动带上的浮点数小尾巴
    if date_str.endswith('.0'):
        date_str = date_str[:-2]
        
    # 规则 1: Date 里是单纯的年份整数 (如 "1968")
    if date_str.isdigit() and len(date_str) == 4:
        if int(date_str) == target_year:
            return date_str, False
        else:
            return str(target_year), False
            
    # 规则 2: Excel 计数类型日期 (如 "40375" 或 "28233")
    if date_str.isdigit() and 10000 < int(date_str) < 80000:
        excel_days = int(date_str)
        # 核心修正逻辑：
        # 标准 Excel 1900 系统是以 1899-12-30 为起点的。
        # 你提到都需要减去一天，所以我们将基准日往前推一天，设定为 1899-12-29
        corrected_date = datetime(1899, 12, 29) + timedelta(days=excel_days)
        
        if corrected_date.year == target_year:
            # 统一格式化为 YYYY-MM-DD
            return corrected_date.strftime('%Y-%m-%d'), True
        else:
            return str(target_year), False
            
    # 规则 3: 字符串日期格式 (如 "11/23/2022")
    try:
        # 让 pandas 智能尝试解析字符串日期
        parsed_date = pd.to_datetime(date_str, errors='coerce')
        if pd.notna(parsed_date):
            if parsed_date.year == target_year:
                # 确保它确实是“完整日期”而不是纯年份（长度 > 4 说明有月份/日期）
                if len(date_str) > 4:
                    return parsed_date.strftime('%Y-%m-%d'), True
                else:
                    return str(target_year), False
            else:
                return str(target_year), False
    except Exception:
        pass
        
    # 兜底规则：如果不符合上述所有情况，退回到只记正确年份
    return str(target_year), False

def main():
    print("🚀 开始读取数据...")
    if not os.path.exists(input_file):
        print(f"❌ 找不到文件: {input_file}，请检查路径。")
        return
        
    df = pd.read_csv(input_file, low_memory=False)
    
    print("🧮 正在按您的规则清洗日期数据 (这可能需要十几秒钟)...")
    # 应用逻辑，逐行处理
    result = df.apply(process_date_logic, axis=1)
    
    # 拆包结果至最后两列
    df['clean_date'] = [res[0] for res in result]
    df['month'] = [res[1] for res in result]
    
    print("💾 正在保存新文件...")
    df.to_csv(output_file, index=False)
    print(f"✅ 处理完成！")
    print(f"数据已安全地另存为: {output_file}")
    print("新增列: 'clean_date' 和 'month' 已追加至末尾。")

if __name__ == '__main__':
    main()