# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import math
import csv
from collections import Counter
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ================= 1. 配置区域 =================
BASE_DIR = "../../"

DATA_FILE = os.path.join(BASE_DIR, "data", "df_IAV_8ORFs_deduplicated_labels_98787_cleaned.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "df_IAV_evolution_metrics_FULL_v4.csv")

SEGMENTS = ['PB2', 'PB1', 'PA', 'HA', 'NP', 'NA', 'M1', 'NS1']

CODON_TABLE = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M', 'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K', 'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L', 'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q', 'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V', 'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E', 'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S', 'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_', 'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W',
}
VALID_CODONS = [c for c, aa in CODON_TABLE.items() if aa != '_']

# ================= 极简宿主清洗逻辑 =================
def categorize_host_v3(row):
    h1 = str(row.get('Host1', '')).lower()
    if 'human' in h1: return 'Human'
    if 'swine' in h1: return 'Swine'
    if 'avian' in h1: return 'Avian'
    return 'Other'

# ================= 核心计算函数 =================
# 【严守红线1】：观测值/期望值 O/E ratio, 1e-9 防零除
def calc_dinuc_bias(seq):
    if not isinstance(seq, str) or len(seq) < 100:
        return np.nan, np.nan
    seq = seq.upper()
    L = len(seq)
    count = Counter(seq)
    fA, fC, fG, fT = count['A']/L, count['C']/L, count['G']/L, count['T']/L
    
    if fC * fG == 0 or fT * fA == 0:
        return np.nan, np.nan
        
    num_CpG = seq.count('CG')
    num_UpA = seq.count('TA')
    
    cpg_oe = (num_CpG / (L - 1)) / (fC * fG + 1e-9)
    upa_oe = (num_UpA / (L - 1)) / (fT * fA + 1e-9)
    return cpg_oe, upa_oe

# 【严守红线2】：拉普拉斯平滑概率提取
def get_laplace_codon_probs(seq_list):
    counter = Counter()
    total_codons = 0
    for seq in seq_list:
        seq = str(seq).upper()
        for i in range(0, len(seq) - 2, 3):
            codon = seq[i:i+3]
            if codon in VALID_CODONS:
                counter[codon] += 1
                total_codons += 1
                
    prob_dict = {}
    for c in VALID_CODONS:
        prob_dict[c] = (counter[c] + 1) / (total_codons + 61)
    return prob_dict

# 【严守红线2】：基于拉普拉斯的DKL计算
def calc_dkl_laplace(seq, ref_probs):
    if not isinstance(seq, str) or len(seq) < 100:
        return np.nan
    seq = seq.upper()
    counter = Counter()
    total_codons = 0
    for i in range(0, len(seq) - 2, 3):
        codon = seq[i:i+3]
        if codon in VALID_CODONS:
            counter[codon] += 1
            total_codons += 1
            
    if total_codons == 0:
        return np.nan
        
    dkl = 0.0
    for c in VALID_CODONS:
        p_obs = (counter[c] + 1) / (total_codons + 61)
        p_ref = ref_probs[c]
        dkl += p_obs * math.log2(p_obs / p_ref)
    return dkl

def main():
    print(f"📥 Loading data from: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE, low_memory=False)
    
    print("🧹 Applying strict Host classification & Parsing Year...")
    df['Host_Clean'] = df.apply(categorize_host_v3, axis=1)
    
    # 年份强制转换，安全过滤脏数据
    if 'Year' in df.columns:
        df['Year_Num'] = pd.to_numeric(df['Year'], errors='coerce')
    else:
        df['Year_Num'] = np.nan
    
    # 【严守红线3】：保留对 NA 神经氨酸酶的防御处理
    df.columns = df.columns.str.strip()
    if 'NA' not in df.columns:
        df['NA'] = ''
    df['NA'] = df['NA'].fillna('')
    
    print("🧬 Extracting Dual-Era Baseline Reference Populations...")
    # 1. 预警质心（全量人类毒株）
    human_prosp_mask = (df['Host_Clean'] == 'Human')
    human_prosp_seqs = [str(r.get('PB2','')) + str(r.get('HA','')) for _, r in df[human_prosp_mask].iterrows()]
    prob_human_prosp = get_laplace_codon_probs(human_prosp_seqs)
    
    # 2. 回溯质心（2009年前人类毒株）
    human_retro_mask = human_prosp_mask & (df['Year_Num'] < 2009)
    human_retro_seqs = [str(r.get('PB2','')) + str(r.get('HA','')) for _, r in df[human_retro_mask].iterrows()]
    prob_human_retro = get_laplace_codon_probs(human_retro_seqs)
    
    # 3. 禽类质心（全量背景）
    avian_seqs = [str(r.get('PB2','')) + str(r.get('HA','')) for _, r in df[df['Host_Clean'] == 'Avian'].iterrows()]
    prob_avian = get_laplace_codon_probs(avian_seqs)

    header = ['strain_name', 'Host_Clean', 'Serotype']
    for seg in SEGMENTS:
        # 注意：增加了 _Retro 后缀用于伴生指标
        header.extend([f'{seg}_CpG', f'{seg}_UpA', 
                       f'{seg}_DKL_Human', f'{seg}_DKL_Human_Retro', f'{seg}_DKL_Avian'])
        
    print("🧮 Calculating Dual-Track Metrics for ALL strains...")
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            s_name = str(row.get('strain_name', '')).strip()
            output_row = [s_name, row['Host_Clean'], row.get('Serotype', '')]
            
            for seg in SEGMENTS:
                seq = str(row.get(seg, '')).strip()
                if len(seq) < 100 or seq.upper() == 'NA' or seq == '':
                    output_row.extend([np.nan, np.nan, np.nan, np.nan, np.nan])
                    continue
                
                cpg, upa = calc_dinuc_bias(seq)
                dkl_h_prosp = calc_dkl_laplace(seq, prob_human_prosp) # 预警用
                dkl_h_retro = calc_dkl_laplace(seq, prob_human_retro) # 回溯用
                dkl_a = calc_dkl_laplace(seq, prob_avian)
                
                output_row.extend([cpg, upa, dkl_h_prosp, dkl_h_retro, dkl_a])
                
            writer.writerow(output_row)

    print(f"✅ Finished! Evolutionary metrics saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()