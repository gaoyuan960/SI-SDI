# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 11:06:36 2021

@author: Jing Li, Small steps make changes. dnt_seq@163.com
"""
import pandas as pd
import numpy as np
import os
# from Bio import SeqIO
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# import seaborn as sns


import numpy as np

nt_table = ['t','c', 'a', 'g']
dnt_table = [nt1+nt2 for nt1 in nt_table for nt2 in nt_table]
dnts_table = [dnt1+dnt2 for dnt1 in dnt_table for dnt2 in dnt_table]
codon_table = [nt1+nt2+nt3 for nt1 in nt_table for nt2 in nt_table for nt3 in nt_table]
codon_table1 = codon_table.copy()

stop_codon = ['taa', 'tag', 'tga']
stop_codonpair = [i+j for i in stop_codon for j in stop_codon]

codonpair_table0 = [codon0 + codon1 for codon0 in codon_table1 for codon1 in codon_table1]
codonpair_table = []
for cp in codonpair_table0:
    if cp not in stop_codonpair:
        codonpair_table.append(cp)


print (len(codonpair_table0))


def count_DCR (cds, dnts_table = dnts_table,\
               codonpair_table = codonpair_table,\
               Num0 = 256, Num3 = 4087, freq = True):
    seq_len = len(cds)
    cds = cds.lower()


#################################################   DNTpair counting
    count_N12M12 = np.zeros(Num0)
    for i in range(0, seq_len, 3):
        cut1 = cds[i:i+2]
        cut2 = cds[i+3:i+5]
        cut = cut1 + cut2
        if cut in dnts_table:
            dnts = dnts_table.index(cut)
            count_N12M12[dnts] += 1

    count_N23M23 = np.zeros(Num0)
    for i in range(0, seq_len, 3):
        cut1 = cds[i+1:i+3]
        cut2 = cds[i+4:i+6]
        cut = cut1 + cut2
        if cut in dnts_table:
            dnts = dnts_table.index(cut)
            count_N23M23[dnts] += 1

    count_N31M31 = np.zeros(Num0)
    for i in range(0, seq_len, 3):
        cut1 = cds[i+2:i+4]
        cut2 = cds[i+5:i+7]
        cut = cut1 + cut2
        if cut in dnts_table:
            dnts = dnts_table.index(cut)
            count_N31M31[dnts] += 1

    count_N12N31 = np.zeros(Num0)
    for i in range(0, seq_len, 3):
        # cut1 = cds[i:i + 2]
        # cut2 = cds[i+2:i + 4]
        cut = cds[i:i+4]
        if cut in dnts_table:
            dnts = dnts_table.index(cut)
            count_N12N31[dnts] += 1

    count_N23M12 = np.zeros(Num0)
    for i in range(0, seq_len, 3):
        cut = cds[i+1:i+5]
        if cut in dnts_table:
            dnts = dnts_table.index(cut)
            count_N23M12[dnts] += 1

    count_N31M23 = np.zeros(Num0)
    for i in range(0, seq_len, 3):
        cut = cds[i+2:i+6]
        if cut in dnts_table:
            dnts = dnts_table.index(cut)
            count_N31M23[dnts] += 1

#################################################   codonpair counting
    count_codonpair = np.zeros(Num3)
    for i in range(0, seq_len, 3):
        cut = cds[i:i+6]
        if cut in codonpair_table:
            codonpair = codonpair_table.index(cut)
            count_codonpair[codonpair] += 1


    count_dnts = np.hstack((count_N12M12, count_N23M23, count_N31M31, count_N12N31, count_N23M12, count_N31M23))

    '''
    ## counting twice for each nt, and twice for each dnt.
    ## Therefore, the freq for a dnt = count_dnts*1536 / (seq_len-1)/2
    '''

    if freq:
        return np.hstack ((count_dnts*256*3 / seq_len, \
                           count_codonpair*4087/seq_len))
    else:
        return np.hstack ((count_dnts, count_codonpair))

################################################## test 
################################################## test 

# Seq_S = 'ATGTTTGTTTTTCTTGTTTTATTGCCACTAGTCTCTAGTCAGTGTGTTAATCTTACAACCAGAACTCAATTACCCCCTGCATACACTAATTCTTTCACACGTGGTGTTTATTACCCTGACAAAGTTTTCAGATCCTCAGTTTTACATTCAACTCAGGACTTGTTCTTACCTTTCTTTTCCAATGTTACTTGGTTCCATGCTATACATGTCTCTGGGACCAATGGTACTAAGAGGTTTGATAACCCTGTCCTACCATTTAATGATGGTGTTTATTTTGCTTCCACTGAGAAGTCTAACATAATAAGAGGCTGGATTTTTGGTACTACTTTAGATTCGAAGACCCAGTCCCTACTTATTGTTAATAACGCTACTAATGTTGTTATTAAAGTCTGTGAATTTCAATTTTGTAATGATCCATTTTTGGGTGTTTATTACCACAAAAACAACAAAAGTTGGATGGAAAGTGAGTTCAGAGTTTATTCTAGTGCGAATAATTGCACTTTTGAATATGTCTCTCAGCCTTTTCTTATGGACCTTGAAGGAAAACAGGGTAATTTCAAAAATCTTAGGGAATTTGTGTTTAAGAATATTGATGGTTATTTTAAAATATATTCTAAGCACACGCCTATTAATTTAGTGCGTGATCTCCCTCAGGGTTTTTCGGCTTTAGAACCATTGGTAGATTTGCCAATAGGTATTAACATCACTAGGTTTCAAACTTTACTTGCTTTACATAGAAGTTATTTGACTCCTGGTGATTCTTCTTCAGGTTGGACAGCTGGTGCTGCAGCTTATTATGTGGGTTATCTTCAACCTAGGACTTTTCTATTAAAATATAATGAAAATGGAACCATTACAGATGCTGTAGACTGTGCACTTGACCCTCTCTCAGAAACAAAGTGTACGTTGAAATCCTTCACTGTAGAAAAAGGAATCTATCAAACTTCTAACTTTAGAGTCCAACCAACAGAATCTATTGTTAGATTTCCTAATATTACAAACTTGTGCCCTTTTGGTGAAGTTTTTAACGCCACCAGATTTGCATCTGTTTATGCTTGGAACAGGAAGAGAATCAGCAACTGTGTTGCTGATTATTCTGTCCTATATAATTCCGCATCATTTTCCACTTTTAAGTGTTATGGAGTGTCTCCTACTAAATTAAATGATCTCTGCTTTACTAATGTCTATGCAGATTCATTTGTAATTAGAGGTGATGAAGTCAGACAAATCGCTCCAGGGCAAACTGGAAAGATTGCTGATTATAATTATAAATTACCAGATGATTTTACAGGCTGCGTTATAGCTTGGAATTCTAACAATCTTGATTCTAAGGTTGGTGGTAATTATAATTACCTGTATAGATTGTTTAGGAAGTCTAATCTCAAACCTTTTGAGAGAGATATTTCAACTGAAATCTATCAGGCCGGTAGCACACCTTGTAATGGTGTTGAAGGTTTTAATTGTTACTTTCCTTTACAATCATATGGTTTCCAACCCACTAATGGTGTTGGTTACCAACCATACAGAGTAGTAGTACTTTCTTTTGAACTTCTACATGCACCAGCAACTGTTTGTGGACCTAAAAAGTCTACTAATTTGGTTAAAAACAAATGTGTCAATTTCAACTTCAATGGTTTAACAGGCACAGGTGTTCTTACTGAGTCTAACAAAAAGTTTCTGCCTTTCCAACAATTTGGCAGAGACATTGCTGACACTACTGATGCTGTCCGTGATCCACAGACACTTGAGATTCTTGACATTACACCATGTTCTTTTGGTGGTGTCAGTGTTATAACACCAGGAACAAATACTTCTAACCAGGTTGCTGTTCTTTATCAGGATGTTAACTGCACAGAAGTCCCTGTTGCTATTCATGCAGATCAACTTACTCCTACTTGGCGTGTTTATTCTACAGGTTCTAATGTTTTTCAAACACGTGCAGGCTGTTTAATAGGGGCTGAACATGTCAACAACTCATATGAGTGTGACATACCCATTGGTGCAGGTATATGCGCTAGTTATCAGACTCAGACTAATTCTCCTCGGCGGGCACGTAGTGTAGCTAGTCAATCCATCATTGCCTACACTATGTCACTTGGTGCAGAAAATTCAGTTGCTTACTCTAATAACTCTATTGCCATACCCACAAATTTTACTATTAGTGTTACCACAGAAATTCTACCAGTGTCTATGACCAAGACATCAGTAGATTGTACAATGTACATTTGTGGTGATTCAACTGAATGCAGCAATCTTTTGTTGCAATATGGCAGTTTTTGTACACAATTAAACCGTGCTTTAACTGGAATAGCTGTTGAACAAGACAAAAACACCCAAGAAGTTTTTGCACAAGTCAAACAAATTTACAAAACACCACCAATTAAAGATTTTGGTGGTTTTAATTTTTCACAAATATTACCAGATCCATCAAAACCAAGCAAGAGGTCATTTATTGAAGATCTACTTTTCAACAAAGTGACACTTGCAGATGCTGGCTTCATCAAACAATATGGTGATTGCCTTGGTGATATTGCTGCTAGAGACCTCATTTGTGCACAAAAGTTTAACGGCCTTACTGTTTTGCCACCTTTGCTCACAGATGAAATGATTGCTCAATACACTTCTGCACTGTTAGCGGGTACAATCACTTCTGGTTGGACCTTTGGTGCAGGTGCTGCATTACAAATACCATTTGCTATGCAAATGGCTTATAGGTTTAATGGTATTGGAGTTACACAGAATGTTCTCTATGAGAACCAAAAATTGATTGCCAACCAATTTAATAGTGCTATTGGCAAAATTCAAGACTCACTTTCTTCCACAGCAAGTGCACTTGGAAAACTTCAAGATGTGGTCAACCAAAATGCACAAGCTTTAAACACGCTTGTTAAACAACTTAGCTCCAATTTTGGTGCAATTTCAAGTGTTTTAAATGATATCCTTTCACGTCTTGACAAAGTTGAGGCTGAAGTGCAAATTGATAGGTTGATCACAGGCAGACTTCAAAGTTTGCAGACATATGTGACTCAACAATTAATTAGAGCTGCAGAAATCAGAGCTTCTGCTAATCTTGCTGCTACTAAAATGTCAGAGTGTGTACTTGGACAATCAAAAAGAGTTGATTTTTGTGGAAAGGGCTATCATCTTATGTCCTTCCCTCAGTCAGCACCTCATGGTGTAGTCTTCTTGCATGTGACTTATGTCCCTGCACAAGAAAAGAACTTCACAACTGCTCCTGCCATTTGTCATGATGGAAAAGCACACTTTCCTCGTGAAGGTGTCTTTGTTTCAAATGGCACACACTGGTTTGTAACACAAAGGAATTTTTATGAACCACAAATCATTACTACAGACAACACATTTGTGTCTGGTAACTGTGATGTTGTAATAGGAATTGTCAACAACACAGTTTATGATCCTTTGCAACCTGAATTAGACTCATTCAAGGAGGAGTTAGATAAATATTTTAAGAATCATACATCACCAGATGTTGATTTAGGTGACATCTCTGGCATTAATGCTTCAGTTGTAAACATTCAAAAAGAAATTGACCGCCTCAATGAGGTTGCCAAGAATTTAAATGAATCTCTCATCGATCTCCAAGAACTTGGAAAGTATGAGCAGTATATAAAATGGCCATGGTACATTTGGCTAGGTTTTATAGCTGGCTTGATTGCCATAGTAATGGTGACAATTATGCTTTGCTGTATGACCAGTTGCTGTAGTTGTCTCAAGGGCTGTTGTTCTTGTGGATCCTGCTGCAAATTTGATGAAGACGACTCTGAGCCAGTGCTCAAAGGAGTCAAATTACATTACACATAA'
# Seq_S = 'ATGATGatg'
# test_count_DCR_S = count_DCR(Seq_S,freq = True)
# testlst = list(test_count_DCR_S)
# testlst = [i for i in testlst if i != 0]
# print ((testlst))
################################################## test 
################################################## test



dnt_category = ['n12', 'n23','n31']
dntpair_category = ['n12m12', 'n23m23','n31m31','n12n31','n23m12','n31m23']
dntpair_cols_list = ['Freq_'+ dnts + '_' + dnts_cat for dnts_cat in dntpair_category for dnts in dnts_table]
codonpair_cols_list = ['Freq_'+ codonpair for codonpair in codonpair_table]


full_cols_list0 = dntpair_cols_list + codonpair_cols_list 
full_cols_list1 = [col[5:] for col in full_cols_list0]


path = r"../data/"
file_list = os.listdir(path)


target_cols = ['HA', 'NP', 'NA ', 'M1', 'NS1']
# target_cols = ['CDS']

for file_obj in file_list:
    
    file_name_target = 'df_IAV_8ORFs_deduplicated_labels_98787.csv'
    
    
    '''
    file_name_target is the file name of the target sequence data
    to count the feature of DCR and codonpair, such as "df_concat_cdsRdRp_host_143654.csv",
    however, a downsampling was suggested before counting.
    '''
    

    if file_obj.startswith (file_name_target) & file_obj.endswith('.csv'):
        
        
        
        df_fs = pd.read_csv (path + file_obj, encoding = 'gb2312')
        
        seqIDlst = df_fs['strain_name'].tolist()
        
        for cds in target_cols:
            cdsLst = df_fs[cds].tolist()
        
        

            '''
            'ID': the index column name of sequence ID, was updatable based on the index column name of user,
            'targetCDS': the column name of sequence, was updatable based on the sequence column name of user,
            
            '''        
           
            seq_num = len(seqIDlst)
            id_list = seqIDlst
            
            dnt_count_list = []
            df_NCR = pd.DataFrame ()
            array_count = np.zeros(shape = (seq_num,1))
            orf_seq_list = cdsLst
            print (len(orf_seq_list))
            for seq_i in range(seq_num):
                my_seq = orf_seq_list[seq_i].lower()        #################### code should be editted if sliding window needed
                                   
                seq_len = (len(my_seq))
                freq_dnt = count_DCR(my_seq)
                dnt_count_list.append(freq_dnt)
            # print (len(dnt_count_list))
    
            # print(len(id_list))
            array_freq_dnt = np.array(dnt_count_list)
            print (array_freq_dnt.shape)
            
            df_NCR['seqID'] = id_list
            df_NCR[full_cols_list0] = array_freq_dnt
            df_NCR = df_NCR.set_index(['seqID'])
    
            df_NCR.to_csv ('df_dcrcp_counting_' + cds + '_' + file_obj[:-4] + '.csv')

###########################################################################
###########################################################################
###########################################################################
