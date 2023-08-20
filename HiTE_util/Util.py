#-- coding: UTF-8 --
import itertools
import os
#from openpyxl.utils import get_column_letter
#from pandas import ExcelWriter
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
#import pandas as pd

def read_fasta(fasta_path):
    contignames = []
    contigs = {}
    if os.path.exists(fasta_path):
        with open(fasta_path, 'r') as rf:
            contigname = ''
            contigseq = ''
            for line in rf:
                if line.startswith('>'):
                    if contigname != '' and contigseq != '':
                        contigs[contigname] = contigseq
                        contignames.append(contigname)
                    contigname = line.strip()[1:].split(" ")[0].split('\t')[0]
                    contigseq = ''
                else:
                    contigseq += line.strip().upper()
            if contigname != '' and contigseq != '':
                contigs[contigname] = contigseq
                contignames.append(contigname)
        rf.close()
    return contignames, contigs

def read_fasta_v1(fasta_path):
    contignames = []
    contigs = {}
    if os.path.exists(fasta_path):
        with open(fasta_path, 'r') as rf:
            contigname = ''
            contigseq = ''
            for line in rf:
                if line.startswith('>'):
                    if contigname != '' and contigseq != '':
                        contigs[contigname] = contigseq
                        contignames.append(contigname)
                    contigname = line.strip()[1:]
                    contigseq = ''
                else:
                    contigseq += line.strip().upper()
            if contigname != '' and contigseq != '':
                contigs[contigname] = contigseq
                contignames.append(contigname)
        rf.close()
    return contignames, contigs


def store_fasta(contigs, file_path):
    with open(file_path, 'w') as f_save:
        for name in contigs.keys():
            seq = contigs[name]
            f_save.write('>'+name+'\n'+seq+'\n')
    f_save.close()

def get_query_copies(cur_segments, query_contigs, subject_path, query_coverage, subject_coverage, query_fixed_extend_base_threshold=1000, subject_fixed_extend_base_threshold=1000, max_copy_num=100):
    all_copies = {}

    if subject_coverage > 0:
        subject_names, subject_contigs = read_fasta(subject_path)

    for item in cur_segments:
        query_name = item[0]
        subject_dict = item[1]

        if query_name == 'GYPSY3-intactLTR_CB':
            print('here')

        longest_queries = []
        for subject_name in subject_dict.keys():
            subject_pos = subject_dict[subject_name]

            # cluster all closed fragments, split forward and reverse records
            forward_pos = []
            reverse_pos = []
            for pos_item in subject_pos:
                if pos_item[2] > pos_item[3]:
                    reverse_pos.append(pos_item)
                else:
                    forward_pos.append(pos_item)
            forward_pos.sort(key=lambda x: (x[2], x[3]))
            reverse_pos.sort(key=lambda x: (-x[2], -x[3]))

            clusters = {}
            cluster_index = 0
            for k, frag in enumerate(forward_pos):
                if not clusters.__contains__(cluster_index):
                    clusters[cluster_index] = []
                cur_cluster = clusters[cluster_index]
                if k == 0:
                    cur_cluster.append(frag)
                else:
                    is_closed = False
                    for exist_frag in reversed(cur_cluster):
                        if (frag[2] - exist_frag[3] < subject_fixed_extend_base_threshold and frag[1] > exist_frag[1]):
                            is_closed = True
                            break
                    if is_closed:
                        cur_cluster.append(frag)
                    else:
                        cluster_index += 1
                        if not clusters.__contains__(cluster_index):
                            clusters[cluster_index] = []
                        cur_cluster = clusters[cluster_index]
                        cur_cluster.append(frag)

            cluster_index += 1
            for k, frag in enumerate(reverse_pos):
                if not clusters.__contains__(cluster_index):
                    clusters[cluster_index] = []
                cur_cluster = clusters[cluster_index]
                if k == 0:
                    cur_cluster.append(frag)
                else:
                    is_closed = False
                    for exist_frag in reversed(cur_cluster):
                        if (exist_frag[3] - frag[2] < subject_fixed_extend_base_threshold and frag[1] > exist_frag[1]):
                            is_closed = True
                            break
                    if is_closed:
                        cur_cluster.append(frag)
                    else:
                        cluster_index += 1
                        if not clusters.__contains__(cluster_index):
                            clusters[cluster_index] = []
                        cur_cluster = clusters[cluster_index]
                        cur_cluster.append(frag)

            for cluster_index in clusters.keys():
                cur_cluster = clusters[cluster_index]
                cur_cluster.sort(key=lambda x: (x[0], x[1]))

                cluster_longest_query_start = -1
                cluster_longest_query_end = -1
                cluster_longest_query_len = -1

                cluster_longest_subject_start = -1
                cluster_longest_subject_end = -1
                cluster_longest_subject_len = -1

                cluster_identity = 0
                cluster_extend_num = 0

                # record visited fragments
                visited_frag = {}
                for i in range(len(cur_cluster)):
                    # keep a longest query start from each fragment
                    origin_frag = cur_cluster[i]
                    if visited_frag.__contains__(origin_frag):
                        continue
                    cur_frag_len = origin_frag[1] - origin_frag[0] + 1
                    cur_longest_query_len = cur_frag_len
                    longest_query_start = origin_frag[0]
                    longest_query_end = origin_frag[1]
                    longest_subject_start = origin_frag[2]
                    longest_subject_end = origin_frag[3]

                    cur_identity = origin_frag[4]
                    cur_extend_num = 0

                    visited_frag[origin_frag] = 1
                    # try to extend query
                    for j in range(i + 1, len(cur_cluster)):
                        ext_frag = cur_cluster[j]
                        if visited_frag.__contains__(ext_frag):
                            continue

                        # could extend
                        # extend right
                        if ext_frag[1] > longest_query_end:
                            # judge subject direction
                            if longest_subject_start < longest_subject_end and ext_frag[2] < ext_frag[3]:
                                # +
                                if ext_frag[3] > longest_subject_end:
                                    # forward extend
                                    if ext_frag[0] - longest_query_end < query_fixed_extend_base_threshold and ext_frag[
                                        2] - longest_subject_end < subject_fixed_extend_base_threshold:
                                        # update the longest path
                                        longest_query_start = longest_query_start
                                        longest_query_end = ext_frag[1]
                                        longest_subject_start = longest_subject_start if longest_subject_start < \
                                                                                         ext_frag[
                                                                                             2] else ext_frag[2]
                                        longest_subject_end = ext_frag[3]
                                        cur_longest_query_len = longest_query_end - longest_query_start

                                        cur_identity += ext_frag[4]
                                        cur_extend_num += 1
                                        visited_frag[ext_frag] = 1
                                    elif ext_frag[0] - longest_query_end >= query_fixed_extend_base_threshold:
                                        break
                            elif longest_subject_start > longest_subject_end and ext_frag[2] > ext_frag[3]:
                                # reverse
                                if ext_frag[3] < longest_subject_end:
                                    # reverse extend
                                    if ext_frag[
                                        0] - longest_query_end < query_fixed_extend_base_threshold and longest_subject_end - \
                                            ext_frag[2] < subject_fixed_extend_base_threshold:
                                        # update the longest path
                                        longest_query_start = longest_query_start
                                        longest_query_end = ext_frag[1]
                                        longest_subject_start = longest_subject_start if longest_subject_start > \
                                                                                         ext_frag[
                                                                                             2] else ext_frag[2]
                                        longest_subject_end = ext_frag[3]
                                        cur_longest_query_len = longest_query_end - longest_query_start

                                        cur_identity += ext_frag[4]
                                        cur_extend_num += 1
                                        visited_frag[ext_frag] = 1
                                    elif ext_frag[0] - longest_query_end >= query_fixed_extend_base_threshold:
                                        break
                    if cur_longest_query_len > cluster_longest_query_len:
                        cluster_longest_query_start = longest_query_start
                        cluster_longest_query_end = longest_query_end
                        cluster_longest_query_len = cur_longest_query_len

                        cluster_longest_subject_start = longest_subject_start
                        cluster_longest_subject_end = longest_subject_end
                        cluster_longest_subject_len = abs(longest_subject_end - longest_subject_start) + 1

                        cluster_identity = cur_identity
                        cluster_extend_num = cur_extend_num
                # keep this longest query
                if cluster_longest_query_len != -1:
                    longest_queries.append((cluster_longest_query_start, cluster_longest_query_end,
                                            cluster_longest_query_len, cluster_longest_subject_start,
                                            cluster_longest_subject_end, cluster_longest_subject_len, subject_name,
                                            cluster_extend_num, cluster_identity))

        longest_queries.sort(key=lambda x: -x[2])
        query_len = len(query_contigs[query_name])
        # query_len = int(query_name.split('-')[1].split('_')[1])
        copies = []
        keeped_copies = set()
        for query in longest_queries:
            if len(copies) > max_copy_num:
                break
            subject_name = query[6]
            subject_start = query[3]
            subject_end = query[4]
            direct = '+'
            if subject_start > subject_end:
                tmp = subject_start
                subject_start = subject_end
                subject_end = tmp
                direct = '-'
            item = (subject_name, subject_start, subject_end)
            if subject_coverage > 0:
                subject_len = len(subject_contigs[subject_name])
                cur_subject_coverage = float(query[5])/subject_len
                if float(query[2])/query_len >= query_coverage and cur_subject_coverage >= subject_coverage and item not in keeped_copies:
                    copies.append((subject_name, subject_start, subject_end, query[2], direct))
                    keeped_copies.add(item)
            else:
                if float(query[2]) / query_len >= query_coverage and item not in keeped_copies:
                    copies.append((subject_name, subject_start, subject_end, query[2], direct))
                    keeped_copies.add(item)
        #copies.sort(key=lambda x: abs(x[3]-(x[2]-x[1]+1)))
        all_copies[query_name] = copies
    return all_copies

def get_copies_v1(blastnResults_path, query_path, subject_path, query_coverage=0.95, subject_coverage=0):
    query_records = {}
    with open(blastnResults_path, 'r') as f_r:
        for idx, line in enumerate(f_r):
            parts = line.split('\t')
            query_name = parts[0]
            subject_name = parts[1]
            identity = float(parts[2])
            alignment_len = int(parts[3])
            q_start = int(parts[6])
            q_end = int(parts[7])
            s_start = int(parts[8])
            s_end = int(parts[9])
            if query_name == subject_name:
                continue
            if not query_records.__contains__(query_name):
                query_records[query_name] = {}
            subject_dict = query_records[query_name]

            if not subject_dict.__contains__(subject_name):
                subject_dict[subject_name] = []
            subject_pos = subject_dict[subject_name]
            subject_pos.append((q_start, q_end, s_start, s_end, identity))
    f_r.close()

    query_names, query_contigs = read_fasta(query_path)
    cur_segments = list(query_records.items())
    all_copies = get_query_copies(cur_segments, query_contigs, subject_path, query_coverage, subject_coverage)

    return all_copies

def multiple_alignment_blast_and_get_copies(repeats_path):
    split_repeats_path = repeats_path[0]
    ref_db = repeats_path[1]
    blastn2Results_path = repeats_path[2]
    os.system('rm -f ' + blastn2Results_path)
    all_copies = None
    repeat_names, repeat_contigs = read_fasta(split_repeats_path)
    if len(repeat_contigs) > 0:
        align_command = 'blastn -db ' + ref_db + ' -num_threads ' \
                        + str(1) + ' -query ' + split_repeats_path + ' -evalue 1e-20 -outfmt 6 >> ' + blastn2Results_path
        os.system(align_command)
        all_copies = get_copies_v1(blastn2Results_path, split_repeats_path, '')
    return all_copies

def get_full_length_copies(query_path, reference):
    blastn2Results_path = query_path + '.blast.out'
    repeats_path = (query_path, reference, blastn2Results_path)
    all_copies = multiple_alignment_blast_and_get_copies(repeats_path)
    return all_copies, blastn2Results_path

def getReverseSequence(sequence):
    base_map = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    res = ''
    length = len(sequence)
    i = length - 1
    while i >= 0:
        base = sequence[i]
        if base not in base_map.keys():
            base = 'N'
        else:
            base = base_map[base]
        res += base
        i -= 1
    return res

def search_confident_tsd(orig_seq, raw_tir_start, raw_tir_end, tsd_search_distance, query_name, plant):
    #将坐标都换成以0开始的
    raw_tir_start -= 1
    raw_tir_end -= 1

    #我们之前的方法时间复杂度是100*100*9=90000
    #我现在希望通过切kmer的方法将时间复杂度降为9*（100-k）*2=1800
    #节省的时间为50倍
    itr_contigs = {}
    orig_seq_len = len(orig_seq)
    TSD_set = set()
    # 1.先取起始、结束位置附近的2*tsd_search_distance序列
    left_start = raw_tir_start - tsd_search_distance
    if left_start < 0:
        left_start = 0
    left_end = raw_tir_start + tsd_search_distance + 1
    left_round_seq = orig_seq[left_start: left_end]
    #获取left_round_seq相对于整条序列的位置偏移，用来校正后面的TSD边界位置
    left_offset = left_start
    right_start = raw_tir_end - tsd_search_distance
    if right_start < 0:
        right_start = 0
    right_end = raw_tir_end + tsd_search_distance + 1
    right_round_seq = orig_seq[right_start: right_end]
    #获取right_round_seq相对于整条序列的位置偏移，用来校正后面的TSD边界位置
    right_offset = right_start

    # 2.将左右两边的序列切成k-mer，用一个dict存起左边的k-mer，然后遍历右边k-mer时，判断是不是和左边k-mer一致，如果一致，则为一个候选的TSD，记录下位置信息
    TIR_TSDs = [11, 10, 9, 8, 6, 5, 4, 3, 2]
    # 记录的位置应该是离原始边界最近的位置
    # exist_tsd -> {'TAA': {'left_pos': 100, 'right_pos': 200}}
    exist_tsd = {}
    for k_num in TIR_TSDs:
        for i in range(len(left_round_seq) - k_num + 1):
            left_kmer = left_round_seq[i: i + k_num]
            cur_pos = left_offset + i + k_num
            if cur_pos < 0 or cur_pos > orig_seq_len-1:
                continue
            if not exist_tsd.__contains__(left_kmer):
                exist_tsd[left_kmer] = {}
            pos_dict = exist_tsd[left_kmer]
            if not pos_dict.__contains__('left_pos'):
                pos_dict['left_pos'] = cur_pos
            else:
                prev_pos = pos_dict['left_pos']
                #判断谁的位置离原始边界更近
                if abs(cur_pos-raw_tir_start) < abs(prev_pos-raw_tir_start):
                    pos_dict['left_pos'] = cur_pos
            exist_tsd[left_kmer] = pos_dict
    for k_num in TIR_TSDs:
        for i in range(len(right_round_seq) - k_num + 1):
            right_kmer = right_round_seq[i: i + k_num]
            cur_pos = right_offset + i - 1
            if cur_pos < 0 or cur_pos > orig_seq_len - 1:
                continue
            if exist_tsd.__contains__(right_kmer):
                #这是一个TSD
                pos_dict = exist_tsd[right_kmer]
                if not pos_dict.__contains__('right_pos'):
                    pos_dict['right_pos'] = cur_pos
                else:
                    prev_pos = pos_dict['right_pos']
                    # 判断谁的位置离原始边界更近
                    if abs(cur_pos - raw_tir_end) < abs(prev_pos - raw_tir_end):
                        pos_dict['right_pos'] = cur_pos
                exist_tsd[right_kmer] = pos_dict
                #判断这个TSD是否满足一些基本要求
                tir_start = pos_dict['left_pos']
                tir_end = pos_dict['right_pos']
                first_3bp = orig_seq[tir_start: tir_start + 3]
                last_3bp = orig_seq[tir_end - 2: tir_end+1]
                if (k_num != 2) or (k_num == 2 and (right_kmer == 'TA' or (plant == 0 and first_3bp == 'CCC' and last_3bp == 'GGG'))):
                    TSD_set.add((right_kmer, tir_start, tir_end))

    # 按照tir_start, tir_end与原始边界的距离进行排序，越近的排在前面
    TSD_set = sorted(TSD_set, key=lambda x: ((abs(x[1] - raw_tir_start) + abs(x[2] - raw_tir_end)), -len(x[0])))

    final_tsd = 'unknown'
    final_tsd_len = 0
    if len(TSD_set) > 0:
        tsd_info = TSD_set[0]
        final_tsd = tsd_info[0]
        final_tsd_len = len(tsd_info[0])
    return final_tsd, final_tsd_len

# def to_excel_auto_column_weight(df: pd.DataFrame, writer: ExcelWriter, sheet_name="Shee1"):
#     """DataFrame保存为excel并自动设置列宽"""
#     # 数据 to 写入器，并指定sheet名称
#     df.to_excel(writer, sheet_name=sheet_name, index=False)
#     #  计算每列表头的字符宽度
#     column_widths = (
#         df.columns.to_series().apply(lambda x: len(str(x).encode('gbk'))).values
#     )
#     #  计算每列的最大字符宽度
#     max_widths = (
#         df.astype(str).applymap(lambda x: len(str(x).encode('gbk'))).agg(max).values
#     )
#     # 取前两者中每列的最大宽度
#     widths = np.max([column_widths, max_widths], axis=0)
#     # 指定sheet，设置该sheet的每列列宽
#     worksheet = writer.sheets[sheet_name]
#     for i, width in enumerate(widths, 1):
#         # openpyxl引擎设置字符宽度时会缩水0.5左右个字符，所以干脆+2使左右都空出一个字宽。
#         worksheet.column_dimensions[get_column_letter(i)].width = width + 2

def replace_non_atcg(sequence):
    return re.sub("[^ATCG]", "", sequence)

def getRMToWicker():
    # 3.2 将Dfam分类名称转成wicker格式
    ## 3.2.1 这个文件里包含了RepeatMasker类别、Repbase、wicker类别的转换
    rmToWicker = {}
    wicker_superfamily_set = set()
    with open('TEClasses.tsv', 'r') as f_r:
        for i, line in enumerate(f_r):
            parts = line.split('\t')
            rm_type = parts[5]
            rm_subtype = parts[6]
            repbase_type = parts[7]
            wicker_type = parts[8]
            wicker_type_parts = wicker_type.split('/')
            # print(rm_type + ',' + rm_subtype + ',' + repbase_type + ',' + wicker_type)
            # if len(wicker_type_parts) != 3:
            #     continue
            wicker_superfamily_parts = wicker_type_parts[-1].strip().split(' ')
            if len(wicker_superfamily_parts) == 1:
                wicker_superfamily = wicker_superfamily_parts[0]
            elif len(wicker_superfamily_parts) > 1:
                wicker_superfamily = wicker_superfamily_parts[1].replace('(', '').replace(')', '')
            rm_full_type = rm_type + '/' + rm_subtype
            if wicker_superfamily == 'ERV':
                wicker_superfamily = 'Retrovirus'
            rmToWicker[rm_full_type] = wicker_superfamily
            wicker_superfamily_set.add(wicker_superfamily)
    # 补充一些元素
    rmToWicker['LINE/R2'] = 'R2'
    rmToWicker['LINE/RTE'] = 'RTE'
    rmToWicker['LTR/ERVL'] = 'Retrovirus'
    rmToWicker['LTR/Ngaro'] = 'DIRS'
    return rmToWicker

def load_repbase_with_TSD(path, domain_path, all_wicker_class):
    rmToWicker = getRMToWicker()
    name_labels = {}
    # 加载domain文件，读取TE包含的domain标签
    with open(domain_path, 'r') as f_r:
        for i, line in enumerate(f_r):
            if i < 2:
                continue
            parts = line.split('\t')
            TE_name = parts[0]
            label = parts[1].split('#')[1]
            if not rmToWicker.__contains__(label):
                label = 'Unknown'
            else:
                wicker_superfamily = rmToWicker[label]
                label = wicker_superfamily
                if not all_wicker_class.__contains__(label):
                    label = 'Unknown'
            if not name_labels.__contains__(TE_name):
                name_labels[TE_name] = set()
            label_set = name_labels[TE_name]
            label_set.add(label)


    names, contigs = read_fasta_v1(path)
    X = []
    Y = {}
    seq_names = []
    for name in names:
        parts = name.split('\t')
        seq_name = parts[0]
        label = parts[1]
        species_name = parts[2]
        TSD_seq = parts[3].split(':')[1]
        TSD_len = int(parts[4].split(':')[1])
        LTR_info = parts[5]
        TIR_info = parts[6]
        # LTR_info = parts[3]
        # TIR_info = parts[4]
        if name_labels.__contains__(seq_name):
            domain_label_set = name_labels[seq_name]
        else:
            domain_label_set = {'Unknown'}
        seq = contigs[name]
        seq = replace_non_atcg(seq)  # undetermined nucleotides in splice
        x_feature = (seq_name, seq, TSD_seq, TSD_len, LTR_info, TIR_info, domain_label_set)
        #x_feature = (seq_name, seq, LTR_info, TIR_info, domain_label_set)
        X.append(x_feature)
        Y[seq_name] = label
        # y_feature = (seq_name, label)
        # Y.append(y_feature)
        seq_names.append((seq_name, label))
    return X, Y, seq_names

##word_seq generates eg. ['AA', 'AT', 'TC', 'CG', 'GT']
def word_seq(seq, k, stride=1):
    i = 0
    words_list = []
    while i <= len(seq) - k:
        words_list.append(seq[i: i + k])
        i += stride
    return (words_list)

def generate_kmer_dic(repeat_num):
    ##initiate a dic to store the kmer dic
    ##kmer_dic = {'ATC':0,'TTC':1,...}
    kmer_dic = {}
    bases = ['A','G','C','T']
    kmer_list = list(itertools.product(bases, repeat=int(repeat_num)))
    for eachitem in kmer_list:
        #print(eachitem)
        each_kmer = ''.join(eachitem)
        kmer_dic[each_kmer] = 0

    return (kmer_dic)

def generate_mat(words_list,kmer_dic):
    for eachword in words_list:
        kmer_dic[eachword] += 1
    num_list = []  ##this dic stores num_dic = [0,1,1,0,3,4,5,8,2...]
    for eachkmer in kmer_dic:
        num_list.append(kmer_dic[eachkmer])
    return (num_list)


def split_and_track_kmers(sequence, p, k):
    # 将序列补齐，让其变成p的整数倍
    remainder = len(sequence) % p
    part_size = len(sequence) // p
    if remainder != 0:
        padded_sequence = "N" * ((part_size + 1) * p - len(sequence))
    else:
        padded_sequence = ''
    sequence += padded_sequence

    seq_len = len(sequence)
    part_size = seq_len // p
    part_endpoints = [i * part_size for i in range(1, p+1)]

    kmer_records = {}
    current_part = 0
    for i in range(0, seq_len - k + 1):
        if i >= part_endpoints[-1]:
            break
        kmer = sequence[i:i+k]
        if i >= part_endpoints[current_part]:
            current_part += 1
        if not kmer_records.__contains__(kmer):
            kmer_records[kmer] = []
        current_parts = kmer_records[kmer]
        current_parts.append(current_part)
    return part_endpoints, kmer_records

def get_kmer_freq_pos_info(sequence, p, k):
    # Step1: 记录每个k-mer出现的频次
    # 生成4^k个不同的k-mer
    kmer_dic = generate_kmer_dic(k)
    # 记录每个k-mer在序列中出现的频次
    words_list = word_seq(sequence, k, stride=1)
    for eachword in words_list:
        kmer_dic[eachword] += 1

    # Step2: 将序列均分成p块，并记录每个k-mer出现在第几块过，这是一个位置列表。将k-mer频次乘上位置列表，得到一个包含位置信息和频率的矩阵。
    part_endpoints, kmer_records = split_and_track_kmers(sequence, p, k)
    flatten_kmer_pos_encoder = []
    kmer_pos_encoder = {}
    # 将每个kmer出现在哪一段用one-hot编码表示，一共是p段，那么数组长度为p
    for kmer in kmer_dic:
        if kmer_records.__contains__(kmer):
            current_parts = kmer_records[kmer]
        else:
            current_parts = []
        pos_encoder = [0] * p
        for i in current_parts:
            pos_encoder[i] = 1
        freq_pos_encoder = [x * kmer_dic[kmer] for x in pos_encoder]
        kmer_pos_encoder[kmer] = freq_pos_encoder
        flatten_kmer_pos_encoder = np.concatenate((flatten_kmer_pos_encoder, freq_pos_encoder))
    return kmer_pos_encoder, flatten_kmer_pos_encoder

def get_batch_kmer_freq(grouped_x, kmer_sizes):
    group_dict = {}
    for x in grouped_x:
        seq_name = x[0]
        seq = x[1]
        TSD_seq = x[2]
        connected_num_list = []
        for kmer_size in kmer_sizes:
            words_list = word_seq(seq, kmer_size, stride=1)
            kmer_dic = generate_kmer_dic(kmer_size)
            num_list = generate_mat(words_list, kmer_dic)
            connected_num_list = np.concatenate((connected_num_list, num_list))

        # padding_length = max_TE_len - len(seq)
        # encoded_sequence = np.zeros(max_TE_len, dtype=np.int8)
        # for i, base in enumerate(seq):
        #     encoded_sequence[i] = base_num[base]
        # for i in range(padding_length):
        #     encoded_sequence[len(seq) + i] = 0
        # num_list = encoded_sequence.tolist()

        # 将TSD序列转成one-hot编码
        encoder = np.eye(4, dtype=np.int8)
        max_length = 11
        padding_length = max_length - len(TSD_seq)
        encoded_TSD = np.zeros((max_length, 4), dtype=np.int8)
        for i, base in enumerate(TSD_seq):
            if base == 'A':
                encoded_TSD[i] = encoder[0]
            elif base == 'T':
                encoded_TSD[i] = encoder[1]
            elif base == 'C':
                encoded_TSD[i] = encoder[2]
            elif base == 'G':
                encoded_TSD[i] = encoder[3]
        for i in range(padding_length):
            encoded_TSD[len(TSD_seq) + i] = np.zeros(4)
        onehot_encoded_flat = encoded_TSD.reshape(-1)
        connected_list = np.concatenate((connected_num_list, onehot_encoded_flat))
        group_dict[seq_name] = connected_list
    return group_dict

def get_batch_kmer_freq_v1(grouped_x, kmer_sizes, all_wicker_class, p):
    group_dict = {}
    for x in grouped_x:
        # 将序列拆分成internal_Seq, LTR, TIR三块组成
        seq_name = x[0]
        seq = x[1]
        TSD_seq = x[2]
        TSD_len = x[3]
        LTR_pos = x[4]
        TIR_pos = x[5]
        domain_label_set = x[6]
        # LTR_pos = x[2]
        # TIR_pos = x[3]
        # domain_label_set = x[4]
        # 获取LTR序列，以及内部序列。如果同时存在LTR和TIR，取LTR的内部序列（因为LTR相对更长，更可靠）
        internal_seq = ''
        LTR_seq = ''
        TIR_seq = ''
        LTR_pos_str = str(LTR_pos.split(':')[1]).strip()
        TIR_pos_str = str(TIR_pos.split(':')[1]).strip()
        if LTR_pos_str == '' and TIR_pos_str == '':
            internal_seq = seq
        if TIR_pos_str != '':
            TIR_parts = TIR_pos_str.split(',')
            left_TIR_start = int(TIR_parts[0].split('-')[0])
            left_TIR_end = int(TIR_parts[0].split('-')[1])
            right_TIR_start = int(TIR_parts[1].split('-')[0])
            right_TIR_end = int(TIR_parts[1].split('-')[1])
            TIR_seq = seq[left_TIR_start-1: left_TIR_end]
            internal_seq = seq[left_TIR_end: right_TIR_start-1]
        if LTR_pos_str != '':
            LTR_parts = LTR_pos_str.split(',')
            left_LTR_start = int(LTR_parts[0].split('-')[0])
            left_LTR_end = int(LTR_parts[0].split('-')[1])
            right_LTR_start = int(LTR_parts[1].split('-')[0])
            right_LTR_end = int(LTR_parts[1].split('-')[1])
            LTR_seq = seq[left_LTR_start-1: left_LTR_end]
            internal_seq = seq[left_LTR_end: right_LTR_start-1]

        # 将internal_seq，LTR表示成kmer频次和位置信息
        connected_num_list = []
        kmer_size = kmer_sizes[0]
        kmer_pos_encoder, flatten_kmer_pos_encoder = get_kmer_freq_pos_info(internal_seq, p, kmer_size)
        connected_num_list = np.concatenate((connected_num_list, flatten_kmer_pos_encoder))

        kmer_size = kmer_sizes[1]
        kmer_pos_encoder, flatten_kmer_pos_encoder = get_kmer_freq_pos_info(LTR_seq, p, kmer_size)
        connected_num_list = np.concatenate((connected_num_list, flatten_kmer_pos_encoder))

        kmer_size = kmer_sizes[2]
        kmer_pos_encoder, flatten_kmer_pos_encoder = get_kmer_freq_pos_info(TIR_seq, p, kmer_size)
        connected_num_list = np.concatenate((connected_num_list, flatten_kmer_pos_encoder))

        # # 将internal_seq，LTR，TIR表示成kmer频次
        # connected_num_list = []
        # kmer_size = kmer_sizes[0]
        # words_list = word_seq(internal_seq, kmer_size, stride=1)
        # kmer_dic = generate_kmer_dic(kmer_size)
        # num_list = generate_mat(words_list, kmer_dic)
        # connected_num_list = np.concatenate((connected_num_list, num_list))
        #
        # kmer_size = kmer_sizes[1]
        # words_list = word_seq(LTR_seq, kmer_size, stride=1)
        # kmer_dic = generate_kmer_dic(kmer_size)
        # num_list = generate_mat(words_list, kmer_dic)
        # connected_num_list = np.concatenate((connected_num_list, num_list))
        #
        # kmer_size = kmer_sizes[2]
        # words_list = word_seq(TIR_seq, kmer_size, stride=1)
        # kmer_dic = generate_kmer_dic(kmer_size)
        # num_list = generate_mat(words_list, kmer_dic)
        # connected_num_list = np.concatenate((connected_num_list, num_list))

        # TSD_len
        connected_num_list = np.append(connected_num_list, TSD_len)

        # 将TSD转成one-hot编码
        encoder = np.eye(4, dtype=np.int8)
        max_length = 11
        padding_length = max_length - len(TSD_seq)
        encoded_TSD = np.zeros((max_length, 4), dtype=np.int8)
        for i, base in enumerate(TSD_seq):
            if base == 'A':
                encoded_TSD[i] = encoder[0]
            elif base == 'T':
                encoded_TSD[i] = encoder[1]
            elif base == 'C':
                encoded_TSD[i] = encoder[2]
            elif base == 'G':
                encoded_TSD[i] = encoder[3]
        for i in range(padding_length):
            encoded_TSD[len(TSD_seq) + i] = np.zeros(4)
        onehot_encoded_flat = encoded_TSD.reshape(-1)
        connected_num_list = np.concatenate((connected_num_list, onehot_encoded_flat))

        # 将domain set转成one-hot编码
        encoder = [0] * 29
        for domain_label in domain_label_set:
            if domain_label == 'Unknown':
                domain_label_num = 28
            else:
                domain_label_num = all_wicker_class[domain_label]
            encoder[domain_label_num] = 1
        connected_num_list = np.concatenate((connected_num_list, encoder))

        # # 将domain标签转成one-hot编码
        # encoder = np.eye(len(all_wicker_class) + 1, dtype=np.int8)
        # if domain_label == 'Unknown':
        #     domain_label_num = 28
        # else:
        #     domain_label_num = all_wicker_class[domain_label]
        # encoded_domain = encoder[domain_label_num]
        # connected_list = np.concatenate((connected_list, encoded_domain))

        group_dict[seq_name] = connected_num_list
    return group_dict

def split_list_into_groups(lst, group_size):
    return [lst[i:i+group_size] for i in range(0, len(lst), group_size)]

def generate_feature_mats(X, Y, seq_names, all_wicker_class, kmer_sizes, threads, p):
    seq_mats = {}
    ex = ProcessPoolExecutor(threads)
    jobs = []
    grouped_X = split_list_into_groups(X, 100)

    for grouped_x in grouped_X:
        job = ex.submit(get_batch_kmer_freq_v1, grouped_x, kmer_sizes, all_wicker_class, p)
        jobs.append(job)
    ex.shutdown(wait=True)

    for job in as_completed(jobs):
        cur_group_dict = job.result()
        seq_mats.update(cur_group_dict)

    final_X = []
    final_Y = []
    for item in seq_names:
        seq_name = item[0]
        x = seq_mats[seq_name]
        final_X.append(x)
        label = Y[seq_name]
        label_num = all_wicker_class[label]
        final_Y.append(label_num)
    return np.array(final_X), np.array(final_Y)

##generate matrix for all samples
def generate_mats(X, seq_names, kmer_sizes, threads):
    seq_mats = {}
    ex = ProcessPoolExecutor(threads)
    jobs = []
    grouped_X = split_list_into_groups(X, 100)

    for grouped_x in grouped_X:
        job = ex.submit(get_batch_kmer_freq, grouped_x, kmer_sizes)
        jobs.append(job)
    ex.shutdown(wait=True)

    for job in as_completed(jobs):
        cur_group_dict = job.result()
        seq_mats.update(cur_group_dict)

    final_X = []
    for seq_name in seq_names:
        x = seq_mats[seq_name]
        final_X.append(x)
    return np.array(final_X)

def conv_labels(Y, all_wicker_class):
    final_Y = []
    final_Y_name = []
    for y in Y:
        seq_name = y[0]
        label = y[1]
        label_num = all_wicker_class[label]
        final_Y.append(label_num)
        final_Y_name.append((seq_name, label_num))
    return np.array(final_Y), np.array(final_Y_name)

def load_repbase(train_path, max_seq_len, threads):
    train_names, train_contigs = read_fasta_v1(train_path)

    # Step 2: 并行化加载数据
    ex = ProcessPoolExecutor(threads)
    jobs = []
    grouped_train_names = split_list_into_groups(train_names, 100)
    grouped_train_items = []
    for train_names in grouped_train_names:
        train_items = []
        for name in train_names:
            seq = train_contigs[name]
            train_items.append((name, seq))
        grouped_train_items.append(train_items)
    for train_items in grouped_train_items:
        job = ex.submit(get_batch_matrix, train_items, max_seq_len)
        jobs.append(job)
    ex.shutdown(wait=True)

    train_data = []
    train_label = []
    train_label_name = []
    for job in as_completed(jobs):
        cur_train_matrix, cur_train_label, cur_train_label_name = job.result()
        train_data.extend(cur_train_matrix)
        train_label.extend(cur_train_label)
        train_label_name.extend(cur_train_label_name)
    return np.array(train_data), np.array(train_label), np.array(train_label_name)

def get_batch_matrix(train_items, max_seq_len):
    # 将所有序列都转为max_seq_len长度，不足的部分补B，每条序列构建一个max_seq_len * 6(A/T/C/G/N/B)的矩阵
    train_data = []
    train_label = []
    for name, seq in train_items:
        seq_name = name.split('\t')[0]
        label = name.split('\t')[1]
        seq = replace_non_atcg(seq)  # 将非ATCG的碱基转为N字符
        seq = pad_string_to_max_length(seq, max_seq_len)  # 将序列补齐到最大长度，不足补’B‘

        row_num = max_seq_len
        col_num = 6
        encoder = np.eye(col_num)
        encoded_seq = np.zeros((row_num, col_num))
        for i, base in enumerate(seq):
            if base == 'A':
                encoded_seq[i] = encoder[0]
            elif base == 'T':
                encoded_seq[i] = encoder[1]
            elif base == 'C':
                encoded_seq[i] = encoder[2]
            elif base == 'G':
                encoded_seq[i] = encoder[3]
            elif base == 'N':
                encoded_seq[i] = encoder[4]
            elif base == 'B':
                encoded_seq[i] = encoder[5]
        train_data.append(encoded_seq)
        train_label.append((seq_name, label))
    # 将label转为分类数字
    all_wicker_class = {'Tc1-Mariner': 0, 'hAT': 1, 'Mutator': 2, 'Merlin': 3, 'Transib': 4, 'P': 5, 'PiggyBac': 6,
                        'PIF-Harbinger': 7, 'CACTA': 8, 'Crypton': 9, 'Helitron': 10, 'Maverick': 11, 'Copia': 12,
                        'Gypsy': 13, 'Bel-Pao': 14, 'Retrovirus': 15, 'DIRS': 16, 'Ngaro': 17, 'VIPER': 18,
                        'Penelope': 19, 'R2': 20, 'RTE': 21, 'Jockey': 22, 'L1': 23, 'I': 24, 'tRNA': 25, '7SL': 26,
                        '5S': 27}
    class_num = len(all_wicker_class)
    train_label, train_label_name = conv_labels(train_label, all_wicker_class)
    # train_label = to_categorical(train_label, int(class_num))
    return train_data, train_label, train_label_name

def pad_string_to_max_length(input_string, max_length, padding_char='B'):
    if len(input_string) >= max_length:
        return input_string
    else:
        return input_string.ljust(max_length, padding_char)

def store2file(data_partition, cur_consensus_path):
    if len(data_partition) > 0:
        with open(cur_consensus_path, 'w') as f_save:
            for item in data_partition:
                f_save.write('>'+item[0]+'\n'+item[1]+'\n')
        f_save.close()

def PET(seq_item, partitions):
    # sort contigs by length
    original = seq_item
    original = sorted(original, key=lambda x: len(x[1]), reverse=True)
    return divided_array(original, partitions)

def divided_array(original_array, partitions):
    final_partitions = [[] for _ in range(partitions)]
    node_index = 0

    read_from_start = True
    read_from_end = False
    i = 0
    j = len(original_array) - 1
    while i <= j:
        # read from file start
        if read_from_start:
            final_partitions[node_index % partitions].append(original_array[i])
            i += 1
        if read_from_end:
            final_partitions[node_index % partitions].append(original_array[j])
            j -= 1
        node_index += 1
        if node_index % partitions == 0:
            # reverse
            read_from_end = bool(1 - read_from_end)
            read_from_start = bool(1 - read_from_start)
    return final_partitions

def get_domain_info(cons, lib, output_table, threads, temp_dir):
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    # 1. 将cons进行划分，每一块利用blastx -num_threads 1 -evalue 1e-20 进行cons与domain进行比对
    partitions_num = int(threads)
    consensus_contignames, consensus_contigs = read_fasta(cons)
    data_partitions = PET(consensus_contigs.items(), partitions_num)
    merge_distance = 100
    file_list = []
    ex = ProcessPoolExecutor(threads)
    jobs = []
    for partition_index, data_partition in enumerate(data_partitions):
        if len(data_partition) <= 0:
            continue
        cur_consensus_path = temp_dir + '/'+str(partition_index)+'.fa'
        store2file(data_partition, cur_consensus_path)
        cur_output = temp_dir + '/'+str(partition_index)+'.out'
        cur_table = temp_dir + '/' + str(partition_index) + '.tbl'
        cur_file = (cur_consensus_path, lib, cur_output, cur_table)
        job = ex.submit(multiple_alignment_blastx_v1, cur_file, merge_distance)
        jobs.append(job)
    ex.shutdown(wait=True)

    # 2. 生成一个query与domain的最佳比对表
    os.system("echo 'TE_name\tdomain_name\tTE_start\tTE_end\tdomain_start\tdomain_end\n' > " + output_table)
    for job in as_completed(jobs):
        cur_table = job.result()
        os.system('cat ' + cur_table + ' >> ' + output_table)

def multiple_alignment_blastx_v1(repeats_path, merge_distance):
    split_repeats_path = repeats_path[0]
    protein_db_path = repeats_path[1]
    blastx2Results_path = repeats_path[2]
    cur_table = repeats_path[3]
    align_command = 'blastx -db ' + protein_db_path + ' -num_threads ' \
                    + str(1) + ' -evalue 1e-20 -query ' + split_repeats_path + ' -outfmt 6 > ' + blastx2Results_path
    os.system(align_command)

    fixed_extend_base_threshold = merge_distance
    # 将分段的blastx比对合并起来
    query_names, query_contigs = read_fasta(split_repeats_path)

    # parse blastn output, determine the repeat boundary
    # query_records = {query_name: {subject_name: [(q_start, q_end, s_start, s_end), (q_start, q_end, s_start, s_end), (q_start, q_end, s_start, s_end)] }}
    query_records = {}
    with open(blastx2Results_path, 'r') as f_r:
        for idx, line in enumerate(f_r):
            # print('current line idx: %d' % (idx))
            parts = line.split('\t')
            query_name = parts[0]
            subject_name = parts[1]
            identity = float(parts[2])
            alignment_len = int(parts[3])
            q_start = int(parts[6])
            q_end = int(parts[7])
            s_start = int(parts[8])
            s_end = int(parts[9])
            if not query_records.__contains__(query_name):
                query_records[query_name] = {}
            subject_dict = query_records[query_name]

            if not subject_dict.__contains__(subject_name):
                subject_dict[subject_name] = []
            subject_pos = subject_dict[subject_name]
            subject_pos.append((q_start, q_end, s_start, s_end))
    f_r.close()

    keep_longest_query = {}
    longest_repeats = {}
    for idx, query_name in enumerate(query_records.keys()):
        query_len = len(query_contigs[query_name])
        # print('total query size: %d, current query name: %s, idx: %d' % (len(query_records), query_name, idx))

        subject_dict = query_records[query_name]

        # if there are more than one longest query overlap with the final longest query over 90%,
        # then it probably the true TE
        longest_queries = []
        for subject_name in subject_dict.keys():
            subject_pos = subject_dict[subject_name]
            # subject_pos.sort(key=lambda x: (x[2], x[3]))

            # cluster all closed fragments, split forward and reverse records
            forward_pos = []
            reverse_pos = []
            for pos_item in subject_pos:
                if pos_item[0] > pos_item[1]:
                    reverse_pos.append(pos_item)
                else:
                    forward_pos.append(pos_item)
            forward_pos.sort(key=lambda x: (x[2], x[3]))
            reverse_pos.sort(key=lambda x: (-x[0], -x[1]))

            clusters = {}
            cluster_index = 0
            for k, frag in enumerate(forward_pos):
                if not clusters.__contains__(cluster_index):
                    clusters[cluster_index] = []
                cur_cluster = clusters[cluster_index]
                if k == 0:
                    cur_cluster.append(frag)
                else:
                    is_closed = False
                    for exist_frag in reversed(cur_cluster):
                        if (frag[0] - exist_frag[1] < fixed_extend_base_threshold):
                            is_closed = True
                            break
                    if is_closed:
                        cur_cluster.append(frag)
                    else:
                        cluster_index += 1
                        if not clusters.__contains__(cluster_index):
                            clusters[cluster_index] = []
                        cur_cluster = clusters[cluster_index]
                        cur_cluster.append(frag)

            cluster_index += 1
            for k, frag in enumerate(reverse_pos):
                if not clusters.__contains__(cluster_index):
                    clusters[cluster_index] = []
                cur_cluster = clusters[cluster_index]
                if k == 0:
                    cur_cluster.append(frag)
                else:
                    is_closed = False
                    for exist_frag in reversed(cur_cluster):
                        if (exist_frag[1] - frag[0] < fixed_extend_base_threshold):
                            is_closed = True
                            break
                    if is_closed:
                        cur_cluster.append(frag)
                    else:
                        cluster_index += 1
                        if not clusters.__contains__(cluster_index):
                            clusters[cluster_index] = []
                        cur_cluster = clusters[cluster_index]
                        cur_cluster.append(frag)

            for cluster_index in clusters.keys():
                cur_cluster = clusters[cluster_index]
                cur_cluster.sort(key=lambda x: (x[2], x[3]))

                cluster_longest_query_start = -1
                cluster_longest_query_end = -1
                cluster_longest_query_len = -1

                cluster_longest_subject_start = -1
                cluster_longest_subject_end = -1
                cluster_longest_subject_len = -1

                cluster_extend_num = 0

                # print('subject pos size: %d' %(len(cur_cluster)))
                # record visited fragments
                visited_frag = {}
                for i in range(len(cur_cluster)):
                    # keep a longest query start from each fragment
                    origin_frag = cur_cluster[i]
                    if visited_frag.__contains__(origin_frag):
                        continue
                    cur_frag_len = abs(origin_frag[1] - origin_frag[0])
                    cur_longest_query_len = cur_frag_len
                    longest_query_start = origin_frag[0]
                    longest_query_end = origin_frag[1]
                    longest_subject_start = origin_frag[2]
                    longest_subject_end = origin_frag[3]

                    cur_extend_num = 0

                    visited_frag[origin_frag] = 1
                    # try to extend query
                    for j in range(i + 1, len(cur_cluster)):
                        ext_frag = cur_cluster[j]
                        if visited_frag.__contains__(ext_frag):
                            continue

                        # could extend
                        # extend right
                        if ext_frag[3] > longest_subject_end:
                            # judge query direction
                            if longest_query_start < longest_query_end and ext_frag[0] < ext_frag[1]:
                                # +
                                if ext_frag[1] > longest_query_end:
                                    # forward extend
                                    if ext_frag[0] - longest_query_end < fixed_extend_base_threshold and ext_frag[
                                        2] - longest_subject_end < fixed_extend_base_threshold / 3:
                                        # update the longest path
                                        longest_query_start = longest_query_start
                                        longest_query_end = ext_frag[1]
                                        longest_subject_start = longest_subject_start if longest_subject_start < \
                                                                                         ext_frag[
                                                                                             2] else ext_frag[2]
                                        longest_subject_end = ext_frag[3]
                                        cur_longest_query_len = longest_query_end - longest_query_start
                                        cur_extend_num += 1
                                        visited_frag[ext_frag] = 1
                                    elif ext_frag[0] - longest_query_end >= fixed_extend_base_threshold:
                                        break
                            elif longest_query_start > longest_query_end and ext_frag[0] > ext_frag[1]:
                                # reverse
                                if ext_frag[1] < longest_query_end:
                                    # reverse extend
                                    if longest_query_end - ext_frag[0] < fixed_extend_base_threshold and ext_frag[
                                        2] - longest_subject_end < fixed_extend_base_threshold / 3:
                                        # update the longest path
                                        longest_query_start = longest_query_start
                                        longest_query_end = ext_frag[1]
                                        longest_subject_start = longest_subject_start if longest_subject_start < \
                                                                                         ext_frag[
                                                                                             2] else ext_frag[2]
                                        longest_subject_end = ext_frag[3]
                                        cur_longest_query_len = longest_query_start - longest_query_end
                                        cur_extend_num += 1
                                        visited_frag[ext_frag] = 1
                                    elif longest_query_end - ext_frag[0] >= fixed_extend_base_threshold:
                                        break
                    if cur_longest_query_len > cluster_longest_query_len:
                        cluster_longest_query_start = longest_query_start
                        cluster_longest_query_end = longest_query_end
                        cluster_longest_query_len = cur_longest_query_len

                        cluster_longest_subject_start = longest_subject_start
                        cluster_longest_subject_end = longest_subject_end
                        cluster_longest_subject_len = longest_subject_end - longest_subject_start

                        cluster_extend_num = cur_extend_num
                # keep this longest query
                if cluster_longest_query_len != -1:
                    longest_queries.append((cluster_longest_query_start, cluster_longest_query_end,
                                            cluster_longest_query_len, cluster_longest_subject_start,
                                            cluster_longest_subject_end, cluster_longest_subject_len, subject_name,
                                            cluster_extend_num))

        # we now consider, we should take some sequences from longest_queries to represent this query sequence.
        # we take the longest sequence by length, if the latter sequence overlap with the former sequence largely (50%),
        # continue find next sequence until the ratio of query sequence over 90% or no more sequences.
        longest_queries.sort(key=lambda x: -x[2])
        keep_longest_query[query_name] = longest_queries
    # print(keep_longest_query)
    # 记录存成table,去掉冗余记录（后一条序列的50%以上的区域在前一条内）
    with open(cur_table, 'w') as f_save:
        for query_name in keep_longest_query.keys():
            domain_array = keep_longest_query[query_name]
            # for domain_info in domain_array:
            #     f_save.write(query_name+'\t'+str(domain_info[6])+'\t'+str(domain_info[0])+'\t'+str(domain_info[1])+'\t'+str(domain_info[3])+'\t'+str(domain_info[4])+'\n')
            merge_domains = []
            # 对domain_array进行合并
            domain_array.sort(key=lambda x: -x[2])
            for domain_info in domain_array:
                if len(merge_domains) == 0:
                    merge_domains.append(domain_info)
                else:
                    is_new_domain = True
                    for pre_domain in merge_domains:
                        pre_start = pre_domain[0]
                        pre_end = pre_domain[1]
                        # 计算overlap
                        if pre_start > pre_end:
                            tmp = pre_start
                            pre_start = pre_end
                            pre_end = tmp
                        cur_start = domain_info[0]
                        cur_end = domain_info[1]
                        if cur_start > cur_end:
                            tmp = cur_start
                            cur_start = cur_end
                            cur_end = tmp
                        if cur_end >= pre_start and cur_end <= pre_end:
                            if cur_start <= pre_start:
                                overlap = cur_end - pre_start
                            else:
                                overlap = cur_end - cur_start
                        elif cur_end > pre_end:
                            if cur_start >= pre_start and cur_start <= pre_end:
                                overlap = pre_end - cur_start
                            else:
                                overlap = 0
                        else:
                            overlap = 0

                        if float(overlap / domain_info[2]) > 0.5:
                            is_new_domain = False
                    if is_new_domain:
                        merge_domains.append(domain_info)

            for domain_info in merge_domains:
                f_save.write(query_name + '\t' + str(domain_info[6]) + '\t' + str(domain_info[0]) + '\t' + str(
                    domain_info[1]) + '\t' + str(domain_info[3]) + '\t' + str(domain_info[4]) + '\n')

    f_save.close()
    return cur_table