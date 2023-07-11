#-- coding: UTF-8 --
import itertools
import os
#from openpyxl.utils import get_column_letter
#from pandas import ExcelWriter
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

def load_repbase_with_TSD(path):
    names, contigs = read_fasta_v1(path)
    X = []
    Y = []
    for name in names:
        parts = name.split('\t')
        seq_name = parts[0]
        label = parts[1]
        species_name = parts[2]
        TSD_seq = parts[3]
        seq = contigs[name]
        seq = seq.replace("Y", "C")  # undetermined nucleotides in splice
        seq = seq.replace("D", "G")
        seq = seq.replace("S", "C")
        seq = seq.replace("R", "G")
        seq = seq.replace("V", "A")
        seq = seq.replace("K", "G")
        seq = seq.replace("N", "T")
        seq = seq.replace("H", "A")
        seq = seq.replace("W", "A")
        seq = seq.replace("M", "C")
        seq = seq.replace("X", "G")
        seq = seq.replace("B", "C")
        x_feature = (seq, TSD_seq)
        X.append(x_feature)
        Y.append(label)
    return X, Y

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

##generate matrix for all samples
def generate_mats(X):
    seq_mats = []
    for x in X:
        seq = x[0]
        tsd_seq = x[1]
        words_list = word_seq(seq, 7, stride=1)  ##change the k to 3
        kmer_dic = generate_kmer_dic(7)  ##this number should be the same as the window slide number
        num_list = generate_mat(words_list, kmer_dic)

        ##store the all the samples into seq_mats
        ##seq_mats = [[0,1,3,4],[3,4,5,6],...]
        seq_mats.append(num_list)
    return (seq_mats)