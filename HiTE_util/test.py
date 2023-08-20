import numpy as np

# def svd_reduce_dim(sequences, target_dim):
#     max_length = max(len(seq) for seq in sequences)
#     padded_sequences = [np.pad(seq, ((0, max_length - len(seq)), (0, 0)), 'constant') for seq in sequences]
#     stacked_matrix = np.vstack(padded_sequences)
#     U, s, VT = np.linalg.svd(stacked_matrix, full_matrices=False)
#     reduced_U = U[:, :target_dim]
#     reduced_s = np.diag(s[:target_dim])
#     reduced_VT = VT[:target_dim, :]
#     reduced_sequences = [np.dot(np.dot(reduced_U.T, seq), reduced_VT) for seq in padded_sequences]
#     return reduced_sequences
#
# # 示例用法
# sequences = [np.random.random((n-6, 4**7)) for n in range(10, 16)]  # 假设有6条序列，长度从10到15
# target_dim = 3  # 目标降维后的维度
#
# reduced_sequences = svd_reduce_dim(sequences, target_dim)
# for seq in reduced_sequences:
#     print(seq.shape)
from HiTE_util.Util import read_fasta_v1, word_seq, generate_kmer_dic, generate_mat


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

def split_sequence_with_padding(sequence, p):
    remainder = len(sequence) % p
    part_size = len(sequence) // p
    if remainder != 0:
        padded_sequence = "N" * ((part_size+1) * p - len(sequence))
    else:
        padded_sequence = ''
    sequence += padded_sequence

    part_size = len(sequence) // p
    parts = [sequence[i:i+part_size] for i in range(0, len(sequence), part_size)]
    return parts

if __name__ == '__main__':
    # ltr_file = '/home/hukang/NeuralTE/data/temp/out_0.fa.ltr'
    # ltr_names, ltr_contigs = read_fasta_v1(ltr_file)
    # LTR_info = {}
    # for ltr_name in ltr_names:
    #     orig_name = ltr_name.split('\t')[0]
    #     terminal_info = ltr_name.split('\t')[2]
    #     LTR_info_parts = terminal_info.split('LTR')[1].split(' ')[0].replace('(', '').replace(')', '').split('..')
    #     LTR_left_pos_parts = LTR_info_parts[0].split(',')
    #     LTR_right_pos_parts = LTR_info_parts[1].split(',')
    #     lLTR_start = int(LTR_left_pos_parts[0])
    #     lLTR_end = int(LTR_left_pos_parts[1])
    #     rLTR_start = int(LTR_right_pos_parts[1])
    #     rLTR_end = int(LTR_right_pos_parts[0])
    #     LTR_info[orig_name] = (lLTR_start, lLTR_end, rLTR_start, rLTR_end)


    # 输入碱基序列、份数 p 和 k-mer 长度 k
    sequence = "ATCATATC"
    p = 10
    k = 2
    kmer_pos_encoder, flatten_kmer_pos_encoder = get_kmer_freq_pos_info(sequence, p, k)
    print(kmer_pos_encoder)



    # # 输入碱基序列、份数 p 和 pad_length
    # sequence = "ATCGTACGTA"
    # p = 3
    #
    # # 调用函数
    # split_parts = split_sequence_with_padding(sequence, p)
    #
    # # 打印结果
    # for i, part in enumerate(split_parts):
    #     print(f"Part {i + 1}: {part}")



