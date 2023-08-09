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
from HiTE_util.Util import read_fasta_v1

if __name__ == '__main__':
    ltr_file = '/home/hukang/HiTE_Classification/data/temp/out_0.fa.ltr'
    ltr_names, ltr_contigs = read_fasta_v1(ltr_file)
    LTR_info = {}
    for ltr_name in ltr_names:
        orig_name = ltr_name.split('\t')[0]
        terminal_info = ltr_name.split('\t')[2]
        LTR_info_parts = terminal_info.split('LTR')[1].split(' ')[0].replace('(', '').replace(')', '').split('..')
        LTR_left_pos_parts = LTR_info_parts[0].split(',')
        LTR_right_pos_parts = LTR_info_parts[1].split(',')
        lLTR_start = int(LTR_left_pos_parts[0])
        lLTR_end = int(LTR_left_pos_parts[1])
        rLTR_start = int(LTR_right_pos_parts[1])
        rLTR_end = int(LTR_right_pos_parts[0])
        LTR_info[orig_name] = (lLTR_start, lLTR_end, rLTR_start, rLTR_end)