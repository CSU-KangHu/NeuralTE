import random
import re
import pandas as pd
import numpy as np

import configs.config
from HiTE_util.Util import read_fasta_v1, word_seq, generate_kmer_dic, generate_mat
from utils.data_util import merge_tsd_terminal_repbase, generate_random_sequences, generate_random_sequence, \
    store_fasta, summary_class_ratio, expandRepBase, generate_terminal_info, to_excel_auto_column_weight, \
    split2TrainTest, replace_non_atcg
from utils.evaluate_util import add_ClassifyTE_classification, filterRepbase, evaluate_RepeatClassifier, \
    plot_3D_param, transform_TERL_data, evaluate_TERL, evaluate_ClassifyTE, transform_repbase_to_DeepTE_input, \
    transform_DeepTE_to_fasta, evaluate_DeepTE


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
    # work_dir = '/home/hukang/NeuralTE/data'
    # tsd_repbase = work_dir + '/repbase_total.tsd.ref'
    # terminal_repbase = work_dir + '/repbase_total.terminal.ref'
    # merge_repbase = work_dir + '/repbase_total.ref'
    # total_domain = work_dir + '/all_repbase.ref_preprocess.ref.update.domain'
    # merge_domain = merge_repbase + '.domain'
    # merge_tsd_terminal_repbase(tsd_repbase, terminal_repbase, merge_repbase, total_domain, merge_domain)

    # num_sequences = 3760
    # sequence_lengths = generate_random_sequences(num_sequences)
    # # 生成序列并写入文件
    # output_file = work_dir + "/random_sequences.ref"
    # with open(output_file, "w") as f:
    #     for i, length in enumerate(sequence_lengths):
    #         header = f"Random_{i}\tUnknown\tRandom\tTSD:\tTSD_len:\tLTR:\tTIR:"
    #         sequence = generate_random_sequence(length)
    #         f.write(f">{header}\n{sequence}\n")
    #
    # print(f"随机生成的序列已保存到文件：{output_file}")

    # work_dir = '/home/hukang/NeuralTE/data'
    # cur_repbase_total_path = work_dir + '/repbase_total.ref'
    # names, contigs = read_fasta_v1(cur_repbase_total_path)
    # # 随机打乱列表
    # random.shuffle(names)
    # shuffle_contigs = {}
    # for name in names:
    #     shuffle_contigs[name] = contigs[name]
    # store_fasta(shuffle_contigs, cur_repbase_total_path)

    # work_dir = '/home/hukang/NeuralTE/data'
    # cur_repbase_train_path = work_dir + '/repbase_train.ref'
    # cur_repbase_test_path = work_dir + '/repbase_test.ref'
    # # 1. 先将Repbase数据按照8-2比例划分成训练集和测试集
    # cur_repbase_path = work_dir + '/repbase_total.ref'
    # names, contigs = read_fasta_v1(cur_repbase_path)
    # # 随机打乱列表
    # random.shuffle(names)
    # # 计算划分的索引位置
    # split_index = int(0.8 * len(names))
    # # 划分成80%和20%的两个列表
    # train_list = names[:split_index]
    # test_list = names[split_index:]
    # train_contigs = {}
    # test_contigs = {}
    # for name in train_list:
    #     train_contigs[name] = contigs[name]
    # for name in test_list:
    #     test_contigs[name] = contigs[name]
    # store_fasta(train_contigs, cur_repbase_train_path)
    # store_fasta(test_contigs, cur_repbase_test_path)
    # summary_class_ratio(cur_repbase_train_path)
    # summary_class_ratio(cur_repbase_test_path)

    # # 评估LTR和TIR的长度分布范围，最小、最长、平均、中位数
    # data_dir = '/home/hukang/NeuralTE/data'
    # data_path = data_dir + '/repbase_total.64.ref.update'
    # names, contigs = read_fasta_v1(data_path)
    # LTR_length = []
    # TIR_length = []
    # for name in names:
    #     parts = name.split('\t')
    #     LTR_info = parts[5]
    #     TIR_info = parts[6]
    #     LTR_pos = LTR_info.split(':')[1]
    #     d_parts = LTR_pos.split(',')
    #     if len(d_parts) == 2:
    #         LTR_len = d_parts[0].split('-')[1]
    #         LTR_length.append(LTR_len)
    #     TIR_pos = TIR_info.split(':')[1]
    #     d_parts = TIR_pos.split(',')
    #     if len(d_parts) == 2:
    #         TIR_len = d_parts[0].split('-')[1]
    #         TIR_length.append(TIR_len)
    #
    # array = np.array(LTR_length, dtype=int)
    # print('LTR:')
    # # 计算最小值
    # min_value = np.min(array)
    # print("最小值:", min_value)
    # # 计算最大值
    # max_value = np.max(array)
    # print("最大值:", max_value)
    # # 计算平均值
    # mean_value = np.mean(array)
    # print("平均值:", mean_value)
    # # 计算中位数
    # median_value = np.median(array)
    # print("中位数:", median_value)
    #
    # array = np.array(TIR_length, dtype=int)
    # print('TIR:')
    # # 计算最小值
    # min_value = np.min(array)
    # print("最小值:", min_value)
    # # 计算最大值
    # max_value = np.max(array)
    # print("最大值:", max_value)
    # # 计算平均值
    # mean_value = np.mean(array)
    # print("平均值:", mean_value)
    # # 计算中位数
    # median_value = np.median(array)
    # print("中位数:", median_value)

    # work_dir = '/home/hukang/NeuralTE/data'
    # repbase_path = work_dir + '/repbase_processed.ref'
    # summary_class_ratio(repbase_path)
    # repbase_path = work_dir + '/repbase_total.non_plant.ref'
    # summary_class_ratio(repbase_path)

    # data_dir = '/home/hukang/test'
    # genome_path = data_dir + '/GCF_000004555.1_ASM455v1_genomic.fna'
    # repbase_path = data_dir + '/test.ref'
    # temp_dir = data_dir + '/temp'
    # threads = 40
    # flanking_len = 20
    # plant = 0
    # species_name = 'Caenorhabditis briggsae'
    # species = species_name.replace(' ', '_')
    # expandRepBase(repbase_path, genome_path, temp_dir, threads, flanking_len, plant, species, is_expanded=1)

    # # 数据增加前后的类别数量
    # before_path = '/home/hukang/NeuralTE/data_bak1/repbase_total.ref'
    # after_path = '/home/hukang/NeuralTE/data/repbase_total.ref'
    # before_class_num, before_class_set = summary_class_ratio(before_path)
    # after_class_num, after_class_set = summary_class_ratio(after_path)
    # total_class_set = before_class_set.union(after_class_set)
    # data = {}
    # class_name_list = configs.config.all_wicker_class.keys()
    # before_class_num_list = []
    # after_class_num_list = []
    # for cur_class_name in class_name_list:
    #     if before_class_num.__contains__(cur_class_name):
    #         cur_before_class_num = before_class_num[cur_class_name]
    #     else:
    #         cur_before_class_num = 0
    #     if after_class_num.__contains__(cur_class_name):
    #         cur_after_class_num = after_class_num[cur_class_name]
    #     else:
    #         cur_after_class_num = 0
    #     #class_name_list.append(cur_class_name)
    #     before_class_num_list.append(cur_before_class_num)
    #     after_class_num_list.append(cur_after_class_num)
    # data[''] = class_name_list
    # data['before'] = before_class_num_list
    # data['after'] = after_class_num_list
    # df = pd.DataFrame(data)
    # # 将 DataFrame 存储到 Excel 文件中
    # with pd.ExcelWriter('/home/hukang/NeuralTE/data/data_augmentation.xlsx', engine="openpyxl") as writer:
    #     to_excel_auto_column_weight(df, writer, f'before and after data augmentation')

    # 评估ClassifyTE结果
    # # 过滤掉Repbase中ClassifyTE不能处理类别的序列
    # repbase_path = '/home/hukang/TE_Classification/ClassifyTE/data/repbase_total.ref'
    # filter_path = filterRepbase(repbase_path, configs.config.ClassifyTE_class)
    # train_path = '/home/hukang/TE_Classification/ClassifyTE/data/repbase_train.ref'
    # test_path = '/home/hukang/TE_Classification/ClassifyTE/data/repbase_test.ref'
    # split2TrainTest(filter_path, train_path, test_path)
    # feature_path = '/home/hukang/TE_Classification/ClassifyTE/data/new_features.csv'
    # list_path = '/home/hukang/TE_Classification/ClassifyTE/new_features/list.txt'
    # list_data_dir = '/home/hukang/TE_Classification/ClassifyTE/new_features/kanalyze-2.0.0/input_data'
    # add_ClassifyTE_classification(feature_path, list_path, list_data_dir)

    # predict_path = '/home/hukang/TE_Classification/ClassifyTE/output/predicted_out_new_features_test.csv'
    # evaluate_ClassifyTE(predict_path)


    # # #获取RepeatClassifier的结果评估
    # classified_path = '/home/hukang/NeuralTE/data/repbase_test.ref.classified.curated_lib'
    # evaluate_RepeatClassifier(classified_path)

    # # 获取TERL的结果评估
    # data_dir = '/home/hukang/TE_Classification/TERL/Data/DS2'
    # train_path = data_dir + '/repbase_train.ref'
    # validate_path = data_dir + '/repbase_validate.ref'
    # test_path = data_dir + '/repbase_test.ref'
    # #split2TrainTest(train_path, train_path, validate_path)
    # #transform_TERL_data(train_path, validate_path, data_dir)
    # predict_path = data_dir + '/TERL_20230831_103135_repbase_test.ref'
    # evaluate_TERL(test_path, predict_path)

    # # 获取DeepTE的结果评估
    # DeepTE的评估流程：
    # 1.将数据集转fasta格式，shuffle之后按照80-20划分训练train_dataset和测试集test_dataset 。
    # 2.下载DeepTE提供的Metazoans_model。
    # 3.对test_dataset 进行domain的识别。
    # 4.使用训练好的模型对test_dataset 进行预测。
    data_dir = '/home/hukang/TE_Classification/DeepTE-master/training_example_dir/input_dir'
    raw_dataset = data_dir + '/ipt_shuffle_All_CNN_data.txt.bak'
    fasta_dataset = data_dir + '/ipt_shuffle_All_CNN_data.fa'
    #transform_DeepTE_to_fasta(raw_dataset, fasta_dataset)
    train_path = data_dir + '/train.ref'
    test_path = data_dir +'/test.ref'
    #split2TrainTest(fasta_dataset, train_path, test_path)
    #DeepTE_train = data_dir + '/ipt_shuffle_All_CNN_data.txt'
    #transform_repbase_to_DeepTE_input(train_path, DeepTE_train)
    predict_path = data_dir + '/results/opt_DeepTE.txt'
    evaluate_DeepTE(test_path, predict_path)


    # # 画一个3D图
    # Node_matrix = np.random.rand(49, 3)
    # data_path = '/home/hukang/TE_Classification/test.xlsx'
    # data_frame = pd.read_excel(data_path, header=None)
    # x = data_frame.iloc[:, 0].values
    # y = data_frame.iloc[:, 1].values
    # z = data_frame.iloc[:, 2].values
    # plot_3D_param(x, y, z)