import os
import random
import re
import sys
current_folder = os.path.dirname(os.path.abspath(__file__))
# 添加 configs 文件夹的路径到 Python 路径
configs_folder = os.path.join(current_folder, "..")  # 需要根据实际目录结构调整
sys.path.append(configs_folder)

import pandas as pd
import numpy as np

import configs.config
from HiTE_util.Util import read_fasta_v1, word_seq, generate_kmer_dic, generate_mat
from configs import config
from utils.data_util import merge_tsd_terminal_repbase, generate_random_sequences, generate_random_sequence, \
    store_fasta, summary_class_ratio, expandRepBase, generate_terminal_info, to_excel_auto_column_weight, \
    split2TrainTest, replace_non_atcg, generate_non_autonomous_data, get_species_TE, generate_only_dna_data, \
    extract_60_species, get_other_species_from_raw_repbase, generate_domain_info, identify_terminals
from utils.evaluate_util import add_ClassifyTE_classification, filterRepbase, evaluate_RepeatClassifier, \
    plot_3D_param, transform_TERL_data, evaluate_TERL, evaluate_ClassifyTE, transform_repbase_to_DeepTE_input, \
    transform_DeepTE_to_fasta, evaluate_DeepTE, evalute_genome_coverage, get_train_except_species, analyze_class_ratio, \
    transform_repbase_bed


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


    # # # #获取RepeatClassifier的结果评估
    # classified_path = '/home/hukang/NeuralTE/data/repbase_total.rice.ref.classified'
    # RC_name_labels = evaluate_RepeatClassifier(classified_path)

    # all_lib = '/home/hukang/NeuralTE/data/repbase_total.rice.ref'
    # names, contigs = read_fasta_v1(all_lib)
    # name_label = {}
    # for name in names:
    #     parts = name.split('\t')
    #     seq_name = parts[0].split('#')[0]
    #     label = parts[1]
    #     name_label[seq_name] = label
    # names, contigs = read_fasta_v1(classified_path)
    # new_contigs = {}
    # for name in names:
    #     parts = name.split('\t')
    #     seq_name = parts[0].split('#')[0]
    #     label = name_label[seq_name]
    #     new_name = name + ' ' + label
    #     new_contigs[new_name] = contigs[name]
    # store_fasta(new_contigs, classified_path)

    # # 去掉RepeatMasker Library中的水稻序列
    # work_dir = '/home/hukang/NeuralTE/data'
    # lib_path = work_dir + '/RepeatMasker.lib.bak'
    # names, contigs = read_fasta_v1(lib_path)
    # new_contigs = {}
    # new_lib_path = work_dir + '/RepeatMasker.lib'
    # for name in names:
    #     if not name.__contains__('Oryza'):
    #         new_contigs[name] = contigs[name]
    # store_fasta(new_contigs, new_lib_path)

    # # 获取TERL的结果评估
    # data_dir = '/home/hukang/TE_Classification/TERL/Data/DS5'
    # train_path = data_dir + '/repbase_total.no_rice.ref'
    # test_path = data_dir + '/repbase_total.rice.ref'
    # #split2TrainTest(train_path, train_path, validate_path)
    # transform_TERL_data(train_path, test_path, data_dir)
    # # predict_path = data_dir + '/TERL_20230909_161534_repbase_new_species.test.ref'
    # # evaluate_TERL(test_path, predict_path)
    # # data_dir = '/home/hukang/TE_Classification/TERL'
    # # predict_path = data_dir + '/TERL_20230904_142534_seven_well_known_TIR.fa'
    # # test_path = data_dir + '/Data/DS3/seven_well_known_TIR.fa'
    # # evaluate_TERL(test_path, predict_path)

    # # # 获取DeepTE的结果评估
    # # DeepTE的评估流程：
    # # 1.将数据集转fasta格式，shuffle之后按照80-20划分训练train_dataset和测试集test_dataset 。
    # # 2.下载DeepTE提供的Metazoans_model。
    # # 3.对test_dataset 进行domain的识别。
    # # 4.使用训练好的模型对test_dataset 进行预测。
    # data_dir = '/home/hukang/TE_Classification/DeepTE-master/training_example_dir/input_dir'
    # raw_dataset = data_dir + '/ipt_shuffle_All_CNN_data.txt.bak'
    # fasta_dataset = data_dir + '/ipt_shuffle_All_CNN_data.fa'
    # #transform_DeepTE_to_fasta(raw_dataset, fasta_dataset)
    # train_path = data_dir + '/train.ref'
    # test_path = data_dir +'/test.ref'
    # #split2TrainTest(fasta_dataset, train_path, test_path)
    # #DeepTE_train = data_dir + '/ipt_shuffle_All_CNN_data.txt'
    # #transform_repbase_to_DeepTE_input(train_path, DeepTE_train)
    # # predict_path = data_dir + '/results/opt_DeepTE.txt'
    # # evaluate_DeepTE(test_path, predict_path)
    # train_wicker_path = data_dir + '/test.wicker.ref'
    # names, contigs = read_fasta_v1(test_path)
    # with open(train_wicker_path, 'w') as f_save:
    #     for name in names:
    #         parts = name.split('\t')
    #         seq_name = parts[0]
    #         label = parts[1]
    #         wicker_label = config.DeepTE_class[label]
    #         new_name = seq_name + '\t' + wicker_label + '\t' + 'Unknown'
    #         f_save.write('>'+new_name+'\n'+contigs[name]+'\n')
    # data_dir = '/home/hukang/TE_Classification/DeepTE-master/data'
    # predict_path = data_dir + '/results/opt_DeepTE.txt'
    # test_path = data_dir + '/repbase_new_species.test.ref'
    # evaluate_DeepTE(test_path, predict_path)

    # # 将Repbase_test数据集处理成不带预处理信息的格式
    # repbase_path = config.work_dir+'/repbase_train.ref'
    # repbase_no_preprocess = config.work_dir+'/repbase_train.no_pre.ref'
    # names, contigs = read_fasta_v1(repbase_path)
    # new_contigs = {}
    # for name in names:
    #     parts = name.split('\t')
    #     new_name = parts[0]+'\t'+parts[1]+'\t'+parts[2]
    #     new_contigs[new_name] = contigs[name]
    # store_fasta(new_contigs, repbase_no_preprocess)

    # # 将ncbi_ref.info转成genome.info，三列格式
    # with open(config.work_dir + '/genome.info', 'w') as f_save:
    #     with open(config.work_dir + '/ncbi_ref.info', 'r') as f_r:
    #         for i, line in enumerate(f_r):
    #             line = line.replace('\n', '')
    #             parts = line.split('\t')
    #             scientific_name = parts[2]
    #             genome_path = parts[3]
    #             is_plant = parts[5]
    #             if str(line).startswith('#'):
    #                 prefix = '#'
    #             else:
    #                 prefix = ''
    #             f_save.write(prefix+scientific_name+'\t'+genome_path+'\t'+is_plant+'\n')

    # # 生成一个非自治的数据集
    # total_repbase = '/home/hukang/NeuralTE/data/repbase_total.ref'
    # out_path = '/home/hukang/NeuralTE/data/repbase_total.non_autonomous.ref'
    # generate_non_autonomous_data(total_repbase, out_path)

    # # 生成一个只有DNA转座子的的数据集
    # total_repbase = '/home/hukang/NeuralTE/data/repbase_total.ref'
    # out_path = '/home/hukang/NeuralTE/data/repbase_total.tir.ref'
    # generate_only_dna_data(total_repbase, out_path)

    # work_dir = os.getcwd() + "/data"
    # before_path = work_dir + "/repbase_new_species.train.ref"
    # before_class_num, before_class_set = summary_class_ratio(before_path)

    # #重新设计数据增强实验，测试集不动，训练集增强
    # work_dir = os.getcwd() + "/data_bak2"
    # after_path = work_dir + "/repbase_total.ref"
    # after_names, after_contigs = read_fasta_v1(after_path)
    #
    # work_dir = os.getcwd() + "/data"
    # before_path = work_dir + "/repbase_train.ref"
    # before_names, before_contigs = read_fasta_v1(before_path)
    # raw_train_seq_names = set()
    # for name in before_names:
    #     seq_name = name.split('\t')[0]
    #     raw_train_seq_names.add(seq_name)
    #
    # train_after_path = work_dir + "/repbase_train.expand.ref"
    # train_after_contigs = {}
    # for name in after_names:
    #     seq_name = name.split('\t')[0].split('-C_')[0]
    #     if seq_name in raw_train_seq_names:
    #         train_after_contigs[name] = after_contigs[name]
    # store_fasta(train_after_contigs, train_after_path)

    # 从463个物种中，抽取10%个物种当作新测试数据集，剩下的作为训练集
    # total_repbase = '/home/hukang/NeuralTE/data/repbase_total.ref'
    # train_species_path = "/home/hukang/NeuralTE/data/repbase_new_species.train.ref"
    # test_species_path = "/home/hukang/NeuralTE/data/repbase_new_species.test.ref"
    # extract_60_species(total_repbase, train_species_path, test_species_path)

    # # 从原始的All repbase中，抽取除了目前463个物种外的所有序列，当做测试集；463个物种当作训练集
    # raw_repbase = '/home/hukang/NeuralTE/data/all_repbase.ref_preprocess.ref.update'
    # total_repbase = '/home/hukang/NeuralTE/data/repbase_total.ref'
    # train_species_path = "/home/hukang/NeuralTE/data/repbase.train_463.ref"
    # test_species_path = "/home/hukang/NeuralTE/data/repbase.test_other.ref"
    # get_other_species_from_raw_repbase(raw_repbase, total_repbase, train_species_path, test_species_path)

    # # 评估某个工具覆盖某个基因组的比例分析
    # total_repbase = '/home/hukang/NeuralTE/data/repbase_total.ref'
    # rice_path = '/home/hukang/NeuralTE/data/repbase_total.rice.ref'
    # rice_merge_path = '/home/hukang/NeuralTE/data/rice.ref'
    # species_name = 'Oryza sativa'
    # # 1. 将Repbase中的玉米TE单独取出来
    #get_species_TE(total_repbase, rice_path, rice_merge_path, species_name)
    # # 2. 使用不同的工具进行预测
    # # 3. 将预测结果融入到seq_name中，以#分割
    # threads = 40
    # temp_dir = '/home/hukang/NeuralTE/data/RM_temp'
    # genome_path = '/home/hukang/NeuralTE/data/genome/GCF_001433935.1_IRGSP-1.0_genomic.fna'
    # evalute_genome_coverage(rice_merge_path, genome_path, temp_dir, threads)
    # 使用除rice之外的，所有其余Repbase数据当做训练集，rice当做测试集，训练NeuralTE, ClassifyTE, DeepTE, TERL
    #train_path = '/home/hukang/NeuralTE/data/repbase_total.no_rice.ref'
    #get_train_except_species(total_repbase, species_name, train_path)
    # # 评估NeuralTE
    # # 1.将Repbase覆盖基因组产生的bed文件的分类标签替换成 NeuralTE 预测的标签
    # repbase_bed = '/home/hukang/NeuralTE/data/genome/GCF_001433935.1_IRGSP-1.0_genomic.fna.out.bed'
    # NeuralTE_bed = '/home/hukang/NeuralTE/data/genome/GCF_001433935.1_IRGSP-1.0_genomic.fna.out.NeuralTE.bed'
    # label_dict = {}
    # NeuralTE_results = '/home/hukang/NeuralTE/data/results.txt'
    # with open(NeuralTE_results, 'r') as f_r:
    #     for line in f_r:
    #         line = line.replace('\n', '').replace('\'', '').replace('(', '').replace(')', '')
    #         parts = line.split(',')
    #         seq_name = parts[0]
    #         true_label = parts[1].strip()
    #         pred_label = parts[2].strip()
    #         label_dict[seq_name] = pred_label
    # transform_repbase_bed(repbase_bed, NeuralTE_bed, label_dict)
    # temp_dir = '/home/hukang/NeuralTE/data/RM_NeuralTE_rice'
    # analyze_class_ratio(NeuralTE_bed, temp_dir)

    # work_dir = '/home/hukang/NeuralTE/data'
    # data = work_dir + '/repbase_total.ref'
    # threads = 40
    #generate_domain_info(data, work_dir + '/RepeatPeps.lib', work_dir, threads)
    # # 将data中的LTR和TIR去掉
    # names, contigs = read_fasta_v1(data)
    # new_contigs = {}
    # for name in names:
    #     parts = name.split('\t')
    #     new_name = parts[0]+'\t'+parts[1]+'\t'+parts[2]+'\t'+parts[3]+'\t'+parts[4]
    #     seq = contigs[name]
    #     #去除小于80bp的序列
    #     if len(seq) >= 80:
    #         new_contigs[new_name] = contigs[name]
    # store_fasta(new_contigs, data)
    # tool_dir = '/home/hukang/NeuralTE/tools'
    # data = generate_terminal_info(data, work_dir, tool_dir, threads)
    # # 识别一下哪些LTR元素没有LTR终端
    # # 检查哪些LTR元素没有domain
    # domain_path = work_dir + '/repbase_total.ref.domain'
    # seq_names = set()
    # with open(domain_path, 'r') as f_r:
    #     for line in f_r:
    #         seq_name = line.split('\t')[0]
    #         seq_names.add(seq_name)
    # no_terminal_ltr = work_dir + '/repbase_total.no_domain.ref'
    # ltr_types = ('Copia', 'Gypsy', 'Bel-Pao', 'Retrovirus')
    # names, contigs = read_fasta_v1(data)
    # new_contigs = {}
    # for name in names:
    #     parts = name.split('\t')
    #     seq_name = parts[0]
    #     label = parts[1]
    #     LTR_info = parts[5].split(':')[1]
    #     if label in ltr_types and seq_name not in seq_names:
    #         print(seq_name)
    #         new_name = parts[0] + '\t' + parts[1] + '\t' + parts[2] + '\t' + parts[3] + '\t' + parts[4]
    #         new_contigs[new_name] = contigs[name]
    # store_fasta(new_contigs, no_terminal_ltr)

    # #遍历找到想要的LTR文件
    # temp_dir = work_dir + '/temp'
    # for file_name in os.listdir(temp_dir):
    #     if file_name.endswith('updated'):
    #         names, contigs = read_fasta_v1(temp_dir+'/'+file_name)
    #         for name in names:
    #             if name.startswith('Gypsy-123_GM-intactLTR'):
    #                 print(file_name)

    # split_file = temp_dir + '/out_16.bak.fa'
    # output_dir = temp_dir
    # identify_terminals(split_file, output_dir, tool_dir)


    # # 评估RepeatClassifier
    # # 1.将Repbase覆盖基因组产生的bed文件的分类标签替换成 RepeatClassifier 预测的标签
    # repbase_bed = '/home/hukang/NeuralTE/data/genome/GCF_001433935.1_IRGSP-1.0_genomic.fna.out.bed'
    # RepeatClassifier_bed = '/home/hukang/NeuralTE/data/genome/GCF_001433935.1_IRGSP-1.0_genomic.fna.out.RepeatClassifier.bed'
    # label_dict = RC_name_labels
    # transform_repbase_bed(repbase_bed, RepeatClassifier_bed, label_dict)
    # temp_dir = '/home/hukang/NeuralTE/data/RM_RepeatClassifier_rice'
    # analyze_class_ratio(RepeatClassifier_bed, temp_dir)


    # # 提取所有VANDAL element
    # work_dir = '/home/hukang/NeuralTE/data'
    # all_repbase_path = work_dir + '/all_repbase.ref'
    # VANDAL_path = work_dir + '/VANDAL.ref'
    # vandal_contigs = {}
    # names, contigs = read_fasta_v1(all_repbase_path)
    # for name in names:
    #     if name.startswith('VANDAL'):
    #         vandal_contigs[name] = contigs[name]
    # store_fasta(vandal_contigs, VANDAL_path)

    # 判断在Repbase数据库中，是否存在以5'-TG...CA-3'的TIR序列
    TIR_class = ('Tc1-Mariner', 'hAT', 'Mutator', 'Merlin', 'Transib', 'P', 'PiggyBac', 'PIF-Harbinger', 'CACTA')
    work_dir = '/home/hukang/NeuralTE/data'
    all_repbase_path = work_dir + '/all_repbase.ref'
    names, contigs = read_fasta_v1(all_repbase_path)
    unusual_list = []
    for name in names:
        label = name.split('\t')[1]
        seq = contigs[name]
        if label in TIR_class and seq.startswith('TG') and seq.endswith('CA'):
            unusual_list.append(name)
            print(name)
    print(len(unusual_list))

    # # 画一个3D图
    # Node_matrix = np.random.rand(49, 3)
    # data_path = '/home/hukang/TE_Classification/test.xlsx'
    # data_frame = pd.read_excel(data_path, header=None)
    # x = data_frame.iloc[:, 0].values
    # y = data_frame.iloc[:, 1].values
    # z = data_frame.iloc[:, 2].values
    # plot_3D_param(x, y, z)