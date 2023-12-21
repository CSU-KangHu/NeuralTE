#-- coding: UTF-8 --
import os
import random
import sys
from collections import defaultdict

import numpy as np

current_folder = os.path.dirname(os.path.abspath(__file__))
# 添加 configs 文件夹的路径到 Python 路径
configs_folder = os.path.join(current_folder, "..")  # 需要根据实际目录结构调整
sys.path.append(configs_folder)

from configs import config
from utils.data_util import read_fasta, store_fasta, read_fasta_v1, replace_non_atcg, get_flanking_copies, \
    get_copies_TSD_info, search_TSD_regular, extract_non_autonomous, transfer_RMOut2BlastnOut
from utils.evaluate_util import generate_TERL_dataset, generate_ClassifyTE_dataset, evaluate_RepeatClassifier, \
    evaluate_TERL, evaluate_DeepTE, transform_DeepTE_to_fasta, add_ClassifyTE_classification, evaluate_ClassifyTE, \
    evaluate_TEsorter, merge_overlapping_intervals, get_metrics_by_label


def connect_LTR(repbase_path):
    # Preprocess. connect LTR and LTR_internal
    # considering reverse complementary sequence
    raw_names, raw_contigs = read_fasta(repbase_path)
    label_names, label_contigs = read_fasta_v1(repbase_path)
    # store repbase name and label
    repbase_labels = {}
    for name in label_names:
        parts = name.split('\t')
        repbase_name = parts[0]
        classification = parts[1]
        species_name = parts[2]
        repbase_labels[repbase_name] = (classification, species_name)

    # 获取所有LTR序列
    LTR_names = set()
    for name in raw_names:
        # find LTR internal
        parts = name.split('-LTR')
        if len(parts) > 1:
            suffix = parts[1]
            prefix = parts[0]
            # find the LTR seq
            ltr_name = prefix + '-LTR' + suffix
            internal_name1 = prefix + '-I' + suffix
            internal_name2 = prefix + '-INT' + suffix
            LTR_names.add(ltr_name)
            LTR_names.add(internal_name1)
            LTR_names.add(internal_name2)

    # 存储分段的LTR与完整LTR的对应关系
    SegLTR2intactLTR = {}
    new_names = []
    new_contigs = {}
    for name in raw_names:
        if name in LTR_names:
            parts = name.split('-LTR')
            if len(parts) > 1:
                # 为LTR终端序列
                suffix = parts[1]
                prefix = parts[0]
                # find the LTR seq
                ltr_name = prefix + '-LTR' + suffix
                internal_name1 = prefix + '-I' + suffix
                internal_name2 = prefix + '-INT' + suffix
                if raw_contigs.__contains__(ltr_name):
                    if raw_contigs.__contains__(internal_name1):
                        internal_name = internal_name1
                        internal_seq = raw_contigs[internal_name1]
                    elif raw_contigs.__contains__(internal_name2):
                        internal_name = internal_name2
                        internal_seq = raw_contigs[internal_name2]
                    else:
                        internal_name = None
                        internal_seq = None
                    if internal_seq is not None:
                        intact_ltr_name = prefix + '-intactLTR' + suffix
                        intact_ltr_seq = raw_contigs[ltr_name] + internal_seq + raw_contigs[ltr_name]
                        new_names.append(intact_ltr_name)
                        new_contigs[intact_ltr_name] = intact_ltr_seq
                        repbase_labels[intact_ltr_name] = repbase_labels[ltr_name]
                        SegLTR2intactLTR[ltr_name] = intact_ltr_name
                        SegLTR2intactLTR[internal_name] = intact_ltr_name
                    else:
                        new_names.append(name)
                        new_contigs[name] = raw_contigs[name]
        else:
            new_names.append(name)
            new_contigs[name] = raw_contigs[name]

    # Step4. store Repbase sequence with classification, species_name, and TSD sequence
    # 去掉processed_TE_path中存在重复的LTR，例如Copia-1_AA-intactLTR1和Copia-1_AA-intactLTR2，取其中具有合法TSD那个。两个都有，则随机去一个；两个都没有，优先取有TSD那个，否则随机取一个。
    # get all classification
    all_classification = set()
    final_repbase_contigs = {}
    duplicate_ltr = set()
    for query_name in new_names:
        label_item = repbase_labels[query_name]
        cur_prefix = query_name.split('-intactLTR')[0]
        if not duplicate_ltr.__contains__(cur_prefix):
            new_name = query_name + '\t' + label_item[0] + '\t' + label_item[1]
            duplicate_ltr.add(cur_prefix)
            final_repbase_contigs[new_name] = new_contigs[query_name]
            all_classification.add(label_item[0])
    store_fasta(final_repbase_contigs, repbase_path)

    # 存储分段的LTR与完整LTR的对应关系
    SegLTR2intactLTRMap = config.work_dir + '/segLTR2intactLTR.map'
    with open(SegLTR2intactLTRMap, 'a+') as f_save:
        for name in SegLTR2intactLTR.keys():
            intact_ltr_name = SegLTR2intactLTR[name]
            f_save.write(name + '\t' + intact_ltr_name + '\n')
    return repbase_path, repbase_labels

def get_all_files(directory):
    all_files = []
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            all_files.append(file_path)
    return all_files

if __name__ == '__main__':
    # 重新训练DeepTE模型
    # 1. 训练LINE模型
    # 1.1 先提取Dataset2中的train.ref中的LINE元素对应的序列，转换成label,sequence格式
    # work_dir = '/home/hukang/NeuralTE_experiment/Dataset2/DeepTE'
    # train = work_dir + '/train.ref'
    # contigNames, contigs = read_fasta_v1(train)
    # LINE_labels = ['R2', 'RTE', 'Jockey', 'L1', 'I']
    # LINE_path = work_dir + '/ipt_shuffle_LINE_CNN_data.txt'
    # unique_labels = set()
    # with open(LINE_path, 'w') as f_save:
    #     for name in contigNames:
    #         label = name.split('\t')[1]
    #
    #         if label in LINE_labels:
    #             f_save.write(label+','+contigs[name]+'\n')
    #             unique_labels.add(label)
    # print(unique_labels)
    # 2. 训练SINE模型
    # train = work_dir + '/train.ref'
    # contigNames, contigs = read_fasta_v1(train)
    # SINE_labels = ['tRNA', '7SL', '5S']
    # SINE_path = work_dir + '/ipt_shuffle_SINE_CNN_data.txt'
    # unique_labels = set()
    # with open(SINE_path, 'w') as f_save:
    #     for name in contigNames:
    #         label = name.split('\t')[1]
    #         if label in SINE_labels:
    #             f_save.write(label+','+contigs[name]+'\n')
    #             unique_labels.add(label)
    # print(unique_labels)
    # 3. 训练LTR模型
    # train = work_dir + '/train.ref'
    # contigNames, contigs = read_fasta_v1(train)
    # LTR_labels = ['Copia', 'Gypsy', 'Bel-Pao', 'Retrovirus']
    # LTR_path = work_dir + '/ipt_shuffle_LTR_CNN_data.txt'
    # unique_labels = set()
    # with open(LTR_path, 'w') as f_save:
    #     for name in contigNames:
    #         label = name.split('\t')[1]
    #         if label in LTR_labels:
    #             f_save.write(label+','+contigs[name]+'\n')
    #             unique_labels.add(label)
    # print(unique_labels)
    # # 4. 训练nLTR模型
    # train = work_dir + '/train.ref'
    # contigNames, contigs = read_fasta_v1(train)
    # nLTR_labels = {'DIRS': 'DIRS', 'Ngaro': 'DIRS', 'VIPER': 'DIRS', 'Penelope': 'PLE', 'R2': 'LINE', 'RTE': 'LINE', 'Jockey': 'LINE', 'L1': 'LINE', 'I': 'LINE', 'tRNA': 'SINE', '7SL': 'SINE', '5S': 'SINE'}
    # nLTR_path = work_dir + '/ipt_shuffle_nLTR_CNN_data.txt'
    # unique_labels = set()
    # with open(nLTR_path, 'w') as f_save:
    #     for name in contigNames:
    #         label = name.split('\t')[1]
    #         if nLTR_labels.__contains__(label):
    #             f_save.write(nLTR_labels[label] + ',' + contigs[name] + '\n')
    #             unique_labels.add(nLTR_labels[label])
    # print(unique_labels)
    # 5. 训练ClassII模型
    # train = work_dir + '/train.ref'
    # contigNames, contigs = read_fasta_v1(train)
    # ClassII_labels = ['Tc1-Mariner', 'hAT', 'Mutator', 'Merlin', 'Transib', 'P', 'PiggyBac', 'PIF-Harbinger', 'CACTA', 'Crypton']
    # ClassII_path = work_dir + '/ipt_shuffle_ClassII_CNN_data.txt'
    # unique_labels = set()
    # with open(ClassII_path, 'w') as f_save:
    #     for name in contigNames:
    #         label = name.split('\t')[1]
    #         if label in ClassII_labels:
    #             f_save.write(label+','+contigs[name]+'\n')
    #             unique_labels.add(label)
    # print(unique_labels)
    # 6. 训练ClassI模型
    # train = work_dir + '/train.ref'
    # contigNames, contigs = read_fasta_v1(train)
    # ClassI_labels = {'Copia': 'LTR', 'Gypsy': 'LTR', 'Bel-Pao': 'LTR', 'Retrovirus': 'LTR', 'DIRS': 'nLTR', 'Ngaro': 'nLTR', 'VIPER': 'nLTR', 'Penelope': 'nLTR', 'R2': 'nLTR', 'RTE': 'nLTR', 'Jockey': 'nLTR', 'L1': 'nLTR', 'I': 'nLTR', 'tRNA': 'nLTR', '7SL': 'nLTR', '5S': 'nLTR'}
    # ClassI_path = work_dir + '/ipt_shuffle_ClassI_CNN_data.txt'
    # unique_labels = set()
    # with open(ClassI_path, 'w') as f_save:
    #     for name in contigNames:
    #         label = name.split('\t')[1]
    #         if ClassI_labels.__contains__(label):
    #             f_save.write(ClassI_labels[label] + ',' + contigs[name] + '\n')
    #             unique_labels.add(ClassI_labels[label])
    # print(unique_labels)
    # 7. 训练All模型
    # train = work_dir + '/train.ref'
    # contigNames, contigs = read_fasta_v1(train)
    # All_labels = {'Tc1-Mariner': 'ClassII', 'hAT': 'ClassII', 'Mutator': 'ClassII', 'Merlin': 'ClassII', 'Transib': 'ClassII', 'P': 'ClassII', 'PiggyBac': 'ClassII',
    #                 'PIF-Harbinger': 'ClassII', 'CACTA': 'ClassII', 'Crypton': 'ClassII', 'Helitron': 'ClassIII', 'Maverick': 'ClassIII', 'Copia': 'ClassI',
    #                 'Gypsy': 'ClassI', 'Bel-Pao': 'ClassI', 'Retrovirus': 'ClassI', 'DIRS': 'ClassI', 'Ngaro': 'ClassI', 'VIPER': 'ClassI',
    #                 'Penelope': 'ClassI', 'R2': 'ClassI', 'RTE': 'ClassI', 'Jockey': 'ClassI', 'L1': 'ClassI', 'I': 'ClassI', 'tRNA': 'ClassI', '7SL': 'ClassI', '5S': 'ClassI'}
    # All_path = work_dir + '/ipt_shuffle_All_CNN_data.txt'
    # unique_labels = set()
    # with open(All_path, 'w') as f_save:
    #     for name in contigNames:
    #         label = name.split('\t')[1]
    #         if All_labels.__contains__(label):
    #             f_save.write(All_labels[label] + ',' + contigs[name] + '\n')
    #             unique_labels.add(All_labels[label])
    # print(unique_labels)

    # data_dir = '/home/hukang/NeuralTE_experiment/Dataset2/DeepTE'
    # test_path = data_dir + '/test.ref'
    # predict_path = data_dir + '/results/opt_DeepTE.txt'
    # evaluate_DeepTE(test_path, predict_path)

    # # 替换非ATCG字符
    # train_path = data_dir + '/test.ref'
    # train_contignames, train_contigs = read_fasta_v1(train_path)
    # for name in train_contignames:
    #     seq = train_contigs[name]
    #     seq = replace_non_atcg(seq)
    #     train_contigs[name] = seq
    # store_fasta(train_contigs, train_path)

    # # 抽出非自治转座子
    # work_dir = '/home/hukang/NeuralTE_dataset/Dataset3'
    # repbase_path = '/home/hukang/NeuralTE_dataset/Dataset2/all_repbase.ref'
    # extract_non_autonomous(repbase_path, work_dir)

    # # 将Dataset2中的非ATCG字符替换成空
    # files = ['/home/hukang/NeuralTE_dataset/Dataset2/all_repbase.ref', '/home/hukang/NeuralTE_dataset/Dataset2/train.ref', '/home/hukang/NeuralTE_dataset/Dataset2/test.ref']
    # for f in files:
    #     names, contigs = read_fasta_v1(f)
    #     new_contigs = {}
    #     for name in names:
    #         seq = contigs[name]
    #         seq = replace_non_atcg(seq)
    #         new_contigs[name] = seq
    #     store_fasta(new_contigs, f)

    # pred_path = '/home/hukang/NeuralTE_experiment/Dataset2/TEsorter/test.ref.rexdb.cls.lib'
    # test_path = '/home/hukang/NeuralTE_experiment/Dataset2/TEsorter/test.ref'
    # evaluate_TEsorter(pred_path, test_path)
    # # 将TEsorter的macro avg由25分类，变成24分类
    # indicators = [0.557, 0.2934, 0.3517]
    # for ind in indicators:
    #     new_ind = 25 * ind / 24
    #     print(round(new_ind, 4))

    # # 1.2 提取Dataset2中的test.ref中的LINE元素对应的序列
    # train = '/home/hukang/NeuralTE_experiment/Dataset2/DeepTE/test.ref'
    # contigNames, contigs = read_fasta_v1(train)
    # LINE_labels = ['R2', 'RTE', 'Jockey', 'L1', 'I']
    # LINE_path = '/home/hukang/NeuralTE_experiment/Dataset2/DeepTE/test.LINE.ref'
    # LINE_contigs = {}
    # unique_labels = set()
    # for name in contigNames:
    #     label = name.split('\t')[1]
    #     if label in LINE_labels:
    #         LINE_contigs[name] = contigs[name]
    #         unique_labels.add(label)
    # print(unique_labels)
    # store_fasta(LINE_contigs, LINE_path)


    # # Dataset5
    # dataset2 = '/home/hukang/NeuralTE_dataset/Dataset2/all_repbase.ref'
    # contigNames, contigs = read_fasta_v1(dataset2)
    # species_arr = set()
    # for name in contigNames:
    #     species_arr.add(name.split('\t')[2])
    # species_arr = list(species_arr)
    # # 打乱数组顺序
    # random.shuffle(species_arr)
    # # 计算划分点
    # split_point = int(len(species_arr) * 0.8)
    # # 划分数组
    # part_1 = species_arr[:split_point]
    # part_2 = species_arr[split_point:]
    # # 输出结果
    # print("第一部分（80%）：", len(part_1))
    # print("第二部分（20%）：", len(part_2))
    # train = '/home/hukang/NeuralTE_dataset/Dataset5/train.ref'
    # test = '/home/hukang/NeuralTE_dataset/Dataset5/test.ref'
    # train_contigs = {}
    # test_contigs = {}
    # for name in contigNames:
    #     species = name.split('\t')[2]
    #     if species in part_1:
    #         train_contigs[name] = contigs[name]
    #     elif species in part_2:
    #         test_contigs[name] = contigs[name]
    # store_fasta(train_contigs, train)
    # store_fasta(test_contigs, test)
    # print(len(train_contigs), len(test_contigs))


    # feature_path = '/home/hukang/TE_Classification/ClassifyTE/data/new_features.csv.train'
    # list_path = '/home/hukang/TE_Classification/ClassifyTE/new_features/list.txt'
    # list_data_dir = '/home/hukang/TE_Classification/ClassifyTE/new_features/kanalyze-2.0.0/input_data'
    # add_ClassifyTE_classification(feature_path, list_path, list_data_dir)

    # predict_path = '/home/hukang/TE_Classification/ClassifyTE/output/predicted_out_new_features_test.csv'
    # evaluate_ClassifyTE(predict_path)

    # work_dir = '/home/hukang/TE_Classification/TERL/Data/DS2'
    # fasta_file = work_dir + '/train.ref'
    # outdir = work_dir + '/Train'
    # generate_TERL_dataset(fasta_file, outdir)
    # fasta_file = work_dir + '/test.ref'
    # outdir = work_dir + '/Test'
    # generate_TERL_dataset(fasta_file, outdir)

    # fasta_file = '/home/hukang/NeuralTE_dataset/Dataset4/all_repbase.ref'
    # generate_ClassifyTE_dataset(fasta_file)

    # # # #获取RepeatClassifier的结果评估
    # classified_path = '/home/hukang/NeuralTE_experiment/Dataset5/RepeatClassifier/test.ref.classified'
    # RC_name_labels = evaluate_RepeatClassifier(classified_path)
    # # 将RepeatClassifier的macro avg由25分类，变成24分类
    # indicators = [0.919, 0.8543, 0.8794]
    # for ind in indicators:
    #     new_ind = 25 * ind / 24
    #     print(round(new_ind, 4))

    # # 将NeuralTE的macro avg由19分类，变成13分类
    # indicators = [0.8799, 0.9122, 0.8931]
    # for ind in indicators:
    #     new_ind = 12 * ind / 11
    #     print(round(new_ind, 4))

    # # 获取NeuralTE分错类的序列，看是否能够改进
    # TE_path = '/home/hukang/NeuralTE_dataset/Dataset2/test.ref'
    # contigNames, contigs = read_fasta(TE_path)
    #
    # work_dir = '/home/hukang/NeuralTE/work/dataset2'
    # info_path = work_dir + '/classified.info'
    # wrong_classified_TE = {}
    # wrong_TE = work_dir + '/wrong_TE.fa'
    # with open(info_path, 'r') as f_r:
    #     for line in f_r:
    #         line = line.replace('\n', '')
    #         if line.startswith('#'):
    #             continue
    #         parts = line.split(',')
    #         if parts[1] != parts[2]:
    #             raw_name = parts[0]
    #             wrong_classified_TE[raw_name] = contigs[raw_name]
    #             # print(line)
    # store_fasta(wrong_classified_TE, wrong_TE)


    # # 我们尝试获取小样本的比对结果，获取比对序列占query比例 > 80%，identity > 80% 的query序列，然后将query的label设置为target label
    # query_path = '/home/hukang/NeuralTE/work/dataset2/test.ref'
    # query_names, query_contigs = read_fasta(query_path)
    # target_path = '/home/hukang/NeuralTE/work/dataset2/minority/train.minority.ref'
    # target_names, target_contigs = read_fasta_v1(target_path)
    # target_labels = {}
    # target_len_dict = {}
    # for name in target_names:
    #     parts = name.split('\t')
    #     target_name = parts[0]
    #     label = parts[1]
    #     target_labels[target_name] = label
    #     target_len_dict[target_name] = len(target_contigs[name])
    #
    # test_minority_out = '/home/hukang/NeuralTE/work/dataset2/minority/test.minority.out'
    # RMOut = '/home/hukang/NeuralTE/work/dataset2/test.ref.out'
    # tools_dir = config.project_dir + '/tools'
    # transfer_RMOut2BlastnOut(RMOut, test_minority_out, tools_dir)
    #
    # query_intervals = {}
    # query_records = {}
    # with open(test_minority_out, 'r') as f_r:
    #     for line in f_r:
    #         parts = line.split('\t')
    #         query_name = parts[0]
    #         subject_name = parts[1]
    #         identity = float(parts[2])
    #         query_start = int(parts[6])
    #         query_end = int(parts[7])
    #         subject_start = int(parts[8])
    #         subject_end = int(parts[9])
    #         e_value = float(parts[10])
    #         if subject_start > subject_end:
    #             temp = subject_start
    #             subject_start = subject_end
    #             subject_end = temp
    #         if e_value > 1e-10:
    #             continue
    #         if not query_intervals.__contains__(query_name):
    #             query_intervals[query_name] = {}
    #         target_intervals = query_intervals[query_name]
    #         if not target_intervals.__contains__(subject_name):
    #             target_intervals[subject_name] = []
    #         intervals = target_intervals[subject_name]
    #         intervals.append((query_start, query_end))
    #
    #         if not query_records.__contains__(query_name):
    #             query_records[query_name] = {}
    #         target_records = query_records[query_name]
    #         if not target_records.__contains__(subject_name):
    #             target_records[subject_name] = []
    #         records = target_records[subject_name]
    #         records.append((query_start, query_end, subject_start, subject_end))
    #
    # query_labels = {}
    # for query_name in query_intervals.keys():
    #     target_intervals = query_intervals[query_name]
    #     target_records = query_records[query_name]
    #     for subject_name in target_intervals.keys():
    #         records = target_records[subject_name]
    #         target_label = target_labels[subject_name]
    #         intervals = target_intervals[subject_name]
    #         merge_intervals = merge_overlapping_intervals(intervals)
    #         # 求总共占比长度
    #         sum_len = 0
    #         for interval in merge_intervals:
    #             sum_len += abs(interval[1] - interval[0])
    #         query_len = len(query_contigs[query_name])
    #         subject_len = target_len_dict[subject_name]
    #         alignment_ratio = float(sum_len) / query_len
    #         if alignment_ratio > 0.8:
    #             if not query_labels.__contains__(query_name):
    #                 query_labels[query_name] = target_label
    #         elif target_label == 'P' or target_label == 'Merlin':
    #             # 如果target是DNA转座子，且query的终端能比对到target的终端，也算
    #             # query的比对是5'-end或者3'-end，同时subject的比对也是5'-end或者3'-end
    #             is_terminal_alignment = False
    #             for record in records:
    #                 if ((record[0] - 1) <= 5 or (query_len - record[1]) <= 5) and ((record[2] - 1) <= 5 or (subject_len - record[3]) <= 5):
    #                     is_terminal_alignment = True
    #                     query_labels[query_name] = target_label
    #                     break
    #
    # print(query_labels)
    # print(len(query_labels))
    #
    wrong_TE = []
    raw_NeuralTE_result = '/home/hukang/NeuralTE/work/dataset5/classified.info'
    y_pred = []
    y_test = []
    with open(raw_NeuralTE_result, 'r') as f_r:
        for line in f_r:
            if line.startswith('#'):
                continue
            line = line.replace('\n', '')
            parts = line.split(',')
            seq_name = parts[0]
            true_label = parts[1]
            pred_label = parts[2]
            y_pred.append(pred_label)
            y_test.append(true_label)
            if true_label != pred_label:
                wrong_TE.append(seq_name)
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    get_metrics_by_label(y_test, y_pred)
    # print(wrong_TE)
    #
    # # 纠正
    # correct_names = []
    # wrong_TE = []
    # y_pred = []
    # y_test = []
    # with open(raw_NeuralTE_result, 'r') as f_r:
    #     for line in f_r:
    #         if line.startswith('#'):
    #             continue
    #         line = line.replace('\n', '')
    #         parts = line.split(',')
    #         seq_name = parts[0]
    #         true_label = parts[1]
    #         pred_label = parts[2]
    #         if query_labels.__contains__(seq_name):
    #             pred_label = query_labels[seq_name]
    #             correct_names.append(seq_name)
    #         y_pred.append(pred_label)
    #         y_test.append(true_label)
    #         if true_label != pred_label:
    #             wrong_TE.append(seq_name)
    # y_test = np.array(y_test)
    # y_pred = np.array(y_pred)
    # get_metrics_by_label(y_test, y_pred)
    # print(correct_names)
    # print(wrong_TE)



    # # 将训练集的小样本数据先抽出来和训练集与测试集进行同源比对
    # minority_labels = ['Crypton', '5S', '7SL', 'Merlin', 'P', 'R2']
    # work_dir = '/home/hukang/NeuralTE_experiment/Dataset2/NeuralTE'
    # minority_path = work_dir + '/minority.ref'
    # train_path = work_dir + '/train.ref'
    # test_path = work_dir + '/test.ref'
    # minority_contigs = {}
    # train_contigNames, train_contigs = read_fasta_v1(train_path)
    # test_contigNames, test_contigs = read_fasta_v1(test_path)
    # # 1. extract minority dataset
    # for name in train_contigNames:
    #     label = name.split('\t')[1]
    #     if label in minority_labels:
    #         minority_contigs[name] = train_contigs[name]
    # store_fasta(minority_contigs, minority_path)
    # # 2. 进行blastn比对
    # blastn2Results_path = work_dir + '/minority.out'
    # os.system('makeblastdb -in ' + minority_path + ' -dbtype nucl')
    # align_command = 'blastn -db ' + minority_path + ' -num_threads ' \
    #                 + str(40) + ' -query ' + train_path + ' -evalue 1e-20 -outfmt 6 > ' + blastn2Results_path
    # os.system(align_command)

    # 我们现在向Dataset2中插入Dataset1的小样本数据
    # minority_labels = ['Crypton', '5S', '7SL', 'Merlin', 'P', 'R2']
    # dataset1 = '/home/hukang/NeuralTE_dataset/Dataset1/all_repbase.ref'
    # dataset2 = '/home/hukang/NeuralTE_dataset/Dataset2/all_repbase.ref'
    # dataset3 = '/home/hukang/NeuralTE_dataset/Dataset3/all_repbase.ref'
    # DS3_contigs = {}
    # DS3_species = set()
    # DS1_contigNames, DS1_contigs = read_fasta_v1(dataset1)
    # DS2_contigNames, DS2_contigs = read_fasta_v1(dataset2)
    # DS2_raw_names = set()
    # for name in DS2_contigNames:
    #     raw_name = name.split('\t')[0]
    #     DS2_raw_names.add(raw_name)
    # for name in DS1_contigNames:
    #     parts = name.split('\t')
    #     raw_name = parts[0]
    #     label = parts[1]
    #     species = parts[2]
    #     label = config.Repbase_wicker_labels[label]
    #     if raw_name not in DS2_raw_names and label in minority_labels:
    #         new_name = parts[0] + '\t' + label + '\t' + parts[2]
    #         DS3_contigs[new_name] = DS1_contigs[name]
    #         if label == 'Merlin' or label == 'P':
    #             DS3_species.add(species)
    # store_fasta(DS3_contigs, dataset3)
    # print(DS3_species)
    # print(len(DS3_species))
    # tsd_species = ['Allomyces macrogynus', 'Ciona savignyi', 'Drosophila bifasciata', 'Eulimnadia texana',
    #                'Locusta migratoria', 'Oxytricha trifallax', 'Puccinia triticina', 'Capitella teleta',
    #                'Corbicula fluminea', 'Drosophila bocqueti', 'Folsomia candida', 'Owenia fusiformis',
    #                'Parhyale hawaiensis', 'Rhizopus arrhizus']
    # DS3_contigNames, DS3_contigs = read_fasta_v1(dataset3)
    # new_contigs = {}
    # for name in DS3_contigNames:
    #     parts = name.split('\t')
    #     species = parts[2]
    #     if species not in tsd_species:
    #         new_name = parts[0] + '\t' + parts[1] + '\t' + parts[2] + '\t' + 'TSD:\tTSD_len:0\t' + parts[5] + '\t' + parts[6]
    #         new_contigs[new_name] = DS3_contigs[name]
    #     else:
    #         new_contigs[name] = DS3_contigs[name]
    # store_fasta(new_contigs, dataset3)

    # DS3_contigNames, DS3_contigs = read_fasta_v1(dataset3)
    # species_set = set()
    # for name in DS3_contigNames:
    #     parts = name.split('\t')
    #     species = parts[2]
    #     species_set.add(species)
    # print(len(species_set), len(DS3_contigNames))

    # # 将小样本数据的TSD改为Unknown和16
    # dataset3 = '/home/hukang/NeuralTE_dataset/Dataset3/all_repbase.ref'
    # new_contigs = {}
    # contigNames, contigs = read_fasta_v1(dataset3)
    # for name in contigNames:
    #     parts = name.split('\t')
    #     new_name = parts[0] + '\t' + parts[1]+ '\t' + parts[2]+ '\t' + 'TSD:Unknown' + '\t' + 'TSD_len:16' + '\t' + parts[5] + '\t' + parts[6]
    #     new_contigs[new_name] = contigs[name]
    # store_fasta(new_contigs, dataset3)


    # # 获取Dataset1, Dataset2 和 Dfam library中少数类别的数量
    # minority_labels = ['Crypton', '5S', '7SL', 'Merlin', 'P']
    # dataset1 = '/home/hukang/NeuralTE_dataset/Dataset1/all_repbase.ref'
    # dataset2 = '/home/hukang/NeuralTE_dataset/Dataset2/all_repbase.ref'
    # contigNames, contigs = read_fasta_v1(dataset2)
    # minority_count = {}
    # for name in contigNames:
    #     label = name.split('\t')[1]
    #     # label = config.Repbase_wicker_labels[label]
    #     if label in minority_labels:
    #         if not minority_count.__contains__(label):
    #             minority_count[label] = 0
    #         prev_count = minority_count[label]
    #         minority_count[label] = prev_count + 1
    # print(minority_count)

    # minority_labels_dict = {'DNA/Crypton': 'Crypton', 'SINE/5S': '5S', 'SINE/7SL': '7SL', 'DNA/Merlin': 'Merlin', 'DNA/P': 'P'}
    # dataset3 = '/home/hukang/miniconda3/envs/HiTE/share/RepeatMasker/Libraries/RepeatMasker.lib.bak'
    # contigNames, contigs = read_fasta(dataset3)
    # minority_count = {}
    # for name in contigNames:
    #     label = name.split('#')[1]
    #     #label = minority_labels_dict[label]
    #     if label in minority_labels_dict:
    #         if not minority_count.__contains__(label):
    #             minority_count[label] = 0
    #         prev_count = minority_count[label]
    #         minority_count[label] = prev_count + 1
    # print(minority_count)


    # # 我们把train.ref中的Crypton序列都提取出来，然后和我们分错类的序列进行比对，看看什么情况
    # train = '/home/hukang/NeuralTE_dataset/Dataset2/train.ref'
    # extract_labels = ['Crypton', '5S', '7SL', 'Merlin', 'P']
    # train_contigNames, train_contigs = read_fasta_v1(train)
    # file_path = '/home/hukang/NeuralTE_dataset/Dataset2/minority.train.ref'
    # cur_contigs = {}
    # for extract_label in extract_labels:
    #     for name in train_contigNames:
    #         if name.split('\t')[1] == extract_label:
    #             cur_contigs[name] = train_contigs[name]
    # store_fasta(cur_contigs, file_path)

    # # 如果过滤掉了不具备LTR的LTR和不具备TIR的TIR序列，看剩下多少。
    # all_path = '/home/hukang/NeuralTE_dataset/Dataset2/all_repbase.ref'
    # ltr_labels = ('Copia', 'Gypsy', 'Bel-Pao', 'Retrovirus')
    # tir_labels = ('Tc1-Mariner', 'hAT', 'Mutator', 'Merlin', 'Transib', 'P', 'PiggyBac', 'PIF-Harbinger', 'CACTA')
    #
    # contigNames, contigs = read_fasta_v1(all_path)
    # print(len(contigs))
    # total_ltr_num = 0
    # delete_ltr_num = 0
    # total_tir_num = 0
    # delete_tir_num = 0
    # for name in contigNames:
    #     parts = name.split('\t')
    #     label = parts[1]
    #     ltr = parts[5]
    #     tir = parts[6]
    #     if label in ltr_labels:
    #         total_ltr_num += 1
    #         if ltr.split(':')[1] == '':
    #             delete_ltr_num += 1
    #             del contigs[name]
    #     elif label in tir_labels:
    #         total_tir_num += 1
    #         if tir.split(':')[1] == '':
    #             delete_tir_num += 1
    #             del contigs[name]
    # print(len(contigs))
    # print(total_ltr_num, delete_ltr_num)
    # print(total_tir_num, delete_tir_num)

    # # 获取水稻Repbase的拷贝，然后我们人工检查一些获得错误TSD的序列，看能否有办法获得对的TSD
    # repbase_path = '/home/hukang/NeuralTE_dataset/Dataset7/test.ref'
    # genome_path = '/home/hukang/Genome/GCF_001433935.1_IRGSP-1.0_genomic.fna'
    # flanking_len = 20
    # temp_dir = '/home/hukang/NeuralTE/work/TSD/rice_tsd'
    # threads = 40
    # batch_member_files = get_flanking_copies(repbase_path, genome_path, flanking_len, temp_dir, threads)
    #
    # species = 'rice'
    # plant = 1
    # is_expanded = 0
    # label_names, label_contigs = read_fasta_v1(repbase_path)
    # # store repbase name and label
    # repbase_labels = {}
    # for name in label_names:
    #     parts = name.split('\t')
    #     repbase_name = parts[0]
    #     classification = parts[1]
    #     species_name = parts[2]
    #     repbase_labels[repbase_name] = (classification, species_name)
    #
    # # cur_member_file = temp_dir + '/MuDR-N208E_OS.blast.bed.fa'
    # # batch_member_files = [('MuDR-N208E_OS', 'GGGTTAATTTGATCCATGCCACTGCAAATTTAGCTATTCAGAAAAATGACATTGCAATTCATCTATTCTTAACCGTGCCACTGAAATTTTGTAAAACTAAAACCGTGCCATTGACGTCACATTTTCCATCCATTCTCTTCCTTTTCCGTCTTCTTCCTTCCTTCTCCCATCTTCTTCCCGGAGTCAAGCCGGAGAGGGAGCTCGCCGGCAAGGTGAACGAACCCAACCTCGAGTGCGGTTGGCGTGGTCGGCGAATCCGGCGGTGGCGGCGTCGGACAATGGTGGCATCGGGACTCGGGCGGAACCAGCTGAGGCCTAGGCTGGGTGTCGAGCGTGATCGACGACGGTGACTCTCTTCTTCCGCGTTGCTGCTCAACCTCGGCTCCCGCTCTGGCCTCCGGGTCGGTGAGCACCTCATGCCGGCCGCTCTCCCTCGCGGCAGTGCTCTCCCCGCACTACTCATCCTTGGACCTCTCCGAGACTCCAACCGCCTCCTCGTCGCCCGCCATGAGCTCCGCAAGTAGCTGGAGCACCTCGCCGCCGTCTTCGAGTCTTCATCGCCTCTGTTGGGTCTGCTCGCGCCATGCAGCGCCAGATCTACCGTTGTTTCCATCGACGTTGGCTCCACCGCCGACACCGTCGAAGCTCGCTGCGGCCATGGATGGATGGACGACCGCCGTTGGCATCGCCGCCGCTGCTCCCGCACGAGATCTCGCCCACTTGGCTCCGGGAAGAAATGGGAGAAAGAAGGAGCCGCTGCCGCCTGCCATGGTTGGGTCGCTGGCAAGCTCCCTCTCCTCCGAGCTCGCCGGCATGACTCCGAGAAGAAATAGGAGAAAGAACCGGGAAGAAATGGGAGAAAGAAGGAAGAAGACGGAAAAGGCAGGGGATGGATGGAAAACGTGACGGCAGTGGCACGGTTCTAATTTTGTAAAATTCTAGTGGCACGGTTACGAATAGACGAATTGTAATGGCATTTTTCTTAATAGACAAATTTGCAGTGGCATAGATCAAATTAACCCTA', cur_member_file)]
    #
    # tsd_info = get_copies_TSD_info(batch_member_files, flanking_len, is_expanded, repbase_labels, threads)
    # # 将所有的序列存成文件，拷贝序列名称为在原有的名称后面加-C_{num}
    # names, contigs = read_fasta(repbase_path)
    # final_repbase_path = temp_dir + '/' + species + '.ref'
    # final_repbase_contigs = {}
    # for query_name in names:
    #     seq = contigs[query_name]
    #     label_item = repbase_labels[query_name]
    #
    #     if tsd_info.__contains__(query_name):
    #         copies_tsd_info = tsd_info[query_name]
    #     else:
    #         copies_tsd_info = [('Unknown', 16, -1)]
    #
    #     # 遍历一边，取出distance为0的TSD；
    #     new_copies_tsd_info = []
    #     for tsd_seq, tsd_len, cur_distance in copies_tsd_info:
    #         if cur_distance <= 0:
    #             new_copies_tsd_info.append((tsd_seq, tsd_len, cur_distance))
    #     copies_tsd_info = new_copies_tsd_info if len(new_copies_tsd_info) > 0 else copies_tsd_info
    #     # 将所有的拷贝对应的TSD存储起来，记录每种长度的TSD对应的出现次数和离原始边界的距离
    #     max_count_TSD = {}
    #     length_count = {}
    #     for tsd_seq, tsd_len, cur_distance in copies_tsd_info:
    #         if not length_count.__contains__(tsd_len):
    #             length_count[tsd_len] = (1, cur_distance)
    #             max_count_TSD[tsd_len] = tsd_seq
    #         else:
    #             prev_count, prev_distance = length_count[tsd_len]
    #             if cur_distance < prev_distance:
    #                 prev_distance = cur_distance
    #                 max_count_TSD[tsd_len] = tsd_seq
    #             length_count[tsd_len] = (prev_count + 1, prev_distance)
    #     # 按照(tsd_len, tsd_seq, 出现次数, 最小距离)存成数组
    #     # 取出现次数最多的TSD，如果有多个出现次数最多，取distance最小的那个，如果distance相同，取最长的那个
    #     all_tsd_set = []
    #     for tsd_len in length_count.keys():
    #         cur_count, cur_distance = length_count[tsd_len]
    #         tsd_seq = max_count_TSD[tsd_len]
    #         all_tsd_set.append((tsd_len, tsd_seq, cur_count, cur_distance))
    #     all_tsd_set = sorted(all_tsd_set, key=lambda x: (-x[2], x[3], -x[0]))
    #     final_tsd_info = all_tsd_set[0]
    #     tsd_seq = final_tsd_info[1]
    #     tsd_len = final_tsd_info[0]
    #     tsd_distance = final_tsd_info[3]
    #     if tsd_distance > 5:
    #         tsd_seq = ''
    #         tsd_len = len(tsd_seq)
    #     new_name = query_name + '\t' + label_item[0] + '\t' + label_item[1] + '\t' + 'TSD:' + str(tsd_seq) + '\t' + 'TSD_len:' + str(tsd_len)
    #     final_repbase_contigs[new_name] = seq
    # store_fasta(final_repbase_contigs, final_repbase_path)


    # # 将RepeatMasker的同源搜索库替换成train.ref，看是否能运行
    # work_dir = '/home/hukang/miniconda3/envs/HiTE/share/RepeatMasker/Libraries'
    # rm_lib = work_dir + '/RepeatMasker.lib'
    # train = '/home/hukang/NeuralTE_dataset/Dataset5/train.ref'
    #
    # wicker2RM = {}
    # # 转换成RepeatMasker标签
    # with open(config.project_dir + '/data/Wicker2RM.info', 'r') as f_r:
    #     for line in f_r:
    #         if line.startswith('#'):
    #             continue
    #         line = line.replace('\n', '')
    #         parts = line.split('\t')
    #         Wicker_Label = parts[0]
    #         RepeatMasker_Label = parts[1]
    #         wicker2RM[Wicker_Label] = RepeatMasker_Label
    #
    # train_contigNames, train_contigs = read_fasta_v1(train)
    # rm_contigs = {}
    # for name in train_contigNames:
    #     parts = name.split('\t')
    #     seq_name = parts[0]
    #     label = parts[1]
    #     species = parts[2].replace(' ', '_')
    #     RepeatMasker_Label = wicker2RM[label]
    #     new_name = seq_name+'#'+RepeatMasker_Label+' @'+species
    #     rm_contigs[new_name] = train_contigs[name]
    # store_fasta(rm_contigs, rm_lib)

    # # 将RepeatMasker的同源搜索Library去掉test数据集，然后测下性能。
    # work_dir = '/home/hukang/miniconda3/envs/HiTE/share/RepeatMasker/Libraries'
    # rm_lib = work_dir + '/RepeatMasker.lib.bak'
    # test_lib = '/home/hukang/NeuralTE_experiment/Dataset2/RepeatClassifier_all-test_lib/test.ref'
    # test_contigNames, test_contigs = read_fasta(test_lib)
    # print(len(test_contigs))
    # all_test_names = set()
    # test_name_dict = {}
    # for name in test_contigNames:
    #     if name.__contains__('intactLTR'):
    #         internal_name = name.replace('intactLTR', 'I')
    #         internal_name1 = name.replace('intactLTR', 'INT')
    #         internal_name2 = name.replace('intactLTR', 'int')
    #         LTR_name = name.replace('intactLTR', 'LTR')
    #
    #         all_test_names.add(LTR_name)
    #         all_test_names.add(internal_name)
    #         all_test_names.add(internal_name1)
    #         all_test_names.add(internal_name2)
    #         test_name_dict[LTR_name] = name
    #         test_name_dict[internal_name] = name
    #         test_name_dict[internal_name1] = name
    #         test_name_dict[internal_name2] = name
    #     else:
    #         all_test_names.add(name)
    #         test_name_dict[name] = name
    #
    # filter_rm_lib = work_dir + '/RepeatMasker.lib'
    # filter_contigs = {}
    # rm_contigNames, rm_contigs = read_fasta_v1(rm_lib)
    # for name in rm_contigNames:
    #     seq_name = name.split('\t')[0].split('#')[0]
    #     if not seq_name in all_test_names:
    #         filter_contigs[name] = rm_contigs[name]
    #     else:
    #         test_name = test_name_dict[seq_name]
    #         if test_contigs.__contains__(test_name):
    #             del test_contigs[test_name]
    # store_fasta(filter_contigs, filter_rm_lib)
    #
    # # 输出Repbase有，但是RM lib中没有的序列
    # print(len(test_contigs))
    # print(test_contigs.keys())

    # # Dataset6制作
    # work_dir = '/home/hukang/NeuralTE_dataset/Dataset7'
    # total = work_dir + '/all_repbase.ref'
    # train = work_dir + '/train.ref'
    # test = work_dir + '/test.ref'
    # names, contigs = read_fasta_v1(total)
    # train_contigs = {}
    # test_contigs = {}
    # for name in names:
    #     species = name.split('\t')[2]
    #     if species == 'Zea mays':
    #         test_contigs[name] = contigs[name]
    #     elif not species.__contains__('Zea mays'):
    #         train_contigs[name] = contigs[name]
    # store_fasta(train_contigs, train)
    # store_fasta(test_contigs, test)

    # work_dir = '/home/hukang/TE_Classification/TERL'
    # test_path = work_dir + '/Data/DS2/test.ref'
    # predict_path = work_dir + '/TERL_20231219_094410_test.ref'
    # evaluate_TERL(test_path, predict_path)

    # # 获取DeepTE的结果评估
    # DeepTE的评估流程：
    # 1.将数据集转fasta格式，并根据Repbase 28.06 (Dataset1)恢复数据集的header
    # 2.使用NeuralTE提供的split_train_test.py 划分训练train_dataset和测试集test_dataset，保持类别的分布一致。
    # 3.下载DeepTE提供的Metazoans_model。
    # 4.对test_dataset 进行domain的识别。
    # 5.使用训练好的模型对test_dataset 进行预测。
    # data_dir = '/home/hukang/TE_Classification/DeepTE-master/training_example_dir/input_dir'
    # repbase_dataset = '/home/hukang/NeuralTE_dataset/Dataset1/all_repbase.ref'
    # raw_dataset = data_dir + '/ipt_shuffle_All_CNN_data.txt.bak'
    # fasta_dataset = data_dir + '/ipt_shuffle_All_CNN_data.fa'
    # transform_DeepTE_to_fasta(raw_dataset, fasta_dataset)
    # train_path = data_dir + '/train.ref'
    # test_path = data_dir +'/test.ref'
    # # transform DeepTE label to wicker label
    # train_wicker_path = data_dir + '/train.wicker.ref'
    # names, contigs = read_fasta_v1(train_path)
    # with open(train_wicker_path, 'w') as f_save:
    #     for name in names:
    #         parts = name.split('\t')
    #         seq_name = parts[0]
    #         label = parts[1]
    #         wicker_label = config.DeepTE_class[label]
    #         new_name = seq_name + '\t' + wicker_label + '\t' + 'Unknown'
    #         f_save.write('>'+new_name+'\n'+contigs[name]+'\n')
    # test_wicker_path = data_dir + '/test.wicker.ref'
    # names, contigs = read_fasta_v1(test_path)
    # with open(test_wicker_path, 'w') as f_save:
    #     for name in names:
    #         parts = name.split('\t')
    #         seq_name = parts[0]
    #         label = parts[1]
    #         wicker_label = config.DeepTE_class[label]
    #         new_name = seq_name + '\t' + wicker_label + '\t' + 'Unknown'
    #         f_save.write('>' + new_name + '\n' + contigs[name] + '\n')
    #
    # predict_path = data_dir + '/results/opt_DeepTE.txt'
    # evaluate_DeepTE(test_wicker_path, predict_path)



    # # 分析一下repbase中到底包含了多少类别
    # work_dir = '/home/hukang/RepBase28.06.fasta'
    # # 获取指定目录下的所有文件
    # files = get_all_files(work_dir)
    # unique_labels = set()
    # for file in files:
    #     names, contigs = read_fasta_v1(file)
    #     for name in names:
    #         parts = name.split('\t')
    #         if len(parts) == 3:
    #             seq_name = parts[0]
    #             if seq_name.__contains__('LTR') and not seq_name.endswith('-LTR') and not seq_name.endswith('_LTR'):
    #                 unique_labels.add(name)
    # print(unique_labels, len(unique_labels))

    # work_dir = '/home/hukang/HiTE_lib/rice_unmask'
    # test_library = work_dir + '/repbase/classified_TE.fa'
    # gold_library = work_dir + '/oryrep.RM.ref'
    # test_names, test_contigs = read_fasta(test_library)
    # gold_names, gold_contigs = read_fasta(gold_library)
    # gold_labels = {}
    # for name in gold_names:
    #     parts = name.split('#')
    #     seq_name = parts[0]
    #     label = parts[1]
    #     gold_labels[seq_name] = label
    # same_labels = {}
    # diff_labels = {}
    # diff_solo_LTR_labels = {}
    # for name in test_names:
    #     parts = name.split('#')
    #     seq_name = parts[0]
    #     label = parts[1]
    #     gold_label = gold_labels[seq_name]
    #     if label == gold_label:
    #         same_labels[seq_name] = label
    #     else:
    #         diff_labels[seq_name] = (gold_label, label)
    #
    # print(diff_labels)
    # print(len(same_labels), len(diff_labels))




    # # `>Gypsy-171_OS-I`和`>Gypsy-171_OS-LTR`
    # data_path = '/home/hukang/HiTE_lib/rice_unmask/rice-families.rename.fa'
    # # 将输入文件格式化为Repbase格式
    # names, contigs = read_fasta(data_path)
    # os.makedirs(os.path.dirname(data_path), exist_ok=True)
    # with open(data_path, 'w') as f_save:
    #     for name in names:
    #         seq = contigs[name]
    #         name = name.split('/')[0].split('#')[0]
    #         new_name = name + '\tUnknown\tUnknown'
    #         f_save.write('>' + new_name + '\n' + seq + '\n')
    #
    # config.work_dir = '/home/hukang/HiTE_lib/rice_unmask'
    # connect_LTR(data_path)

    # file_path = '/home/hukang/HiTE_lib/rice_unmask/confident_TE.cons.fa'
    # contigNames, contigs = read_fasta(file_path)
    # new_contigs = {}
    # for contigname in contigNames:
    #     seq = contigs[contigname]
    #     if contigname.endswith('_LTR') or contigname.endswith('_INT'):
    #         contigname = contigname.replace('_LTR', '-LTR').replace('_INT', '-INT')
    #     new_contigs[contigname] = seq
    # store_fasta(new_contigs, file_path)


    # work_dir = '/home/hukang/NeuralTE_dataset/Dataset7'
    # data_path = work_dir + '/test.ref'
    # wicker2RM = {}
    # # 转换成RepeatMasker标签
    # with open(config.project_dir + '/data/Wicker2RM.info', 'r') as f_r:
    #     for line in f_r:
    #         if line.startswith('#'):
    #             continue
    #         line = line.replace('\n', '')
    #         parts = line.split('\t')
    #         Wicker_Label = parts[0]
    #         RepeatMasker_Label = parts[1]
    #         wicker2RM[Wicker_Label] = RepeatMasker_Label
    #
    # filter_labels = ['SAT', 'Multicopy gene', 'Satellite', 'REP-10_OS', 'REP-1_OS', 'snRNA', 'RCH2', 'KRISPIE']
    # orig_names, orig_contigs = read_fasta_v1(data_path)
    # classified_data = work_dir + '/test.RM.ref'
    # classified_contigs = {}
    # for name in orig_names:
    #     parts = name.split('\t')
    #     map_name = parts[0]
    #     predict_label = parts[1]
    #     if predict_label in filter_labels:
    #         continue
    #     predict_label = wicker2RM[predict_label]
    #     new_name = map_name + '#' + predict_label
    #     classified_contigs[new_name] = orig_contigs[name]
    # store_fasta(classified_contigs, classified_data)

    # work_dir = '/home/hukang/Genome'
    # genome_path = work_dir + '/Oryza_sativa.IRGSP-1.0.dna.toplevel.fa'
    # ref_names, ref_contigs = read_fasta(genome_path)
    # store_fasta(ref_contigs, genome_path)

    # # 2. 在test数据集上评估RepeatClassifier
    # # 2.2 将Dfam分类名称转成wicker格式
    # # 2.2.1 这个文件里包含了RepeatMasker类别、Repbase、wicker类别的转换
    # rmToWicker = {}
    # WickerToRM = {}
    # wicker_superfamily_set = set()
    # with open(config.project_dir + '/data/TEClasses.tsv', 'r') as f_r:
    #     for i, line in enumerate(f_r):
    #         parts = line.split('\t')
    #         rm_type = parts[5]
    #         rm_subtype = parts[6]
    #         repbase_type = parts[7]
    #         wicker_type = parts[8]
    #         wicker_type_parts = wicker_type.split('/')
    #         #print(rm_type + ',' + rm_subtype + ',' + repbase_type + ',' + wicker_type)
    #         if len(wicker_type_parts) != 3:
    #             continue
    #         wicker_superfamily_parts = wicker_type_parts[-1].strip().split(' ')
    #         if len(wicker_superfamily_parts) == 1:
    #             wicker_superfamily = wicker_superfamily_parts[0]
    #         elif len(wicker_superfamily_parts) > 1:
    #             wicker_superfamily = wicker_superfamily_parts[1].replace('(', '').replace(')', '')
    #         rm_full_type = rm_type + '/' + rm_subtype
    #         if wicker_superfamily == 'ERV':
    #             wicker_superfamily = 'Retrovirus'
    #         if wicker_superfamily == 'Viper':
    #             wicker_superfamily = 'VIPER'
    #         if wicker_superfamily == 'H':
    #             wicker_superfamily = 'Helitron'
    #         rmToWicker[rm_full_type] = (wicker_superfamily, repbase_type)
    #         WickerToRM[wicker_superfamily] = rm_full_type
    #         wicker_superfamily_set.add(wicker_superfamily)
    # # 补充一些元素
    # rmToWicker['LINE/R2'] = 'R2'
    # rmToWicker['DNA/Crypton'] = 'Crypton'
    # rmToWicker['Unknown'] = 'Unknown'
    # # 固定wicker对应的RM标签
    # WickerToRM['Retrovirus'] = 'LTR/ERV'
    # WickerToRM['DIRS'] = 'LTR/DIRS'
    # WickerToRM['R2'] = 'LINE/R2'
    # WickerToRM['RTE'] = 'LINE/RTE-RTE'
    # WickerToRM['L1'] = 'LINE/L1'
    # WickerToRM['I'] = 'LINE/I'
    # WickerToRM['tRNA'] = 'SINE/tRNA'
    # WickerToRM['7SL'] = 'SINE/7SL'
    # WickerToRM['5S'] = 'SINE/5S'
    # WickerToRM['Helitron'] = 'RC/Helitron'
    # WickerToRM['Maverick'] = 'DNA/Maverick'
    # WickerToRM['Crypton'] = 'DNA/Crypton'
    # WickerToRM['Tc1-Mariner'] = 'DNA/TcMar'
    # WickerToRM['hAT'] = 'DNA/hAT'
    # WickerToRM['Mutator'] = 'DNA/MULE'
    # WickerToRM['P'] = 'DNA/P'
    # WickerToRM['PiggyBac'] = 'DNA/PiggyBac'
    # WickerToRM['PIF-Harbinger'] = 'DNA/PIF-Harbinger'
    # WickerToRM['CACTA'] = 'DNA/CMC-EnSpm'
    # print(rmToWicker)
    # #print(len(rmToWicker))
    # #print(wicker_superfamily_set)
    # #print(len(wicker_superfamily_set))
    # ClassifySystem = config.project_dir + '/data/Wicker2RM.info'
    # with open(ClassifySystem, 'w') as f_save:
    #     f_save.write('#Wicker_Label\tRepeatMasker_Label\n')
    #     for name in WickerToRM.keys():
    #         f_save.write(name + '\t' + WickerToRM[name] + '\n')