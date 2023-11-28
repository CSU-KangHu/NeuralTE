import codecs
import json
import os.path
import matplotlib
matplotlib.use('pdf')
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata, interpolate
from matplotlib import cm

from configs import config
from utils.data_util import read_fasta_v1, word_seq, generate_kmer_dic, generate_mat, store_fasta, replace_non_atcg, \
    get_species_TE

import tensorflow as tf

# def multi_category_focal_loss1(alpha, gamma=2.0):
#     """
#     focal loss for multi category of multi label problem
#     适用于多分类或多标签问题的focal loss
#     alpha用于指定不同类别/标签的权重，数组大小需要与类别个数一致
#     当你的数据集不同类别/标签之间存在偏斜，可以尝试适用本函数作为loss
#     Usage:
#      model.compile(loss=[multi_category_focal_loss1(alpha=[1,2,3,2], gamma=2)], metrics=["accuracy"], optimizer=adam)
#     """
#     epsilon = 1.e-7
#     alpha = tf.constant(alpha, dtype=tf.float32)
#     #alpha = tf.constant([[1],[1],[1],[1],[1]], dtype=tf.float32)
#     #alpha = tf.constant_initializer(alpha)
#     gamma = float(gamma)
#     def multi_category_focal_loss1_fixed(y_true, y_pred):
#         y_true = tf.cast(y_true, tf.float32)
#         y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
#         y_t = tf.multiply(y_true, y_pred) + tf.multiply(1-y_true, 1-y_pred)
#         ce = -tf.log(y_t)
#         weight = tf.pow(tf.subtract(1., y_t), gamma)
#         fl = tf.matmul(tf.multiply(weight, ce), alpha)
#         loss = tf.reduce_mean(fl)
#         return loss
#     return multi_category_focal_loss1_fixed

from keras import backend as K
def focal_loss(gamma=2.):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        return -K.sum(K.pow(1. - pt_1, gamma) * K.log(pt_1))
    return focal_loss_fixed

def plot_confusion_matrix(y_pred, y_test):

    y_pred_set = set(y_pred)
    y_test_set = set(y_test)
    class_list = list(y_pred_set | y_test_set)
    #class_list = list(y_test_set)

    class_names = []
    all_class_list = list(config.all_wicker_class.keys())
    for class_num in all_class_list:
        if class_num in class_list:
            label = class_num
            class_names.append(label)
    # 计算混淆矩阵
    conf_matrix = confusion_matrix(y_test, y_pred, labels=class_names)
    class_report = classification_report(y_test, y_pred)
    print(class_report)

    # 绘制混淆矩阵图表
    plt.figure(figsize=(15, 15))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.yticks(rotation=360)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(config.work_dir + '/confusion_matrix.png', format='png')
    #plt.show()

def get_metrics(y_pred, y_test, seq_names):
    predicted_classes = np.argmax(np.round(y_pred), axis=1)
    # transfer the prop less than a threshold to be unknown for a class
    prop_thr = 0
    max_value_predicted_classes = np.amax(y_pred, axis=1)
    order = -1
    ls_thr_order_list = []
    for i in range(len(max_value_predicted_classes)):
        order += 1
        if max_value_predicted_classes[i] < float(prop_thr):
            ls_thr_order_list.append(order)

    predicted_classes_list = []
    order = -1
    for i in range(len(predicted_classes)):
        order += 1
        if order in ls_thr_order_list:
            new_class = 28  # unknown class label
        else:
            new_class = predicted_classes[i]
        predicted_classes_list.append(new_class)

    if seq_names is not None:
        store_results_dic = {}
        for i in range(0, len(predicted_classes_list)):
            predicted_class = predicted_classes_list[i]
            seq_name = seq_names[i]
            if predicted_class != 28:
                store_results_dic[seq_name] = str(seq_name) + ',' + config.inverted_all_wicker_class[predicted_class]
            else:
                store_results_dic[seq_name] = str(seq_name) + ',' + 'Unknown'

        with open(config.work_dir + '/results.txt', 'w+') as opt:
            for eachid in store_results_dic:
                opt.write(store_results_dic[eachid] + '\n')

    # 计算准确率
    accuracy = accuracy_score(y_test, predicted_classes_list)
    # 计算精确率
    precision = precision_score(y_test, predicted_classes_list, average='macro')
    # 计算召回率
    recall = recall_score(y_test, predicted_classes_list, average='macro')
    # 计算F1值
    f1 = f1_score(y_test, predicted_classes_list, average='macro')

    # 将数值标签转为字符标签
    inverted_all_wicker_class = config.inverted_all_wicker_class
    predicted_labels = []
    test_labels = []
    for label_num in predicted_classes_list:
        predicted_labels.append(inverted_all_wicker_class[label_num])
    for y_num in y_test:
        test_labels.append(inverted_all_wicker_class[y_num])
    # plot confusion matrix
    plot_confusion_matrix(predicted_labels, test_labels)

    return round(accuracy, 4), round(precision, 4), round(recall, 4), round(f1, 4)

# ClassifyTE 训练数据特征的加入
# 因为 ClassifyTE 对于训练数据的特征提取没有加入classification标签，因此如果要重新训练模型，需要额外对特征提取结果进行处理，加入classification列
def add_ClassifyTE_classification(feature_path, list_path, list_data_dir):
    new_feature_path = feature_path + '.train'
    labels = ['classification']
    with open(list_path, 'r') as f_r:
        for i, line in enumerate(f_r):
            seq_name = line.replace('\n', '')
            seq_path = list_data_dir + '/' + seq_name
            names, contigs = read_fasta_v1(seq_path)
            name = names[0]
            label = name.split('\t')[1]
            label_num = config.ClassifyTE_class[label]
            labels.append(label_num)

    with open(new_feature_path, 'w') as f_save:
        with open(feature_path, 'r') as f_r:
            for i, line in enumerate(f_r):
                line = line.replace('\n', '')
                newline = line + ',' + labels[i]
                f_save.write(newline+'\n')

# 对Repbase数据集进行过滤，只保留 TE_class 中所包含的类型
def filterRepbase(repbase_path, TE_class):
    filter_contigs = {}
    names, contigs = read_fasta_v1(repbase_path)
    for name in names:
        label = name.split('\t')[1]
        if label in TE_class:
            filter_contigs[name] = contigs[name]
    filter_path = repbase_path+'.filter'
    store_fasta(filter_contigs, filter_path)
    return filter_path

def get_metrics_by_label(y_test, y_pred):
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", round(accuracy, 4))
    # 计算精确率
    precision = precision_score(y_test, y_pred, average='macro')
    print("Precision:", round(precision, 4))
    # 计算召回率
    recall = recall_score(y_test, y_pred, average='macro')
    print("Recall:", round(recall, 4))
    # 计算F1值
    f1 = f1_score(y_test, y_pred, average='macro')
    print("F1:", round(f1, 4))

    plot_confusion_matrix(y_pred, y_test)

def evaluate_RepeatClassifier(classified_path):
    # 2. 在test数据集上评估RepeatClassifier
    # 2.2 将Dfam分类名称转成wicker格式
    # 2.2.1 这个文件里包含了RepeatMasker类别、Repbase、wicker类别的转换
    rmToWicker = {}
    wicker_superfamily_set = set()
    with open(config.work_dir + '/TEClasses.tsv', 'r') as f_r:
        for i,line in enumerate(f_r):
            parts = line.split('\t')
            rm_type = parts[5]
            rm_subtype = parts[6]
            repbase_type = parts[7]
            wicker_type = parts[8]
            wicker_type_parts = wicker_type.split('/')
            #print(rm_type + ',' + rm_subtype + ',' + repbase_type + ',' + wicker_type)
            if len(wicker_type_parts) != 3:
                continue
            wicker_superfamily_parts = wicker_type_parts[-1].strip().split(' ')
            if len(wicker_superfamily_parts) == 1:
                wicker_superfamily = wicker_superfamily_parts[0]
            elif len(wicker_superfamily_parts) > 1:
                wicker_superfamily = wicker_superfamily_parts[1].replace('(', '').replace(')', '')
            rm_full_type = rm_type+'/'+rm_subtype
            if wicker_superfamily == 'ERV':
                wicker_superfamily = 'Retrovirus'
            if wicker_superfamily == 'Viper':
                wicker_superfamily = 'VIPER'
            if wicker_superfamily == 'H':
                wicker_superfamily = 'Helitron'
            rmToWicker[rm_full_type] = wicker_superfamily
            wicker_superfamily_set.add(wicker_superfamily)
    #补充一些元素
    rmToWicker['LINE/R2'] = 'R2'
    rmToWicker['Unknown'] = 'Unknown'
    print(rmToWicker)
    print(wicker_superfamily_set)
    print(len(wicker_superfamily_set))

    ## 2.3 获取RepeatClassifier分类后的序列标签，对于未能标注到superfamily的标签或者错误的标签，我们直接标为Unknown （因为我们这里的数据集标签是直接打到了superfamily层级）
    names, contigs = read_fasta_v1(classified_path)
    RC_name_labels = {}
    for name in names:
        label = name.split('#')[1].split(' ')[0]
        if not rmToWicker.__contains__(label):
            label = 'Unknown'
        else:
            wicker_superfamily = rmToWicker[label]
            label = wicker_superfamily
        RC_name_labels[name.split('#')[0]] = label

    # 2.4 获取test数据的name与标签，然后与RepeatClassifier预测的标签进行评估
    names, contigs = read_fasta_v1(classified_path)
    sequence_names = []
    y_labels = []
    for name in names:
        parts = name.split('\t')
        seq_name = parts[0]
        seq_name_parts = seq_name.split(' ')
        if len(seq_name_parts) > 1:
            seq_name = seq_name_parts[0].split('#')[0]
            label = seq_name_parts[1]
        else:
            label = parts[1]
        sequence_names.append(seq_name)
        y_labels.append(label)

    y_predicts = []
    for name in sequence_names:
        y_predicts.append(RC_name_labels[name])

    print(y_labels)
    print(len(y_labels))
    print(y_predicts)
    print(len(y_predicts))
    y_test = np.array(y_labels)
    y_pred = np.array(y_predicts)
    get_metrics_by_label(y_test, y_pred)
    return RC_name_labels


def transform_TERL_data(train_path, test_path, data_dir):
    # 1.首先将数据集切分成TERL需要的结构
    train_dir = data_dir + '/Train'
    test_dir = data_dir + '/Test'
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    train_names, train_contigs = read_fasta_v1(train_path)
    test_names, test_contigs = read_fasta_v1(test_path)
    train_contigs_dict = {}
    for name in train_names:
        label = name.split('\t')[1]
        if not train_contigs_dict.__contains__(label):
            train_contigs_dict[label] = {}
        cur_train_contigs = train_contigs_dict[label]
        cur_train_contigs[name] = train_contigs[name]
    for label in train_contigs_dict.keys():
        cur_path = train_dir + '/' + label + '.fa'
        store_fasta(train_contigs_dict[label], cur_path)

    test_contigs_dict = {}
    for name in test_names:
        label = name.split('\t')[1]
        if not test_contigs_dict.__contains__(label):
            test_contigs_dict[label] = {}
        cur_test_contigs = test_contigs_dict[label]
        cur_test_contigs[name] = test_contigs[name]
    for label in test_contigs_dict.keys():
        cur_path = test_dir + '/' + label + '.fa'
        store_fasta(test_contigs_dict[label], cur_path)

# 评估TERL的性能
def evaluate_TERL(test_path, predict_path):
    # 在test数据上评估TERL的准确性
    names, contigs = read_fasta_v1(test_path)
    y_labels = []
    for name in names:
        parts = name.split('\t')
        label = parts[1]
        y_labels.append(label)

    names, contigs = read_fasta_v1(predict_path)
    y_predicts = []
    for name in names:
        parts = name.split('\t')
        label = parts[-2]
        y_predicts.append(label)

    print(y_labels)
    print(len(y_labels))
    print(y_predicts)
    print(len(y_predicts))
    y_test = np.array(y_labels)
    y_pred = np.array(y_predicts)
    get_metrics_by_label(y_test, y_pred)

# 评估ClassifyTE的性能
def evaluate_ClassifyTE(predict_path):
    y_labels = []
    y_predicts = []
    y_predicts_set = set()
    not_superfamily_labels = ('LTR', 'SubclassI', 'LINE', 'SINE')
    with open(predict_path, 'r') as f_r:
        for i, line in enumerate(f_r):
            if i == 0:
                continue
            line = line.replace('\n', '')
            parts = line.split(',')
            raw_name = parts[0]
            y_label = raw_name.split('\t')[1]
            y_predict = parts[1]
            y_predicts_set.add(y_predict)
            if y_predict == 'gypsy':
                y_predict = 'Gypsy'
            if y_predict in not_superfamily_labels:
                y_predict = 'Unknown'
            y_labels.append(y_label)
            y_predicts.append(y_predict)
    print(y_predicts_set)
    print(y_labels)
    print(len(y_labels))
    print(y_predicts)
    print(len(y_predicts))
    y_test = np.array(y_labels)
    y_pred = np.array(y_predicts)
    get_metrics_by_label(y_test, y_pred)

# 将 repbase 转换成 DeepTE training 模型的输入
def transform_repbase_to_DeepTE_input(repbase_path, DeepTE_input):
    names, contigs = read_fasta_v1(repbase_path)
    DeepTE_set = set()
    with open(DeepTE_input, 'w') as f_save:
        for name in names:
            label = name.split('\t')[1]
            seq = contigs[name]
            f_save.write(label+','+seq+'\n')
    print(DeepTE_set)
    print(len(DeepTE_set))

def transform_DeepTE_to_fasta(raw_dataset, outpt):
    node_index = 0
    new_contigs = {}
    filter_labels = ['nLTR_LINE', 'nLTR_SINE']
    with open(raw_dataset, 'r') as f_r:
        for line in f_r:
            line = line.replace('\n', '')
            parts = line.split(',')
            label = parts[0]
            if label in filter_labels:
                continue
            seq = parts[1]
            new_contigs['node_'+str(node_index)+'\t'+label] = seq
            node_index += 1
    store_fasta(new_contigs, outpt)

def evaluate_DeepTE(test_path, predict_path):
    names, contigs = read_fasta_v1(test_path)
    y_labels_dict = {}
    for name in names:
        parts = name.split('\t')
        seq_name = parts[0]
        label = parts[1]
        # wicker_label = config.DeepTE_class[label]
        # y_labels_dict[seq_name] = wicker_label
        y_labels_dict[seq_name] = label

    ## 4.2 将DeepTE的分类标签转成superfamily级别，如果没到superfamily，则为unknown
    DeepTE_labels = {'ClassII_DNA_Mutator_unknown': 'Mutator', 'ClassII_DNA_TcMar_nMITE': 'Tc1-Mariner',
                     'ClassII_DNA_hAT_unknown': 'hAT', 'ClassII_DNA_P_MITE': 'P', 'ClassI_nLTR': 'Unknown',
                     'ClassIII_Helitron': 'Helitron', 'ClassI_LTR_Gypsy': 'Gypsy', 'ClassI_LTR': 'Unknown',
                     'ClassII_DNA_Mutator_MITE': 'Mutator', 'ClassI_LTR_Copia': 'Copia', 'ClassI_nLTR_LINE': 'Unknown',
                     'ClassII_DNA_CACTA_unknown': 'CACTA', 'ClassI_nLTR_LINE_I': 'I', 'ClassI_nLTR_DIRS': 'DIRS',
                     'ClassII_MITE': 'Unknown', 'unknown': 'Unknown', 'ClassII_DNA_TcMar_unknown': 'Tc1-Mariner',
                     'ClassII_DNA_CACTA_MITE': 'CACTA', 'ClassII_DNA_Harbinger_unknown': 'PIF-Harbinger',
                     'ClassII_DNA_hAT_nMITE': 'hAT', 'ClassI': 'Unknown', 'ClassI_nLTR_SINE_7SL': '7SL',
                     'ClassII_DNA_Harbinger_nMITE': 'PIF-Harbinger', 'ClassII_DNA_Mutator_nMITE': 'Mutator',
                     'ClassII_DNA_hAT_MITE': 'hAT', 'ClassII_DNA_CACTA_nMITE': 'CACTA', 'ClassI_nLTR_SINE_tRNA': 'tRNA',
                     'ClassII_DNA_TcMar_MITE': 'Tc1-Mariner', 'ClassII_DNA_P_nMITE': 'P', 'ClassI_nLTR_PLE': 'Penelope',
                     'ClassII_DNA_Harbinger_MITE': 'PIF-Harbinger', 'ClassI_nLTR_LINE_L1': 'L1', 'ClassII_nMITE': 'Unknown',
                     'ClassI_LTR_ERV': 'Retrovirus', 'ClassI_LTR_BEL': 'Bel-Pao', 'ClassI_nLTR_LINE_RTE': 'RTE', 'ClassI_nLTR_LINE_R2': 'R2',
                     'ClassII_DNA_Transib_nMITE': 'Transib', 'ClassII_DNA_PiggyBac_nMITE': 'PiggyBac', 'ClassI_nLTR_LINE_Jockey': 'Jockey',
                     'ClassI_nLTR_SINE_5S': '5S'}

    y_predict_seq_names = []
    y_predicts = []
    with open(predict_path, 'r') as f_r:
        for line in f_r:
            line = line.replace('\n', '')
            parts = line.split('\t')
            seq_name = parts[0]
            predict = parts[1]
            predict = DeepTE_labels[predict]
            y_predict_seq_names.append(seq_name)
            y_predicts.append(predict)

    y_labels = []
    for seq_name in y_predict_seq_names:
        y_labels.append(y_labels_dict[seq_name])

    print(y_labels)
    print(len(y_labels))
    print(y_predicts)
    print(len(y_predicts))
    y_test = np.array(y_labels)
    y_pred = np.array(y_predicts)
    get_metrics_by_label(y_test, y_pred)

def transform_repbase_bed(repbase_bed, NeuralTE_bed, label_dict):
    new_lines = []
    with open(repbase_bed, 'r') as f_r:
        for line in f_r:
            if line.startswith('#'):
                continue
            line = line.replace('\n', '')
            parts = line.split('\t')
            chr_name = parts[0]
            chr_start = int(parts[1])
            chr_end = int(parts[2])
            info = parts[3]
            info_parts = info.split(';')
            te_name = info_parts[9]
            te_class = info_parts[10]
            new_label = label_dict[te_name]
            new_string = ''
            for i, item in enumerate(info_parts):
                if i == 10:
                    new_string += new_label
                else:
                    new_string += item
                if i != len(info_parts)-1:
                    new_string += ';'
            new_line = ''
            for i, item in enumerate(parts):
                if i == 3:
                    new_line += new_string
                else:
                    new_line += item
                if i != len(parts)-1:
                    new_line += '\t'
                else:
                    new_line += '\n'
            new_lines.append(new_line)
    #print(new_lines)
    with open(NeuralTE_bed, 'w') as f_save:
        for line in new_lines:
            f_save.write(line)

def evalute_genome_coverage(mazie_merge_path, genome_path, temp_dir, threads):
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    # 1. 将序列用RepeatMasker比对到基因组上。
    RepeatMasker_command = 'cd ' + temp_dir + ' && RepeatMasker -e ncbi -pa ' + str(threads) \
                           + ' -q -no_is -norna -nolow -div 40 -gff -lib ' + mazie_merge_path + ' -cutoff 225 ' \
                           + genome_path
    print(RepeatMasker_command)
    os.system(RepeatMasker_command)
    out_file = genome_path + '.out'
    # 2. 将.out文件转.bed文件
    convert2bed_command = 'perl ' + config.project_dir + '/tools/RMout_to_bed.pl ' + out_file + ' base1'
    print(convert2bed_command)
    os.system(convert2bed_command)

    bed_file = out_file + '.bed'
    # 3. 用程序分析比对.bed，分析比例
    analyze_class_ratio(bed_file, temp_dir)

def analyze_class_ratio(bed_file, work_dir):
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    # 1.将.out转换成可分析格式，例如.gff
    # 2、获取每个element对应的染色体以及染色体上的起始、终止位置，位置可能会有overlap，合并有overlap的坐标，此时得到每个element对应的不重叠的位置
    # te_chrs -> {te_name: {chr_name: []}}
    te_chrs = {}
    #class_chrs = -> {class_name: {chr_name: []}}
    class_chrs = {}
    # main_class_chrs = -> {class_name: {chr_name: []}}
    # main_class_chrs = {}

    te_class_dict = {}

    #保存每个te对应的碱基数量
    te_bases = {}
    #保存每个 class 对应的碱基数量
    class_bases = {}
    # 保存每个 main_class 对应的碱基数量
    main_class_bases = {}
    with open(bed_file, 'r') as f_r:
        for line in f_r:
            if line.startswith('#'):
                continue
            parts = line.split('\t')
            chr_name = parts[0]
            chr_start = int(parts[1])
            chr_end = int(parts[2])
            info = parts[3]
            info_parts = info.split(';')
            te_name = info_parts[9]
            te_class = info_parts[10]
            if not te_chrs.__contains__(te_name):
                te_chrs[te_name] = {}
            chr_dict = te_chrs[te_name]
            if not chr_dict.__contains__(chr_name):
                chr_dict[chr_name] = []
            chr_pos = chr_dict[chr_name]
            chr_pos.append((chr_start, chr_end))

            if not class_chrs.__contains__(te_class):
                class_chrs[te_class] = {}
            chr_dict = class_chrs[te_class]
            if not chr_dict.__contains__(chr_name):
                chr_dict[chr_name] = []
            chr_pos = chr_dict[chr_name]
            chr_pos.append((chr_start, chr_end))

            # if te_class.__contains__('LTR'):
            #     main_class = 'LTR'
            # elif te_class.__contains__('Helitron'):
            #     main_class = 'Helitron'
            # elif te_class.__contains__('DNA'):
            #     main_class = 'DNA'
            # elif te_class.__contains__('MITE'):
            #     main_class = 'MITE'
            #
            # if not main_class_chrs.__contains__(main_class):
            #     main_class_chrs[main_class] = {}
            # chr_dict = main_class_chrs[main_class]
            # if not chr_dict.__contains__(chr_name):
            #     chr_dict[chr_name] = []
            # chr_pos = chr_dict[chr_name]
            # chr_pos.append((chr_start, chr_end))

            te_class_dict[te_name] = te_class

    for te_name in te_chrs.keys():
        chr_dict = te_chrs[te_name]
        total_bases = 0
        for chr_name in chr_dict.keys():
            chr_pos = chr_dict[chr_name]
            merged_intervals = merge_overlapping_intervals(chr_pos)
            for segment in merged_intervals:
                total_bases += abs(segment[1] - segment[0])
        te_bases[te_name] = total_bases
    #print(te_bases)

    # #统计占比最高的前10个DNA转座子
    # te_base_list = list(te_bases.items())
    # te_base_list.sort(key=lambda x: -x[1])
    # top_dna = []
    # for item in te_base_list:
    #     te_name = item[0]
    #     class_name = te_class_dict[te_name]
    #     if not class_name.__contains__('Helitron') and (class_name.__contains__('DNA') or class_name.__contains__('MITE')):
    #         top_dna.append((te_name, class_name, item[1]))

    for class_name in class_chrs.keys():
        chr_dict = class_chrs[class_name]
        total_bases = 0
        for chr_name in chr_dict.keys():
            chr_pos = chr_dict[chr_name]
            merged_intervals = merge_overlapping_intervals(chr_pos)
            for segment in merged_intervals:
                total_bases += abs(segment[1] - segment[0])
        class_bases[class_name] = total_bases
    #print(class_bases)

    # for class_name in main_class_chrs.keys():
    #     chr_dict = main_class_chrs[class_name]
    #     total_bases = 0
    #     for chr_name in chr_dict.keys():
    #         chr_pos = chr_dict[chr_name]
    #         merged_intervals = merge_overlapping_intervals(chr_pos)
    #         for segment in merged_intervals:
    #             total_bases += abs(segment[1] - segment[0])
    #     main_class_bases[class_name] = total_bases

    # store for testing
    te_base_file = work_dir + '/te_base.json'
    with codecs.open(te_base_file, 'w', encoding='utf-8') as f:
        json.dump(te_bases, f)
    class_base_file = work_dir + '/class_base.json'
    with codecs.open(class_base_file, 'w', encoding='utf-8') as f:
        json.dump(class_bases, f)
    # main_class_base_file = work_dir + '/main_class_base.json'
    # with codecs.open(main_class_base_file, 'w', encoding='utf-8') as f:
    #     json.dump(main_class_bases, f)
    # top_dna_file = work_dir + '/top_dna.json'
    # with codecs.open(top_dna_file, 'w', encoding='utf-8') as f:
    #     json.dump(top_dna, f)

def merge_overlapping_intervals(intervals):
    # 按照起始位置对基因组位置数组进行排序
    sorted_intervals = sorted(intervals, key=lambda x: x[0])

    merged_intervals = []
    current_interval = sorted_intervals[0]

    # 遍历排序后的基因组位置数组
    for interval in sorted_intervals[1:]:
        # 如果当前基因组位置与下一个基因组位置有重叠
        if current_interval[1] >= interval[0]:
            # 更新当前基因组位置的结束位置
            current_interval = (current_interval[0], max(current_interval[1], interval[1]))
        else:
            # 如果没有重叠，则将当前基因组位置添加到结果数组中，并更新当前基因组位置为下一个基因组位置
            merged_intervals.append(current_interval)
            current_interval = interval

    # 将最后一个基因组位置添加到结果数组中
    merged_intervals.append(current_interval)

    return merged_intervals

def get_train_except_species(total_repbase, species_name, train_path):
    names, contigs = read_fasta_v1(total_repbase)
    train_contigs = {}
    for name in names:
        parts = name.split('\t')
        cur_species_name = parts[2]
        if cur_species_name != species_name:
            train_contigs[name] = contigs[name]
    store_fasta(train_contigs, train_path)

# 画一个3D图，用于展示参数变化后的f1值
def plot_3D_param(x, y, z):
    # 计算边界值
    print(x.shape, y.shape, z.shape)

    # 创建网格点坐标
    xi = np.linspace(min(x), max(x))
    yi = np.linspace(min(y), max(y))
    X, Y = np.meshgrid(xi, yi)

    # 插值
    Z = griddata((x, y), z, (X, Y), method='cubic')

    # 绘制三维图形
    fig = plt.figure()
    ax3 = plt.axes(projection='3d')

    plt.rcParams['font.sans-serif'] = ['FangSong']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # 作图
    surf=ax3.plot_surface(X, Y, Z, cmap='BuPu', linewidth=0, antialiased=False)
    fig.colorbar(surf)
    plt.show()