# from sklearn import datasets
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import numpy as np

from Util import read_fasta_v1, store_fasta
import random

def get_metrics(y_test, y_pred):
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    # 计算精确率
    precision = precision_score(y_test, y_pred, average='macro')
    print("Precision:", precision)
    # 计算召回率
    recall = recall_score(y_test, y_pred, average='macro')
    print("Recall:", recall)
    # 计算F1值
    f1 = f1_score(y_test, y_pred, average='macro')
    print("F1:", f1)


work_dir = '/home/hukang/NeuralTE/data'
threads = 40
cur_repbase_train_path = work_dir + '/repbase_train.ref'
cur_repbase_test_path = work_dir + '/repbase_test.ref'
# 1. 先将Repbase数据按照8-2比例划分成训练集和测试集
cur_repbase_path = work_dir + '/all_repbase.ref_preprocess.ref.update'
names, contigs = read_fasta_v1(cur_repbase_path)
# 随机打乱列表
random.shuffle(names)
# 计算划分的索引位置
split_index = int(0.8 * len(names))
# 划分成80%和20%的两个列表
train_list = names[:split_index]
test_list = names[split_index:]
train_contigs = {}
test_contigs = {}
for name in train_list:
    train_contigs[name] = contigs[name]
for name in test_list:
    test_contigs[name] = contigs[name]
store_fasta(train_contigs, cur_repbase_train_path)
store_fasta(test_contigs, cur_repbase_test_path)

# # 2. 在test数据集上评估RepeatClassifier
## 2.1 运行RepeatClassifier
# HiTE_home = '/home/hukang/HiTE'
# TEClass_home = HiTE_home + '/classification'
# test_home = HiTE_home + '/module'
# protein_lib_path = HiTE_home + '/library/RepeatPeps.lib'
# classify_lib_command = 'cd ' + test_home + ' && python3 ' + test_home + '/get_classified_lib.py' \
#                        + ' --confident_TE_consensus ' + cur_repbase_test_path \
#                        + ' -t ' + str(threads) + ' --tmp_output_dir ' + work_dir \
#                        + ' --classified ' + str(1) + ' --domain ' + str(0) + ' --TEClass_home ' + str(TEClass_home) \
#                        + ' --protein_path ' + str(protein_lib_path) \
#                        + ' --debug ' + str(0)
# os.system(classify_lib_command)

# 2.2 将Dfam分类名称转成wicker格式
## 2.2.1 这个文件里包含了RepeatMasker类别、Repbase、wicker类别的转换
# rmToWicker = {}
# wicker_superfamily_set = set()
# with open('TEClasses.tsv', 'r') as f_r:
#     for i,line in enumerate(f_r):
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
#         rm_full_type = rm_type+'/'+rm_subtype
#         if wicker_superfamily == 'ERV':
#             wicker_superfamily = 'Retrovirus'
#         rmToWicker[rm_full_type] = wicker_superfamily
#         wicker_superfamily_set.add(wicker_superfamily)
# #补充一些元素
# rmToWicker['LINE/R2'] = 'R2'
# print(wicker_superfamily_set)
# print(len(wicker_superfamily_set))
#
# ## 2.3 获取RepeatClassifier分类后的序列标签，对于未能标注到superfamily的标签或者错误的标签，我们直接标为Unknown （因为我们这里的数据集标签是直接打到了superfamily层级）
# classified_path = cur_repbase_test_path + '.classified'
# names, contigs = read_fasta_v1(classified_path)
# RC_name_labels = {}
# for name in names:
#     label = name.split('#')[1]
#     if not rmToWicker.__contains__(label):
#         label = 'Unknown'
#     else:
#         wicker_superfamily = rmToWicker[label]
#         label = wicker_superfamily
#     RC_name_labels[name.split('#')[0]] = label
#
# # 2.4 获取test数据的name与标签，然后与RepeatClassifier预测的标签进行评估
# names, contigs = read_fasta_v1(cur_repbase_test_path)
# sequence_names = []
# y_labels = []
# for name in names:
#     parts = name.split('\t')
#     sequence_names.append(parts[0])
#     label = parts[1]
#     y_labels.append(label)
#
# y_predicts = []
# for name in sequence_names:
#     y_predicts.append(RC_name_labels[name])
#
# print(y_labels)
# print(len(y_labels))
# print(y_predicts)
# print(len(y_predicts))
# y_test = np.array(y_labels)
# y_pred = np.array(y_predicts)
# get_metrics(y_test, y_pred)


# # 3.在test数据集上评估domain结果
# # 3.2 将Dfam分类名称转成wicker格式
# ## 3.2.1 这个文件里包含了RepeatMasker类别、Repbase、wicker类别的转换
# rmToWicker = {}
# wicker_superfamily_set = set()
# with open('TEClasses.tsv', 'r') as f_r:
#     for i,line in enumerate(f_r):
#         parts = line.split('\t')
#         rm_type = parts[5]
#         rm_subtype = parts[6]
#         repbase_type = parts[7]
#         wicker_type = parts[8]
#         wicker_type_parts = wicker_type.split('/')
#         #print(rm_type + ',' + rm_subtype + ',' + repbase_type + ',' + wicker_type)
#         # if len(wicker_type_parts) != 3:
#         #     continue
#         wicker_superfamily_parts = wicker_type_parts[-1].strip().split(' ')
#         if len(wicker_superfamily_parts) == 1:
#             wicker_superfamily = wicker_superfamily_parts[0]
#         elif len(wicker_superfamily_parts) > 1:
#             wicker_superfamily = wicker_superfamily_parts[1].replace('(', '').replace(')', '')
#         rm_full_type = rm_type+'/'+rm_subtype
#         if wicker_superfamily == 'ERV':
#             wicker_superfamily = 'Retrovirus'
#         rmToWicker[rm_full_type] = wicker_superfamily
#         wicker_superfamily_set.add(wicker_superfamily)
# #补充一些元素
# rmToWicker['LINE/R2'] = 'R2'
# rmToWicker['LTR/ERVL'] = 'Retrovirus'
# rmToWicker['LTR/Ngaro'] = 'DIRS'
# print(wicker_superfamily_set)
# print(len(wicker_superfamily_set))
#
# # 3.1 解析domain文件，获取TE_name, label
# data_dir = '/home/hukang/NeuralTE/data'
# domain_path = data_dir + '/repbase_test_part.64.ref.update.domain'
# RC_name_labels = {}
# with open(domain_path, 'r') as f_r:
#     for i, line in enumerate(f_r):
#         if i < 2:
#             continue
#         parts = line.split('\t')
#         TE_name = parts[0]
#         label = parts[1].split('#')[1]
#         if RC_name_labels.__contains__(TE_name):
#             continue
#         if not rmToWicker.__contains__(label):
#             label = 'Unknown'
#         else:
#             wicker_superfamily = rmToWicker[label]
#             label = wicker_superfamily
#         RC_name_labels[TE_name] = label
#
# # 3.2 获取test数据的name与标签，然后与domain预测的标签进行评估
# cur_repbase_test_path = data_dir + '/repbase_test_part.64.ref.update'
# names, contigs = read_fasta_v1(cur_repbase_test_path)
# sequence_names = []
# TE_labels = {}
# y_labels = []
# for name in names:
#     parts = name.split('\t')
#     sequence_names.append(parts[0])
#     label = parts[1]
#     TE_labels[parts[0]] = label
#     y_labels.append(label)
#
# y_predicts = []
# for name in sequence_names:
#     if not RC_name_labels.__contains__(name):
#         predict_label = 'Unknown'
#     else:
#         predict_label = RC_name_labels[name]
#     y_predicts.append(predict_label)
#     # label = TE_labels[name]
#     # y_labels.append(label)
#     # if name == 'ERV3-1-intactLTR1_XT':
#     #     print('here')
#     # if label != predict_label:
#     #     print(name, label, predict_label)
#
# print(y_labels)
# print(len(y_labels))
# print(y_predicts)
# print(len(y_predicts))
# y_test = np.array(y_labels)
# y_pred = np.array(y_predicts)
# get_metrics(y_test, y_pred)



# 3.在test数据集上评估ClassifyTE
## 3.1 运行ClassifyTE，预测test数据的标签
#ClassifyTE_home = '/public/home/hpc194701009/TE_Classification/ClassifyTE'
#os.system('cd ' + ClassifyTE_home + ' && python generate_feature_file.py -f repbase_test_part.ref -d repbase_test_features -o repbase_test_features.csv && python evaluate.py -f repbase_test_features.csv -n node.txt -d repbase_test_features -m ClassifyTE_combined.pkl -a lcpnb')
## 3.2 与标注的标签进行比较
### 3.2.1 获取分类后的序列标签，对于未能标注到superfamily的标签或者错误的标签，我们直接标为Unknown （因为我们这里的数据集标签是直接打到了superfamily层级）
# ClassifyTE_result = '/home/hukang/TE_Classification/ClassifyTE/output/predicted_out_repbase_test_features.csv'
# y_labels = []
# y_predicts = []
# not_superfamily_labels = ('LTR', 'SubclassI', 'LINE', 'SINE')
# with open(ClassifyTE_result, 'r') as f_r:
#     for i, line in enumerate(f_r):
#         if i == 0:
#             continue
#         parts = line.replace('\n', '').split(',')
#         raw_name = parts[0]
#         y_label = raw_name.split('\t')[1]
#         y_predict = parts[1]
#         if y_predict == 'gypsy':
#             y_predict = 'Gypsy'
#         if y_predict in not_superfamily_labels:
#             y_predict = 'Unknown'
#         y_labels.append(y_label)
#         y_predicts.append(y_predict)
# print(y_labels)
# print(len(y_labels))
# print(y_predicts)
# print(len(y_predicts))
# y_test = np.array(y_labels)
# y_pred = np.array(y_predicts)
# get_metrics(y_test, y_pred)


# # 4.在test数据集上评估DeepTE
# ## 4.1 对test数据集
# DeepTE_home = '/public/home/hpc194701009/TE_Classification/DeepTE-master'
# #os.system('cd ' + DeepTE_home + ' && python DeepTE.py -d test_work -o test_result -i example_data/input_test.fasta -sp P -m_dir models/Plants_model/')
# repbase_test_path = '/public/home/hpc194701009/TE_Classification/DeepTE-master/example_data/repbase_test_part.ref'
# names, contigs = read_fasta_v1(repbase_test_path)
# y_labels_dict = {}
# for name in names:
#     parts = name.split('\t')
#     seq_name = parts[0]
#     label = parts[1]
#     y_labels_dict[seq_name] = label
#
# ## 4.2 将DeepTE的分类标签转成superfamily级别，如果没到superfamily，则为unknown
# DeepTE_labels = {'ClassII_DNA_Mutator_unknown': 'Mutator', 'ClassII_DNA_TcMar_nMITE': 'Tc1-Mariner',
#                  'ClassII_DNA_hAT_unknown': 'hAT', 'ClassII_DNA_P_MITE': 'P', 'ClassI_nLTR': 'Unknown',
#                  'ClassIII_Helitron': 'Helitron', 'ClassI_LTR_Gypsy': 'Gypsy', 'ClassI_LTR': 'Unknown',
#                  'ClassII_DNA_Mutator_MITE': 'Mutator', 'ClassI_LTR_Copia': 'Copia', 'ClassI_nLTR_LINE': 'Unknown',
#                  'ClassII_DNA_CACTA_unknown': 'CACTA', 'ClassI_nLTR_LINE_I': 'I', 'ClassI_nLTR_DIRS': 'DIRS',
#                  'ClassII_MITE': 'Unknown', 'unknown': 'Unknown', 'ClassII_DNA_TcMar_unknown': 'Tc1-Mariner',
#                  'ClassII_DNA_CACTA_MITE': 'CACTA', 'ClassII_DNA_Harbinger_unknown': 'PIF-Harbinger',
#                  'ClassII_DNA_hAT_nMITE': 'hAT', 'ClassI': 'Unknown', 'ClassI_nLTR_SINE_7SL': '7SL',
#                  'ClassII_DNA_Harbinger_nMITE': 'PIF-Harbinger', 'ClassII_DNA_Mutator_nMITE': 'Mutator',
#                  'ClassII_DNA_hAT_MITE': 'hAT', 'ClassII_DNA_CACTA_nMITE': 'CACTA', 'ClassI_nLTR_SINE_tRNA': 'tRNA',
#                  'ClassII_DNA_TcMar_MITE': 'Tc1-Mariner', 'ClassII_DNA_P_nMITE': 'P', 'ClassI_nLTR_PLE': 'Unknown',
#                  'ClassII_DNA_Harbinger_MITE': 'PIF-Harbinger', 'ClassI_nLTR_LINE_L1': 'L1', 'ClassII_nMITE': 'Unknown'}
#
# predict_path = '/public/home/hpc194701009/TE_Classification/DeepTE-master/repbase_test_result/opt_DeepTE.txt'
# y_predict_seq_names = []
# y_predicts = []
# with open(predict_path, 'r') as f_r:
#     for line in f_r:
#         line = line.replace('\n', '')
#         parts = line.split('\t')
#         seq_name = parts[0]
#         predict = parts[1]
#         predict = DeepTE_labels[predict]
#         y_predict_seq_names.append(seq_name)
#         y_predicts.append(predict)
#
# y_labels = []
# for seq_name in y_predict_seq_names:
#     y_labels.append(y_labels_dict[seq_name])
#
# print(y_labels)
# print(len(y_labels))
# print(y_predicts)
# print(len(y_predicts))
# y_test = np.array(y_labels)
# y_pred = np.array(y_predicts)
# get_metrics(y_test, y_pred)


# # 5.在test数据上评估TERL的准确性
# repbase_test_path = '/public/home/hpc194701009/TE_Classification/TERL/Data/DS1/repbase_test_part.ref'
# names, contigs = read_fasta_v1(repbase_test_path)
# y_labels = []
# for name in names:
#     parts = name.split('\t')
#     label = parts[1]
#     y_labels.append(label)
#
# predict_path = '/public/home/hpc194701009/TE_Classification/TERL/TERL_20230720_173316_repbase_test_part.ref'
# names, contigs = read_fasta_v1(predict_path)
# y_predicts = []
# for name in names:
#     parts = name.split('\t')
#     label = parts[-2]
#     y_predicts.append(label)
#
# print(y_labels)
# print(len(y_labels))
# print(y_predicts)
# print(len(y_predicts))
# y_test = np.array(y_labels)
# y_pred = np.array(y_predicts)
# get_metrics(y_test, y_pred)




# # 加载数据集
# iris = datasets.load_iris()
# X = iris.data
# y = iris.target
#
# # 划分数据集为训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # 创建分类器
# classifier = LogisticRegression()
#
# # 在训练集上训练分类器
# classifier.fit(X_train, y_train)
#
# # 在测试集上进行预测
# y_pred = classifier.predict(X_test)
#

