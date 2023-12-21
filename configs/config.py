#-- coding: UTF-8 --
# config.py：本文件存储NeuralTE中定义的变量和参数，修改之前确保你理解它的作用，否则建议保持默认值

import os
from multiprocessing import cpu_count
current_folder = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(current_folder, "..")

# 1.数据预处理参数
## 是否使用相应的特征进行分类，所有的特征已被证明对分类有帮助。
use_kmers = 1   # 使用k-mer特征
use_terminal = 1    # 使用LTR、TIR终端特征
use_TSD = 0     # 使用TSD特征
use_domain = 1  # 使用TE domain特征
use_ends = 1    # 使用TE 5'端和3'端各5bp特征

use_minority = 1 # 使用少数样本进行纠错
is_train = 0  # 当前是否为模型训练阶段.
is_predict = 1  # Enable prediction mode. Setting to 0 requires the input FASTA file to be in Repbase format (seq_name\tLabel\tspecies).
is_wicker = 1   # Use Wicker classification labels. Setting to 0 will output RepeatMasker classification labels."
is_plant = 0 # Is the input genome of a plant? 0 represents non-plant, while 1 represents plant.

# 2.程序和模型参数
threads = int(cpu_count())  # 使用多线程加载数据
internal_kmer_sizes = [1, 2, 4]   # 内部序列转k-mer频率特征所使用的k-mer大小
terminal_kmer_sizes = [1, 2, 4] # 终端序列转k-mer频率特征所使用的k-mer大小
## CNN model参数
cnn_num_convs = 3 # CNN卷积层数量
cnn_filters_array = [32, 32, 32] # CNN每层卷积层的filter数量
cnn_kernel_sizes_array = [3, 3, 3] # CNN每层卷积层的kernel size大小, 二维卷积层设置为[(3, 3), ...]
cnn_dropout = 0.5 # CNN Dropout阈值
## 训练参数
batch_size = 32 # 训练的batch size大小
epochs = 50 # 训练的epoch次数
use_checkpoint = 0  # 是否使用断点训练，使用设置为1；当使用断点训练，模型会从上次失败的参数继续训练，避免从头训练


###################################################分界线，下面无需修改参数######################################################################
version_num = '1.0.0'
work_dir = project_dir + '/work' # 数据预处理时临时工作目录

# 小样本标签
#minority_labels_class = {'Crypton': 0, '5S': 1, '7SL': 2, 'Merlin': 3, 'P': 4, 'R2': 5, 'Unknown': 6}
minority_labels_class = {'Crypton': 0, '5S': 1, 'Merlin': 2, 'P': 3, 'R2': 4, 'Unknown': 5}

## 根据wicker分类的superfamily标签
all_wicker_class = {'Tc1-Mariner': 0, 'hAT': 1, 'Mutator': 2, 'Merlin': 3, 'Transib': 4, 'P': 5, 'PiggyBac': 6,
                    'PIF-Harbinger': 7, 'CACTA': 8, 'Crypton': 9, 'Helitron': 10, 'Maverick': 11, 'Copia': 12,
                    'Gypsy': 13, 'Bel-Pao': 14, 'Retrovirus': 15, 'DIRS': 16, 'Ngaro': 17, 'VIPER': 18,
                    'Penelope': 19, 'R2': 20, 'RTE': 21, 'Jockey': 22, 'L1': 23, 'I': 24, 'tRNA': 25, '7SL': 26, '5S': 27, 'Unknown': 28}
class_num = len(all_wicker_class)
print('Total Class num: ' + str(class_num))
inverted_all_wicker_class = {value: key for key, value in all_wicker_class.items()}
# 最大的 TSD 长度
max_tsd_length = 15
# 获取CNN输入维度
X_feature_len = 0
# TE seq/internal_seq 维度
if use_kmers != 0:
    for kmer_size in internal_kmer_sizes:
        X_feature_len += pow(4, kmer_size)
    if use_terminal != 0:
        for i in range(2):
            for kmer_size in terminal_kmer_sizes:
                X_feature_len += pow(4, kmer_size)
if use_TSD != 0:
    X_feature_len += max_tsd_length * 4 + 1
# if use_minority != 0:
#     X_feature_len += len(minority_labels_class)
if use_domain != 0:
    X_feature_len += len(all_wicker_class)
if use_ends != 0:
    X_feature_len += 10 * 4

## Repbase label对应到wicker label
Repbase_wicker_labels = {'Mariner/Tc1': 'Tc1-Mariner', 'mariner/Tc1 superfamily': 'Tc1-Mariner', 'hAT': 'hAT',
                         'HAT superfamily': 'hAT', 'MuDR': 'Mutator', 'Merlin': 'Merlin', 'Transib': 'Transib',
                         'P': 'P', 'P-element': 'P', 'PiggyBac': 'PiggyBac', 'Harbinger': 'PIF-Harbinger',
                         'EnSpm/CACTA': 'CACTA', 'Crypton': 'Crypton', 'CryptonF': 'Crypton', 'CryptonS': 'Crypton',
                         'CryptonI': 'Crypton', 'CryptonV': 'Crypton', 'CryptonA': 'Crypton', 'Helitron': 'Helitron',
                         'HELITRON superfamily': 'Helitron', 'Copia': 'Copia', 'Gypsy': 'Gypsy',
                         'GYPSY superfamily': 'Gypsy', 'Gypsy retrotransposon': 'Gypsy', 'BEL': 'Bel-Pao',
                         'Caulimoviridae': 'Retrovirus',
                         'ERV1': 'Retrovirus', 'ERV2': 'Retrovirus', 'ERV3': 'Retrovirus', 'Lentivirus': 'Retrovirus',
                         'ERV4': 'Retrovirus', 'Lokiretrovirus': 'Retrovirus', 'DIRS': 'DIRS', 'Penelope': 'Penelope',
                         'Penelope/Poseidon': 'Penelope', 'Neptune': 'Penelope', 'Nematis': 'Penelope',
                         'Athena': 'Penelope', 'Coprina': 'Penelope', 'Hydra': 'Penelope', 'Naiad/Chlamys': 'Penelope',
                         'R2': 'R2', 'RTE': 'RTE', 'Jockey': 'Jockey', 'L1': 'L1', 'I': 'I', 'SINE2/tRNA': 'tRNA',
                         'SINE1/7SL': '7SL', 'SINE3/5S': '5S', 'Unknown': 'Unknown'}

## 对每种Repbase数据进行扩增
expandClassNum = {'Merlin': 20, 'Transib': 10, 'P': 10, 'Crypton': 10, 'Penelope': 5, 'R2': 20, 'RTE': 8, 'Jockey': 10, 'I': 10}

## ClassifyTE 工具中能分类的superfamily
ClassifyTE_class = {'Tc1-Mariner': '2.1.1.1', 'hAT': '2.1.1.2', 'Mutator': '2.1.1.3',
                'Merlin': '2.1.1.4', 'Transib': '2.1.1.5', 'P': '2.1.1.6',
                'PiggyBac': '2.1.1.7', 'PIF-Harbinger': '2.1.1.8', 'CACTA': '2.1.1.9',
                'Copia': '1.1.1', 'Gypsy': '1.1.2', 'Bel-Pao': '1.1.3',
                'DIRS': '1.2', 'R2': '1.4.1', 'RTE': '1.4.2',
                'Jockey': '1.4.3', 'L1': '1.4.4', 'I': '1.4.5',
                'tRNA': '1.5.1', '7SL': '1.5.2', '5S': '1.5.3'}

## Repbase的标签转为DeepTE的标签
DeepTE_class = {'DNA_MITE_Tc': 'Tc1-Mariner', 'DNA_MITE_Harbinger': 'PIF-Harbinger',
             'DNA_MITE_hAT': 'hAT', 'DNA_MITE_CACTA': 'CACTA', 'DNA_MITE_MuDR': 'Mutator',
             'DNA_nMITE_Tc': 'Tc1-Mariner', 'DNA_nMITE_Harbinger': 'PIF-Harbinger', 'DNA_nMITE_hAT': 'hAT',
             'DNA_nMITE_CACTA': 'CACTA', 'DNA_nMITE_MuDR': 'Mutator', 'LTR_Copia': 'Copia', 'LTR_Gypsy': 'Gypsy',
             'LTR_ERV': 'Retrovirus', 'LTR_BEL': 'Bel-Pao', 'DIRS_DIRS': 'DIRS', 'RC_Helitron': 'Helitron'}
inverted_DeepTE_class = {'Tc1-Mariner': 'DNA_nMITE_Tc', 'PIF-Harbinger': 'DNA_nMITE_Harbinger', 'hAT': 'DNA_nMITE_hAT',
                         'CACTA': 'DNA_nMITE_CACTA', 'Mutator': 'DNA_nMITE_MuDR', 'Copia': 'LTR_Copia',
                         'Gypsy': 'LTR_Gypsy', 'Retrovirus': 'LTR_ERV', 'Bel-Pao': 'LTR_BEL', 'DIRS': 'DIRS_DIRS',
                         'Helitron': 'RC_Helitron'}