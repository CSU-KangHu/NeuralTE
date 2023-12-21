# 此程序是加载数据，根据配置文件中使用哪些特征，进而判断数据是否需要进行相应的预处理
import os
import sys

current_folder = os.path.dirname(os.path.abspath(__file__))
# 添加 configs 文件夹的路径到 Python 路径
configs_folder = os.path.join(current_folder, "..")  # 需要根据实际目录结构调整
sys.path.append(configs_folder)

from configs import config
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from utils.data_util import read_fasta_v1, generate_TSD_info, generate_domain_info, generate_terminal_info, \
    load_repbase_with_TSD, generate_feature_mats, read_fasta, connect_LTR, store_fasta, generate_minority_info


class DataProcessor:
    def __init__(self):
        self.project_dir = config.project_dir
        self.tool_dir = self.project_dir + '/tools'
        self.work_dir = config.work_dir
        self.threads = config.threads
        self.minority_labels_class = config.minority_labels_class
        self.all_wicker_class = config.all_wicker_class
        self.use_terminal = config.use_terminal
        self.use_TSD = config.use_TSD
        self.use_domain = config.use_domain
        self.use_minority = config.use_minority
        self.use_ends = config.use_ends
        self.is_train = config.is_train
        self.is_predict = config.is_predict

        self.ex = ProcessPoolExecutor(self.threads)


    def load_data(self, internal_kmer_sizes, terminal_kmer_sizes, data_path):
        # 将输入文件拷贝到工作目录中
        if not os.path.exists(data_path):
            print('Input file not exist: ' + data_path)
            exit(-1)
        os.makedirs(config.work_dir, exist_ok=True)
        os.system('cp ' + data_path + ' ' + config.work_dir)
        genome_info_path = config.work_dir + '/genome.info'
        data_path = config.work_dir + '/' + os.path.basename(data_path)
        domain_train_path = data_path + '.domain'

        minority_temp = config.work_dir + '/minority'
        if not os.path.exists(minority_temp):
            os.makedirs(minority_temp)
        minority_train_path = minority_temp + '/train.minority.ref'
        minority_out = minority_temp + '/train.minority.out'

        data_path = self.preprocess_data(data_path, domain_train_path, minority_train_path, minority_out, self.work_dir,
                                                    self.project_dir, self.tool_dir, self.threads,
                                                    self.use_TSD, self.use_domain, self.use_minority, self.use_terminal,
                                                    self.is_predict, self.is_train, genome_info_path)

        X, Y, seq_names = load_repbase_with_TSD(data_path, domain_train_path, minority_train_path, minority_out,
                                                    self.all_wicker_class, self.project_dir+'/data/TEClasses.tsv')

        X, Y = generate_feature_mats(X, Y, seq_names, self.minority_labels_class, self.all_wicker_class,
                                                    internal_kmer_sizes, terminal_kmer_sizes, self.ex)

        # 将数据reshape成模型接收的格式
        X = X.reshape(X.shape[0], config.X_feature_len, 1)
        X = X.astype('float64')
        return X, Y, seq_names, data_path

    def preprocess_data(self, data, domain_train_path, minority_train_path, minority_out, work_dir, project_dir,
                        tool_dir, threads, use_TSD, use_domain, use_minority, use_terminal, is_predict, is_train, genome_info_path):
        # 删除上一次运行保留结果
        SegLTR2intactLTRMap = config.work_dir + '/segLTR2intactLTR.map'
        os.system('rm -f ' + SegLTR2intactLTRMap)

        if is_predict:
            # 将输入文件格式化为Repbase格式
            names, contigs = read_fasta(data)
            os.makedirs(os.path.dirname(data), exist_ok=True)
            with open(data, 'w') as f_save:
                for name in names:
                    seq = contigs[name]
                    name = name.split('/')[0].split('#')[0]
                    new_name = name + '\tUnknown\tUnknown'
                    f_save.write('>' + new_name + '\n' + seq + '\n')

            # 将输入的TE library中LTR序列与LTR internal序列连接起来成为一个完整的LTR序列
            data, repbase_labels = connect_LTR(data)

            # 删除domain文件，确保每次重新生成domain文件
            os.system('rm -f ' + domain_train_path)

        names, contigs = read_fasta_v1(data)
        # 将Repbase label转成wicker格式
        converted_contigs = {}
        unconverted_contigs = {}
        for name in names:
            parts = name.split('\t')
            if len(parts) >= 3:
                label = parts[1]
                if config.all_wicker_class.__contains__(label):
                    new_label = label
                elif config.Repbase_wicker_labels.__contains__(label):
                    new_label = config.Repbase_wicker_labels[label]
                else:
                    new_label = None
                if new_label is not None:
                    new_name = '\t'.join([parts[0], new_label] + parts[2:])
                    converted_contigs[new_name] = contigs[name]
                else:
                    unconverted_contigs[name] = contigs[name]
            else:
                unconverted_contigs[name] = contigs[name]
        store_fasta(converted_contigs, data)
        if len(unconverted_contigs) > 0:
            unconverted_data = config.work_dir + '/unconverted_TE.fa'
            store_fasta(unconverted_contigs, unconverted_data)
            print('Warning: The input TE library contains unknown superfamily labels, total size = ' + str(len(unconverted_contigs)) + ', which saved at ' + os.path.realpath(unconverted_data))

        # 取出一条记录，看它是否包含相应的信息，如果没有，则需要对应处理
        if len(names) > 0:
            cur_name = names[0]
            if use_TSD != 0 and ('TSD:' not in cur_name or 'TSD_len:' not in cur_name):
                # 使用TSD特征，但是TSD信息在数据集中不存在，则需要重新生成TSD特征信息
                # 请确保修改 ‘data/ncbi_ref.info’ 文件中Genome Path为正确的位置信息
                is_expanded = 0 # 只有在平衡Repbase数据集时，设置is_expanded=1，否则is_expanded=0
                keep_raw = 0 # 保留原始序列，默认为1; 如果希望只保留具有TSD的序列，则设置为0，训练TSD模型的时候使用。
                data = generate_TSD_info(data, genome_info_path, work_dir, is_expanded, keep_raw, threads)
            if use_domain != 0 and not os.path.exists(domain_train_path):
                # 使用domain特征，但是domain信息文件却不存在，则需要重新比对生成domain信息
                generate_domain_info(data, project_dir+'/data/RepeatPeps.lib', work_dir, threads)
            if use_minority != 0:
                generate_minority_info(data, minority_train_path, minority_out, threads, is_train)
            if use_terminal != 0 and ('LTR:' not in cur_name or 'TIR:' not in cur_name):
                data = generate_terminal_info(data, work_dir, tool_dir, threads)
        return data


def main():
    # 实例化 DataProcessor 类
    data_processor = DataProcessor()
    data_path = '/public/home/hpc194701009/TE_Classification/NeuralTE/data/repbase_total.tsd.ref.update'
    # 加载数据
    X, y, seq_names = data_processor.load_data(config.internal_kmer_sizes, config.terminal_kmer_sizes, data_path)
    print(X.shape, y.shape)

if __name__ == '__main__':
    main()