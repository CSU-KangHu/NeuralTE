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
    load_repbase_with_TSD, generate_feature_mats


class DataProcessor:
    def __init__(self):
        self.project_dir = config.project_dir
        self.tool_dir = self.project_dir + '/tools'
        self.work_dir = config.work_dir
        self.threads = config.threads
        self.all_wicker_class = config.all_wicker_class
        self.use_terminal = config.use_terminal
        self.use_TSD = config.use_TSD
        self.use_domain = config.use_domain
        self.use_ends = config.use_ends

        self.ex = ProcessPoolExecutor(self.threads)


    def load_data(self, internal_kmer_sizes, terminal_kmer_sizes, data_path):
        domain_train_path = data_path + '.domain'
        data_path = self.preprocess_data(data_path, domain_train_path, self.work_dir,
                                                    self.project_dir, self.tool_dir, self.threads,
                                                    self.use_TSD, self.use_domain, self.use_terminal)
        X, Y, seq_names = load_repbase_with_TSD(data_path, domain_train_path, self.all_wicker_class, self.project_dir+'/data/TEClasses.tsv')
        X, Y = generate_feature_mats(X, Y, seq_names, self.all_wicker_class, internal_kmer_sizes, terminal_kmer_sizes, self.ex)
        # 将数据reshape成模型接收的格式
        X = X.reshape(X.shape[0], config.X_feature_len, 1)
        X = X.astype('float64')
        return X, Y, seq_names

    def preprocess_data(self, data, domain_train_path, work_dir, project_dir,
                        tool_dir, threads, use_TSD, use_domain, use_terminal):
        names, contigs = read_fasta_v1(data)
        # 取出一条记录，看它是否包含相应的信息，如果没有，则需要对应处理
        if len(names) > 0:
            cur_name = names[0]
            if use_TSD != 0 and ('TSD:' not in cur_name or 'TSD_len:' not in cur_name):
                # 使用TSD特征，但是TSD信息在数据集中不存在，则需要重新生成TSD特征信息
                # 请确保修改 ‘data/ncbi_ref.info’ 文件中Genome Path为正确的位置信息
                is_expanded = 0 # 只有在平衡Repbase数据集时，设置is_expanded=1，否则is_expanded=0
                data = generate_TSD_info(data, project_dir+'/data/ncbi_ref.info', work_dir, is_expanded, threads)
            if use_domain != 0 and not os.path.exists(domain_train_path):
                # 使用domain特征，但是domain信息文件却不存在，则需要重新比对生成domain信息
                generate_domain_info(data, project_dir+'/data/RepeatPeps.lib', work_dir, threads)
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