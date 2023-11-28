import argparse
import json
import os
import sys

current_folder = os.path.dirname(os.path.abspath(__file__))
# 添加 configs 文件夹的路径到 Python 路径
configs_folder = os.path.join(current_folder, "..")  # 需要根据实际目录结构调整
sys.path.append(configs_folder)

from utils.show_util import showToolName, showTestParams
import numpy as np
from keras.models import load_model
from keras.utils import np_utils
from configs import config
from DataProcessor import DataProcessor
from utils.evaluate_util import get_metrics
from utils.data_util import get_feature_len



class Classifier:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def predict(self, X, y, seq_names):
        # 预测概率
        y_pred = self.model.predict(X)
        accuracy, precision, recall, f1 = get_metrics(y_pred, y, seq_names)

        return accuracy, precision, recall, f1


def main():
    showToolName()

    # 1.parse args
    describe_info = '########################## NeuralTE, version ' + str(config.version_num) + ' ##########################'
    parser = argparse.ArgumentParser(description=describe_info)
    parser.add_argument('--data', metavar='data', help='Input fasta file used to predict, header format: seq_name\tlabel\tspecies_name, refer to "data/test.example.fa" for example.')
    parser.add_argument('--outdir', metavar='output_dir', help='Output directory, store temporary files')
    parser.add_argument('--model_path', metavar='model_path', help='Input the path of trained model, absolute path.')
    parser.add_argument('--use_terminal', metavar='use_terminal', help='Whether to use LTR, TIR terminal features, 1: true, 0: false. default = [ ' + str(config.use_terminal) + ' ]')
    parser.add_argument('--use_TSD', metavar='use_TSD', help='Whether to use TSD features, 1: true, 0: false. default = [ ' + str(config.use_TSD) + ' ]')
    parser.add_argument('--use_domain', metavar='use_domain', help='Whether to use domain features, 1: true, 0: false. default = [ ' + str(config.use_domain) + ' ]')
    parser.add_argument('--use_ends', metavar='use_ends', help='Whether to use 5-bp terminal ends features, 1: true, 0: false. default = [ ' + str(config.use_ends) + ' ]')
    parser.add_argument('--threads', metavar='thread_num', help='Input thread num, default = [ ' + str(config.threads) + ' ]')
    parser.add_argument('--internal_kmer_sizes', metavar='internal_kmer_sizes', help='The k-mer size used to convert internal sequences to k-mer frequency features, default = [ ' + str(config.internal_kmer_sizes) + ' MB ]')
    parser.add_argument('--terminal_kmer_sizes', metavar='terminal_kmer_sizes', help='The k-mer size used to convert terminal sequences to k-mer frequency features, default = [ ' + str(config.terminal_kmer_sizes) + ' ]')

    args = parser.parse_args()

    data_path = args.data
    outdir = args.outdir
    model_path = args.model_path
    use_terminal = args.use_terminal
    use_TSD = args.use_TSD
    use_domain = args.use_domain
    use_ends = args.use_ends
    threads = args.threads
    internal_kmer_sizes = args.internal_kmer_sizes
    terminal_kmer_sizes = args.terminal_kmer_sizes

    if outdir is not None:
        config.work_dir = outdir
    if use_terminal is not None:
        config.use_terminal = int(use_terminal)
    if use_TSD is not None:
        config.use_TSD = int(use_TSD)
    if use_domain is not None:
        config.use_domain = int(use_domain)
    if use_ends is not None:
        config.use_ends = int(use_ends)
    if threads is not None:
        config.threads = int(threads)
    if internal_kmer_sizes is not None:
        config.internal_kmer_sizes = json.loads(internal_kmer_sizes)
    if terminal_kmer_sizes is not None:
        config.terminal_kmer_sizes = json.loads(terminal_kmer_sizes)

    showTestParams(data_path, model_path)

    X_feature_len = get_feature_len()
    config.X_feature_len = X_feature_len

    # 实例化 DataProcessor 类
    data_processor = DataProcessor()
    # 加载数据
    X, y, seq_names = data_processor.load_data(config.internal_kmer_sizes, config.terminal_kmer_sizes, data_path)
    print(X.shape, y.shape)

    # 实例化 Classifier 类
    classifier = Classifier(model_path=model_path)

    # 进行预测
    accuracy, precision, recall, f1 = classifier.predict(X, y, seq_names)
    print('accuracy, precision, recall, f1:', accuracy, precision, recall, f1)


if __name__ == '__main__':
    main()