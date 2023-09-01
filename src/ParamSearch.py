import os
import sys

# 获取当前文件的路径
from itertools import product

from src.CrossValidator import CrossValidator

current_folder = os.path.dirname(os.path.abspath(__file__))
# 添加 configs 文件夹的路径到 Python 路径
configs_folder = os.path.join(current_folder, "..")  # 需要根据实际目录结构调整
sys.path.append(configs_folder)

import numpy as np
from keras.models import load_model
from keras.utils import np_utils
from configs import config
from CNN_Model import CNN_Model
from DataProcessor import DataProcessor
from src.Classifier import Classifier
from src.Trainer import Trainer
from utils.evaluate_util import get_metrics

def test_use_all_repbase_no_tsd():
    # 测试分别使用不同的特征变化，五折交叉验证

    feature = (1, 0, 1, 1)
    #加载数据前，先设定使用哪些特征，并定义好特征维度

    print('feature:', feature)
    config.use_terminal = feature[0]  # 使用LTR、TIR终端特征
    config.use_TSD = feature[1]  # 使用TSD特征
    config.use_domain = feature[2]  # 使用TE domain特征
    config.use_ends = feature[3]  # 使用TE 5'端和3'端各5bp特征

    # 获取CNN输入维度
    X_feature_len = 0
    # TE seq/internal_seq 维度
    for kmer_size in config.internal_kmer_sizes:
        X_feature_len += pow(4, kmer_size)
    if config.use_terminal != 0:
        for i in range(2):
            for kmer_size in config.terminal_kmer_sizes:
                X_feature_len += pow(4, kmer_size)
    if config.use_TSD != 0:
        X_feature_len += 11 * 4 + 1
    if config.use_domain != 0:
        X_feature_len += 29
    if config.use_ends != 0:
        X_feature_len += 10 * 4
    config.X_feature_len = X_feature_len


    # 实例化 DataProcessor 类
    data_processor = DataProcessor()
    # 加载数据
    # 请确保下面数据的header格式为Repbase格式，即'TE_name  Superfamily Species'，以'\t'分割
    cv_train_data_path = config.work_dir + "/all_repbase.ref_preprocess.ref.update"  # 交叉验证训练数据路径
    X, y, seq_names = data_processor.load_data(config.internal_kmer_sizes, config.terminal_kmer_sizes, cv_train_data_path)
    print(X.shape, y.shape)

    # 实例化 CrossValidator 类
    validator = CrossValidator(num_folds=5)

    # 进行交叉验证
    means, stdvs = validator.evaluate(X, y)
    print('accuracy, precision, recall, f1:')
    print("Mean array:", means)
    print("stdv array:", stdvs)


def test_use_features():
    # 测试分别使用不同的特征变化，五折交叉验证
    features = [(0, 0, 0, 0), (0, 1, 0, 0), (1, 1, 0, 0), (1, 1, 0, 1), (1, 1, 1, 1)]
    #加载数据前，先设定使用哪些特征，并定义好特征维度
    for feature in features:
        print('feature:', feature)
        config.use_terminal = feature[0]  # 使用LTR、TIR终端特征
        config.use_TSD = feature[1]  # 使用TSD特征
        config.use_domain = feature[2]  # 使用TE domain特征
        config.use_ends = feature[3]  # 使用TE 5'端和3'端各5bp特征

        # 获取CNN输入维度
        X_feature_len = 0
        # TE seq/internal_seq 维度
        for kmer_size in config.internal_kmer_sizes:
            X_feature_len += pow(4, kmer_size)
        if config.use_terminal != 0:
            for i in range(2):
                for kmer_size in config.terminal_kmer_sizes:
                    X_feature_len += pow(4, kmer_size)
        if config.use_TSD != 0:
            X_feature_len += config.max_tsd_length * 4 + 1
        if config.use_domain != 0:
            X_feature_len += 29
        if config.use_ends != 0:
            X_feature_len += 10 * 4
        config.X_feature_len = X_feature_len


        # 实例化 DataProcessor 类
        data_processor = DataProcessor()
        # 加载数据
        # 请确保下面数据的header格式为Repbase格式，即'TE_name  Superfamily Species'，以'\t'分割
        cv_train_data_path = config.work_dir + "/repbase_total.ref"  # 交叉验证训练数据路径
        X, y, seq_names = data_processor.load_data(config.internal_kmer_sizes, config.terminal_kmer_sizes, cv_train_data_path)
        print(X.shape, y.shape)

        # 实例化 CrossValidator 类
        validator = CrossValidator(num_folds=5)

        # 进行交叉验证
        means, stdvs = validator.evaluate(X, y)
        print('accuracy, precision, recall, f1:')
        print("Mean array:", means)
        print("stdv array:", stdvs)


def test_cnn_filters():
    # 固定kmer_size = [1, 2, 4], CNN卷积层数量 3, 改变滤波器数量，测试滤波器数量的变化对结果的影响
    cnn_filters_arrays = [[32, 32, 32], ] # CNN每层卷积层的filter数量
    numbers = [32, 64, 128]
    cnn_filters_arrays = list(product(numbers, repeat=3))
    filter_arrays = [(32, 32, 32), (32, 32, 64), (32, 32, 128), (32, 64, 32)]
    filtered_cnn_filters_arrays = [p for p in cnn_filters_arrays if p not in filter_arrays]
    for cnn_filters_array in filtered_cnn_filters_arrays:
        print('cnn_filters_array:', cnn_filters_array)
        data_path = config.work_dir + "/repbase_train.ref"
        # 实例化 DataProcessor 类
        data_processor = DataProcessor()
        # 加载数据
        X, y, seq_names = data_processor.load_data(config.internal_kmer_sizes, config.terminal_kmer_sizes, data_path)
        print(X.shape, y.shape)

        # 实例化 Trainer 类
        trainer = Trainer()

        # 进行训练
        model_path = trainer.train(X, y, config.cnn_num_convs, cnn_filters_array)
        # print('Trained model is stored in:', model_path)

        # 实例化 DataProcessor 类
        data_path = config.work_dir + "/repbase_test.ref"
        # 加载数据
        X_test, y_test, seq_names = data_processor.load_data(config.internal_kmer_sizes, config.terminal_kmer_sizes, data_path)
        print(X_test.shape, y_test.shape)

        # 实例化 Classifier 类
        model_path = config.project_dir + '/models/model.h5'
        classifier = Classifier(model_path=model_path)

        # 进行预测
        accuracy, precision, recall, f1 = classifier.predict(X_test, y_test)
        print('accuracy, precision, recall, f1:', accuracy, precision, recall, f1)

def test_cnn_layers():
    # 固定kmer_size = [1, 2, 4], 滤波器数量 = 32， 测试卷积层数的变化对结果的影响
    # cnn_num_convs = 4 # CNN卷积层数量
    # cnn_filters_array = [32, 32, 32, 32] # CNN每层卷积层的filter数量
    cnn_num_convs_array = [5, 6, 7]
    for cnn_num_convs in cnn_num_convs_array:
        print('cnn_num_convs:', cnn_num_convs)

        # 实例化 DataProcessor 类
        data_processor = DataProcessor()
        # 加载数据
        X, y, seq_names = data_processor.load_data(config.internal_kmer_sizes, config.terminal_kmer_sizes)
        print(X.shape, y.shape)

        # 实例化 Trainer 类
        trainer = Trainer()

        # 进行训练
        model_path = trainer.train(X, y, cnn_num_convs, config.cnn_filters_array)
        # print('Trained model is stored in:', model_path)

        # 实例化 DataProcessor 类
        data_processor = DataProcessor(run_type='predict')
        # 加载数据
        X, y, seq_names = data_processor.load_data(config.internal_kmer_sizes, config.terminal_kmer_sizes)
        print(X.shape, y.shape)

        # 实例化 Classifier 类
        model_path = config.project_dir + '/models/model.h5'
        classifier = Classifier(model_path=model_path)

        # 进行预测
        accuracy, precision, recall, f1 = classifier.predict(X, y)
        print('accuracy, precision, recall, f1:', accuracy, precision, recall, f1)

def test_kmer_size():
    # 即固定卷积层数和滤波器数量，测试kmer组合的变化对结果的影响，此时只使用kmer特征
    # cnn_num_convs = 4 # CNN卷积层数量
    # cnn_filters_array = [32, 32, 32, 32] # CNN每层卷积层的filter数量
    kmer_sizes_array = [
                        [1, 2], [1, 3], [1, 4], [1, 5],
                        [2, 3], [2, 4], [2, 5], [3, 4], [3, 5],
                        [1, 2, 3], [1, 2, 4], [1, 2, 5], [1, 3, 4], [1, 3, 5], [2, 3, 4], [2, 3, 5], [3, 4, 5],
                        [1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 4, 5], [1, 3, 4, 5], [2, 3, 4, 5]]
    for kmer_sizes in kmer_sizes_array:
        print('kmer sizes:', kmer_sizes)
        # 获取CNN输入维度
        X_feature_len = 0
        # TE seq/internal_seq 维度
        for kmer_size in kmer_sizes:
            X_feature_len += pow(4, kmer_size)
        if config.use_terminal != 0:
            for i in range(2):
                for kmer_size in kmer_sizes:
                    X_feature_len += pow(4, kmer_size)
        if config.use_TSD != 0:
            X_feature_len += 11 * 4 + 1
        if config.use_domain != 0:
            X_feature_len += 29
        if config.use_ends != 0:
            X_feature_len += 10 * 4
        config.X_feature_len = X_feature_len

        # 实例化 DataProcessor 类
        data_processor = DataProcessor()
        # 加载数据
        X, y, seq_names = data_processor.load_data(kmer_sizes)
        print(X.shape, y.shape)

        # 实例化 Trainer 类
        trainer = Trainer()

        # 进行训练
        model_path = trainer.train(X, y, config.cnn_num_convs, config.cnn_filters_array)
        # print('Trained model is stored in:', model_path)

        # 实例化 DataProcessor 类
        data_processor = DataProcessor()
        # 加载数据
        X, y, seq_names = data_processor.load_data(kmer_sizes)
        print(X.shape, y.shape)

        # 实例化 Classifier 类
        model_path = config.project_dir + '/models/model.h5'
        classifier = Classifier(model_path=model_path)

        # 进行预测
        accuracy, precision, recall, f1 = classifier.predict(X, y)
        print('accuracy, precision, recall, f1:', accuracy, precision, recall, f1)

def main():
    #test_kmer_size()
    #test_cnn_layers()
    #test_cnn_filters()
    test_use_features()
    #test_use_all_repbase_no_tsd()

if __name__ == '__main__':
    main()