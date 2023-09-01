import os
import sys
current_folder = os.path.dirname(os.path.abspath(__file__))
# 添加 configs 文件夹的路径到 Python 路径
configs_folder = os.path.join(current_folder, "..")  # 需要根据实际目录结构调整
sys.path.append(configs_folder)

import numpy as np
from sklearn.model_selection import KFold
from keras.utils import np_utils
from configs import config
from CNN_Model import CNN_Model
from DataProcessor import DataProcessor
from utils.evaluate_util import get_metrics

class CrossValidator:
    def __init__(self, num_folds=5):
        self.num_folds = num_folds
        self.kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    def evaluate(self, X, y):
        accuracy_array = []
        precision_array = []
        recall_array = []
        f1_array = []
        # 循环每个K折
        for fold, (train_index, test_index) in enumerate(self.kf.split(X)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            y_train_one_hot = np_utils.to_categorical(y_train, int(config.class_num))
            y_train_one_hot = np.array(y_train_one_hot)
            y_test_one_hot = np_utils.to_categorical(y_test, int(config.class_num))
            y_test_one_hot = np.array(y_test_one_hot)

            cnn_model = CNN_Model(config.X_feature_len, config.class_num)
            model = cnn_model.build_model(config.cnn_num_convs, config.cnn_filters_array)

            # 训练模型
            model.fit(X_train, y_train_one_hot, batch_size=config.batch_size, epochs=config.epochs, verbose=1)
            # 保存模型
            model_path = config.project_dir + '/models/' + f'model_fold_{fold}.h5'
            model.save(model_path)

            # 预测概率
            y_pred = model.predict(X_test)
            accuracy, precision, recall, f1 = get_metrics(y_pred, y_test, None)
            print("Fold:", fold)
            print("Accuracy:", accuracy)
            print("Precision:", precision)
            print("Recall:", recall)
            print("F1:", f1)
            accuracy_array.append(accuracy)
            precision_array.append(precision)
            recall_array.append(recall)
            f1_array.append(f1)
            accuracies = np.array(accuracy_array)
            precisions = np.array(precision_array)
            recalls = np.array(recall_array)
            f1s = np.array(f1_array)
            # 计算平均值和样本标准差
            accuracy_mean = round(np.mean(accuracies), 4)
            accuracy_stdv = round(np.std(accuracies, ddof=1), 4)  # 使用 ddof=1 表示计算样本标准差
            precision_mean = round(np.mean(precisions), 4)
            precision_stdv = round(np.std(precisions, ddof=1), 4)  # 使用 ddof=1 表示计算样本标准差
            recall_mean = round(np.mean(recalls), 4)
            recall_stdv = round(np.std(recalls, ddof=1), 4)  # 使用 ddof=1 表示计算样本标准差
            f1_mean = round(np.mean(f1s), 4)
            f1_stdv = round(np.std(f1s, ddof=1), 4)  # 使用 ddof=1 表示计算样本标准差
        return [accuracy_mean, precision_mean, recall_mean, f1_mean], [accuracy_stdv, precision_stdv, recall_stdv, f1_stdv]


def main():
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


if __name__ == '__main__':
    main()