import argparse
import json
import os
import sys
import time

current_folder = os.path.dirname(os.path.abspath(__file__))
# Add the path to the 'configs' folder to the Python path
configs_folder = os.path.join(current_folder, "..")
sys.path.append(configs_folder)

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from keras.utils import np_utils
from configs import config, gpu_config
from CNN_Model import CNN_Model
from DataProcessor import DataProcessor
from utils.evaluate_util import get_metrics
from utils.show_util import showToolName, showTrainParams
from utils.data_util import get_feature_len, get_gpu_config


class CrossValidator:
    def __init__(self, num_folds=5):
        self.num_folds = num_folds
        self.sss = StratifiedShuffleSplit(n_splits=num_folds, test_size=0.2, random_state=42)

    def evaluate(self, X, y):
        accuracy_array = []
        precision_array = []
        recall_array = []
        f1_array = []
        # Loop through each K-fold
        for fold, (train_index, test_index) in enumerate(self.sss.split(X, y)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            y_train_one_hot = np_utils.to_categorical(y_train, int(config.class_num))
            y_train_one_hot = np.array(y_train_one_hot)
            y_test_one_hot = np_utils.to_categorical(y_test, int(config.class_num))
            y_test_one_hot = np.array(y_test_one_hot)

            cnn_model = CNN_Model(config.X_feature_len, config.class_num)
            model = cnn_model.build_model(config.cnn_num_convs, config.cnn_filters_array)

            # Train the model
            model.fit(X_train, y_train_one_hot, batch_size=config.batch_size, epochs=config.epochs, verbose=1)
            # Save the model
            model_path = config.project_dir + '/models/' + f'model_fold_{fold}.h5'
            model.save(model_path)

            # Predict probabilities
            y_pred = model.predict(X_test)
            accuracy, precision, recall, f1 = get_metrics(y_pred, y_test, None, None)
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
        # Calculate mean and sample standard deviation
        accuracy_mean = round(np.mean(accuracies), 4)
        accuracy_stdv = round(np.std(accuracies, ddof=1), 4)  # Use ddof=1 to calculate sample standard deviation
        precision_mean = round(np.mean(precisions), 4)
        precision_stdv = round(np.std(precisions, ddof=1), 4)
        recall_mean = round(np.mean(recalls), 4)
        recall_stdv = round(np.std(recalls, ddof=1), 4)
        f1_mean = round(np.mean(f1s), 4)
        f1_stdv = round(np.std(f1s, ddof=1), 4)
        return [accuracy_mean, precision_mean, recall_mean, f1_mean], [accuracy_stdv, precision_stdv, recall_stdv, f1_stdv]


def main():
    showToolName()

    # 1.parse args
    describe_info = '########################## NeuralTE-CrossValidator, version ' + str(config.version_num) + ' ##########################'
    parser = argparse.ArgumentParser(description=describe_info)
    parser.add_argument('--data', metavar='data', help='Input fasta file containing TSDs information used to CrossValidator model')
    parser.add_argument('--outdir', metavar='output_dir', help='Output directory, store temporary files')
    parser.add_argument('--use_TSD', metavar='use_TSD', help='Whether to use TSD features, 1: true, 0: false. default = [ ' + str(config.use_TSD) + ' ]')
    parser.add_argument('--is_predict', metavar='is_predict', help='Enable prediction mode, 1: true, 0: false. default = [ ' + str(config.is_predict) + ' ]')

    parser.add_argument('--start_gpu_num', metavar='start_gpu_num', help='The starting index for using GPUs. default = [ ' + str(gpu_config.start_gpu_num) + ' ]')
    parser.add_argument('--use_gpu_num', metavar='use_gpu_num', help='Specifying the number of GPUs in use. default = [ ' + str(gpu_config.use_gpu_num) + ' ]')
    parser.add_argument('--only_preprocess', metavar='only_preprocess', help='Whether to only perform data preprocessing, 1: true, 0: false.')
    parser.add_argument('--keep_raw', metavar='keep_raw', help='Whether to retain the raw input sequence, 1: true, 0: false; only save species having TSDs. default = [ ' + str(config.keep_raw) + ' ]')
    parser.add_argument('--genome', metavar='genome', help='Genome path, use to search for TSDs')
    parser.add_argument('--use_kmers', metavar='use_kmers', help='Whether to use kmers features, 1: true, 0: false. default = [ ' + str(config.use_kmers) + ' ]')
    parser.add_argument('--use_terminal', metavar='use_terminal', help='Whether to use LTR, TIR terminal features, 1: true, 0: false. default = [ ' + str(config.use_terminal) + ' ]')
    parser.add_argument('--use_minority', metavar='use_minority', help='Whether to use minority features, 1: true, 0: false. default = [ ' + str(config.use_minority) + ' ]')
    parser.add_argument('--use_domain', metavar='use_domain', help='Whether to use domain features, 1: true, 0: false. default = [ ' + str(config.use_domain) + ' ]')
    parser.add_argument('--use_ends', metavar='use_ends', help='Whether to use 5-bp terminal ends features, 1: true, 0: false. default = [ ' + str(config.use_ends) + ' ]')
    parser.add_argument('--threads', metavar='thread_num', help='Input thread num, default = [ ' + str(config.threads) + ' ]')

    parser.add_argument('--internal_kmer_sizes', metavar='internal_kmer_sizes', help='The k-mer size used to convert internal sequences to k-mer frequency features, default = [ ' + str(config.internal_kmer_sizes) + ' MB ]')
    parser.add_argument('--terminal_kmer_sizes', metavar='terminal_kmer_sizes', help='The k-mer size used to convert terminal sequences to k-mer frequency features, default = [ ' + str(config.terminal_kmer_sizes) + ' ]')
    parser.add_argument('--cnn_num_convs', metavar='cnn_num_convs', help='The number of CNN convolutional layers. default = [ ' + str(config.cnn_num_convs) + ' ]')
    parser.add_argument('--cnn_filters_array', metavar='cnn_filters_array', help='The number of filters in each CNN convolutional layer. default = [ ' + str(config.cnn_filters_array) + ' ]')
    parser.add_argument('--cnn_kernel_sizes_array', metavar='cnn_kernel_sizes_array', help='The kernel size in each of CNN convolutional layer. default = [ ' + str(config.cnn_kernel_sizes_array) + ' ]')
    parser.add_argument('--cnn_dropout', metavar='cnn_dropout', help='The threshold of CNN Dropout. default = [ ' + str(config.cnn_dropout) + ' ]')
    parser.add_argument('--batch_size', metavar='batch_size', help='The batch size in training model. default = [ ' + str(config.batch_size) + ' ]')
    parser.add_argument('--epochs', metavar='epochs', help='The number of epochs in training model. default = [ ' + str(config.epochs) + ' ]')
    parser.add_argument('--use_checkpoint', metavar='use_checkpoint', help='Whether to use breakpoint training. 1: true, 0: false. The model will continue training from the last failed parameters to avoid training from head. default = [ ' + str(config.use_checkpoint) + ' ]')

    args = parser.parse_args()

    data_path = args.data
    outdir = args.outdir
    use_kmers = args.use_kmers
    use_terminal = args.use_terminal
    use_TSD = args.use_TSD
    use_domain = args.use_domain
    use_ends = args.use_ends
    is_predict = args.is_predict
    threads = args.threads
    start_gpu_num = args.start_gpu_num
    use_gpu_num = args.use_gpu_num
    only_preprocess = args.only_preprocess
    keep_raw = args.keep_raw
    genome = args.genome
    use_minority = args.use_minority

    internal_kmer_sizes = args.internal_kmer_sizes
    terminal_kmer_sizes = args.terminal_kmer_sizes
    cnn_num_convs = args.cnn_num_convs
    cnn_filters_array = args.cnn_filters_array
    cnn_kernel_sizes_array = args.cnn_kernel_sizes_array
    cnn_dropout = args.cnn_dropout
    batch_size = args.batch_size
    epochs = args.epochs
    use_checkpoint = args.use_checkpoint

    if outdir is not None:
        config.work_dir = outdir
    if use_terminal is not None:
        config.use_terminal = int(use_terminal)
    if use_kmers is not None:
        config.use_kmers = int(use_kmers)
    if use_TSD is not None:
        config.use_TSD = int(use_TSD)
    if use_domain is not None:
        config.use_domain = int(use_domain)
    if use_ends is not None:
        config.use_ends = int(use_ends)
    if is_predict is not None:
        config.is_predict = int(is_predict)
    if threads is not None:
        config.threads = int(threads)
    if start_gpu_num is not None:
        gpu_config.start_gpu_num = int(start_gpu_num)
    if use_gpu_num is not None:
        gpu_config.use_gpu_num = int(use_gpu_num)
    if only_preprocess is not None:
        config.only_preprocess = int(only_preprocess)
    if keep_raw is not None:
        config.keep_raw = int(keep_raw)
    if use_minority is not None:
        config.use_minority = int(use_minority)

    if internal_kmer_sizes is not None:
        config.internal_kmer_sizes = json.loads(internal_kmer_sizes)
    if terminal_kmer_sizes is not None:
        config.terminal_kmer_sizes = json.loads(terminal_kmer_sizes)
    if cnn_num_convs is not None:
        config.cnn_num_convs = int(cnn_num_convs)
    if cnn_filters_array is not None:
        config.cnn_filters_array = json.loads(cnn_filters_array)
    if cnn_kernel_sizes_array is not None:
        config.cnn_kernel_sizes_array = json.loads(cnn_kernel_sizes_array)
    if cnn_dropout is not None:
        config.cnn_dropout = float(cnn_dropout)
    if batch_size is not None:
        config.batch_size = int(batch_size)
    if epochs is not None:
        config.epochs = int(epochs)
    if use_checkpoint is not None:
        config.use_checkpoint = int(use_checkpoint)


    if genome is not None:
        os.makedirs(config.work_dir, exist_ok=True)
        genome_info_path = config.work_dir + '/genome.info'
        if str(genome).__contains__('genome.info'):
            os.system('cp ' + genome + ' ' + genome_info_path)
        else:
            with open(genome_info_path, 'w') as f_save:
                f_save.write('#Scientific Name\tGenome Path\tIs Plant\n')
                f_save.write('Unknown\t'+genome+'\t'+str(config.is_plant)+'\n')

    params = {}
    params['data_path'] = data_path
    params['outdir'] = outdir
    params['genome'] = genome
    showTrainParams(params)

    X_feature_len = get_feature_len()
    config.X_feature_len = X_feature_len

    # reload GPU config
    get_gpu_config(gpu_config.start_gpu_num, gpu_config.use_gpu_num)

    # Instantiate the DataProcessor class
    data_processor = DataProcessor()
    # Load the data
    # Make sure the header format for the following data is in Repbase format, i.e., 'TE_name  Superfamily Species', separated by '\t'
    cv_train_data_path = data_path  # Path to cross-validation training data
    X, y, seq_names, data_path = data_processor.load_data(config.internal_kmer_sizes, config.terminal_kmer_sizes, cv_train_data_path)
    print(X.shape, y.shape)

    if not config.only_preprocess:
        starttime2 = time.time()

        # Instantiate the CrossValidator class
        validator = CrossValidator(num_folds=5)
        # Perform cross-validation
        means, stdvs = validator.evaluate(X, y)
        print('accuracy, precision, recall, f1:')
        print("Mean array:", means)
        print("stdv array:", stdvs)

        endtime2 = time.time()
        dtime2 = endtime2 - starttime2
        print("Running time of model Trainer: %.8s s" % (dtime2))


if __name__ == '__main__':
    main()