# 融合了Seq, LTR, TIR, TSD, Domain等多种特征进行训练，同时包含k-mer的位置信息：
# 进行K折交叉验证，并计算AUC, ROC：


import os

import tensorflow as tf

from keras.utils import np_utils
from keras.models import Sequential, load_model, Model
from keras.layers import Layer, Input, Dense, Dropout, Activation, Flatten, LSTM, Conv1D, Conv2D,\
    MaxPooling1D, MaxPooling2D, Bidirectional, Embedding, GlobalAveragePooling1D, concatenate, \
    MultiHeadAttention, LayerNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np


from plot_confusion_matrix import plot_confusion_matrix
from Util import load_repbase_with_TSD, generate_mats, conv_labels, load_repbase, read_fasta_v1, generate_feature_mats, \
    generate_hybrid_feature_mats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import StratifiedKFold, KFold

from attn_augconv import augmented_conv2d, augmented_conv1d

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Set the GPUs you want to use
gpus = tf.config.experimental.list_physical_devices('GPU')
# For GPU memory growth
for device in gpus:
    tf.config.experimental.set_memory_growth(device, True)


all_devices = ["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3"]
start_gpu_num = 0
use_gpu_num = 1
use_devices = all_devices[start_gpu_num: start_gpu_num + use_gpu_num]

tf.config.experimental.set_visible_devices(gpus[start_gpu_num: start_gpu_num + use_gpu_num], 'GPU')
# Create a MirroredStrategy to use multiple GPUs
strategy = tf.distribute.MirroredStrategy(devices=use_devices)
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# Prepare a directory to store all the checkpoints.
checkpoint_dir = "./ckpt"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

def make_or_restore_model():
    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.
    checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring from", latest_checkpoint)
        return load_model(latest_checkpoint)
    print("Creating a new model")
    return get_compiled_model()

def run_training(X_num, X_seq, Y, batch_size=32, epochs=1, use_checkpoint=1):
    # 定义K折交叉验证
    n_splits = 5
    #skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # # 初始化AUC列表
    # 循环每个K折
    for fold, (train_index, test_index) in enumerate(kf.split(X_num)):
        X_num_train, X_num_test = X_num[train_index], X_num[test_index]
        X_seq_train, X_seq_test = X_seq[train_index], X_seq[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        y_train_one_hot = np_utils.to_categorical(y_train, int(class_num))
        y_train_one_hot = np.array(y_train_one_hot)
        y_test_one_hot = np_utils.to_categorical(y_test, int(class_num))
        y_test_one_hot = np.array(y_test_one_hot)

        # 构建模型
        if use_checkpoint == 0:
            os.system('cd ' + checkpoint_dir + ' && rm -rf ckpt*')
        # Create a MirroredStrategy.
        strategy = tf.distribute.MirroredStrategy()
        # Open a strategy scope and create/restore the model
        with strategy.scope():
            model = make_or_restore_model()

        # 训练模型
        model.fit([X_num_train, X_seq_train], y_train_one_hot, batch_size=batch_size, epochs=epochs, verbose=1)
        # 预测概率
        y_pred = model.predict([X_num_test, X_seq_test])
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
                new_class = 28    #unknown class label
            else:
                new_class = predicted_classes[i]
            predicted_classes_list.append(new_class)

        # 计算准确率
        accuracy = accuracy_score(y_test, predicted_classes_list)
        print("Accuracy:", accuracy)
        # 计算精确率
        precision = precision_score(y_test, predicted_classes_list, average='macro')
        print("Precision:", precision)
        # 计算召回率
        recall = recall_score(y_test, predicted_classes_list, average='macro')
        print("Recall:", recall)
        # 计算F1值
        f1 = f1_score(y_test, predicted_classes_list, average='macro')
        print("F1:", f1)

        # # 绘制ROC曲线（可选）
        # import matplotlib.pyplot as plt
        # # 计算多类别的ROC曲线和AUC
        # fpr = dict()
        # tpr = dict()
        # roc_auc = dict()
        #
        # for class_idx in range(y_pred.shape[1]):
        #     fpr[class_idx], tpr[class_idx], _ = roc_curve(y_test == class_idx, y_pred[:, class_idx])
        #     roc_auc[class_idx] = auc(fpr[class_idx], tpr[class_idx])
        #
        # # 绘制多类别的ROC曲线
        # plt.figure(figsize=(10, 10))
        # for class_idx in range(y_pred.shape[1]):
        #     plt.plot(fpr[class_idx], tpr[class_idx], label=f'Class {class_idx} (AUC = {roc_auc[class_idx]:.2f})')
        #
        # plt.plot([0, 1], [0, 1], 'k--')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Multi-Class ROC Curve')
        # plt.legend(loc="lower right")
        # plt.show()

    return model

# 构建 CNN 模型
def cnn_model(input_shape):
    # 输入层
    input_layer = Input(shape=input_shape)
    # 添加卷积层
    conv1 = Conv1D(32, 3, activation='relu')(input_layer)
    conv2 = Conv1D(32, 3, activation='relu')(conv1)
    conv3 = Conv1D(32, 3, activation='relu')(conv2)
    conv4 = Conv1D(32, 3, activation='relu')(conv3)
    dropout1 = Dropout(0.5)(conv4)
    # 添加展平层和全连接层
    flatten = Flatten()(dropout1)
    # 构建模型
    model = Model(inputs=input_layer, outputs=flatten)
    return model


# # 构建 Transformer 模型
# def transformer_model(input_shape, vocab_size):
#     input_layer = Input(shape=input_shape)
#     embedding_layer = Embedding(input_dim=vocab_size, output_dim=64)(input_layer)
#
#     # 自注意层
#     attention_layer = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=16)(embedding_layer, embedding_layer)
#     pooling_layer = GlobalAveragePooling1D()(attention_layer)
#     return Model(inputs=input_layer, outputs=pooling_layer)

# 构建 Transformer 模型
def transformer_model(input_shape, vocab_size, num_heads, d_model, num_layers, max_seq_length):
    inputs = Input(shape=input_shape)
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=d_model)(inputs)

    # 生成位置编码
    position_encoding = positional_encoding(max_seq_length, d_model)
    encoded_inputs = embedding_layer + position_encoding

    # 创建多个 Transformer 层
    for _ in range(num_layers):
        attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(value=encoded_inputs,
                                                                                    query=encoded_inputs,
                                                                                    key=encoded_inputs)
        attention_output = LayerNormalization(epsilon=1e-6)(attention_output + encoded_inputs)

        feedforward_output = feed_forward(attention_output, d_model)
        encoded_inputs = LayerNormalization(epsilon=1e-6)(feedforward_output + attention_output)

    # 对最终输出进行全局平均池化
    pooling_layer = GlobalAveragePooling1D()(encoded_inputs)

    return Model(inputs=inputs, outputs=pooling_layer)

# 生成位置编码
def positional_encoding(max_seq_length, d_model):
    position = np.arange(max_seq_length)[:, np.newaxis]
    angle_rates = 1 / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :]) // 2) / d_model)
    position_encoding = position * angle_rates

    # 偶数索引使用正弦函数，奇数索引使用余弦函数
    position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
    position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

    return position_encoding


# 前馈神经网络
def feed_forward(x, d_model):
    d_ff = 4 * d_model  # 前馈层维度
    ff_layer = tf.keras.Sequential([
        Dense(d_ff, activation='relu'),
        Dense(d_model)
    ])
    return ff_layer(x)


# 构建并编译混合模型
def hybrid_model(cnn_model, transformer_model):
    cnn_input_shape = (X_feature_len, 1)
    max_TSD_seq_length = 11
    vocab_size = 5
    num_heads = 4
    d_model = 64
    num_layers = 2
    transformer_input_shape = (max_TSD_seq_length,)

    cnn = cnn_model(cnn_input_shape)
    transformer = transformer_model(transformer_input_shape, vocab_size, num_heads, d_model, num_layers, max_TSD_seq_length)

    combined_output = concatenate([cnn.output, transformer.output])
    dense_layer = Dense(128, activation='relu')(combined_output)
    #dense_layer = Dense(128, activation='relu')(cnn.output)
    output_layer = Dense(int(class_num), activation='softmax')(dense_layer)

    model = Model(inputs=[cnn.input, transformer.input], outputs=output_layer)
    #model = Model(inputs=cnn.input, outputs=output_layer)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def get_compiled_model():
    # # 1. CNN model
    # model = Sequential()
    # model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(1, X_feature_len, 1)))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.5))
    # model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(int(class_num), activation='softmax'))

    # # 2. CNN model
    # model = Sequential()
    # model.add(Conv2D(100, (1, 3), activation='relu', input_shape=(1, X_feature_len, 1)))
    # model.add(MaxPooling2D(pool_size=(1, 2)))
    # model.add(Conv2D(150, (1, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(1, 2)))
    # model.add(Conv2D(225, (1, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(1, 2)))
    # model.add(Dropout(0.5))
    #
    # model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.5))
    #
    # model.add(Dense(int(class_num), activation='softmax'))
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 32-32-32：F1: 0.91124188158888
    # 32-32-32-32： F1: 0.9120153577045595
    # 128-64-32-32： F1: 0.9047548731675614

    # # 2. CNN model
    # model = Sequential()
    # model.add(Conv1D(32, 3, activation='relu', input_shape=(X_feature_len, 1)))
    # #model.add(MaxPooling1D(pool_size=2))
    # model.add(Conv1D(32, 3, activation='relu'))
    # #model.add(MaxPooling1D(pool_size=2))
    # model.add(Conv1D(32, 3, activation='relu'))
    # #model.add(MaxPooling1D(pool_size=2))
    # model.add(Conv1D(32, 3, activation='relu'))
    # #model.add(MaxPooling1D(pool_size=2))
    # model.add(Dropout(0.5))
    #
    # model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.5))
    #
    # model.add(Dense(int(class_num), activation='softmax'))
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 构建混合模型
    model = hybrid_model(cnn_model, transformer_model)

    # from keras.layers import Conv1D, Flatten, Dense, Dropout, Input
    # from keras.models import Model
    # from keras.layers import Attention

    # 100-150-225: F1: 0.9061523877354777
    # 32: F1: 0.8803469039813364
    # 32-32: F1: 0.914593253763968
    # 32-32-32: F1: 0.9177968092796981
    # 32-64-128: F1: 0.9067107094594844
    # 32-64-32: F1: 0.914074828497396

    # # 输入层
    # input_layer = Input(shape=(X_feature_len, 1))
    # # 添加卷积层
    # conv1 = Conv1D(32, 3, activation='relu')(input_layer)
    # # # 添加自注意力层
    # # self_attention = Attention()([conv1, conv1])  # 注意力机制输入为[query, value]
    # # conv2 = Conv1D(150, 3, activation='relu')(self_attention)
    # conv2 = Conv1D(64, 3, activation='relu')(conv1)
    # conv3 = Conv1D(32, 3, activation='relu')(conv2)
    # dropout1 = Dropout(0.5)(conv3)
    # # 添加展平层和全连接层
    # flatten = Flatten()(dropout1)
    # dense1 = Dense(128, activation='relu')(flatten)
    # dropout2 = Dropout(0.5)(dense1)
    # # 输出层
    # output_layer = Dense(int(class_num), activation='softmax')(dropout2)
    # # 构建模型
    # model = Model(inputs=input_layer, outputs=output_layer)
    # # 编译模型
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # 打印模型摘要
    # model.summary()


    # # LSTM model
    # model = Sequential()
    # model.add(Bidirectional(LSTM(32, return_sequences=False, input_shape=(X_feature_len, 6))))
    # #model.add(Bidirectional(LSTM(units=64, return_sequences=True, input_shape=(X_feature_len, 6))))
    # #model.add(Bidirectional(LSTM(32, return_sequences=False)))
    # model.add(Dropout(0.2))
    # model.add(Dense(int(class_num), activation='softmax'))
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def get_metrics(y_test, y_pred, all_class_list):
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

    # plot confusion matrix
    plot_confusion_matrix(y_pred, y_test)




if __name__ == '__main__':
    # Step 0: 配置参数
    batch_size = 32
    epochs = 50
    use_checkpoint = 0

    # Step 1: 获取所有superfamily的标签
    all_wicker_class = {'Tc1-Mariner': 0, 'hAT': 1, 'Mutator': 2, 'Merlin': 3, 'Transib': 4, 'P': 5, 'PiggyBac': 6,
                        'PIF-Harbinger': 7, 'CACTA': 8, 'Crypton': 9, 'Helitron': 10, 'Maverick': 11, 'Copia': 12,
                        'Gypsy': 13, 'Bel-Pao': 14, 'Retrovirus': 15, 'DIRS': 16, 'Ngaro': 17, 'VIPER': 18,
                        'Penelope': 19, 'R2': 20, 'RTE': 21, 'Jockey': 22, 'L1': 23, 'I': 24, 'tRNA': 25, '7SL': 26, '5S': 27}
    class_num = len(all_wicker_class)
    print('Total Class num: ' + str(class_num))
    inverted_all_wicker_class = {value: key for key, value in all_wicker_class.items()}

    # Step 2: 将Repbase序列和TSD数据， 以及label 转换成特征
    threads = 40
    data_dir = '/home/hukang/NeuralTE/data'

    data_path = data_dir + '/repbase_total.64.ref.shuffle.update'
    domain_path = data_dir + '/repbase_total.64.ref.update.domain'

    # Step 3: 加载repbase数据。将序列划分成internal_seq, LTR, TIR，TSD四个部分，分别用k=5, 4, 3频次表示internal_seq, LTR, TIR;TSD用one-hot编码表示，用多1位表示TSD_length;将domain转为One-hot编码
    parts = ['internal_seq', 'LTR', 'TIR']
    kmer_sizes = [1, 2, 3, 4]
    X_feature_len = 1 + 29
    for part in parts:
        for kmer_size in kmer_sizes:
            X_feature_len += pow(4, kmer_size)

    X, Y, seq_names = load_repbase_with_TSD(data_path, domain_path, all_wicker_class)
    #X, Y = generate_feature_mats(X, Y, seq_names, all_wicker_class, kmer_sizes, threads)
    X_num, X_seq, Y = generate_hybrid_feature_mats(X, Y, seq_names, all_wicker_class, kmer_sizes, threads)

    # Step 4: 将数据reshape成模型接收的格式
    X_num = X_num.reshape(X_num.shape[0], X_feature_len, 1)
    X_num = X_num.astype('float64')
    X_seq = X_seq.astype('float64')
    print('X_num, X_seq and Y shape: ')
    print(X_num.shape, X_seq.shape, Y.shape)
    print('batch size: ' + str(batch_size))

    # Step 5: 训练模型, 使用K折交叉验证
    # Running the first time creates the model
    model = run_training(X_num, X_seq, Y, batch_size=batch_size, epochs=epochs, use_checkpoint=use_checkpoint)

    # Step 6: 保存模型
    model_path = 'model/' + 'test_model.h5'
    model.save(model_path)

    # # Step 8: 在测试集上进行测试，并输出所有评测指标
    # model_path = 'model/' + 'test_model.h5'
    # model = load_model(model_path)
    #
    # # Step 9: 评估模型
    # loss, accuracy = model.evaluate(X_test, Y_test_one_hot, batch_size=batch_size, verbose=1)
    # print("\nloss=" + str(loss) + ', accuracy=' + str(accuracy))
    #
    # prop_thr = 0
    #
    # ##Step 9.2: generate the predict class
    # Y_pred = model.predict(X_test)
    #
    # predicted_classes = np.argmax(np.round(Y_pred), axis=1)
    # predicted_classes_list = predicted_classes.tolist()
    # # print(predicted_classes_list)
    # # print(Y_pred)
    # # print(Y_pred.shape)
    #
    # ##Step 9.3 transfer the prop less than a threshold to be unknown for a class
    # max_value_predicted_classes = np.amax(Y_pred, axis=1)
    # order = -1
    # ls_thr_order_list = []
    # for i in range(len(max_value_predicted_classes)):
    #     order += 1
    #     if max_value_predicted_classes[i] < float(prop_thr):
    #         ls_thr_order_list.append(order)
    #
    # new_predicted_classes_list = []
    # order = -1
    # for i in range(len(predicted_classes)):
    #     order += 1
    #     if order in ls_thr_order_list:
    #         new_class = 28    #unknown class label
    #     else:
    #         new_class = predicted_classes[i]
    #     new_predicted_classes_list.append(new_class)
    # # print(new_predicted_classes_list)
    # # print(len(new_predicted_classes_list))
    #
    # y_test_labels = []
    # y_test_predicts = []
    # store_results_dic = {}
    # for i in range(0, len(new_predicted_classes_list)):
    #     predicted_class = new_predicted_classes_list[i]
    #     y_test_tuple = Y_test_name[i]
    #     seq_name = y_test_tuple[0]
    #     label = y_test_tuple[1]
    #     y_test_labels.append(label)
    #     if predicted_class != 28:
    #         store_results_dic[seq_name] = str(seq_name) + ',' + str(label) + ',' + inverted_all_wicker_class[predicted_class]
    #         y_test_predicts.append(inverted_all_wicker_class[predicted_class])
    #     else:
    #         store_results_dic[seq_name] = str(seq_name) + ',' + str(label) + ',' + 'Unknown'
    #         y_test_predicts.append('Unknown')
    #
    #
    # with open ('results/' + 'test_results.txt', 'w+') as opt:
    #     for eachid in store_results_dic:
    #         opt.write(store_results_dic[eachid] + '\n')
    #
    # # print(y_test_labels)
    # # print(len(y_test_labels))
    # # print(y_test_predicts)
    # # print(len(y_test_predicts))
    # y_test = np.array(Y_test)
    # y_pred = np.array(new_predicted_classes_list)
    # get_metrics(y_test, y_pred, inverted_all_wicker_class)
