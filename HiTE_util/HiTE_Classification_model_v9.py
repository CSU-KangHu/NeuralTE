# 融合了Seq, LTR, TIR, Domain等多种特征进行训练，去掉TSD，基于全部的数据进行训练和测试，结果为：
# Accuracy: 0.9454438795463433
# Precision: 0.9324762706055223
# Recall: 0.846982565233756
# F1: 0.8756815502079845

import json
import os

import tensorflow as tf

from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Bidirectional
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np

from HiTE_util.plot_confusion_matrix import plot_confusion_matrix
from Util import load_repbase_with_TSD, generate_mats, conv_labels, load_repbase, read_fasta_v1, generate_feature_mats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

def run_training(batch_size=32, epochs=1, use_checkpoint=1):
    if use_checkpoint == 0:
        os.system('cd ' + checkpoint_dir + ' && rm -rf ckpt*')
    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()

    # Open a strategy scope and create/restore the model
    with strategy.scope():
        model = make_or_restore_model()

    callbacks = [
        # This callback saves a SavedModel every epoch
        # We include the current epoch in the folder name.
        ModelCheckpoint(
            filepath=checkpoint_dir + "/ckpt-{epoch}", save_freq="epoch"
        ),
        EarlyStopping(monitor='val_loss', patience=20)
    ]

    model.fit(X_train, Y_train_one_hot, validation_data=(X_validate, Y_validate_one_hot), callbacks=callbacks,
              batch_size=batch_size, epochs=epochs, verbose=1)
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

    # # 2. CNN model
    # model = Sequential()
    # model.add(Conv1D(100, 3, activation='relu', input_shape=(X_feature_len, 1)))
    # #model.add(MaxPooling1D(pool_size=2))
    # model.add(Conv1D(150, 3, activation='relu'))
    # #model.add(MaxPooling1D(pool_size=2))
    # model.add(Conv1D(225, 3, activation='relu'))
    # #model.add(MaxPooling1D(pool_size=2))
    # model.add(Dropout(0.5))
    #
    # model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.5))
    #
    # model.add(Dense(int(class_num), activation='softmax'))
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    from keras.layers import Conv1D, Flatten, Dense, Dropout, Input
    from keras.models import Model
    from keras.layers import Attention

    # 100-150-225: F1: 0.9061523877354777
    # 32: F1: 0.8803469039813364
    # 32-32: F1: 0.914593253763968
    # 32-32-32: F1: 0.9177968092796981
    # 32-64-128: F1: 0.9067107094594844
    # 32-64-32: F1: 0.914074828497396

    # 输入层
    input_layer = Input(shape=(X_feature_len, 1))
    # 添加卷积层
    conv1 = Conv1D(32, 3, activation='relu')(input_layer)
    # # 添加自注意力层
    # self_attention = Attention()([conv1, conv1])  # 注意力机制输入为[query, value]
    # conv2 = Conv1D(150, 3, activation='relu')(self_attention)
    conv2 = Conv1D(64, 3, activation='relu')(conv1)
    conv3 = Conv1D(32, 3, activation='relu')(conv2)
    dropout1 = Dropout(0.5)(conv3)
    # 添加展平层和全连接层
    flatten = Flatten()(dropout1)
    dense1 = Dense(128, activation='relu')(flatten)
    dropout2 = Dropout(0.5)(dense1)
    # 输出层
    output_layer = Dense(int(class_num), activation='softmax')(dropout2)
    # 构建模型
    model = Model(inputs=input_layer, outputs=output_layer)
    # 编译模型
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # 打印模型摘要
    #model.summary()

    # # 3. attention CNN
    # from tensorflow.keras.layers import Input
    # from tensorflow.keras.models import Model
    # ip = Input()
    # x = augmented_conv1d(ip, shape=(X_feature_len, 1), filters=20)
    #
    # model = Model(ip, x)
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.summary()

    # from tensorflow.keras.layers import Input
    # from tensorflow.keras.models import Model
    # tf.compat.v1.disable_eager_execution()
    # ip = Input(shape=(None, X_feature_len))
    # x = Conv1D(100, 3, activation='relu')(ip)
    # x = MaxPooling1D(pool_size=2)(x)
    # x = Conv1D(150, 3, activation='relu')(x)
    # x = MaxPooling1D(pool_size=2)(x)
    # x = Dropout(0.5)(x)
    # x = augmented_conv1d(x, shape=(X_feature_len, 150), filters=20)
    # x = Flatten()(x)
    # x = Dense(128, activation='relu')(x)
    # op = Dropout(0.5)(x)
    # model = Model(ip, op)
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


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
    epochs = 100
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
    #data_dir = '/public/home/hpc194701009/TE_Classification/NeuralTE/data'
    train_path = data_dir + '/repbase_train.ref'
    test_path = data_dir + '/repbase_test.ref'

    domain_path = data_dir + '/all_repbase.ref_preprocess.ref.update.domain'

    # train_path = data_dir + '/test.ref.update'
    # test_path = data_dir + '/test.ref.update'

    # Step 1: 加载repbase数据。将序列划分成seq, LTR, TIR，TSD四个部分，分别用k=5, 4, 3频次表示seq, LTR, TIR;TSD用one-hot编码表示，用多1位表示TSD_length;将domain转为One-hot编码
    kmer_sizes = [5, 5, 5]
    #X_feature_len = 11 * 4 + 1 + 29
    X_feature_len = 29
    for kmer_size in kmer_sizes:
        X_feature_len += pow(4, kmer_size)

    X, Y, seq_names = load_repbase_with_TSD(train_path, domain_path, all_wicker_class)
    X, Y = generate_feature_mats(X, Y, seq_names, all_wicker_class, kmer_sizes, threads)


    # Step 3: 抽取训练集的20%当做验证集
    divide_data_most_part = int(X.shape[0] * 0.8)
    X_train = np.array(X[0:divide_data_most_part])  ##change the sample
    X_validate = np.array(X[divide_data_most_part:])
    Y_train = np.array(Y[0:divide_data_most_part])
    Y_validate = np.array(Y[divide_data_most_part:])

    # Step 4: 将数据reshape成模型接收的格式
    # 训练和验证集
    X_train = X_train.reshape(X_train.shape[0], X_feature_len, 1)
    X_validate = X_validate.reshape(X_validate.shape[0], X_feature_len, 1)
    X_train = X_train.astype('float64')
    X_validate = X_validate.astype('float64')
    Y_train_one_hot = np_utils.to_categorical(Y_train, int(class_num))
    Y_validate_one_hot = np_utils.to_categorical(Y_validate, int(class_num))

    X_train = np.array(X_train)
    Y_train_one_hot = np.array(Y_train_one_hot)
    X_validate = np.array(X_validate)
    Y_validate_one_hot = np.array(Y_validate_one_hot)

    # 测试集
    X_test, Y_test, Y_test_name = load_repbase_with_TSD(test_path, domain_path, all_wicker_class)
    X_test, Y_test = generate_feature_mats(X_test, Y_test, Y_test_name, all_wicker_class, kmer_sizes, threads)
    X_test = X_test.reshape(X_test.shape[0], X_feature_len, 1)
    X_test = X_test.astype('float64')
    Y_test_one_hot = np_utils.to_categorical(Y_test, int(class_num))
    X_test = np.array(X_test)
    Y_test_one_hot = np.array(Y_test_one_hot)

    print('X_train and Y_train_one_hot shape: ')
    print(X_train.shape, Y_train_one_hot.shape)
    print('X_validate and Y_validate_one_hot shape: ')
    print(X_validate.shape, Y_validate_one_hot.shape)
    print('X_test and Y_test_one_hot shape: ')
    print(X_test.shape, Y_test_one_hot.shape)
    print('batch size: ' + str(batch_size))


    # Step 6: 训练模型
    # Running the first time creates the model
    model = run_training(batch_size=batch_size, epochs=epochs, use_checkpoint=use_checkpoint)

    # Step 7: 保存模型
    model_path = 'model/' + 'test_model.h5'
    model.save(model_path)

    # Step 8: 在测试集上进行测试，并输出所有评测指标
    model_path = 'model/' + 'test_model.h5'
    model = load_model(model_path)

    # Step 9: 评估模型
    loss, accuracy = model.evaluate(X_test, Y_test_one_hot, batch_size=batch_size, verbose=1)
    print("\nloss=" + str(loss) + ', accuracy=' + str(accuracy))

    prop_thr = 0

    ##Step 9.2: generate the predict class
    Y_pred = model.predict(X_test)

    predicted_classes = np.argmax(np.round(Y_pred), axis=1)
    predicted_classes_list = predicted_classes.tolist()
    # print(predicted_classes_list)
    # print(Y_pred)
    # print(Y_pred.shape)

    ##Step 9.3 transfer the prop less than a threshold to be unknown for a class
    max_value_predicted_classes = np.amax(Y_pred, axis=1)
    order = -1
    ls_thr_order_list = []
    for i in range(len(max_value_predicted_classes)):
        order += 1
        if max_value_predicted_classes[i] < float(prop_thr):
            ls_thr_order_list.append(order)

    new_predicted_classes_list = []
    order = -1
    for i in range(len(predicted_classes)):
        order += 1
        if order in ls_thr_order_list:
            new_class = 28    #unknown class label
        else:
            new_class = predicted_classes[i]
        new_predicted_classes_list.append(new_class)
    # print(new_predicted_classes_list)
    # print(len(new_predicted_classes_list))

    y_test_labels = []
    y_test_predicts = []
    store_results_dic = {}
    for i in range(0, len(new_predicted_classes_list)):
        predicted_class = new_predicted_classes_list[i]
        y_test_tuple = Y_test_name[i]
        seq_name = y_test_tuple[0]
        label = y_test_tuple[1]
        y_test_labels.append(label)
        if predicted_class != 28:
            store_results_dic[seq_name] = str(seq_name) + ',' + str(label) + ',' + inverted_all_wicker_class[predicted_class]
            y_test_predicts.append(inverted_all_wicker_class[predicted_class])
        else:
            store_results_dic[seq_name] = str(seq_name) + ',' + str(label) + ',' + 'Unknown'
            y_test_predicts.append('Unknown')


    with open ('results/' + 'test_results.txt', 'w+') as opt:
        for eachid in store_results_dic:
            opt.write(store_results_dic[eachid] + '\n')

    # print(y_test_labels)
    # print(len(y_test_labels))
    # print(y_test_predicts)
    # print(len(y_test_predicts))
    y_test = np.array(Y_test)
    y_pred = np.array(new_predicted_classes_list)
    get_metrics(y_test, y_pred, inverted_all_wicker_class)
