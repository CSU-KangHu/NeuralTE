#注：将序列转为7mer,特征为7mer频次。
#测试结果：

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
import numpy as np

from Util import load_repbase_with_TSD, generate_mats, conv_labels
from evaluate import get_metrics


# Step 1: 获取所有superfamily的标签
all_wicker_class = {'Tc1-Mariner': 0, 'hAT': 1, 'Mutator': 2, 'Merlin': 3, 'Transib': 4, 'P': 5, 'PiggyBac': 6,
                    'PIF-Harbinger': 7, 'CACTA': 8, 'Crypton': 9, 'Helitron': 10, 'Maverick': 11, 'Copia': 12,
                    'Gypsy': 13, 'Bel-Pao': 14, 'Retrovirus': 15, 'DIRS': 16, 'Ngaro': 17, 'VIPER': 18,
                    'Penelope': 19, 'R2': 20, 'RTE': 21, 'Jockey': 22, 'L1': 23, 'I': 24, 'tRNA': 25, '7SL': 26, '5S': 27}
class_num = len(all_wicker_class)
print(class_num)
inverted_all_wicker_class = {value: key for key, value in all_wicker_class.items()}

# Step 2: 将Repbase序列和TSD数据， 以及label 转换成特征
threads = 40
work_dir = '/public/home/hpc194701009/TE_Classification/DeepTE-master/example_data/model_test'
#work_dir = '/home/hukang/TE_Classification/DeepTE-master/example_data/model_test'
repbase_train = work_dir + '/repbase_train_part.ref'
repbase_test = work_dir + '/repbase_test_part.ref'
# repbase_train = work_dir + '/train.ref'
# repbase_test = work_dir + '/test.ref'
kmer_size = 7
X_feature_len = pow(4, kmer_size)

X, Y, seq_names = load_repbase_with_TSD(repbase_train)
#X -> {seq_name: [0, 4, 2, 3,...], }
X = generate_mats(X, seq_names, kmer_size, threads)
#print(X)
print(X.shape)

Y = conv_labels(Y, all_wicker_class)
#print(Y)
print(Y.shape)

# Step 3: 抽取训练集的20%当做验证集
divide_data_most_part = int(X.shape[0] * 0.8)
X_train = np.array(X[0:divide_data_most_part])  ##change the sample
X_validate = np.array(X[divide_data_most_part:])
Y_train = np.array(Y[0:divide_data_most_part])
Y_validate = np.array(Y[divide_data_most_part:])
# print(X_train.shape)
# print(X_validate.shape)
# print(Y_train.shape)
# print(Y_validate.shape)

# Step 4: 将数据reshape成CNN接收的格式
X_train = X_train.reshape(X_train.shape[0], 1, X_feature_len, 1)
X_validate = X_validate.reshape(X_validate.shape[0], 1, X_feature_len, 1)
X_train = X_train.astype('float64')
X_validate = X_validate.astype('float64')

Y_train_one_hot = np_utils.to_categorical(Y_train, int(class_num))
Y_validate_one_hot = np_utils.to_categorical(Y_validate, int(class_num))
# print(Y_train_one_hot)
# print(Y_test_one_hot)

# Step 5: 定义模型的架构
model = Sequential()

model.add(Conv2D(100, (1, 3), activation='relu', input_shape=(1, X_feature_len, 1)))
model.add(MaxPooling2D(pool_size=(1, 2)))
model.add(Conv2D(150, (1, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(1, 2)))
model.add(Conv2D(225, (1, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(1, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(int(class_num), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Step 6: 训练模型
model.fit(X_train, Y_train_one_hot, validation_data=(X_validate, Y_validate_one_hot),
          batch_size=32, epochs=10, verbose=1)   ##epoch An epoch is an iteration over the entire x and y data provided
                                                ##batch size if we have 1000 samples, we set batch size to 100, so we will
                                                ##run 100 first and then the second 100, so this will help us to reduce the
                                                ##the memory we use

# Step 7: 评估模型
loss, accuracy = model.evaluate(X_validate, Y_validate_one_hot, verbose=1)
print ("\nloss=" + str(loss) + ', accuracy=' + str(accuracy))


# Step 8: 保存模型
model_path = work_dir + '/' + 'test_model.h5'
model.save(model_path)

# Step 9: 在测试集上进行测试，并输出所有评测指标
##Step 9.1: generate the input data
model_path = work_dir + '/' + 'test_model.h5'
model = load_model(model_path)

prop_thr = 0

X_test, Y_test, seq_names = load_repbase_with_TSD(repbase_test)
X_test = generate_mats(X_test, seq_names, kmer_size, threads)
X_test = X_test.reshape(X_test.shape[0], 1, X_feature_len, 1)
X_test = X_test.astype('float64')

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
        new_class = -1    #unknown class label
    else:
        new_class = predicted_classes[i]
    new_predicted_classes_list.append(new_class)
print(new_predicted_classes_list)
print(len(new_predicted_classes_list))

y_test_labels = []
y_test_predicts = []
store_results_dic = {}
for i in range(0, len(new_predicted_classes_list)):
    predicted_class = new_predicted_classes_list[i]
    y_test_tuple = Y_test[i]
    seq_name = y_test_tuple[0]
    label = y_test_tuple[1]
    y_test_labels.append(label)
    if predicted_class != -1:
        store_results_dic[seq_name] = str(seq_name) + ',' + str(label) + ',' + inverted_all_wicker_class[predicted_class]
        y_test_predicts.append(inverted_all_wicker_class[predicted_class])
    else:
        store_results_dic[seq_name] = str(seq_name) + ',' + str(label) + ',' + 'Unknown'
        y_test_predicts.append('Unknown')


with open (work_dir + '/' + 'test_results.txt', 'w+') as opt:
    for eachid in store_results_dic:
        opt.write(store_results_dic[eachid] + '\n')

print(y_test_labels)
print(len(y_test_labels))
print(y_test_predicts)
print(len(y_test_predicts))
y_test = np.array(y_test_labels)
y_pred = np.array(y_test_predicts)
get_metrics(y_test, y_pred)