import os
import atexit
from keras.models import Sequential, load_model, Model
from keras.layers import Layer, Input, Dense, Dropout, Activation, Flatten, LSTM, Conv1D, Conv2D,\
    MaxPooling1D, MaxPooling2D, Bidirectional, Embedding, GlobalAveragePooling1D, concatenate, \
    MultiHeadAttention, LayerNormalization
import numpy as np
from configs import config
from configs import gpu_config
import tensorflow as tf

class CNN_Model:
    def __init__(self, input_features_num, output_class_num):
        self.num_features = input_features_num
        self.class_num = output_class_num

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def build_model(self, cnn_num_convs, cnn_filters_array):
        # 构建模型
        if config.use_checkpoint == 0:
            os.system('cd ' + gpu_config.checkpoint_dir + ' && rm -rf ckpt*')
        # Create a MirroredStrategy.
        strategy = tf.distribute.MirroredStrategy()
        # Open a strategy scope and create/restore the model
        with strategy.scope():
            # Either restore the latest model, or create a fresh one
            # if there is no checkpoint available.
            checkpoints = [gpu_config.checkpoint_dir + "/" + name for name in os.listdir(gpu_config.checkpoint_dir)]
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=os.path.getctime)
                print("Restoring from", latest_checkpoint)
                return load_model(latest_checkpoint)
            print("Creating a new model")

            # CNN model
            # 输入层
            input_layer = Input(shape=(self.num_features, 1))
            conv_input_layer = input_layer
            # 创建多个卷积层
            for i in range(cnn_num_convs):
                # 添加卷积层
                conv = Conv1D(cnn_filters_array[i], config.cnn_kernel_sizes_array[i], activation='relu')(conv_input_layer)
                conv_input_layer = conv
            dropout1 = Dropout(0.5)(conv_input_layer)
            # 添加展平层和全连接层
            flatten = Flatten()(dropout1)
            dense1 = Dense(128, activation='relu')(flatten)
            dropout2 = Dropout(config.cnn_dropout)(dense1)
            # 输出层
            output_layer = Dense(int(config.class_num), activation='softmax')(dropout2)
            # 构建模型
            model = Model(inputs=input_layer, outputs=output_layer)
            # 编译模型
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            # 打印模型摘要
            #model.summary()
        atexit.register(strategy._extended._collective_ops._pool.close)  # type: ignore
        return model