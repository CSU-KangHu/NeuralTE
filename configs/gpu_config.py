# 1. 配置使用多GPU
start_gpu_num = 1 # 开始GPU编号
use_gpu_num = 1 # 使用GPU数量，start_gpu_num=0，use_gpu_num=2表示使用GPU0，GPU1，共两个GPU
all_devices = ["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3", "/gpu:4", "/gpu:5", "/gpu:6", "/gpu:7"] # 机器上所有GPU编号，如果你的机器GPU数量超过8个，依次再后面添加 "/gpu:8", "/gpu:9" ...


# 2. 无需修改参数
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
# Set the GPUs you want to use
gpus = tf.config.experimental.list_physical_devices('GPU')
# For GPU memory growth
for device in gpus:
    tf.config.experimental.set_memory_growth(device, True)
use_devices = all_devices[start_gpu_num: start_gpu_num + use_gpu_num]
tf.config.experimental.set_visible_devices(gpus[start_gpu_num: start_gpu_num + use_gpu_num], 'GPU')
# Create a MirroredStrategy to use multiple GPUs
strategy = tf.distribute.MirroredStrategy(devices=use_devices)
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
# Prepare a directory to store all the checkpoints.
checkpoint_dir = "./ckpt"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)