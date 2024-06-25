import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler
import pandas as pd
import numpy as np
torch.manual_seed(2024)
import torch.nn.functional as F
from mymodel_neuralLTR import CNNCAT 
import os
from tqdm import tqdm
import sys
import argparse
import math
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
import itertools
import re
import multiprocessing
from PIL import Image
from collections import OrderedDict

def generate_kmer_dic(repeat_num):
    ##initiate a dic to store the kmer dic
    ##kmer_dic = {'ATC':0,'TTC':1,...}
    kmer_dic = {}
    bases = ['A','G','C','T', '-']
    kmer_list = list(itertools.product(bases, repeat=int(repeat_num)))
    reduce_mer = '-' * repeat_num # 去掉全空的kmer
    for eachitem in kmer_list:
        #print(eachitem)
        each_kmer = ''.join(eachitem)
        if each_kmer == reduce_mer:
            continue
        kmer_dic[each_kmer] = 0
    return (kmer_dic)

def get_freq_feature(char_array, Ks):
    freq_features = []
    for k in Ks:
        kmer_dic = generate_kmer_dic(k)
        kmer_list_len = char_array.shape[1] - k + 1
        concatenated_strings = np.array([''.join(row[i:i+k]) for row in char_array for i in range(kmer_list_len)])
        concatenated_strings = concatenated_strings.reshape(-1, kmer_list_len)
        mers = []
        for i in range(kmer_list_len):
            mers += concatenated_strings[:,i].tolist()
        reduce_mer = '-' * k
        for mer in mers:
            if mer == reduce_mer:
                continue
            kmer_dic[mer] += 1
        num_list = []  ##this dic stores num_dic = [0,1,1,0,3,4,5,8,2...]
        for eachkmer in kmer_dic:
            num_list.append(kmer_dic[eachkmer])
        freq_features += num_list
    freq_features = np.array(freq_features)
    return freq_features

def get_block_img(char_array):
    block_array = np.full(char_array.shape, 100, dtype=int)
    # 将char_array中为'-'的位置在new_array中设置为0
    block_array[char_array == '-'] = 0
    return block_array

def get_support_img(char_array):
    counts_array = np.full(char_array.shape, 100, dtype=int)
    letters = set('AGTC')
    # 遍历每一列
    for col in range(char_array.shape[1]):
        # 对每一列进行计数
        total_count = sum(np.array([np.sum(char_array[:, col] == letter) for letter in letters]))
        for letter in letters:  # 遍历该列中出现的不同字符
            # print(char_array[:, col])
            if total_count == 0:
                counts_array[char_array[:, col] == '-', col] = 0
                break
            else:
                count = np.sum(char_array[:, col] == letter)
                # print(count, total_count)
                counts_array[char_array[:, col] == letter, col] = int((count / total_count) * 100)
    counts_array[char_array == '-'] = 0
    return counts_array

def get_base_img(char_array):
    base_dic = {'G': 100, 'C': 75, 'A': 50, 'T': 25, '-': 0}
    base_array = np.array([[base_dic[base] for base in row] for row in char_array])
    return base_array

def extract_features(file_path):
    ltr_names = []
    chunk_features = []
    chunk_freqs = []
    Ks = [1,2,3,4,5]
    for file in tqdm(file_path):
        ltr_name = file.split('.')[0].split('/')[-1]
        ltr_names.append(ltr_name)
        with open(file, 'r') as fr:
            lines = fr.readlines()
            lines = [re.sub(r'[^ATCG]', '-', line.replace('\t', '').strip()) for line in lines]
            char_array = np.array([list(s) for s in lines])

            # 不足100行填充100行
            padding_size = 100 - char_array.shape[0]
            if padding_size > 0:  # 填充数组直到它有 100 行
                pad_array = np.full((padding_size, char_array.shape[1]), '-', dtype=char_array.dtype)
                char_array = np.concatenate((char_array, pad_array), axis=0)# 沿着第一个维度（行）进行填充

            char_array_left = char_array[:, :100]
            char_array_right = char_array[:, 100:]
            block_img_left = get_block_img(char_array_left)
            support_img_left = get_support_img(char_array_left)
            base_img_left = get_base_img(char_array_left)
            block_img_right = get_block_img(char_array_right)
            support_img_right = get_support_img(char_array_right)
            base_img_right = get_base_img(char_array_right)
            
            freq_left = get_freq_feature(char_array_left, Ks) # kmer频次特征
            freq_right = get_freq_feature(char_array_right, Ks) 
            freq_feature = np.stack((freq_left, freq_right), axis=0)
            # 中间填上一列 0
            mid_img = np.zeros((100, 1), dtype=int)
            block_img = np.concatenate((block_img_left, mid_img, block_img_right), axis=1)
            support_img = np.concatenate((support_img_left, mid_img, support_img_right), axis=1)
            base_img = np.concatenate((base_img_left, mid_img, base_img_right), axis=1)

            features = np.stack((block_img, support_img, base_img), axis=2) # [100, 200, 3]

            chunk_features.append(features)
            chunk_freqs.append(freq_feature)
            
    return chunk_features, chunk_freqs, ltr_names


# 读取文件
def get_files(file_path):
    target_files = []
    for root, dirs, files in os.walk(file_path):
        for file in files:
            # 检查文件是否以'.matrix'结尾
            if file.endswith('.matrix'):
                # 构建完整路径
                file_name = os.path.join(root, file)
                # 存储路径
                target_files.append(file_name)
    return target_files

def chunk_data(pos_data, chunk_size1):
    for i in range(0, len(pos_data), chunk_size1):
        pos_chunk = pos_data[i:i + chunk_size1] if i is not None else []
        yield pos_chunk

def align2pileup(file_path):
    # 进程数
    num_processes = 20
    target_files = get_files(file_path)
    print(len(target_files))
    chunk_size = len(target_files) // num_processes 
    data_chunks = list(chunk_data(target_files, chunk_size))
    # 创建进程池
    # pool = multiprocessing.Pool(processes=num_processes)
    # results = pool.map(extract_features, data_chunks)
    with multiprocessing.Pool(processes=num_processes) as pool:
        # 使用 map 方法将 process_array 函数应用于每个数组
        results = pool.map(extract_features,  data_chunks)
    img_features = torch.empty((0, 100, 201, 3), dtype=torch.float32)
    freq_features = torch.empty((0, 2, 3900), dtype=torch.float32)
    seq_names = []
    for result in results:
        chunk_feature, chunk_freq, chunk_name = result
        chunk_feature = torch.tensor(np.array(chunk_feature)) # 图像特征
        chunk_freq = torch.tensor(np.array(chunk_freq))  # kmer频次特征
        img_features = torch.cat((img_features, chunk_feature), dim=0)
        freq_features = torch.cat((freq_features, chunk_freq), dim=0)
        seq_names.extend(chunk_name)
    img_features = img_features.squeeze(0)
    img_features = img_features.permute(0, 3, 1, 2)
    freq_features = torch.unsqueeze(freq_features, dim=2)
    return img_features, freq_features, seq_names

def spilt_filename(pred_file, img_dir):
    name_to_value = {}
    with open(pred_file, 'r') as file:
        for line in file:
            name, value = line.strip().split()
            name_to_value[name] = value
    for filename in os.listdir(img_dir):
        # 检查文件名是否在name_to_value字典中
        seq_id = filename.split('_')[0]
        if seq_id in name_to_value:
            # 根据字典中的值来决定添加0还是1
            prefix = '0' if name_to_value[seq_id] == '0' else '1'
            # 构建新的文件名
            new_filename = f"{prefix}_{filename}"
            # 重命名文件
            os.rename(os.path.join(img_dir, filename), os.path.join(img_dir, new_filename))

def choice_device():
    if torch.cuda.is_available():
        devices = []
        for i in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(i)
            memory_stats = torch.cuda.memory_stats(i)
            total_memory = properties.total_memory # 获取总显存
            allocated_memory_current = memory_stats['allocated_bytes.all.current'] # 当前已分配的显存
            available_memory = total_memory - allocated_memory_current # 估算当前可用显存
            devices.append((i, available_memory))
        device, _ = max(devices, key=lambda x: x[1])
        max_available_memory = devices[device][1]
        print(f"Using GPU {device} with the most memory available.")
        print(f"the available memory is {max_available_memory / (1024 ** 3) }GB.")

        single_datasize = 3 * 200 * 100 + 2 * 1900
        batch_size = min(128, max_available_memory // single_datasize // 2)
        print(f"batch_size is {batch_size}.")
    else:
        device = torch.device("cpu")
        print("Using CPU.")
        batch_size = 33
   
    return device, batch_size

def main():
    # 从命令行接收参数
    parser = argparse.ArgumentParser(description="接收data_dir参数")
    parser.add_argument("--data_dir", type=str, help="数据目录路径")
    parser.add_argument("--out_dir", type=str, help="输出路径")    
    parser.add_argument("--model_path", type=str, help="模型路径")
    parser.add_argument("--threads", type=str, help="模型路径")

    args = parser.parse_args()
    outpath = args.out_dir
    file_path = args.data_dir
    model_path = args.model_path
    print(model_path)

    device_id = 0
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    model = CNNCAT()
    state_dict = torch.load(model_path, map_location=device)

    if next(iter(state_dict)).startswith("module."): # 4 剥除权重文件中的module层
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        state_dict = new_state_dict

    # model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)
    # model = model.module.cpu()
    # device=torch.device('cuda')
    model = model.to(device)
    print('model loaded done!')

    img_features, freq_features, ltr_names = align2pileup(file_path)
    print(img_features.shape)
    img_features = F.normalize(img_features, p=2, dim=1)
    freq_features = F.normalize(freq_features, p=2, dim=1)
    dataset = torch.utils.data.TensorDataset(img_features, freq_features)
    
    batch_size = 32
    # device, batch_size = choice_device()
    
    model.eval()
    # batch_size = 32
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds = []
    probs_array = []
    with torch.no_grad():
        for batch_x, batch_kmer in tqdm(val_loader):
            batch_x = batch_x.to(device)
            batch_kmer = batch_kmer.to(device)
            outputs = model(batch_x, batch_kmer)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(probs, 1)
            probs_array.extend(probs)


            # probs = F.softmax(outputs, dim=1)
            # prob_class1 = probs[:, 1]
            # prob_class0 = probs[:, 0]
            # prob_diff = prob_class1 - prob_class0
            # predicted = (prob_diff > 0.7).int()

            preds.extend(predicted.cpu().numpy())
    with open(outpath + '/is_LTR_deep.txt', 'w') as fw:
        for i in range(len(ltr_names)):
            fw.write(ltr_names[i] + '\t' + str(preds[i]) + '\n')

    with open(outpath + '/probs.txt', 'w') as fw:
        for i in range(len(ltr_names)):
            fw.write(f'{ltr_names[i]}\t{probs_array[i]}\n')

if __name__ == "__main__":
    main()