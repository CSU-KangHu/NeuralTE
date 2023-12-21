import argparse
import os
import random
import re
import sys

current_folder = os.path.dirname(os.path.abspath(__file__))
# 添加 configs 文件夹的路径到 Python 路径
configs_folder = os.path.join(current_folder, "..")  # 需要根据实际目录结构调整
sys.path.append(configs_folder)

from configs import config
from utils.data_util import read_fasta_v1, read_fasta, store_fasta


def split_fasta(cur_path, output_dir, num_chunks):
    split_files = []
    os.system('rm -rf ' + output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    names, contigs = read_fasta_v1(cur_path)
    num_names = len(names)
    chunk_size = num_names // num_chunks

    for i in range(num_chunks):
        chunk_start = i * chunk_size
        chunk_end = chunk_start + chunk_size if i < num_chunks - 1 else num_names
        chunk = names[chunk_start:chunk_end]
        output_path = output_dir + '/out_' + str(i) + '.fa'
        with open(output_path, 'w') as out_file:
            for name in chunk:
                seq = contigs[name]
                out_file.write('>'+name+'\n'+seq+'\n')
        split_files.append(output_path)
    return split_files


def identify_terminals(split_file, output_dir, tool_dir):
    ltrsearch_command = 'cd ' + output_dir + ' && ' + tool_dir + '/ltrsearch ' + split_file
    itrsearch_command = 'cd ' + output_dir + ' && ' + tool_dir + '/itrsearch -i 0.7 -l 7 ' + split_file
    os.system(ltrsearch_command)
    os.system(itrsearch_command)
    ltr_file = split_file + '.ltr'
    tir_file = split_file + '.itr'

    # 读取ltr和itr文件，获取ltr和itr开始和结束位置
    ltr_names, ltr_contigs = read_fasta_v1(ltr_file)
    tir_names, tir_contigs = read_fasta_v1(tir_file)
    LTR_info = {}
    for ltr_name in ltr_names:
        orig_name = ltr_name.split('\t')[0]
        terminal_info = ltr_name.split('\t')[2]
        LTR_info_parts = terminal_info.split('LTR')[1].split(' ')[0].replace('(', '').replace(')', '').split('..')
        LTR_left_pos_parts = LTR_info_parts[0].split(',')
        LTR_right_pos_parts = LTR_info_parts[1].split(',')
        lLTR_start = int(LTR_left_pos_parts[0])
        lLTR_end = int(LTR_left_pos_parts[1])
        rLTR_start = int(LTR_right_pos_parts[1])
        rLTR_end = int(LTR_right_pos_parts[0])
        LTR_info[orig_name] = (lLTR_start, lLTR_end, rLTR_start, rLTR_end)
    TIR_info = {}
    for tir_name in tir_names:
        orig_name = tir_name.split('\t')[0]
        terminal_info = tir_name.split('\t')[2]
        TIR_info_parts = terminal_info.split('ITR')[1].split(' ')[0].replace('(', '').replace(')', '').split('..')
        TIR_left_pos_parts = TIR_info_parts[0].split(',')
        TIR_right_pos_parts = TIR_info_parts[1].split(',')
        lTIR_start = int(TIR_left_pos_parts[0])
        lTIR_end = int(TIR_left_pos_parts[1])
        rTIR_start = int(TIR_right_pos_parts[1])
        rTIR_end = int(TIR_right_pos_parts[0])
        TIR_info[orig_name] = (lTIR_start, lTIR_end, rTIR_start, rTIR_end)

    # 更新split_file的header,添加两列 LTR:1-206,4552-4757  TIR:1-33,3869-3836
    update_split_file = split_file + '.updated'
    update_contigs = {}
    names, contigs = read_fasta_v1(split_file)
    for name in names:
        orig_name = name.split('\t')[0]
        LTR_str = 'LTR:'
        if LTR_info.__contains__(orig_name):
            lLTR_start, lLTR_end, rLTR_start, rLTR_end = LTR_info[orig_name]
            LTR_str += str(lLTR_start) + '-' + str(lLTR_end) + ',' + str(rLTR_start) + '-' + str(rLTR_end)
        TIR_str = 'TIR:'
        if TIR_info.__contains__(orig_name):
            lTIR_start, lTIR_end, rTIR_start, rTIR_end = TIR_info[orig_name]
            TIR_str += str(lTIR_start) + '-' + str(lTIR_end) + ',' + str(rTIR_start) + '-' + str(rTIR_end)
        update_name = name + '\t' + LTR_str + '\t' + TIR_str
        update_contigs[update_name] = contigs[name]
    store_fasta(update_contigs, update_split_file)
    return update_split_file


def connect_LTR(repbase_path):
    # Preprocess. connect LTR and LTR_internal
    # considering reverse complementary sequence
    raw_names, raw_contigs = read_fasta(repbase_path)
    label_names, label_contigs = read_fasta_v1(repbase_path)
    # store repbase name and label
    repbase_labels = {}
    for name in label_names:
        parts = name.split('\t')
        repbase_name = parts[0]
        classification = parts[1]
        species_name = parts[2]
        repbase_labels[repbase_name] = (classification, species_name)

    # 获取所有LTR序列
    LTR_names = set()
    for name in raw_names:
        # 识别LTR终端序列，并获得对应的内部序列名称
        pattern = r'\b(\w+(-|_)?)LTR((-|_)?\w*)\b'
        matches = re.findall(pattern, name)
        if matches:
            replacement = r'\1I\3'
            internal_name1 = re.sub(pattern, replacement, name)
            replacement = r'\1INT\3'
            internal_name2 = re.sub(pattern, replacement, name)
            LTR_names.add(name)
            LTR_names.add(internal_name1)
            LTR_names.add(internal_name2)

    # 存储分段的LTR与完整LTR的对应关系
    SegLTR2intactLTR = {}
    new_names = []
    new_contigs = {}
    for name in raw_names:
        if name in LTR_names:
            # 当前序列是LTR，判断内部序列是否存在
            pattern = r'\b(\w+(-|_)?)LTR((-|_)?\w*)\b'
            matches = re.findall(pattern, name)
            if matches:
                ltr_name = name
                replacement = r'\1I\3'
                internal_name1 = re.sub(pattern, replacement, name)
                replacement = r'\1INT\3'
                internal_name2 = re.sub(pattern, replacement, name)

                if raw_contigs.__contains__(ltr_name):
                    if raw_contigs.__contains__(internal_name1):
                        internal_name = internal_name1
                        internal_seq = raw_contigs[internal_name1]
                    elif raw_contigs.__contains__(internal_name2):
                        internal_name = internal_name2
                        internal_seq = raw_contigs[internal_name2]
                    else:
                        internal_name = None
                        internal_seq = None
                    if internal_seq is not None:
                        replacement = r'\1intactLTR\3'
                        intact_ltr_name = re.sub(pattern, replacement, name)
                        intact_ltr_seq = raw_contigs[ltr_name] + internal_seq + raw_contigs[ltr_name]
                        new_names.append(intact_ltr_name)
                        new_contigs[intact_ltr_name] = intact_ltr_seq
                        repbase_labels[intact_ltr_name] = repbase_labels[ltr_name]
                        SegLTR2intactLTR[ltr_name] = intact_ltr_name
                        SegLTR2intactLTR[internal_name] = intact_ltr_name
        else:
            # 如果当前序列是INT，直接丢弃，因为具有LTR的INT肯定会被识别出来，而没有LTR的INT应该被当做不完整的LTR丢弃
            pattern = r'\b(\w+(-|_)?)INT((-|_)?\w*)\b'
            matches = re.findall(pattern, name)
            if not matches:
                # 保留其余类型的转座子
                new_names.append(name)
                new_contigs[name] = raw_contigs[name]

    # Step4. store Repbase sequence with classification, species_name, and TSD sequence
    # get all classification
    final_repbase_contigs = {}
    for query_name in new_names:
        label_item = repbase_labels[query_name]
        new_name = query_name + '\t' + label_item[0] + '\t' + label_item[1]
        final_repbase_contigs[new_name] = new_contigs[query_name]
    store_fasta(final_repbase_contigs, repbase_path)

    # 存储分段的LTR与完整LTR的对应关系
    SegLTR2intactLTRMap = config.work_dir + '/segLTR2intactLTR.map'
    with open(SegLTR2intactLTRMap, 'a+') as f_save:
        for name in SegLTR2intactLTR.keys():
            intact_ltr_name = SegLTR2intactLTR[name]
            f_save.write(name + '\t' + intact_ltr_name + '\n')
    return repbase_path, repbase_labels

def get_all_files(directory):
    all_files = []
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            all_files.append(file_path)
    return all_files

def split_dataset(sequences, train_output_file, test_output_file, split_ratio=0.8):
    # 随机划分训练集和测试集
    all_ids = list(sequences.keys())
    random.shuffle(all_ids)
    train_ids = all_ids[:int(split_ratio * len(all_ids))]
    test_ids = all_ids[int(split_ratio * len(all_ids)):]

    # 写入训练集fasta文件
    with open(train_output_file, 'w') as f:
        for seq_id in train_ids:
            f.write('>' + seq_id + '\n')
            f.write(sequences[seq_id] + '\n')

    # 写入测试集fasta文件
    with open(test_output_file, 'w') as f:
        for seq_id in test_ids:
            f.write('>' + seq_id + '\n')
            f.write(sequences[seq_id] + '\n')


def print_dataset_info(repbase_path):
    repbase_names, repbase_contigs = read_fasta_v1(repbase_path)
    # 统计其中序列个数，物种数量
    unique_species = set()
    for name in repbase_names:
        unique_species.add(name.split('\t')[2])
    print('pre-processed Repbase database sequence size: ' + str(len(repbase_names)) + ', total species num: ' + str(
        len(unique_species)))


def main():
    # 1.parse args
    describe_info = '########################## NeuralTE-preprocess_repbase, version ' + str(config.version_num) + ' ##########################'
    parser = argparse.ArgumentParser(description=describe_info)
    parser.add_argument('--repbase_dir', metavar='repbase_dir', help='Input Repbase directory')
    parser.add_argument('--out_dir', metavar='out_dir', help='Output directory')

    args = parser.parse_args()

    repbase_dir = args.repbase_dir
    out_dir = args.out_dir

    repbase_dir = os.path.realpath(repbase_dir)
    out_dir = os.path.realpath(out_dir)
    repbase_path = out_dir + '/all_repbase.ref'

    # 1.合并Repbase目录下的所有Repbase文件, 只保留header格式为seq_name\tlabel\tspecies_name的序列
    # 2. 保留能够转换为 Wicker superfamily 标签的序列，其余的序列很难确定其 superfamily 类别
    files = get_all_files(repbase_dir)
    all_repbase_contigs = {}
    for file in files:
        names, contigs = read_fasta_v1(file)
        for name in names:
            parts = name.split('\t')
            if len(parts) == 3 and config.Repbase_wicker_labels.__contains__(parts[1]):
                all_repbase_contigs[name] = contigs[name]
    store_fasta(all_repbase_contigs, repbase_path)
    # 3. 将repbase序列的LTR和Internal连接起来，过滤掉那些不完整的LTR序列
    repbase_path, repbase_labels = connect_LTR(repbase_path)
    print_dataset_info(repbase_path)


if __name__ == '__main__':
    main()