import os
import sys

current_folder = os.path.dirname(os.path.abspath(__file__))
# 添加 configs 文件夹的路径到 Python 路径
configs_folder = os.path.join(current_folder, "..")  # 需要根据实际目录结构调整
sys.path.append(configs_folder)

import argparse

from configs import config
from utils.data_util import read_fasta


def main():
    # 1.parse args
    describe_info = '########################## NeuralTE-reFormat, version ' + str(config.version_num) + ' ##########################'
    parser = argparse.ArgumentParser(description=describe_info)
    parser.add_argument('-i', metavar='input_fasta', help='Input the file path to be reformatted, fasta format')
    parser.add_argument('-o', metavar='output_fasta', help='Output the reformatted file, fasta format')

    args = parser.parse_args()

    data_path = args.i
    output_path = args.o
    names, contigs = read_fasta(data_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f_save:
        for name in names:
            new_name = name + '\tUnknown\tUnknown'
            f_save.write('>'+new_name+'\n'+contigs[name]+'\n')


if __name__ == '__main__':
    main()