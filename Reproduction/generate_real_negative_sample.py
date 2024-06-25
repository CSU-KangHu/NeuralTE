import os
import time

from Util import read_fasta, store_fasta, get_full_length_copies, getReverseSequence, generate_msa, read_fasta_v1, \
    generate_both_ends_frame_from_seq, rename_reference, find_files_recursively, random_downsample
from concurrent.futures import ProcessPoolExecutor, as_completed


if __name__ == '__main__':
    # # species_list = ['Picea_abies']
    # # reference_list = ['GCA_900067695.1_Pabies01_genomic.fna']
    # # rename_reference_list = ['GCA_900067695.1_Pabies01_genomic.rename.fna']
    # # species_list = ['human']
    # # reference_list = ['GCF_000001405.39_GRCh38.p13_genomic.fna']
    # # rename_reference_list = ['GCF_000001405.39_GRCh38.p13_genomic.rename.fna']
    # species_list = ['Sorghum_bicolor']
    # reference_list = ['GCF_000003195.3_Sorghum_bicolor_NCBIv3_genomic.fna']
    # rename_reference_list = ['GCF_000003195.3_Sorghum_bicolor_NCBIv3_genomic.rename.fna']
    # # num_indexes = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]
    # num_indexes = [0, 5000, 10000, 15000, 20000]
    # threads = 30
    # LtrDetector_home = '/home/hukang/repeat_detect_tools/LtrDetector-master/bin'
    # for i in range(len(species_list)):
    #     species = species_list[i]
    #     reference_name = reference_list[i]
    #     rename_reference_name = rename_reference_list[i]
    #
    #     # 将参考基因组变成Chr+number格式
    #     tmp_output_dir = '/home/hukang/LTR_Benchmarking/LTR_libraries/LtrDetector/' + species
    #     reference = tmp_output_dir + '/' + reference_name
    #     # rename reference
    #     genome_dir = '/home/hukang/LTR_Benchmarking/LTR_libraries/LtrDetector/' + species + '/genome'
    #     if not os.path.exists(genome_dir):
    #         os.makedirs(genome_dir)
    #
    #     genome_path = genome_dir + '/' + rename_reference_name
    #     chr_name_map = tmp_output_dir + '/chr_name.map'
    #     rename_reference(reference, genome_path, chr_name_map)
    #
    #     tmp_output_dir = '/home/hukang/LTR_Benchmarking/LTR_libraries/LtrDetector/'+species+'/output/left_LTR_sample'
    #     if not os.path.exists(tmp_output_dir):
    #         os.makedirs(tmp_output_dir)
    #
    #     # Step1. 调用 LtrDetector 生成 left LTR 序列
    #     left_ltr_cons = tmp_output_dir + '/left_LTR.cons'
    #     if not os.path.exists(left_ltr_cons):
    #         parallel_LtrDetector_command = 'python ' + os.getcwd() + '/run_LtrDetector_parallel.py --genome ' + genome_path + ' --genome_dir ' + genome_dir + ' --out_dir ' + tmp_output_dir + ' --LtrDetector_home ' + LtrDetector_home
    #         print(parallel_LtrDetector_command)
    #         os.system(parallel_LtrDetector_command)
    #
    #     split_ref_dir = genome_dir + '/ref_chr'
    #     ref_contigs = {}
    #     for name in os.listdir(split_ref_dir):
    #         if name.endswith('.fa'):
    #             cur_genome = split_ref_dir + '/' + name
    #             cur_ref_names, cur_ref_contigs = read_fasta(cur_genome)
    #             ref_contigs.update(cur_ref_contigs)
    #
    #     raw_ltr_names, raw_ltr_contigs = read_fasta(left_ltr_cons)
    #
    #     for num_index in num_indexes:
    #         if num_index > len(raw_ltr_names):
    #             break
    #
    #         # 控制 left LTR 数量，以防无法生成 FP 结果
    #         limited_left_ltr_cons = tmp_output_dir + '/left_LTR.limit.cons'
    #         # max_left_ltr_num = 1000
    #         # os.system('head -n ' + str(2 * max_left_ltr_num) + ' ' + left_ltr_cons + ' > ' + limited_left_ltr_cons)
    #         #os.system('sed -n \'4000,6000p\' ' + left_ltr_cons + ' > ' + limited_left_ltr_cons)
    #         interval = 5000
    #         start_index = 2 * num_index
    #         end_index = 2 * (num_index + interval)
    #         print('sed -n \'' + str(start_index+1) + ',' + str(end_index) + 'p\' ' + left_ltr_cons + ' > ' + limited_left_ltr_cons)
    #         os.system('sed -n \'' + str(start_index+1) + ',' + str(end_index) + 'p\' ' + left_ltr_cons + ' > ' + limited_left_ltr_cons)
    #
    #         # Step2. 调用 BM_HiTE 生成 FP ltr
    #         BM_HiTE_command = 'python ' + os.getcwd() + '/benchmarking.py --BM_RM2 0 --BM_EDTA 0 --BM_HiTE 1 -t ' + \
    #                           str(threads) + ' --TE_lib ' + limited_left_ltr_cons + ' -r ' + genome_path + \
    #                           ' --EDTA_home /home/hukang/EDTA/ --species ' + species + ' --tmp_output_dir ' + tmp_output_dir
    #         os.system(BM_HiTE_command)
    #
    #
    #         FP_path = tmp_output_dir + '/FP.blastn.out'
    #         # Step3. 获取 Ltrdetector FP 终端序列 的 左侧框
    #         FP_ltr_seq = tmp_output_dir + '/FP_ltr.fa'
    #         ltr_names, ltr_contigs = read_fasta(limited_left_ltr_cons)
    #         FP_ltr_contigs = {}
    #         ltr_terminal_names = set()
    #         with open(FP_path, 'r') as f_r:
    #             for line in f_r:
    #                 parts = line.split('\t')
    #                 raw_name = parts[0]
    #                 chr_start = parts[2]
    #                 chr_end = parts[3]
    #                 if raw_name.endswith('-lLTR') and (chr_start in raw_name or chr_end in raw_name):
    #                     cur_name = raw_name+'#LTR'
    #                     seq = ltr_contigs[cur_name]
    #                     FP_ltr_contigs[raw_name] = seq
    #         store_fasta(FP_ltr_contigs, FP_ltr_seq)
    #
    #         # Step4. 获取 Ltrdetector FP LTR 终端序列 的 左侧框
    #         temp_dir = tmp_output_dir + '/FP_ltr'
    #         output_dir = tmp_output_dir + '/negative_both_ends_' + str(start_index) + '_' + str(end_index)
    #         # get_left_frame(FP_ltr_seq, ref_contigs, threads, temp_dir, output_dir, split_ref_dir)
    #         generate_both_ends_frame_from_seq(FP_ltr_seq, ref_contigs, threads, temp_dir, output_dir, split_ref_dir)

    # 对负样本进行下采样
    negative_dir = '/home/hukang/left_LTR_real_dataset/negative'
    file_extension = '.matrix'
    all_matrix_files = find_files_recursively(negative_dir, file_extension)
    keep_files = random_downsample(all_matrix_files, 13785)
    # 遍历原始列表，删除未被选中的文件
    for file_path in all_matrix_files:
        if file_path not in keep_files:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleted: {file_path}")


