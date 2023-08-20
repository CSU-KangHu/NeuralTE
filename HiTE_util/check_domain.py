import os

from Util import read_fasta, store_fasta, get_domain_info
from concurrent.futures import ProcessPoolExecutor, as_completed

def divided_array(original_array, partitions):
    final_partitions = [[] for _ in range(partitions)]
    node_index = 0

    read_from_start = True
    read_from_end = False
    i = 0
    j = len(original_array) - 1
    while i <= j:
        # read from file start
        if read_from_start:
            final_partitions[node_index % partitions].append(original_array[i])
            i += 1
        if read_from_end:
            final_partitions[node_index % partitions].append(original_array[j])
            j -= 1
        node_index += 1
        if node_index % partitions == 0:
            # reverse
            read_from_end = bool(1 - read_from_end)
            read_from_start = bool(1 - read_from_start)
    return final_partitions

def multiple_alignment_blastx(repeats_path):
    split_repeats_path = repeats_path[0]
    protein_db_path = repeats_path[1]
    blastx2Results_path = repeats_path[2]
    align_command = 'blastx -db ' + protein_db_path + ' -num_threads ' \
                    + str(1) + ' -query ' + split_repeats_path + ' -outfmt 6 > ' + blastx2Results_path
    os.system(align_command)
    return blastx2Results_path

def multi_process_alignx(query_path, subject_path, blastnResults_path, tmp_output_dir, threads):

    tmp_blast_dir = tmp_output_dir + '/tmp_blast_test'
    os.system('rm -rf ' + tmp_blast_dir)
    if not os.path.exists(tmp_blast_dir):
        os.makedirs(tmp_blast_dir)

    orig_names, orig_contigs = read_fasta(query_path)

    longest_repeat_files = []
    segments_cluster = divided_array(list(orig_contigs.items()), threads)
    for partition_index, cur_segments in enumerate(segments_cluster):
        single_tmp_dir = tmp_blast_dir + '/' + str(partition_index)
        print('current partition_index: ' + str(partition_index))
        if not os.path.exists(single_tmp_dir):
            os.makedirs(single_tmp_dir)
        split_repeat_file = single_tmp_dir + '/repeats_split.fa'
        cur_contigs = {}
        for item in cur_segments:
            cur_contigs[item[0]] = item[1]
        store_fasta(cur_contigs, split_repeat_file)
        repeats_path = (split_repeat_file, subject_path, single_tmp_dir + '/temp.out')
        longest_repeat_files.append(repeats_path)

    ex = ProcessPoolExecutor(threads)
    jobs = []
    for file in longest_repeat_files:
        job = ex.submit(multiple_alignment_blastx, file)
        jobs.append(job)
    ex.shutdown(wait=True)

    if os.path.exists(blastnResults_path):
        os.remove(blastnResults_path)

    for job in as_completed(jobs):
        cur_blastn2Results_path = job.result()
        os.system('cat ' + cur_blastn2Results_path + ' >> ' + blastnResults_path)

if __name__ == '__main__':
    data_dir = '/home/hukang/HiTE/library'
    domain_path = data_dir + '/RepeatPeps.lib'
    # # 所有蛋白质类型
    # names, contigs = read_fasta(domain_path)
    # protein_set = set()
    # for name in names:
    #     seq = contigs[name]
    #     raw_name = name.split('#')[0]
    #     label = name.split('#')[1]
    #     protein_type = raw_name.split('_')[-1]
    #     protein_set.add(protein_type)
    # print(protein_set)
    # print(len(protein_set))

    input_dir = '/home/hukang/NeuralTE/data'
    input_path = input_dir + '/all_repbase.ref_preprocess.ref.update'
    blastnResults_path = input_dir + '/all_repbase.ref_preprocess.ref.update.out'
    # input_path = input_dir + '/repbase_test_part.64.ref.update'
    # blastnResults_path = input_dir + '/repbase_test_part.out'
    threads = 40
    output_table = input_path + '.domain'
    temp_dir = input_dir + '/domain'
    get_domain_info(input_path, domain_path, output_table, threads, temp_dir)