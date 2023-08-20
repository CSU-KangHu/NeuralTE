import os

from concurrent.futures import ProcessPoolExecutor, as_completed
from HiTE_util.Util import read_fasta_v1, store_fasta, read_fasta, getReverseSequence


def split_fasta(input_file, output_dir, num_chunks):
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
        # find LTR internal
        parts = name.split('LTR')
        if len(parts) > 1:
            suffix = parts[1]
            prefix = parts[0]
            # find the LTR seq
            ltr_name = prefix + 'LTR' + suffix
            internal_name = prefix + 'I' + suffix
            LTR_names.add(ltr_name)
            LTR_names.add(internal_name)

    new_names = []
    new_contigs = {}
    for name in raw_names:
        if name in LTR_names:
            parts = name.split('LTR')
            if len(parts) > 1:
                suffix = parts[1]
                prefix = parts[0]
                # find the LTR seq
                ltr_name = prefix + 'LTR' + suffix
                internal_name = prefix + 'I' + suffix
                if raw_contigs.__contains__(ltr_name) and raw_contigs.__contains__(internal_name):
                    intact_ltr_name = prefix + 'intactLTR' + suffix
                    intact_ltr_seq = raw_contigs[ltr_name] + raw_contigs[internal_name] + raw_contigs[ltr_name]
                    new_names.append(intact_ltr_name)
                    new_contigs[intact_ltr_name] = intact_ltr_seq
                    repbase_labels[intact_ltr_name] = repbase_labels[ltr_name]
        elif not name.endswith('-I') and not name.endswith('-LTR'):
            # 丢弃（只存在内部序列，不存在终端序列，或者只有终端序列，不存在内部序列）
            new_names.append(name)
            new_contigs[name] = raw_contigs[name]
    processed_TE_path = repbase_path + '_preprocess.ref'

    # Step4. store Repbase sequence with classification, species_name, and TSD sequence
    # 去掉processed_TE_path中存在重复的LTR，例如Copia-1_AA-intactLTR1和Copia-1_AA-intactLTR2，取其中具有合法TSD那个。两个都有，则随机去一个；两个都没有，优先取有TSD那个，否则随机取一个。
    # get all classification
    all_classification = set()
    final_repbase_contigs = {}
    duplicate_ltr = set()
    for query_name in new_names:
        label_item = repbase_labels[query_name]
        cur_prefix = query_name.split('-intactLTR')[0]
        if not duplicate_ltr.__contains__(cur_prefix):
            new_name = query_name + '\t' + label_item[0] + '\t' + label_item[1]
            duplicate_ltr.add(cur_prefix)
            final_repbase_contigs[new_name] = new_contigs[query_name]
            all_classification.add(label_item[0])
    store_fasta(final_repbase_contigs, processed_TE_path)

    return processed_TE_path


if __name__ == '__main__':
    data_dir = '/home/hukang/NeuralTE/data'
    output_dir = data_dir + '/temp'
    #filenames = ['repbase_train_part.64.ref', 'repbase_test_part.64.ref']
    filenames = ['all_repbase.ref']
    tool_dir = '/home/hukang/HiTE/tools'
    threads = 40
    for filename in filenames:
        cur_path = data_dir + '/' + filename
        # 将文件中的LTR序列连接起来
        cur_path = connect_LTR(cur_path)

        # 将文件切分成threads块
        split_files = split_fasta(cur_path, output_dir, threads)
        # 并行化识别LTR和TIR
        ex = ProcessPoolExecutor(threads)
        jobs = []
        for split_file in split_files:
            job = ex.submit(identify_terminals, split_file, output_dir, tool_dir)
            jobs.append(job)
        ex.shutdown(wait=True)

        cur_update_path = cur_path + '.update'
        os.system('rm -f ' + cur_update_path)
        for job in as_completed(jobs):
            update_split_file = job.result()
            os.system('cat ' + update_split_file + ' >> ' + cur_update_path)