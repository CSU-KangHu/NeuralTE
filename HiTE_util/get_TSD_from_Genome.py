#-- coding: UTF-8 --
import codecs
import json
import os
from Util import read_fasta, store_fasta, get_full_length_copies, getReverseSequence, search_confident_tsd, \
    read_fasta_v1
from concurrent.futures import ProcessPoolExecutor, as_completed


def getRepBaseTSDFromGenome(repbase_path, genome_path, temp_dir, threads, flanking_len, plant, species):
    # function input: fasta file, genome file
    # function output: a list of TSD length corresponding to fasta sequence
    os.system('rm -rf ' + temp_dir)
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

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

    #获取所有LTR序列
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
                    intact_ltr_name1 = prefix + 'intactLTR1' + suffix
                    intact_ltr_seq1 = raw_contigs[ltr_name] + raw_contigs[internal_name] + raw_contigs[ltr_name]
                    intact_ltr_name2 = prefix + 'intactLTR2' + suffix
                    intact_ltr_seq2 = raw_contigs[ltr_name] + getReverseSequence(raw_contigs[internal_name]) + raw_contigs[ltr_name]
                    new_names.append(intact_ltr_name1)
                    new_contigs[intact_ltr_name1] = intact_ltr_seq1
                    new_names.append(intact_ltr_name2)
                    new_contigs[intact_ltr_name2] = intact_ltr_seq2
                    repbase_labels[intact_ltr_name1] = repbase_labels[ltr_name]
                    repbase_labels[intact_ltr_name2] = repbase_labels[ltr_name]
        else:
            new_names.append(name)
            new_contigs[name] = raw_contigs[name]
    processed_TE_path = temp_dir + '/' + species + '_preprocess.ref'
    store_fasta(new_contigs, processed_TE_path)

    names, contigs = read_fasta(processed_TE_path)
    # Step1. align the TEs to genome, and get copies
    os.system('makeblastdb -in ' + genome_path + ' -dbtype nucl')
    batch_size = 10
    batch_id = 0
    total_names = set(names)
    split_files = []
    cur_contigs = {}
    for i, name in enumerate(names):
        cur_file = temp_dir + '/' + str(batch_id) + '.fa'
        cur_contigs[name] = contigs[name]
        if len(cur_contigs) == batch_size:
            store_fasta(cur_contigs, cur_file)
            split_files.append(cur_file)
            cur_contigs = {}
            batch_id += 1
    if len(cur_contigs) > 0:
        cur_file = temp_dir + '/' + str(batch_id) + '.fa'
        store_fasta(cur_contigs, cur_file)
        split_files.append(cur_file)
        batch_id += 1

    blastn2Results_path = temp_dir + '/all.out'
    os.system('rm -f ' + blastn2Results_path)
    ex = ProcessPoolExecutor(threads)
    jobs = []
    for cur_split_files in split_files:
        job = ex.submit(get_full_length_copies, cur_split_files, genome_path)
        jobs.append(job)
    ex.shutdown(wait=True)
    all_copies = {}
    for job in as_completed(jobs):
        cur_all_copies, cur_blastn2Results_path = job.result()
        all_copies.update(cur_all_copies)
        os.system('cat ' + cur_blastn2Results_path + ' >> ' + blastn2Results_path)

    # Step2. flank all copies to obtain TSDs.
    ref_names, ref_contigs = read_fasta(genome_path)
    batch_member_files = []
    new_all_copies = {}
    for query_name in all_copies.keys():
        copies = all_copies[query_name]
        for copy in copies:
            ref_name = copy[0]
            copy_ref_start = int(copy[1])
            copy_ref_end = int(copy[2])
            direct = copy[4]
            copy_len = copy_ref_end - copy_ref_start + 1
            if copy_ref_start - 1 - flanking_len < 0 or copy_ref_end + flanking_len > len(ref_contigs[ref_name]):
                continue
            copy_seq = ref_contigs[ref_name][copy_ref_start - 1 - flanking_len: copy_ref_end + flanking_len]
            if direct == '-':
                copy_seq = getReverseSequence(copy_seq)
            if len(copy_seq) < 100:
                continue
            new_name = ref_name + ':' + str(copy_ref_start) + '-' + str(copy_ref_end) + '(' + direct + ')'
            if not new_all_copies.__contains__(query_name):
                new_all_copies[query_name] = {}
            copy_contigs = new_all_copies[query_name]
            copy_contigs[new_name] = copy_seq
            new_all_copies[query_name] = copy_contigs
            if len(cur_contigs) >= 100:
                break
    for query_name in new_all_copies.keys():
        copy_contigs = new_all_copies[query_name]
        cur_member_file = temp_dir + '/' + query_name + '.blast.bed.fa'
        store_fasta(copy_contigs, cur_member_file)
        query_seq = contigs[query_name]
        batch_member_files.append((query_name, query_seq, cur_member_file))

    # Step3. search TSDs in all flanked copies. The TSD length that occurs most times is regarded as the TSD length of the TE
    # store {query_name: tsd_len}
    tsd_path = temp_dir + '/' + species + '.tsd_info'
    tsd_info = {}
    for batch_member_file in batch_member_files:
        cur_query_name = batch_member_file[0]
        cur_member_file = batch_member_file[2]
        cur_contignames, cur_contigs = read_fasta(cur_member_file)
        # summarize the count of tsd_len, get the one occurs most times
        tsd_len_count = {}
        for query_name in cur_contignames:
            seq = cur_contigs[query_name]
            tir_start = flanking_len + 1
            tir_end = len(seq) - flanking_len
            tsd_search_distance = flanking_len
            cur_tsd, cur_tsd_len = search_confident_tsd(seq, tir_start, tir_end, tsd_search_distance, query_name, plant)
            if not tsd_len_count.__contains__(cur_tsd_len):
                tsd_len_count[cur_tsd_len] = (0, cur_tsd)
            cur_tsd_count = tsd_len_count[cur_tsd_len]
            cur_count = cur_tsd_count[0]
            cur_count += 1
            tsd_len_count[cur_tsd_len] = (cur_count, cur_tsd_count[1])
        final_tsd_len = 0
        final_tsd = ''
        max_count = 0
        for cur_tsd_len in tsd_len_count.keys():
            cur_tsd_count = tsd_len_count[cur_tsd_len]
            cur_count = cur_tsd_count[0]
            if cur_count > max_count:
                max_count = cur_count
                final_tsd_len = cur_tsd_len
                final_tsd = cur_tsd_count[1]
        tsd_info[cur_query_name] = (final_tsd, final_tsd_len)
    with codecs.open(tsd_path, 'w', encoding='utf-8') as f:
        json.dump(tsd_info, f)

    # Step4. store Repbase sequence with classification, species_name, and TSD sequence
    # 去掉processed_TE_path中存在重复的LTR，例如Copia-1_AA-intactLTR1和Copia-1_AA-intactLTR2，取其中具有合法TSD那个。两个都有，则随机去一个；两个都没有，优先取有TSD那个，否则随机取一个。
    # get all classification
    all_classification = set()
    names, contigs = read_fasta(processed_TE_path)
    final_repbase_path = temp_dir + '/' + species + '.ref'
    final_repbase_contigs = {}
    duplicate_ltr = set()
    for query_name in names:
        label_item = repbase_labels[query_name]
        if query_name.__contains__('intactLTR1'):
            other_ltr_name = query_name.replace('intactLTR1', 'intactLTR2')
            if tsd_info.__contains__(query_name):
                (final_tsd, final_tsd_len) = tsd_info[query_name]
            elif tsd_info.__contains__(other_ltr_name):
                (final_tsd, final_tsd_len) = tsd_info[other_ltr_name]
                query_name = other_ltr_name
            else:
                final_tsd = ''
                final_tsd_len = 0
        elif query_name.__contains__('intactLTR2'):
            other_ltr_name = query_name.replace('intactLTR2', 'intactLTR1')
            if tsd_info.__contains__(query_name):
                (final_tsd, final_tsd_len) = tsd_info[query_name]
            elif tsd_info.__contains__(other_ltr_name):
                (final_tsd, final_tsd_len) = tsd_info[other_ltr_name]
                query_name = other_ltr_name
            else:
                final_tsd = ''
                final_tsd_len = 0
        else:
            if tsd_info.__contains__(query_name):
                (final_tsd, final_tsd_len) = tsd_info[query_name]
            else:
                final_tsd = ''
                final_tsd_len = 0
        cur_prefix = query_name.split('-intactLTR')[0]
        if not duplicate_ltr.__contains__(cur_prefix):
            new_name = query_name + '\t' + label_item[0] + '\t' + label_item[1] + '\t' + 'TSD:' + str(final_tsd) + '\t' + 'TSD_len:' + str(final_tsd_len)
            duplicate_ltr.add(cur_prefix)
            final_repbase_contigs[new_name] = contigs[query_name]
            all_classification.add(label_item[0])
    store_fasta(final_repbase_contigs, final_repbase_path)
    #print(all_classification)
    return final_repbase_path

if __name__ == '__main__':
    # # read all repbase library, get all unique classification
    # valid_sequence_count = 0
    # repeat_seq_count = 0
    # all_repbase_contigs = {}
    # #work_dir = '/homeb/hukang/TE_Classification_test/curated_lib'
    # work_dir = '/public/home/hpc194701009/TE_Classification_test/curated_lib'
    # all_repbase_path = work_dir + '/all_repbase.ref'
    # temp_dir = work_dir + '/temp'
    # if not os.path.exists(temp_dir):
    #     os.makedirs(temp_dir)
    # all_classification = set()
    # repbase_dir = '/homeb/hukang/KmerRepFinder_test/library/curated_lib/RepBase28.06.fasta'
    # for filename in os.listdir(repbase_dir):
    #     file_path = repbase_dir + '/' + filename
    #     #print(file_path)
    #     if os.path.isfile(file_path):
    #         label_names, label_contigs = read_fasta_v1(file_path)
    #         for name in label_names:
    #             #print(name)
    #             parts = name.split('\t')
    #             if len(parts) < 3:
    #                 continue
    #             repbase_name = parts[0]
    #             classification = parts[1]
    #             all_classification.add(classification)
    #             valid_sequence_count += 1
    #             if all_repbase_contigs.__contains__(name):
    #                 repeat_seq_count += 1
    #                 #print(file_path, name)
    #             all_repbase_contigs[name] = label_contigs[name]
    #     else:
    #         for new_filename in os.listdir(file_path):
    #             new_file_path = file_path + '/' + new_filename
    #             if os.path.isfile(new_file_path):
    #                 label_names, label_contigs = read_fasta_v1(new_file_path)
    #                 for name in label_names:
    #                     parts = name.split('\t')
    #                     if len(parts) < 3:
    #                         continue
    #                     repbase_name = parts[0]
    #                     classification = parts[1]
    #                     all_classification.add(classification)
    #                     valid_sequence_count += 1
    #                     if all_repbase_contigs.__contains__(name):
    #                         repeat_seq_count += 1
    #                         #print(file_path, name)
    #                     all_repbase_contigs[name] = label_contigs[name]

    # transfer all classification label of Repbase To wicker's label, structures:
    # repbaseLabelToWickerLabel
    repbaseToWicker = {'Mariner/Tc1': 'Tc1-Mariner', 'mariner/Tc1 superfamily': 'Tc1-Mariner', 'hAT': 'hAT',
                       'HAT superfamily': 'hAT', 'MuDR': 'Mutator', 'Merlin': 'Merlin', 'Transib': 'Transib',
                       'P': 'P', 'P-element': 'P', 'PiggyBac': 'PiggyBac', 'Harbinger': 'PIF-Harbinger',
                       'EnSpm/CACTA': 'CACTA', 'Crypton': 'Crypton', 'CryptonF': 'Crypton', 'CryptonS': 'Crypton',
                       'CryptonI': 'Crypton', 'CryptonV': 'Crypton', 'CryptonA': 'Crypton', 'Helitron': 'Helitron',
                       'HELITRON superfamily': 'Helitron', 'Copia': 'Copia', 'Gypsy': 'Gypsy',
                       'GYPSY superfamily': 'Gypsy', 'Gypsy retrotransposon': 'Gypsy', 'BEL': 'Bel-Pao',
                       'ERV1': 'Retrovirus', 'ERV2': 'Retrovirus', 'ERV3': 'Retrovirus', 'ERV4': 'Retrovirus',
                       'Lentivirus': 'Retrovirus', 'Lokiretrovirus': 'Retrovirus', 'DIRS': 'DIRS',
                       'Penelope': 'Penelope', 'Penelope/Poseidon': 'Penelope', 'Neptune': 'Penelope',
                       'Nematis': 'Penelope', 'Athena': 'Penelope', 'Coprina': 'Penelope', 'Hydra': 'Penelope',
                       'Naiad/Chlamys': 'Penelope', 'R2': 'R2', 'RTE': 'RTE', 'Jockey': 'Jockey', 'L1': 'L1', 'I': 'I',
                       'SINE2/tRNA': 'tRNA', 'SINE1/7SL': '7SL', 'SINE3/5S': '5S'}
    # new_all_repbase_contigs = {}
    # new_all_classification = set()
    # all_species = set()
    # for name in all_repbase_contigs.keys():
    #     parts = name.split('\t')
    #     repbase_name = parts[0]
    #     classification = parts[1]
    #     species_name = parts[2]
    #     if repbaseToWicker.__contains__(classification):
    #         wicker_classification = repbaseToWicker[classification]
    #         new_name = repbase_name + '\t' + wicker_classification + '\t' + species_name
    #         new_all_repbase_contigs[new_name] = all_repbase_contigs[name]
    #         new_all_classification.add(wicker_classification)
    #         all_species.add(species_name)
    # store_fasta(new_all_repbase_contigs, all_repbase_path)
    # print('all unique wicker label:')
    # print(new_all_classification)
    # print(len(new_all_classification))
    #
    # print('all unique species name:')
    # print(all_species)
    # print(len(all_species))

    # #按照物种，统计序列
    # names, contigs = read_fasta_v1(all_repbase_path)
    # speices_TE_contigs = {}
    # species_TE_summary = {}
    # for name in names:
    #     parts = name.split('\t')
    #     species_name = parts[2]
    #     label = parts[1]
    #     TE_name = parts[0]
    #     if not species_TE_summary.__contains__(species_name):
    #         species_TE_summary[species_name] = {}
    #     label_TE_summary = species_TE_summary[species_name]
    #     if not label_TE_summary.__contains__(label):
    #         label_TE_summary[label] = 0
    #     label_count = label_TE_summary[label]
    #     label_count += 1
    #     label_TE_summary[label] = label_count
    #
    #     if not speices_TE_contigs.__contains__(species_name):
    #         speices_TE_contigs[species_name] = {}
    #     TE_contigs = speices_TE_contigs[species_name]
    #     TE_contigs[name] = contigs[name]
    # #print(species_TE_summary)
    # #print(len(species_TE_summary))
    #
    #
    # all_sequence_count = 0
    # species_count = []
    # species_count_dict = {}
    # for species_name in species_TE_summary:
    #     label_TE_summary = species_TE_summary[species_name]
    #     total_num = 0
    #     for label in label_TE_summary.keys():
    #         total_num += label_TE_summary[label]
    #     all_sequence_count += total_num
    #     species_count.append((species_name, total_num))
    #     species_count_dict[species_name] = total_num
    # species_count.sort(key=lambda x: -x[1])
    # #print('all sequence count:' + str(all_sequence_count))
    #
    # data = {}
    # species_names = []
    # TE_sequences_nums = []
    # for item in species_count:
    #     species_names.append(item[0])
    #     TE_sequences_nums.append(item[1])
    # data['Species Name'] = species_names
    # data['Total TE number'] = TE_sequences_nums
    #
    # df = pd.DataFrame(data)
    # # 将 DataFrame 存储到 Excel 文件中
    # with pd.ExcelWriter(temp_dir + '/data.xlsx', engine="openpyxl") as writer:
    #     to_excel_auto_column_weight(df, writer, f'novel TIR information')
    #
    # # 看看前top 100个物种中，有多少个物种我们实际上已经有基因组了
    # keep_species_names = set()
    # species_genome = {}
    # with open('ncbi_ref.info', 'r') as f_r:
    #     for line in f_r:
    #         if line.startswith('#'):
    #             continue
    #         species_name = line.split('\t')[2]
    #         genome = line.split('\t')[3]
    #         is_plant = line.split('\t')[5]
    #         species_genome[species_name] = (genome, is_plant)
    #         keep_species_names.add(species_name)
    #
    # top_num = 100
    # top_seq_count = 0
    # not_keep_species_names = []
    # for i, species_name in enumerate(species_names):
    #     if i > top_num:
    #         break
    #     if species_name not in keep_species_names:
    #         not_keep_species_names.append((species_name, species_count_dict[species_name]))
    #     top_seq_count += species_count_dict[species_name]
    # #print(not_keep_species_names)
    # #print(len(not_keep_species_names))
    # print('top num:' + str(top_num) + ', all sequence count:' + str(top_seq_count))
    #
    # # 把TE序列按照物种名称，存成不同的文件
    # species_dir = work_dir + '/species'
    # processed_species_dir = work_dir + '/species_processed'
    # if not os.path.exists(species_dir):
    #     os.makedirs(species_dir)
    # if not os.path.exists(processed_species_dir):
    #     os.makedirs(processed_species_dir)
    # species_TE_files = {}
    # for species_name in speices_TE_contigs.keys():
    #     TE_contigs = speices_TE_contigs[species_name]
    #     species = species_name.replace(' ', '_')
    #     species_TE_files[species_name] = species_dir + '/' + species + '.ref'
    #     store_fasta(TE_contigs, species_dir + '/' + species + '.ref')
    # for i, species_name in enumerate(species_names):
    #     if species_genome.__contains__(species_name):
    #         print('current species name:' + species_name)
    #         genome_path = species_genome[species_name][0]
    #         is_plant = int(species_genome[species_name][1])
    #
    #         species = species_name.replace(' ', '_')
    #         repbase_path = species_TE_files[species_name]
    #         print(repbase_path)
    #
    #         threads = 40
    #         flanking_len = 20
    #         final_repbase_path = getRepBaseTSDFromGenome(repbase_path, genome_path, temp_dir, threads, flanking_len, is_plant, species)
    #         os.system('mv ' + final_repbase_path + ' ' + processed_species_dir)

    # 统计训练集和测试集中不同TE类别的数量
    work_dir = '/home/hukang/NeuralTE/data'
    train_path = work_dir + '/repbase_train.ref'
    test_path = work_dir + '/repbase_test.ref'
    train_names, train_contigs = read_fasta_v1(train_path)
    test_names, test_contigs = read_fasta_v1(test_path)
    train_class_num = {}
    species_set = set()
    for name in test_names:
        class_name = name.split('\t')[1]
        species_name = name.split('\t')[2]
        if not train_class_num.__contains__(class_name):
            train_class_num[class_name] = 0
        class_num = train_class_num[class_name]
        train_class_num[class_name] = class_num + 1
        species_set.add(species_name)
    print(train_class_num)
    print(len(species_set))

    # #统计train里面序列长度超过15K的有多少
    # K15 = 15000
    # count = 0
    # for name in train_names:
    #     seq = train_contigs[name]
    #     seq_len = len(seq)
    #     if seq_len >= K15:
    #         count += 1
    # print(count)
