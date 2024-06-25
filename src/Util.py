import json
import os
import random
import re
import subprocess
import time
import logging
from logging import handlers
from fuzzysearch import find_near_matches
from concurrent.futures import ProcessPoolExecutor, as_completed

class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }

    def __init__(self,filename,level='info',when='D',backCount=3,fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)
        self.logger.setLevel(self.level_relations.get(level))
        sh = logging.StreamHandler()
        sh.setFormatter(format_str)
        th = handlers.TimedRotatingFileHandler(filename=filename,when=when,backupCount=backCount,encoding='utf-8')
        th.setFormatter(format_str)
        self.logger.addHandler(sh)
        self.logger.addHandler(th)

def multi_line(fasta_path, line_len):
    tmp_fasta_path = fasta_path + ".tmp"
    contigNames, contigs = read_fasta(fasta_path)
    with open(tmp_fasta_path, 'w') as f_w:
        for contigName in contigNames:
            contig = contigs[contigName]
            # line = '>' + contigName + '\t' + contig + '\n'
            # f_w.write(line)
            start = 0
            end = len(contig)
            while start < end:
                # add extra kmer length
                seg = contig[start:start+line_len]
                line = '>' + contigName + '\t' + str(start) + '\t' + seg + '\n'
                f_w.write(line)
                start += line_len
    f_w.close()
    return tmp_fasta_path


def save_dict_to_fasta(data_dict, fasta_file):
    with open(fasta_file, 'w') as file:
        for identifier, sequence in data_dict.items():
            # 写入FASTA头部
            file.write(f">{identifier}\n")

            # 分割序列以确保每行不超过70个字符
            seq_lines = [sequence[i:i + 70] for i in range(0, len(sequence), 70)]

            # 写入序列数据
            for line in seq_lines:
                file.write(line + "\n")

def split_dict_into_blocks(chromosomes_dict, threads):
    total_length = sum(len(seq) for seq in chromosomes_dict.values())
    target_length = total_length // threads

    blocks = []
    current_block = {}
    current_length = 0

    for chrom, seq in chromosomes_dict.items():
        current_block[chrom] = seq
        current_length += len(seq)

        if current_length >= target_length:
            blocks.append(current_block)
            current_block = {}
            current_length = 0

    if current_block:
        blocks.append(current_block)

    return blocks

def convertToUpperCase_v1(reference):
    contigNames = []
    contigs = {}
    with open(reference, "r") as f_r:
        contigName = ''
        contigseq = ''
        for line in f_r:
            if line.startswith('>'):
                if contigName != '' and contigseq != '':
                    contigs[contigName] = contigseq
                    contigNames.append(contigName)
                contigName = line.strip()[1:].split(' ')[0]
                contigseq = ''
            else:
                contigseq += line.strip().upper()
        contigs[contigName] = contigseq
        contigNames.append(contigName)
    f_r.close()

    # (dir, filename) = os.path.split(reference)
    # (name, extension) = os.path.splitext(filename)
    # reference_pre = dir + '/' + name + '_preprocess' + extension
    with open(reference, "w") as f_save:
        for contigName in contigNames:
            contigseq = contigs[contigName]
            f_save.write(">" + contigName + '\n' + contigseq + '\n')
    f_save.close()
    return reference

def read_scn(scn_file, log, remove_dup=False):
    ltr_candidates = {}
    ltr_lines = {}
    candidate_index = 0
    existing_records = set()
    remove_count = 0
    total_lines = 0
    with open(scn_file, 'r') as f_r:
        for line in f_r:
            if line.startswith('#') or line.strip() == '':
                continue
            line = line.replace('\n', '')
            parts = line.split(' ')
            ltr_start = int(parts[0])
            ltr_end = int(parts[1])
            chr_name = parts[11]
            total_lines += 1
            if remove_dup:
                cur_record = (ltr_start, ltr_end, chr_name)
                # 过滤掉冗余的记录
                if cur_record in existing_records:
                    remove_count += 1
                    continue
                existing_records.add(cur_record)
            left_ltr_start = int(parts[3])
            left_ltr_end = int(parts[4])
            right_ltr_start = int(parts[6])
            right_ltr_end = int(parts[7])

            ltr_candidates[candidate_index] = (chr_name, left_ltr_start, left_ltr_end, right_ltr_start, right_ltr_end)
            ltr_lines[candidate_index] = line
            candidate_index += 1
    if remove_dup:
        log.logger.debug('Total LTR num: ' + str(total_lines))
        log.logger.debug('Remove ' + str(remove_count) + ' replicate LTR in scn file, remaining LTR num: ' + str(len(ltr_candidates)))
    return ltr_candidates, ltr_lines

def store_scn(confident_lines, confident_scn):
    with open(confident_scn, 'w') as f_save:
        for line in confident_lines:
            f_save.write(line + '\n')

def rename_fasta(input, output, header='N'):
    names, contigs = read_fasta(input)
    node_index = 0
    with open(output, 'w') as f_save:
        for name in names:
            seq = contigs[name]
            f_save.write('>'+header+'_'+str(node_index)+'\n'+seq+'\n')
            node_index += 1
    f_save.close()

def read_fasta(fasta_path):
    contignames = []
    contigs = {}
    if os.path.exists(fasta_path):
        with open(fasta_path, 'r') as rf:
            contigname = ''
            contigseq = ''
            for line in rf:
                if line.startswith('>'):
                    if contigname != '' and contigseq != '':
                        contigs[contigname] = contigseq
                        contignames.append(contigname)
                    contigname = line.strip()[1:].split(" ")[0].split('\t')[0]
                    contigseq = ''
                else:
                    contigseq += line.strip().upper()
            if contigname != '' and contigseq != '':
                contigs[contigname] = contigseq
                contignames.append(contigname)
        rf.close()
    return contignames, contigs

def read_fasta_v1(fasta_path):
    contignames = []
    contigs = {}
    if os.path.exists(fasta_path):
        with open(fasta_path, 'r') as rf:
            contigname = ''
            contigseq = ''
            for line in rf:
                if line.startswith('>'):
                    if contigname != '' and contigseq != '':
                        contigs[contigname] = contigseq
                        contignames.append(contigname)
                    contigname = line.strip()[1:]
                    contigseq = ''
                else:
                    contigseq += line.strip().upper()
            if contigname != '' and contigseq != '':
                contigs[contigname] = contigseq
                contignames.append(contigname)
        rf.close()
    return contignames, contigs

def store_fasta(contigs, file_path):
    with open(file_path, 'w') as f_save:
        for name in contigs.keys():
            seq = contigs[name]
            f_save.write('>'+name+'\n'+seq+'\n')
    f_save.close()

def get_full_length_copies_batch(confident_ltr_internal, split_ref_dir, threads, temp_dir):
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    batch_size = 10
    batch_id = 0
    names, contigs = read_fasta(confident_ltr_internal)
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

    ex = ProcessPoolExecutor(threads)
    jobs = []
    for cur_split_files in split_files:
        job = ex.submit(get_full_length_copies, cur_split_files, split_ref_dir, debug=0)
        jobs.append(job)
    ex.shutdown(wait=True)
    all_copies = {}
    for job in as_completed(jobs):
        cur_all_copies = job.result()
        all_copies.update(cur_all_copies)
    return all_copies


def flank_region_align_v5(candidate_sequence_path, flanking_len, reference, split_ref_dir, TE_type, tmp_output_dir, threads, ref_index, log, subset_script_path, plant, debug, iter_num, result_type='cons'):
    log.logger.info('------Determination of homology in regions outside the boundaries of ' + TE_type + ' copies')
    starttime = time.time()
    temp_dir = tmp_output_dir + '/' + TE_type + '_copies_' + str(ref_index) + '_' + str(iter_num)
    os.system('rm -rf ' + temp_dir)
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # We are considering that the current running time is too long, maybe it is related to submitting one sequence for Blastn alignment at a time.
    # We will try to combine 10 sequences together and run Blastn once.
    # To increase CPU utilization, we will submit one thread to process 10 sequences.
    batch_size = 10
    batch_id = 0
    names, contigs = read_fasta(candidate_sequence_path)
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

    ref_names, ref_contigs = read_fasta(reference)
    ex = ProcessPoolExecutor(threads)
    jobs = []
    for cur_split_files in split_files:
        job = ex.submit(get_full_length_copies, cur_split_files, split_ref_dir, debug)
        jobs.append(job)
    ex.shutdown(wait=True)
    all_copies = {}
    for job in as_completed(jobs):
        cur_all_copies = job.result()
        all_copies.update(cur_all_copies)
    # extend copies
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
    for query_name in new_all_copies.keys():
        copy_contigs = new_all_copies[query_name]
        cur_member_file = temp_dir + '/' + query_name + '.blast.bed.fa'
        store_fasta(copy_contigs, cur_member_file)
        query_seq = contigs[query_name]
        batch_member_files.append((query_name, query_seq, cur_member_file))

    # Determine whether the multiple sequence alignment of each copied file satisfies the homology rule
    ex = ProcessPoolExecutor(threads)
    jobs = []
    for batch_member_file in batch_member_files:
        job = ex.submit(run_find_members_v8, batch_member_file, temp_dir, subset_script_path,
                        plant, TE_type, debug, result_type)
        jobs.append(job)
    ex.shutdown(wait=True)

    true_tes = {}
    for job in as_completed(jobs):
        cur_name, is_TE = job.result()
        true_tes[cur_name] = is_TE

    if debug != 1:
        os.system('rm -rf ' + temp_dir)
    endtime = time.time()
    dtime = endtime - starttime
    log.logger.info("Running time of determination of homology in regions outside the boundaries of  " + TE_type + " copies: %.8s s" % (dtime))
    return true_tes

def get_full_length_copies(query_path, split_ref_dir, debug):
    blastn2Results_path = query_path + '.blast.out'
    repeats_path = (query_path, split_ref_dir, blastn2Results_path)
    all_copies = multiple_alignment_blast_and_get_copies_v1(repeats_path)
    if debug != 1:
        os.remove(blastn2Results_path)
    return all_copies

def multiple_alignment_blast_and_get_copies_v1(repeats_path):
    split_repeats_path = repeats_path[0]
    split_ref_dir = repeats_path[1]
    blastn2Results_path = repeats_path[2]
    os.system('rm -f ' + blastn2Results_path)
    all_copies = {}
    repeat_names, repeat_contigs = read_fasta(split_repeats_path)
    remain_contigs = repeat_contigs
    for chr_name in os.listdir(split_ref_dir):
        if len(remain_contigs) > 0:
            if not str(chr_name).endswith('.fa'):
                continue
            chr_path = split_ref_dir + '/' + chr_name
            align_command = 'blastn -db ' + chr_path + ' -num_threads ' \
                            + str(1) + ' -query ' + split_repeats_path + ' -evalue 1e-20 -outfmt 6 > ' + blastn2Results_path
            os.system(align_command)
            # 由于我们只需要100个拷贝，因此如果有序列已经满足了，就不需要进行后续的比对了，这样在mouse这样的高拷贝大基因组上减少运行时间
            cur_all_copies = get_copies_v1(blastn2Results_path, split_repeats_path, '')
            for query_name in cur_all_copies.keys():
                copy_list = cur_all_copies[query_name]
                if query_name in all_copies:
                    prev_copy_list = all_copies[query_name]
                else:
                    prev_copy_list = []
                update_copy_list = prev_copy_list + copy_list
                all_copies[query_name] = update_copy_list
                if len(update_copy_list) >= 100:
                    del repeat_contigs[query_name]
            remain_contigs = repeat_contigs
            store_fasta(remain_contigs, split_repeats_path)

    # all_copies = get_copies_v1(blastn2Results_path, split_repeats_path, '')
    return all_copies

def get_copies_v1(blastnResults_path, query_path, subject_path, query_coverage=0.95, subject_coverage=0):
    query_records = {}
    with open(blastnResults_path, 'r') as f_r:
        for idx, line in enumerate(f_r):
            parts = line.split('\t')
            query_name = parts[0]
            subject_name = parts[1]
            identity = float(parts[2])
            alignment_len = int(parts[3])
            q_start = int(parts[6])
            q_end = int(parts[7])
            s_start = int(parts[8])
            s_end = int(parts[9])
            if query_name == subject_name:
                continue
            if not query_records.__contains__(query_name):
                query_records[query_name] = {}
            subject_dict = query_records[query_name]

            if not subject_dict.__contains__(subject_name):
                subject_dict[subject_name] = []
            subject_pos = subject_dict[subject_name]
            subject_pos.append((q_start, q_end, s_start, s_end, identity))
    f_r.close()

    query_names, query_contigs = read_fasta(query_path)
    cur_segments = list(query_records.items())
    all_copies = get_query_copies(cur_segments, query_contigs, subject_path, query_coverage, subject_coverage)

    return all_copies

def get_query_copies(cur_segments, query_contigs, subject_path, query_coverage, subject_coverage, query_fixed_extend_base_threshold=200, subject_fixed_extend_base_threshold=200, max_copy_num=100):
    all_copies = {}

    if subject_coverage > 0:
        subject_names, subject_contigs = read_fasta(subject_path)

    for item in cur_segments:
        query_name = item[0]
        subject_dict = item[1]

        longest_queries = []
        for subject_name in subject_dict.keys():
            subject_pos = subject_dict[subject_name]

            # cluster all closed fragments, split forward and reverse records
            forward_pos = []
            reverse_pos = []
            for pos_item in subject_pos:
                if pos_item[2] > pos_item[3]:
                    reverse_pos.append(pos_item)
                else:
                    forward_pos.append(pos_item)
            forward_pos.sort(key=lambda x: (x[2], x[3]))
            reverse_pos.sort(key=lambda x: (-x[2], -x[3]))

            clusters = {}
            cluster_index = 0
            for k, frag in enumerate(forward_pos):
                if not clusters.__contains__(cluster_index):
                    clusters[cluster_index] = []
                cur_cluster = clusters[cluster_index]
                if k == 0:
                    cur_cluster.append(frag)
                else:
                    is_closed = False
                    for exist_frag in reversed(cur_cluster):
                        if (frag[2] - exist_frag[3] < subject_fixed_extend_base_threshold and frag[1] > exist_frag[1]):
                            is_closed = True
                            break
                    if is_closed:
                        cur_cluster.append(frag)
                    else:
                        cluster_index += 1
                        if not clusters.__contains__(cluster_index):
                            clusters[cluster_index] = []
                        cur_cluster = clusters[cluster_index]
                        cur_cluster.append(frag)

            cluster_index += 1
            for k, frag in enumerate(reverse_pos):
                if not clusters.__contains__(cluster_index):
                    clusters[cluster_index] = []
                cur_cluster = clusters[cluster_index]
                if k == 0:
                    cur_cluster.append(frag)
                else:
                    is_closed = False
                    for exist_frag in reversed(cur_cluster):
                        if (exist_frag[3] - frag[2] < subject_fixed_extend_base_threshold and frag[1] > exist_frag[1]):
                            is_closed = True
                            break
                    if is_closed:
                        cur_cluster.append(frag)
                    else:
                        cluster_index += 1
                        if not clusters.__contains__(cluster_index):
                            clusters[cluster_index] = []
                        cur_cluster = clusters[cluster_index]
                        cur_cluster.append(frag)

            for cluster_index in clusters.keys():
                cur_cluster = clusters[cluster_index]
                cur_cluster.sort(key=lambda x: (x[0], x[1]))

                cluster_longest_query_start = -1
                cluster_longest_query_end = -1
                cluster_longest_query_len = -1

                cluster_longest_subject_start = -1
                cluster_longest_subject_end = -1
                cluster_longest_subject_len = -1

                cluster_identity = 0
                cluster_extend_num = 0

                # record visited fragments
                visited_frag = {}
                for i in range(len(cur_cluster)):
                    # keep a longest query start from each fragment
                    origin_frag = cur_cluster[i]
                    if visited_frag.__contains__(origin_frag):
                        continue
                    cur_frag_len = origin_frag[1] - origin_frag[0] + 1
                    cur_longest_query_len = cur_frag_len
                    longest_query_start = origin_frag[0]
                    longest_query_end = origin_frag[1]
                    longest_subject_start = origin_frag[2]
                    longest_subject_end = origin_frag[3]

                    cur_identity = origin_frag[4]
                    cur_extend_num = 0

                    visited_frag[origin_frag] = 1
                    # try to extend query
                    for j in range(i + 1, len(cur_cluster)):
                        ext_frag = cur_cluster[j]
                        if visited_frag.__contains__(ext_frag):
                            continue

                        # could extend
                        # extend right
                        if ext_frag[1] > longest_query_end:
                            # judge subject direction
                            if longest_subject_start < longest_subject_end and ext_frag[2] < ext_frag[3]:
                                # +
                                if ext_frag[3] > longest_subject_end:
                                    # forward extend
                                    if ext_frag[0] - longest_query_end < query_fixed_extend_base_threshold and ext_frag[
                                        2] - longest_subject_end < subject_fixed_extend_base_threshold:
                                        # update the longest path
                                        longest_query_start = longest_query_start
                                        longest_query_end = ext_frag[1]
                                        longest_subject_start = longest_subject_start if longest_subject_start < \
                                                                                         ext_frag[
                                                                                             2] else ext_frag[2]
                                        longest_subject_end = ext_frag[3]
                                        cur_longest_query_len = longest_query_end - longest_query_start

                                        cur_identity += ext_frag[4]
                                        cur_extend_num += 1
                                        visited_frag[ext_frag] = 1
                                    elif ext_frag[0] - longest_query_end >= query_fixed_extend_base_threshold:
                                        break
                            elif longest_subject_start > longest_subject_end and ext_frag[2] > ext_frag[3]:
                                # reverse
                                if ext_frag[3] < longest_subject_end:
                                    # reverse extend
                                    if ext_frag[
                                        0] - longest_query_end < query_fixed_extend_base_threshold and longest_subject_end - \
                                            ext_frag[2] < subject_fixed_extend_base_threshold:
                                        # update the longest path
                                        longest_query_start = longest_query_start
                                        longest_query_end = ext_frag[1]
                                        longest_subject_start = longest_subject_start if longest_subject_start > \
                                                                                         ext_frag[
                                                                                             2] else ext_frag[2]
                                        longest_subject_end = ext_frag[3]
                                        cur_longest_query_len = longest_query_end - longest_query_start

                                        cur_identity += ext_frag[4]
                                        cur_extend_num += 1
                                        visited_frag[ext_frag] = 1
                                    elif ext_frag[0] - longest_query_end >= query_fixed_extend_base_threshold:
                                        break
                    if cur_longest_query_len > cluster_longest_query_len:
                        cluster_longest_query_start = longest_query_start
                        cluster_longest_query_end = longest_query_end
                        cluster_longest_query_len = cur_longest_query_len

                        cluster_longest_subject_start = longest_subject_start
                        cluster_longest_subject_end = longest_subject_end
                        cluster_longest_subject_len = abs(longest_subject_end - longest_subject_start) + 1

                        cluster_identity = cur_identity
                        cluster_extend_num = cur_extend_num
                # keep this longest query
                if cluster_longest_query_len != -1:
                    longest_queries.append((cluster_longest_query_start, cluster_longest_query_end,
                                            cluster_longest_query_len, cluster_longest_subject_start,
                                            cluster_longest_subject_end, cluster_longest_subject_len, subject_name,
                                            cluster_extend_num, cluster_identity))

        longest_queries.sort(key=lambda x: -x[2])
        query_len = len(query_contigs[query_name])
        # query_len = int(query_name.split('-')[1].split('_')[1])
        copies = []
        keeped_copies = set()
        for query in longest_queries:
            if len(copies) > max_copy_num:
                break
            subject_name = query[6]
            subject_start = query[3]
            subject_end = query[4]
            direct = '+'
            if subject_start > subject_end:
                tmp = subject_start
                subject_start = subject_end
                subject_end = tmp
                direct = '-'
            item = (subject_name, subject_start, subject_end)
            if subject_coverage > 0:
                subject_len = len(subject_contigs[subject_name])
                cur_subject_coverage = float(query[5])/subject_len
                if float(query[2])/query_len >= query_coverage and cur_subject_coverage >= subject_coverage and item not in keeped_copies:
                    copies.append((subject_name, subject_start, subject_end, query[2], direct))
                    keeped_copies.add(item)
            else:
                if float(query[2]) / query_len >= query_coverage and item not in keeped_copies:
                    copies.append((subject_name, subject_start, subject_end, query[2], direct))
                    keeped_copies.add(item)
        #copies.sort(key=lambda x: abs(x[3]-(x[2]-x[1]+1)))
        all_copies[query_name] = copies
    return all_copies


def generate_left_frame_from_seq(candidate_sequence_path, reference, threads, temp_dir, output_dir, split_ref_dir):
    debug = 0
    flanking_len = 50
    starttime = time.time()
    if os.path.exists(temp_dir):
        os.system('rm -rf ' + temp_dir)
    os.makedirs(temp_dir)
    if os.path.exists(output_dir):
        os.system('rm -rf ' + output_dir)
    os.makedirs(output_dir)

    # We are considering that the current running time is too long, maybe it is related to submitting one sequence for Blastn alignment at a time.
    # We will try to combine 10 sequences together and run Blastn once.
    # To increase CPU utilization, we will submit one thread to process 10 sequences.
    batch_size = 10
    batch_id = 0
    names, contigs = read_fasta(candidate_sequence_path)
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

    ref_names, ref_contigs = read_fasta(reference)
    ex = ProcessPoolExecutor(threads)
    jobs = []
    for cur_split_files in split_files:
        job = ex.submit(get_full_length_copies, cur_split_files, split_ref_dir, debug)
        jobs.append(job)
    ex.shutdown(wait=True)
    all_copies = {}
    for job in as_completed(jobs):
        cur_all_copies = job.result()
        all_copies.update(cur_all_copies)
    # extend copies
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
    for query_name in new_all_copies.keys():
        copy_contigs = new_all_copies[query_name]
        cur_member_file = temp_dir + '/' + query_name + '.blast.bed.fa'
        store_fasta(copy_contigs, cur_member_file)
        query_seq = contigs[query_name]
        batch_member_files.append((query_name, query_seq, cur_member_file))


    subset_script_path = os.getcwd() + '/tools/ready_for_MSA.sh'
    # Determine whether the multiple sequence alignment of each copied file satisfies the homology rule
    ex = ProcessPoolExecutor(threads)
    jobs = []
    for batch_member_file in batch_member_files:
        job = ex.submit(generate_msa, batch_member_file, temp_dir, output_dir, subset_script_path, debug)
        jobs.append(job)
    ex.shutdown(wait=True)

    all_left_frames = []
    for job in as_completed(jobs):
        left_frame_path = job.result()
        all_left_frames.append(left_frame_path)

    endtime = time.time()
    dtime = endtime - starttime
    return all_left_frames

def generate_both_ends_frame_from_seq(candidate_sequence_path, reference, threads, temp_dir, output_dir, split_ref_dir):
    debug = 0
    flanking_len = 50
    starttime = time.time()
    if os.path.exists(temp_dir):
        os.system('rm -rf ' + temp_dir)
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    if os.path.exists(output_dir):
        os.system('rm -rf ' + output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # We are considering that the current running time is too long, maybe it is related to submitting one sequence for Blastn alignment at a time.
    # We will try to combine 10 sequences together and run Blastn once.
    # To increase CPU utilization, we will submit one thread to process 10 sequences.
    batch_size = 10
    batch_id = 0
    names, contigs = read_fasta(candidate_sequence_path)
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

    ref_names, ref_contigs = read_fasta(reference)
    ex = ProcessPoolExecutor(threads)
    jobs = []
    for cur_split_files in split_files:
        job = ex.submit(get_full_length_copies, cur_split_files, split_ref_dir, debug)
        jobs.append(job)
    ex.shutdown(wait=True)
    all_copies = {}
    for job in as_completed(jobs):
        cur_all_copies = job.result()
        all_copies.update(cur_all_copies)
    # extend copies
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
    for query_name in new_all_copies.keys():
        copy_contigs = new_all_copies[query_name]
        cur_member_file = temp_dir + '/' + query_name + '.blast.bed.fa'
        store_fasta(copy_contigs, cur_member_file)
        batch_member_files.append((query_name, cur_member_file))

    # subset_script_path = config.project_dir + '/tools/ready_for_MSA.sh'
    # Determine whether the multiple sequence alignment of each copied file satisfies the homology rule
    ex = ProcessPoolExecutor(threads)
    jobs = []
    for batch_member_file in batch_member_files:
        job = ex.submit(generate_msa, batch_member_file, temp_dir, output_dir, debug)
        jobs.append(job)
    ex.shutdown(wait=True)

    for job in as_completed(jobs):
        left_frame_path = job.result()

    endtime = time.time()
    dtime = endtime - starttime

def merge_terminals(confident_ltr_terminal_cons, threads):
    blastn_command = 'blastn -query ' + confident_ltr_terminal_cons + ' -subject ' + confident_ltr_terminal_cons + ' -num_threads ' + str(threads) + ' -outfmt 6 '
    contignames, contigs = read_fasta(confident_ltr_terminal_cons)

    result = subprocess.run(blastn_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            executable='/bin/bash')

    remove_frag_ltr = set()
    duplicate_records = set()
    if result.returncode == 0:
        lines = result.stdout.split('\n')
        for line in lines:
            parts = line.split('\t')
            if len(parts) != 12:
                continue
            query_name = parts[0]
            subject_name = parts[1]
            if query_name == subject_name:
                continue
            query_len = len(contigs[query_name])
            subject_len = len(contigs[subject_name])

            query_start = int(parts[6])
            query_end = int(parts[7])
            if abs(query_end - query_start) / query_len >= 0.95:
                if (query_name, subject_name) not in duplicate_records and (subject_name, query_name) not in duplicate_records:
                    if query_len < subject_len:
                        remove_frag_ltr.add(query_name)
                    else:
                        remove_frag_ltr.add(subject_name)
                    duplicate_records.add((query_name, subject_name))
                    duplicate_records.add((subject_name, query_name))

    print('remove fragmented LTR num: ' + str(len(remove_frag_ltr)))

    for ltr_name in remove_frag_ltr:
        del contigs[ltr_name]
    store_fasta(contigs, confident_ltr_terminal_cons)



def extract_copies(member_file, max_num):
    contignames, contigs = read_fasta(member_file)
    new_contigs = {}
    for name in contignames[:max_num]:
        new_contigs[name] = contigs[name]
    # sorted_contigs = dict(sorted(contigs.items(), key=lambda item: len(item[1]))[:max_num])
    member_file += '.rdmSubset.fa'
    store_fasta(new_contigs, member_file)
    return member_file

def generate_msa(batch_member_file, temp_dir, output_dir, debug):
    (query_name, member_file) = batch_member_file

    member_names, member_contigs = read_fasta(member_file)
    if len(member_names) > 100:
        # 抽取100条最长的 拷贝
        max_num = 100
        member_file = extract_copies(member_file, max_num)
    if not os.path.exists(member_file):
        return (query_name, False)
    align_file = member_file + '.maf.fa'
    align_command = 'cd ' + temp_dir + ' && mafft --preservecase --quiet --thread 1 ' + member_file + ' > ' + align_file
    os.system(align_command)
    # left_frame_path = get_left_frame(query_name, cur_seq, align_file, output_dir, debug)
    if len(member_names) >= 1:
        cur_seq = member_contigs[member_names[0]][50:-50]
        both_end_frame_path = get_both_ends_frame(query_name, cur_seq, align_file, output_dir, debug)
    else:
        both_end_frame_path = ''
    return both_end_frame_path

def get_both_ends_frame(query_name, cur_seq, align_file, output_dir, debug):
    anchor_len = 20
    first_10bp = cur_seq[0:anchor_len]
    last_10bp = cur_seq[-anchor_len:]
    align_names, align_contigs = read_fasta(align_file)
    align_start = -1
    align_end = -1
    for name in align_names:
        raw_align_seq = align_contigs[name]
        align_seq = ''
        position_reflex = {}
        cur_align_index = 0
        for i, base in enumerate(raw_align_seq):
            if base == '-':
                continue
            else:
                align_seq += base
                position_reflex[cur_align_index] = i
                cur_align_index += 1

        start_dist = 2
        last_dist = 2
        first_matches = find_near_matches(first_10bp, align_seq, max_l_dist=start_dist)
        last_matches = find_near_matches(last_10bp, align_seq, max_l_dist=last_dist)
        last_matches = last_matches[::-1]
        if len(first_matches) > 0 and len(last_matches) > 0:
            align_no_gap_start = first_matches[0].start
            align_no_gap_end = last_matches[0].end - 1
            align_start = position_reflex[align_no_gap_start]
            align_end = position_reflex[align_no_gap_end]
            break
    if debug:
        print(align_file, align_start, align_end)
    if align_start == -1 or align_end == -1:
        if debug:
            print('not found boundary:' + align_file)
        return False

    align_names, align_contigs = read_fasta(align_file)
    if len(align_names) <= 0:
        if debug:
            print('align file size = 0, ' + align_file)
        return False

    # 取 align_start 的外侧 100 bp
    out_threshold = 100
    start_align_file = output_dir + '/' + query_name + '.matrix'
    with open(start_align_file, 'w') as f_save:
        for name in align_names:
            raw_align_seq = align_contigs[name]
            # 取左侧 100 bp frame
            if align_start - out_threshold < 0:
                start_pos = 0
            else:
                start_pos = align_start - out_threshold
            start_seq = raw_align_seq[start_pos: align_start]
            # 补全不足 100 bp的部分
            seq1 = '-' * (out_threshold - len(start_seq)) + start_seq

            # 取右侧侧 100 bp frame
            if align_end + out_threshold > len(raw_align_seq):
                end_pos = len(raw_align_seq)
            else:
                end_pos = align_end + out_threshold
            end_seq = raw_align_seq[align_end: end_pos]
            # 补全不足 100 bp的部分
            seq2 = end_seq + '-' * (out_threshold - len(end_seq))

            f_save.write(seq1+'\t'+seq2+'\n')
    return start_align_file

def get_left_frame(query_name, cur_seq, align_file, output_dir, debug):
    anchor_len = 20
    first_10bp = cur_seq[0:anchor_len]
    last_10bp = cur_seq[-anchor_len:]
    align_names, align_contigs = read_fasta(align_file)
    align_start = -1
    align_end = -1
    for name in align_names:
        raw_align_seq = align_contigs[name]
        align_seq = ''
        position_reflex = {}
        cur_align_index = 0
        for i, base in enumerate(raw_align_seq):
            if base == '-':
                continue
            else:
                align_seq += base
                position_reflex[cur_align_index] = i
                cur_align_index += 1

        start_dist = 2
        last_dist = 2
        first_matches = find_near_matches(first_10bp, align_seq, max_l_dist=start_dist)
        last_matches = find_near_matches(last_10bp, align_seq, max_l_dist=last_dist)
        last_matches = last_matches[::-1]
        if len(first_matches) > 0 and len(last_matches) > 0:
            align_no_gap_start = first_matches[0].start
            align_no_gap_end = last_matches[0].end - 1
            align_start = position_reflex[align_no_gap_start]
            align_end = position_reflex[align_no_gap_end]
            break
    if debug:
        print(align_file, align_start, align_end)
    if align_start == -1 or align_end == -1:
        if debug:
            print('not found boundary:' + align_file)
        return False

    align_names, align_contigs = read_fasta(align_file)
    if len(align_names) <= 0:
        if debug:
            print('align file size = 0, ' + align_file)
        return False

    # 取 align_start 的外侧 100 bp
    out_threshold = 100
    start_align_file = output_dir + '/' + query_name + '.matrix'
    with open(start_align_file, 'w') as f_save:
        for name in align_names:
            raw_align_seq = align_contigs[name]
            if align_start - out_threshold < 0:
                start_pos = 0
            else:
                start_pos = align_start - out_threshold
            start_seq = raw_align_seq[start_pos: align_start]
            # 补全不足 100 bp的部分
            seq = '-' * (out_threshold - len(start_seq)) + start_seq
            f_save.write(seq+'\n')
    return start_align_file





def run_find_members_v8(batch_member_file, temp_dir, subset_script_path, plant, TE_type, debug, result_type):
    (query_name, cur_seq, member_file) = batch_member_file

    member_names, member_contigs = read_fasta(member_file)
    if len(member_names) > 100:
        sub_command = 'cd ' + temp_dir + ' && sh ' + subset_script_path + ' ' + member_file + ' 100 100 ' + ' > /dev/null 2>&1'
        os.system(sub_command)
        member_file += '.rdmSubset.fa'
    if not os.path.exists(member_file):
        return (query_name, False)
    align_file = member_file + '.maf.fa'
    align_command = 'cd ' + temp_dir + ' && mafft --preservecase --quiet --thread 1 ' + member_file + ' > ' + align_file
    os.system(align_command)

    is_TE = judge_boundary_v9(cur_seq, align_file, debug, TE_type, plant, result_type)
    return query_name, is_TE


def judge_boundary_v9(cur_seq, align_file, debug, TE_type, plant, result_type):
    # 1. Based on the 'remove gap' multi-alignment file, locate the position of the original sequence (anchor point).
    #     # Extend 20bp on both sides from the anchor point, extract the effective columns, and determine their homology.
    #     If it contradicts our rule, it is a false positive sequence.
    #     # --First, locate the TIR boundary position of the first sequence in the alignment file as the anchor point.
    #     # Take the first and last 20bp of the original sequence, and search on the aligned sequence without gaps.

    anchor_len = 20
    first_10bp = cur_seq[0:anchor_len]
    last_10bp = cur_seq[-anchor_len:]
    align_names, align_contigs = read_fasta(align_file)
    align_start = -1
    align_end = -1
    for name in align_names:
        raw_align_seq = align_contigs[name]
        align_seq = ''
        position_reflex = {}
        cur_align_index = 0
        for i, base in enumerate(raw_align_seq):
            if base == '-':
                continue
            else:
                align_seq += base
                position_reflex[cur_align_index] = i
                cur_align_index += 1

        start_dist = 2
        last_dist = 2
        first_matches = find_near_matches(first_10bp, align_seq, max_l_dist=start_dist)
        last_matches = find_near_matches(last_10bp, align_seq, max_l_dist=last_dist)
        last_matches = last_matches[::-1]
        if len(first_matches) > 0 and len(last_matches) > 0:
            align_no_gap_start = first_matches[0].start
            align_no_gap_end = last_matches[0].end - 1
            align_start = position_reflex[align_no_gap_start]
            align_end = position_reflex[align_no_gap_end]
            break
    if debug:
        print(align_file, align_start, align_end)
    if align_start == -1 or align_end == -1:
        if debug:
            print('not found boundary:' + align_file)
        return False

    align_names, align_contigs = read_fasta(align_file)
    if len(align_names) <= 0:
        if debug:
            print('align file size = 0, ' + align_file)
        return False

    # 3. Take the full-length sequence to generate a consensus sequence.
    # There should be bases both up and down by 10bp at the anchor point.
    full_length_member_names = []
    full_length_member_contigs = {}
    anchor_len = 10
    for name in align_names:
        if len(full_length_member_names) > 100:
            break
        align_seq = align_contigs[name]
        if align_start - anchor_len >= 0:
            anchor_start = align_start - anchor_len
        else:
            anchor_start = 0
        anchor_start_seq = align_seq[anchor_start: align_start + anchor_len]
        if align_end + anchor_len < len(align_seq):
            anchor_end = align_end + anchor_len
        else:
            anchor_end = len(align_seq)
        anchor_end_seq = align_seq[align_end - anchor_len: anchor_end]

        if not all(c == '-' for c in list(anchor_start_seq)) and not all(c == '-' for c in list(anchor_end_seq)):
            full_length_member_names.append(name)
            full_length_member_contigs[name] = align_seq

    first_seq = full_length_member_contigs[full_length_member_names[0]]
    col_num = len(first_seq)
    row_num = len(full_length_member_names)
    if row_num <= 1:
        if debug:
            print('full length number = 1, ' + align_file)
        return False
    matrix = [[''] * col_num for i in range(row_num)]
    for row, name in enumerate(full_length_member_names):
        seq = full_length_member_contigs[name]
        for col in range(len(seq)):
            matrix[row][col] = seq[col]

    # Starting from column 'align_start', search for 15 effective columns to the left.
    # Count the base composition of each column, in the format of {40: {A: 10, T: 5, C: 7, G: 9, '-': 20}},
    # which indicates the number of different bases in the current column.
    # Based on this, it is easy to determine whether the current column is effective and whether it is a homologous column.
    sliding_window_size = 20
    valid_col_threshold = int(row_num/2)

    if row_num <= 2:
        homo_threshold = 0.95
    elif row_num <= 5:
        homo_threshold = 0.9
    else:
        homo_threshold = 0.8

    homology_boundary_shift_threshold = 10

    homo_boundary_start = search_boundary_homo_v3(valid_col_threshold, align_start, matrix, row_num,
                                             col_num, 'start', homo_threshold, debug, sliding_window_size)
    if homo_boundary_start == -1 or abs(homo_boundary_start - align_start) > homology_boundary_shift_threshold:
        return False

    homo_boundary_end = search_boundary_homo_v3(valid_col_threshold, align_end, matrix, row_num,
                                                col_num, 'end', homo_threshold, debug, sliding_window_size)

    if homo_boundary_end == -1 or abs(homo_boundary_end - align_end) > homology_boundary_shift_threshold:
        return False

    # Generate a consensus sequence.
    model_seq = ''
    # Record the base composition of each column.
    col_base_map = {}
    for col_index in range(col_num):
        if not col_base_map.__contains__(col_index):
            col_base_map[col_index] = {}
        base_map = col_base_map[col_index]
        # Calculate the base composition ratio in the current column.
        if len(base_map) == 0:
            for row in range(row_num):
                cur_base = matrix[row][col_index]
                if not base_map.__contains__(cur_base):
                    base_map[cur_base] = 0
                cur_count = base_map[cur_base]
                cur_count += 1
                base_map[cur_base] = cur_count
        if not base_map.__contains__('-'):
            base_map['-'] = 0
    for col_index in range(homo_boundary_start, homo_boundary_end+1):
        base_map = col_base_map[col_index]
        # Identify the most frequent base that exceeds the threshold 'valid_col_threshold'.
        max_base_count = 0
        max_base = ''
        for cur_base in base_map.keys():
            cur_count = base_map[cur_base]
            if cur_count > max_base_count:
                max_base_count = cur_count
                max_base = cur_base
        if max_base_count >= int(row_num/2):
            if max_base != '-':
                model_seq += max_base
            else:
                continue
        else:
            max_base_count = 0
            max_base = ''
            for cur_base in base_map.keys():
                if cur_base == '-':
                    continue
                cur_count = base_map[cur_base]
                if cur_count > max_base_count:
                    max_base_count = cur_count
                    max_base = cur_base
            model_seq += max_base

    if model_seq == '' or len(model_seq) < 80:
        is_TE = False
    else:
        is_TE = True

    if debug:
        print(align_file, is_TE, homo_boundary_start, homo_boundary_end)
    return is_TE

def judge_boundary_v10(cur_seq, align_file, debug, TE_type, plant, result_type):
    # 1. Based on the 'remove gap' multi-alignment file, locate the position of the original sequence (anchor point).
    #     # Extend 20bp on both sides from the anchor point, extract the effective columns, and determine their homology.
    #     If it contradicts our rule, it is a false positive sequence.
    #     # --First, locate the TIR boundary position of the first sequence in the alignment file as the anchor point.
    #     # Take the first and last 20bp of the original sequence, and search on the aligned sequence without gaps.

    anchor_len = 20
    first_10bp = cur_seq[0:anchor_len]
    last_10bp = cur_seq[-anchor_len:]
    align_names, align_contigs = read_fasta(align_file)
    align_start = -1
    align_end = -1
    for name in align_names:
        raw_align_seq = align_contigs[name]
        align_seq = ''
        position_reflex = {}
        cur_align_index = 0
        for i, base in enumerate(raw_align_seq):
            if base == '-':
                continue
            else:
                align_seq += base
                position_reflex[cur_align_index] = i
                cur_align_index += 1

        start_dist = 2
        last_dist = 2
        first_matches = find_near_matches(first_10bp, align_seq, max_l_dist=start_dist)
        last_matches = find_near_matches(last_10bp, align_seq, max_l_dist=last_dist)
        last_matches = last_matches[::-1]
        if len(first_matches) > 0 and len(last_matches) > 0:
            align_no_gap_start = first_matches[0].start
            align_no_gap_end = last_matches[0].end - 1
            align_start = position_reflex[align_no_gap_start]
            align_end = position_reflex[align_no_gap_end]
            break
    if debug:
        print(align_file, align_start, align_end)
    if align_start == -1 or align_end == -1:
        if debug:
            print('not found boundary:' + align_file)
        return False

    align_names, align_contigs = read_fasta(align_file)
    if len(align_names) <= 0:
        if debug:
            print('align file size = 0, ' + align_file)
        return False

    # 3. Take the full-length sequence to generate a consensus sequence.
    # There should be bases both up and down by 10bp at the anchor point.
    full_length_member_names = []
    full_length_member_contigs = {}
    anchor_len = 10
    for name in align_names:
        if len(full_length_member_names) > 100:
            break
        align_seq = align_contigs[name]
        if align_start - anchor_len >= 0:
            anchor_start = align_start - anchor_len
        else:
            anchor_start = 0
        anchor_start_seq = align_seq[anchor_start: align_start + anchor_len]
        if align_end + anchor_len < len(align_seq):
            anchor_end = align_end + anchor_len
        else:
            anchor_end = len(align_seq)
        anchor_end_seq = align_seq[align_end - anchor_len: anchor_end]

        if not all(c == '-' for c in list(anchor_start_seq)) and not all(c == '-' for c in list(anchor_end_seq)):
            full_length_member_names.append(name)
            full_length_member_contigs[name] = align_seq

    first_seq = full_length_member_contigs[full_length_member_names[0]]
    col_num = len(first_seq)
    row_num = len(full_length_member_names)
    if row_num <= 1:
        if debug:
            print('full length number = 1, ' + align_file)
        return False
    matrix = [[''] * col_num for i in range(row_num)]
    for row, name in enumerate(full_length_member_names):
        seq = full_length_member_contigs[name]
        for col in range(len(seq)):
            matrix[row][col] = seq[col]

    # Starting from column 'align_start', search for 15 effective columns to the left.
    # Count the base composition of each column, in the format of {40: {A: 10, T: 5, C: 7, G: 9, '-': 20}},
    # which indicates the number of different bases in the current column.
    # Based on this, it is easy to determine whether the current column is effective and whether it is a homologous column.
    sliding_window_size = 100
    valid_col_threshold = int(row_num/2)

    if row_num <= 2:
        homo_threshold = 0.95
    elif row_num <= 5:
        homo_threshold = 0.9
    else:
        homo_threshold = 0.8

    homology_boundary_shift_threshold = 10

    is_TE = (search_boundary_homo_v4(valid_col_threshold, align_start, matrix, row_num, col_num, 'start',
                                    homo_threshold, debug, sliding_window_size)
             and search_boundary_homo_v4(valid_col_threshold, align_end, matrix, row_num, col_num, 'end',
                                         homo_threshold, debug, sliding_window_size))



    # Generate a consensus sequence.
    model_seq = ''
    # Record the base composition of each column.
    col_base_map = {}
    for col_index in range(col_num):
        if not col_base_map.__contains__(col_index):
            col_base_map[col_index] = {}
        base_map = col_base_map[col_index]
        # Calculate the base composition ratio in the current column.
        if len(base_map) == 0:
            for row in range(row_num):
                cur_base = matrix[row][col_index]
                if not base_map.__contains__(cur_base):
                    base_map[cur_base] = 0
                cur_count = base_map[cur_base]
                cur_count += 1
                base_map[cur_base] = cur_count
        if not base_map.__contains__('-'):
            base_map['-'] = 0
    for col_index in range(align_start, align_end+1):
        base_map = col_base_map[col_index]
        # Identify the most frequent base that exceeds the threshold 'valid_col_threshold'.
        max_base_count = 0
        max_base = ''
        for cur_base in base_map.keys():
            cur_count = base_map[cur_base]
            if cur_count > max_base_count:
                max_base_count = cur_count
                max_base = cur_base
        if max_base_count >= int(row_num/2):
            if max_base != '-':
                model_seq += max_base
            else:
                continue
        else:
            max_base_count = 0
            max_base = ''
            for cur_base in base_map.keys():
                if cur_base == '-':
                    continue
                cur_count = base_map[cur_base]
                if cur_count > max_base_count:
                    max_base_count = cur_count
                    max_base = cur_base
            model_seq += max_base


    if debug:
        print(align_file, is_TE, align_start, align_end)
    return is_TE


def judge_boundary_v11(cur_seq, align_file, debug, TE_type, plant, result_type):
    # 我们取 align_start 和 align_end 的外侧 100 bp，然后使用 Ninja 进行聚类
    # 假阳性的聚类数量应该比较少，而真实LTR 聚类数量会较多

    anchor_len = 20
    first_10bp = cur_seq[0:anchor_len]
    last_10bp = cur_seq[-anchor_len:]
    align_names, align_contigs = read_fasta(align_file)
    align_start = -1
    align_end = -1
    for name in align_names:
        raw_align_seq = align_contigs[name]
        align_seq = ''
        position_reflex = {}
        cur_align_index = 0
        for i, base in enumerate(raw_align_seq):
            if base == '-':
                continue
            else:
                align_seq += base
                position_reflex[cur_align_index] = i
                cur_align_index += 1

        start_dist = 2
        last_dist = 2
        first_matches = find_near_matches(first_10bp, align_seq, max_l_dist=start_dist)
        last_matches = find_near_matches(last_10bp, align_seq, max_l_dist=last_dist)
        last_matches = last_matches[::-1]
        if len(first_matches) > 0 and len(last_matches) > 0:
            align_no_gap_start = first_matches[0].start
            align_no_gap_end = last_matches[0].end - 1
            align_start = position_reflex[align_no_gap_start]
            align_end = position_reflex[align_no_gap_end]
            break
    if debug:
        print(align_file, align_start, align_end)
    if align_start == -1 or align_end == -1:
        if debug:
            print('not found boundary:' + align_file)
        return False

    align_names, align_contigs = read_fasta(align_file)
    if len(align_names) <= 0:
        if debug:
            print('align file size = 0, ' + align_file)
        return False

    is_TE = True
    # 取 align_start 和 align_end 的外侧 100 bp
    out_threshold = 100

    start_align_file = align_file + '.start'
    start_contigs = {}
    for name in align_names:
        raw_align_seq = align_contigs[name]
        if align_start - out_threshold < 0:
            start_pos = 0
        else:
            start_pos = align_start - out_threshold
        start_seq = raw_align_seq[start_pos: align_start]
        start_contigs[name] = start_seq
    save_dict_to_fasta(start_contigs, start_align_file)

    # 调用 Ninja 对多序列比对再次聚类
    # align_start_out = '/home/hukang/LTR_Benchmarking/LTR_libraries/LtrHomo/dmel_ltr_detector/tmp_dir/LTR_copies_0_0/Chr1:63812-65704.start.maf.fa'
    cluster_file = align_file + '.dat'
    Ninja_command = 'Ninja --in ' + align_file + ' --out ' + cluster_file + ' --out_type c --corr_type m --cluster_cutoff 0.2 --threads 1'
    #os.system(Ninja_command + ' > /dev/null 2>&1')
    # os.system(Ninja_command)
    print(Ninja_command)
    run_command(Ninja_command)

    # 解析聚类文件，生成不同簇
    clusters = read_Ninja_clusters(cluster_file)
    print(len(clusters))

    if debug:
        print(align_file, is_TE, align_start, align_end)
    return is_TE

def search_boundary_homo_v3(valid_col_threshold, pos, matrix, row_num, col_num,
                            type, homo_threshold, debug, sliding_window_size):
    # We need a program that takes an alignment file 'align_file' and boundary positions 'start_pos' and 'end_pos' as inputs, and extracts effective 20 columns around the boundaries. It also checks if these 20 columns exhibit homology.
    # Key Definitions:
    # ① What is an effective column? A column that has at least half of the total copy count, i.e., at least total/2 non-empty bases.
    # ② How is homology calculated? If consistent bases exceed 80% of the total sequence count, the column is considered homologous; otherwise, it is not.
    # If there is homology in 10 out of 15bp outside the boundary, it is likely to be a false positive.

    # Functionality:
    # Given an alignment matrix and a starting column, search for effective columns, homologous columns (homologous columns are always effective columns) towards both ends, and count the number of homologous columns, continuous homologous columns, and continuous non-homologous columns.
    # If there are consecutive non-homologous columns within the boundary or consecutive homologous columns outside the boundary beyond the threshold, it is considered a false positive.
    # Record the base composition of each column.
    col_base_map = {}
    for col_index in range(col_num):
        if not col_base_map.__contains__(col_index):
            col_base_map[col_index] = {}
        base_map = col_base_map[col_index]
        # Calculate the base composition ratio in the current column.
        if len(base_map) == 0:
            for row in range(row_num):
                cur_base = matrix[row][col_index]
                if not base_map.__contains__(cur_base):
                    base_map[cur_base] = 0
                cur_count = base_map[cur_base]
                cur_count += 1
                base_map[cur_base] = cur_count
        if not base_map.__contains__('-'):
            base_map['-'] = 0

    search_len = 100
    if type == 'start':
        valid_col_count = 0
        homo_col_count = 0

        max_con_homo = 0
        con_homo = 0
        prev_homo = False

        max_con_no_homo = 0
        con_no_homo = 0
        prev_non_homo = False

        col_index = pos
        homo_cols = []
        while valid_col_count < search_len and col_index < col_num / 2:
            # Starting from position 'pos', search for 15 effective columns to the right.
            # Determine if the current column is effective.
            is_homo_col = False
            base_map = col_base_map[col_index]
            no_gap_num = row_num - base_map['-']
            max_homo_ratio = 0
            gap_num = base_map['-']
            # If the number of gaps in the current column is <= half of the copy count, then it is an effective column.
            if gap_num <= valid_col_threshold:
                valid_col_count += 1
                # Determine if the effective column is homologous.
                for base in base_map.keys():
                    if base == '-':
                        continue
                    # 修正bug，row_num 替换成 no_gap_num
                    cur_homo_ratio = float(base_map[base]) / row_num
                    if cur_homo_ratio > max_homo_ratio:
                        max_homo_ratio = cur_homo_ratio
                    if cur_homo_ratio >= homo_threshold:
                        homo_col_count += 1
                        # Check for consecutive homologous columns.
                        if prev_homo:
                            con_homo += 1
                        is_homo_col = True
                        break
                if not is_homo_col:
                    max_con_homo = con_homo if con_homo > max_con_homo else max_con_homo
                    con_homo = 0

                    if prev_non_homo:
                        con_no_homo += 1
                    else:
                        max_con_no_homo = con_no_homo if con_no_homo > max_con_no_homo else max_con_no_homo
                        con_no_homo = 0
                    is_no_homo_col = True
                    prev_non_homo = True
                    prev_homo = False
                else:
                    max_con_no_homo = con_no_homo if con_no_homo > max_con_no_homo else max_con_no_homo
                    prev_homo = True
                    prev_non_homo = False
                    con_no_homo = 0
                    is_no_homo_col = False
                homo_cols.append(
                    (col_index, is_homo_col, con_homo, is_no_homo_col, con_no_homo,
                     max_homo_ratio))
            col_index += 1
        max_con_homo = con_homo if con_homo > max_con_homo else max_con_homo
        max_con_no_homo = con_no_homo if con_no_homo > max_con_no_homo else max_con_no_homo
        # if debug:
        #     print('align start right: ' + str(homo_col_count) + ', max continous homology bases: ' + str(max_con_homo)
        #           + ', max continous no-homology bases: ' + str(max_con_no_homo))
        #     print(homo_cols)

        # Use a sliding window to calculate the average homology of 10 consecutive bases starting from the left. Determine if it exceeds the threshold.
        # If it exceeds the threshold, obtain the first column with homology above the threshold within the 10bp, and consider it as the homologous boundary.
        cur_boundary = pos
        new_boundary_start = -1
        for i in range(len(homo_cols) - sliding_window_size + 1):
            window = homo_cols[i:i + sliding_window_size]
            avg_homo_ratio = 0
            first_candidate_boundary = -1
            for item in window:
                cur_homo_ratio = item[5]
                if cur_homo_ratio >= homo_threshold-0.1 and first_candidate_boundary == -1:
                    first_candidate_boundary = item[0]
                avg_homo_ratio += cur_homo_ratio
            avg_homo_ratio = float(avg_homo_ratio) / sliding_window_size
            if avg_homo_ratio >= homo_threshold:
                # If homology in the sliding window exceeds the threshold, find the boundary.
                new_boundary_start = first_candidate_boundary
                break
        if new_boundary_start != cur_boundary and new_boundary_start != -1:
            if debug:
                print('align start right non-homology, new boundary: ' + str(new_boundary_start))
        cur_boundary = new_boundary_start

        col_index = cur_boundary
        valid_col_count = 0
        homo_col_count = 0

        max_con_homo = 0
        con_homo = 0
        prev_homo = False

        max_con_no_homo = 0
        con_no_homo = 0
        prev_non_homo = False

        homo_cols = []
        while valid_col_count < search_len and col_index >= 0:
            # Starting from position 'pos', search for 15 effective columns to the left.
            # Determine if the current column is effective.
            is_homo_col = False
            base_map = col_base_map[col_index]
            max_homo_ratio = 0
            no_gap_num = row_num - base_map['-']
            gap_num = base_map['-']
            # If the number of gaps in the current column is <= half of the copy count, then it is an effective column.
            if gap_num <= valid_col_threshold:
                valid_col_count += 1
                # Determine if the effective column is homologous.
                for base in base_map.keys():
                    if base == '-':
                        continue
                    # 修正bug，row_num 替换成 no_gap_num
                    cur_homo_ratio = float(base_map[base]) / row_num
                    if cur_homo_ratio > max_homo_ratio:
                        max_homo_ratio = cur_homo_ratio
                    if cur_homo_ratio >= homo_threshold:
                        homo_col_count += 1
                        # Check for consecutive homologous columns.
                        if prev_homo:
                            con_homo += 1
                        is_homo_col = True
                        break
                if not is_homo_col:
                    max_con_homo = con_homo if con_homo > max_con_homo else max_con_homo
                    con_homo = 0

                    if prev_non_homo:
                        con_no_homo += 1
                    else:
                        max_con_no_homo = con_no_homo if con_no_homo > max_con_no_homo else max_con_no_homo
                        con_no_homo = 0
                    is_no_homo_col = True
                    prev_non_homo = True
                    prev_homo = False
                else:
                    max_con_no_homo = con_no_homo if con_no_homo > max_con_no_homo else max_con_no_homo
                    prev_homo = True
                    prev_non_homo = False
                    con_no_homo = 0
                    is_no_homo_col = False
                homo_cols.append(
                    (col_index, is_homo_col, con_homo, is_no_homo_col, con_no_homo, max_homo_ratio))
            col_index -= 1
        max_con_homo = con_homo if con_homo > max_con_homo else max_con_homo
        max_con_no_homo = con_no_homo if con_no_homo > max_con_no_homo else max_con_no_homo
        # if debug:
        #     print('align start left: ' + str(homo_col_count) + ', max continous homology bases: ' + str(max_con_homo)
        #           + ', max continous no-homology bases: ' + str(max_con_no_homo))
        #     print(homo_cols)

        # Use a sliding window to calculate the average homology of 10 consecutive bases starting from the left. Determine if it exceeds the threshold.
        # If it exceeds the threshold, obtain the first column with homology above the threshold within the 10bp, and consider it as the homologous boundary.
        homo_cols.reverse()
        new_boundary_start = -1
        for i in range(len(homo_cols) - sliding_window_size + 1):
            window = homo_cols[i:i + sliding_window_size]
            avg_homo_ratio = 0
            first_candidate_boundary = -1
            for item in window:
                cur_homo_ratio = item[5]
                if cur_homo_ratio >= homo_threshold-0.1 and first_candidate_boundary == -1:
                    first_candidate_boundary = item[0]
                avg_homo_ratio += cur_homo_ratio
            avg_homo_ratio = float(avg_homo_ratio)/sliding_window_size
            if avg_homo_ratio >= homo_threshold:
                # If homology in the sliding window exceeds the threshold, find the boundary.
                new_boundary_start = first_candidate_boundary
                break
        if new_boundary_start != cur_boundary and new_boundary_start != -1:
            if debug:
                print('align start left homology, new boundary: ' + str(new_boundary_start))
            cur_boundary = new_boundary_start

        return cur_boundary
    else:
        valid_col_count = 0
        homo_col_count = 0

        max_con_homo = 0
        con_homo = 0
        prev_homo = False

        max_con_no_homo = 0
        con_no_homo = 0
        prev_non_homo = False

        col_index = pos
        homo_cols = []
        while valid_col_count < search_len and col_index < col_num:
            # Starting from position 'pos', search for 15 effective columns to the right.
            # Determine if the current column is effective.
            is_homo_col = False
            base_map = col_base_map[col_index]
            # If the number of non-empty rows exceeds the threshold, then it is an effective row.
            no_gap_num = row_num - base_map['-']
            max_homo_ratio = 0
            gap_num = base_map['-']
            # If the number of gaps in the current column is <= half of the copy count, then it is an effective column.
            if gap_num <= valid_col_threshold:
                valid_col_count += 1
                # Determine if the effective column is homologous.
                for base in base_map.keys():
                    if base == '-':
                        continue
                    # 修正bug，row_num 替换成 no_gap_num
                    cur_homo_ratio = float(base_map[base]) / row_num
                    if cur_homo_ratio > max_homo_ratio:
                        max_homo_ratio = cur_homo_ratio
                    if cur_homo_ratio >= homo_threshold:
                        homo_col_count += 1
                        # Check for consecutive homologous columns.
                        if prev_homo:
                            con_homo += 1
                        is_homo_col = True
                        break
                if not is_homo_col:
                    max_con_homo = con_homo if con_homo > max_con_homo else max_con_homo
                    con_homo = 0

                    if prev_non_homo:
                        con_no_homo += 1
                    else:
                        max_con_no_homo = con_no_homo if con_no_homo > max_con_no_homo else max_con_no_homo
                        con_no_homo = 0
                    is_no_homo_col = True
                    prev_non_homo = True
                    prev_homo = False
                else:
                    max_con_no_homo = con_no_homo if con_no_homo > max_con_no_homo else max_con_no_homo
                    prev_homo = True
                    prev_non_homo = False
                    con_no_homo = 0
                    is_no_homo_col = False
                homo_cols.append(
                    (col_index, is_homo_col, con_homo, is_no_homo_col, con_no_homo, max_homo_ratio))
            col_index += 1
        max_con_homo = con_homo if con_homo > max_con_homo else max_con_homo
        max_con_no_homo = con_no_homo if con_no_homo > max_con_no_homo else max_con_no_homo
        # if debug:
        #     print('align end right: ' + str(homo_col_count) + ', max continous homology bases: ' + str(max_con_homo)
        #           + ', max continous no-homology bases: ' + str(max_con_no_homo))
        #     print(homo_cols)

        # Use a sliding window to calculate the average homology of 10 consecutive bases starting from the right. Determine if it exceeds the threshold.
        # If it exceeds the threshold, obtain the first column with homology above the threshold within the 10bp, and consider it as the homologous boundary.
        cur_boundary = pos
        homo_cols.reverse()
        new_boundary_end = -1
        for i in range(len(homo_cols) - sliding_window_size + 1):
            window = homo_cols[i:i + sliding_window_size]
            avg_homo_ratio = 0
            first_candidate_boundary = -1
            for item in window:
                cur_homo_ratio = item[5]
                if cur_homo_ratio >= homo_threshold-0.1 and first_candidate_boundary == -1:
                    first_candidate_boundary = item[0]
                avg_homo_ratio += cur_homo_ratio
            avg_homo_ratio = float(avg_homo_ratio) / sliding_window_size
            if avg_homo_ratio >= homo_threshold:
                # If homology in the sliding window exceeds the threshold, find the boundary.
                new_boundary_end = first_candidate_boundary
                break
        if new_boundary_end != cur_boundary and new_boundary_end != -1:
            if debug:
                print('align end right homology, new boundary: ' + str(new_boundary_end))
            cur_boundary = new_boundary_end

        col_index = cur_boundary
        valid_col_count = 0
        homo_col_count = 0

        max_con_homo = 0
        con_homo = 0
        prev_homo = False

        max_con_no_homo = 0
        con_no_homo = 0
        prev_non_homo = False

        homo_cols = []
        while valid_col_count < search_len and col_index >= col_num / 2:
            # Starting from position 'pos', search for 20 effective columns to the left.
            # Determine if the current column is effective.
            is_homo_col = False
            base_map = col_base_map[col_index]
            # If the number of non-empty rows exceeds the threshold, then it is an effective row.
            no_gap_num = row_num - base_map['-']
            max_homo_ratio = 0
            gap_num = base_map['-']
            # If the number of gaps in the current column is <= half of the copy count, then it is an effective column.
            if gap_num <= valid_col_threshold:
                valid_col_count += 1
                # Determine if the effective column is homologous.
                for base in base_map.keys():
                    if base == '-':
                        continue
                    # 修正bug，row_num 替换成 no_gap_num
                    cur_homo_ratio = float(base_map[base]) / row_num
                    if cur_homo_ratio > max_homo_ratio:
                        max_homo_ratio = cur_homo_ratio
                    if cur_homo_ratio >= homo_threshold:
                        homo_col_count += 1
                        # Check for consecutive homologous columns.
                        if prev_homo:
                            con_homo += 1
                        is_homo_col = True
                        break
                if not is_homo_col:
                    max_con_homo = con_homo if con_homo > max_con_homo else max_con_homo
                    con_homo = 0

                    if prev_non_homo:
                        con_no_homo += 1
                    else:
                        max_con_no_homo = con_no_homo if con_no_homo > max_con_no_homo else max_con_no_homo
                        con_no_homo = 0
                    is_no_homo_col = True
                    prev_non_homo = True
                    prev_homo = False
                else:
                    max_con_no_homo = con_no_homo if con_no_homo > max_con_no_homo else max_con_no_homo
                    prev_homo = True
                    prev_non_homo = False
                    con_no_homo = 0
                    is_no_homo_col = False
                homo_cols.append(
                    (col_index, is_homo_col, con_homo, is_no_homo_col, con_no_homo, max_homo_ratio))
            col_index -= 1
        max_con_homo = con_homo if con_homo > max_con_homo else max_con_homo
        max_con_no_homo = con_no_homo if con_no_homo > max_con_no_homo else max_con_no_homo
        # if debug:
        #     print('align end left: ' + str(homo_col_count) + ', max continous homology bases: ' + str(max_con_homo)
        #           + ', max continous no-homology bases: ' + str(max_con_no_homo))
        #     print(homo_cols)

        # Use a sliding window to calculate the average homology of 10 consecutive bases starting from the right. Determine if it exceeds the threshold.
        # If it exceeds the threshold, obtain the first column with homology above the threshold within the 10bp, and consider it as the homologous boundary.
        new_boundary_end = -1
        for i in range(len(homo_cols) - sliding_window_size + 1):
            window = homo_cols[i:i + sliding_window_size]
            avg_homo_ratio = 0
            first_candidate_boundary = -1
            for item in window:
                cur_homo_ratio = item[5]
                if cur_homo_ratio >= homo_threshold-0.1 and first_candidate_boundary == -1:
                    first_candidate_boundary = item[0]
                avg_homo_ratio += cur_homo_ratio
            avg_homo_ratio = float(avg_homo_ratio) / sliding_window_size
            if avg_homo_ratio >= homo_threshold:
                # If homology in the sliding window exceeds the threshold, find the boundary.
                new_boundary_end = first_candidate_boundary
                break
        if new_boundary_end != cur_boundary and new_boundary_end != -1:
            if debug:
                print('align end left non-homology, new boundary: ' + str(new_boundary_end))
        cur_boundary = new_boundary_end

        return cur_boundary

def search_boundary_homo_v4(valid_col_threshold, pos, matrix, row_num, col_num,
                            type, homo_threshold, debug, sliding_window_size):
    # We need a program that takes an alignment file 'align_file' and boundary positions 'start_pos' and 'end_pos' as inputs, and extracts effective 20 columns around the boundaries. It also checks if these 20 columns exhibit homology.
    # Key Definitions:
    # ① What is an effective column? A column that has at least half of the total copy count, i.e., at least total/2 non-empty bases.
    # ② How is homology calculated? If consistent bases exceed 80% of the total sequence count, the column is considered homologous; otherwise, it is not.
    # If there is homology in 10 out of 15bp outside the boundary, it is likely to be a false positive.

    # Functionality:
    # Given an alignment matrix and a starting column, search for effective columns, homologous columns (homologous columns are always effective columns) towards both ends, and count the number of homologous columns, continuous homologous columns, and continuous non-homologous columns.
    # If there are consecutive non-homologous columns within the boundary or consecutive homologous columns outside the boundary beyond the threshold, it is considered a false positive.
    # Record the base composition of each column.
    col_base_map = {}
    for col_index in range(col_num):
        if not col_base_map.__contains__(col_index):
            col_base_map[col_index] = {}
        base_map = col_base_map[col_index]
        # Calculate the base composition ratio in the current column.
        if len(base_map) == 0:
            for row in range(row_num):
                cur_base = matrix[row][col_index]
                if not base_map.__contains__(cur_base):
                    base_map[cur_base] = 0
                cur_count = base_map[cur_base]
                cur_count += 1
                base_map[cur_base] = cur_count
        if not base_map.__contains__('-'):
            base_map['-'] = 0

    search_len = 120
    if type == 'start':
        valid_col_count = 0
        homo_col_count = 0

        max_con_homo = 0
        con_homo = 0
        prev_homo = False

        max_con_no_homo = 0
        con_no_homo = 0
        prev_non_homo = False

        is_TE = True

        col_index = pos
        homo_cols = []
        while valid_col_count < search_len and col_index < col_num / 2:
            # Starting from position 'pos', search for 15 effective columns to the right.
            # Determine if the current column is effective.
            is_homo_col = False
            base_map = col_base_map[col_index]
            no_gap_num = row_num - base_map['-']
            max_homo_ratio = 0
            gap_num = base_map['-']
            # If the number of gaps in the current column is <= half of the copy count, then it is an effective column.
            if gap_num <= valid_col_threshold:
                valid_col_count += 1
                # Determine if the effective column is homologous.
                for base in base_map.keys():
                    if base == '-':
                        continue
                    # 修正bug，row_num 替换成 no_gap_num
                    cur_homo_ratio = float(base_map[base]) / no_gap_num
                    if cur_homo_ratio > max_homo_ratio:
                        max_homo_ratio = cur_homo_ratio
                    if cur_homo_ratio >= homo_threshold:
                        homo_col_count += 1
                        # Check for consecutive homologous columns.
                        if prev_homo:
                            con_homo += 1
                        is_homo_col = True
                        break
                if not is_homo_col:
                    max_con_homo = con_homo if con_homo > max_con_homo else max_con_homo
                    con_homo = 0

                    if prev_non_homo:
                        con_no_homo += 1
                    else:
                        max_con_no_homo = con_no_homo if con_no_homo > max_con_no_homo else max_con_no_homo
                        con_no_homo = 0
                    is_no_homo_col = True
                    prev_non_homo = True
                    prev_homo = False
                else:
                    max_con_no_homo = con_no_homo if con_no_homo > max_con_no_homo else max_con_no_homo
                    prev_homo = True
                    prev_non_homo = False
                    con_no_homo = 0
                    is_no_homo_col = False
                homo_cols.append(
                    (col_index, is_homo_col, con_homo, is_no_homo_col, con_no_homo,
                     max_homo_ratio))
            col_index += 1
        max_con_homo = con_homo if con_homo > max_con_homo else max_con_homo
        max_con_no_homo = con_no_homo if con_no_homo > max_con_no_homo else max_con_no_homo
        # if debug:
        #     print('align start right: ' + str(homo_col_count) + ', max continous homology bases: ' + str(max_con_homo)
        #           + ', max continous no-homology bases: ' + str(max_con_no_homo))
        #     print(homo_cols)

        # Use a sliding window to calculate the average homology of 10 consecutive bases starting from the left. Determine if it exceeds the threshold.
        # If it exceeds the threshold, obtain the first column with homology above the threshold within the 10bp, and consider it as the homologous boundary.
        window = homo_cols[0:0 + sliding_window_size]
        if len(window) > 0:
            # 计算窗口中的同源列的比例
            homo_col_num = 0
            for item in window:
                if item[1]:
                    homo_col_num += 1
            if float(homo_col_num)/len(window) >= homo_threshold:
                is_TE &= True
            else:
                is_TE &= False
        else:
            is_TE &= False

        col_index = pos
        valid_col_count = 0
        homo_col_count = 0

        max_con_homo = 0
        con_homo = 0
        prev_homo = False

        max_con_no_homo = 0
        con_no_homo = 0
        prev_non_homo = False

        homo_cols = []
        while valid_col_count < search_len and col_index >= 0:
            # Starting from position 'pos', search for 15 effective columns to the left.
            # Determine if the current column is effective.
            is_homo_col = False
            base_map = col_base_map[col_index]
            max_homo_ratio = 0
            no_gap_num = row_num - base_map['-']
            gap_num = base_map['-']
            # If the number of gaps in the current column is <= half of the copy count, then it is an effective column.
            if gap_num <= valid_col_threshold:
                valid_col_count += 1
                # Determine if the effective column is homologous.
                for base in base_map.keys():
                    if base == '-':
                        continue
                    # 修正bug，row_num 替换成 no_gap_num
                    cur_homo_ratio = float(base_map[base]) / no_gap_num
                    if cur_homo_ratio > max_homo_ratio:
                        max_homo_ratio = cur_homo_ratio
                    if cur_homo_ratio >= homo_threshold:
                        homo_col_count += 1
                        # Check for consecutive homologous columns.
                        if prev_homo:
                            con_homo += 1
                        is_homo_col = True
                        break
                if not is_homo_col:
                    max_con_homo = con_homo if con_homo > max_con_homo else max_con_homo
                    con_homo = 0

                    if prev_non_homo:
                        con_no_homo += 1
                    else:
                        max_con_no_homo = con_no_homo if con_no_homo > max_con_no_homo else max_con_no_homo
                        con_no_homo = 0
                    is_no_homo_col = True
                    prev_non_homo = True
                    prev_homo = False
                else:
                    max_con_no_homo = con_no_homo if con_no_homo > max_con_no_homo else max_con_no_homo
                    prev_homo = True
                    prev_non_homo = False
                    con_no_homo = 0
                    is_no_homo_col = False
                homo_cols.append(
                    (col_index, is_homo_col, con_homo, is_no_homo_col, con_no_homo, max_homo_ratio))
            col_index -= 1
        max_con_homo = con_homo if con_homo > max_con_homo else max_con_homo
        max_con_no_homo = con_no_homo if con_no_homo > max_con_no_homo else max_con_no_homo
        # if debug:
        #     print('align start left: ' + str(homo_col_count) + ', max continous homology bases: ' + str(max_con_homo)
        #           + ', max continous no-homology bases: ' + str(max_con_no_homo))
        #     print(homo_cols)

        # Use a sliding window to calculate the average homology of 10 consecutive bases starting from the left. Determine if it exceeds the threshold.
        # If it exceeds the threshold, obtain the first column with homology above the threshold within the 10bp, and consider it as the homologous boundary.
        window = homo_cols[0:0 + sliding_window_size]
        if len(window) > 0:
            # 计算窗口中的同源列的比例
            homo_col_num = 0
            for item in window:
                if item[1]:
                    homo_col_num += 1
            if float(homo_col_num) / len(window) >= homo_threshold:
                is_TE &= False
            else:
                is_TE &= True
        else:
            is_TE &= False

        return is_TE
    else:
        valid_col_count = 0
        homo_col_count = 0

        max_con_homo = 0
        con_homo = 0
        prev_homo = False

        max_con_no_homo = 0
        con_no_homo = 0
        prev_non_homo = False

        is_TE = True

        col_index = pos
        homo_cols = []
        while valid_col_count < search_len and col_index < col_num:
            # Starting from position 'pos', search for 15 effective columns to the right.
            # Determine if the current column is effective.
            is_homo_col = False
            base_map = col_base_map[col_index]
            # If the number of non-empty rows exceeds the threshold, then it is an effective row.
            no_gap_num = row_num - base_map['-']
            max_homo_ratio = 0
            gap_num = base_map['-']
            # If the number of gaps in the current column is <= half of the copy count, then it is an effective column.
            if gap_num <= valid_col_threshold:
                valid_col_count += 1
                # Determine if the effective column is homologous.
                for base in base_map.keys():
                    if base == '-':
                        continue
                    # 修正bug，row_num 替换成 no_gap_num
                    cur_homo_ratio = float(base_map[base]) / no_gap_num
                    if cur_homo_ratio > max_homo_ratio:
                        max_homo_ratio = cur_homo_ratio
                    if cur_homo_ratio >= homo_threshold:
                        homo_col_count += 1
                        # Check for consecutive homologous columns.
                        if prev_homo:
                            con_homo += 1
                        is_homo_col = True
                        break
                if not is_homo_col:
                    max_con_homo = con_homo if con_homo > max_con_homo else max_con_homo
                    con_homo = 0

                    if prev_non_homo:
                        con_no_homo += 1
                    else:
                        max_con_no_homo = con_no_homo if con_no_homo > max_con_no_homo else max_con_no_homo
                        con_no_homo = 0
                    is_no_homo_col = True
                    prev_non_homo = True
                    prev_homo = False
                else:
                    max_con_no_homo = con_no_homo if con_no_homo > max_con_no_homo else max_con_no_homo
                    prev_homo = True
                    prev_non_homo = False
                    con_no_homo = 0
                    is_no_homo_col = False
                homo_cols.append(
                    (col_index, is_homo_col, con_homo, is_no_homo_col, con_no_homo, max_homo_ratio))
            col_index += 1
        max_con_homo = con_homo if con_homo > max_con_homo else max_con_homo
        max_con_no_homo = con_no_homo if con_no_homo > max_con_no_homo else max_con_no_homo
        # if debug:
        #     print('align end right: ' + str(homo_col_count) + ', max continous homology bases: ' + str(max_con_homo)
        #           + ', max continous no-homology bases: ' + str(max_con_no_homo))
        #     print(homo_cols)

        # Use a sliding window to calculate the average homology of 10 consecutive bases starting from the right. Determine if it exceeds the threshold.
        # If it exceeds the threshold, obtain the first column with homology above the threshold within the 10bp, and consider it as the homologous boundary.
        window = homo_cols[0:0 + sliding_window_size]
        if len(window) > 0:
            # 计算窗口中的同源列的比例
            homo_col_num = 0
            for item in window:
                if item[1]:
                    homo_col_num += 1
            if float(homo_col_num) / len(window) >= homo_threshold:
                is_TE &= False
            else:
                is_TE &= True
        else:
            is_TE &= False

        col_index = pos
        valid_col_count = 0
        homo_col_count = 0

        max_con_homo = 0
        con_homo = 0
        prev_homo = False

        max_con_no_homo = 0
        con_no_homo = 0
        prev_non_homo = False

        homo_cols = []
        while valid_col_count < search_len and col_index >= col_num / 2:
            # Starting from position 'pos', search for 20 effective columns to the left.
            # Determine if the current column is effective.
            is_homo_col = False
            base_map = col_base_map[col_index]
            # If the number of non-empty rows exceeds the threshold, then it is an effective row.
            no_gap_num = row_num - base_map['-']
            max_homo_ratio = 0
            gap_num = base_map['-']
            # If the number of gaps in the current column is <= half of the copy count, then it is an effective column.
            if gap_num <= valid_col_threshold:
                valid_col_count += 1
                # Determine if the effective column is homologous.
                for base in base_map.keys():
                    if base == '-':
                        continue
                    # 修正bug，row_num 替换成 no_gap_num
                    cur_homo_ratio = float(base_map[base]) / no_gap_num
                    if cur_homo_ratio > max_homo_ratio:
                        max_homo_ratio = cur_homo_ratio
                    if cur_homo_ratio >= homo_threshold:
                        homo_col_count += 1
                        # Check for consecutive homologous columns.
                        if prev_homo:
                            con_homo += 1
                        is_homo_col = True
                        break
                if not is_homo_col:
                    max_con_homo = con_homo if con_homo > max_con_homo else max_con_homo
                    con_homo = 0

                    if prev_non_homo:
                        con_no_homo += 1
                    else:
                        max_con_no_homo = con_no_homo if con_no_homo > max_con_no_homo else max_con_no_homo
                        con_no_homo = 0
                    is_no_homo_col = True
                    prev_non_homo = True
                    prev_homo = False
                else:
                    max_con_no_homo = con_no_homo if con_no_homo > max_con_no_homo else max_con_no_homo
                    prev_homo = True
                    prev_non_homo = False
                    con_no_homo = 0
                    is_no_homo_col = False
                homo_cols.append(
                    (col_index, is_homo_col, con_homo, is_no_homo_col, con_no_homo, max_homo_ratio))
            col_index -= 1
        max_con_homo = con_homo if con_homo > max_con_homo else max_con_homo
        max_con_no_homo = con_no_homo if con_no_homo > max_con_no_homo else max_con_no_homo
        # if debug:
        #     print('align end left: ' + str(homo_col_count) + ', max continous homology bases: ' + str(max_con_homo)
        #           + ', max continous no-homology bases: ' + str(max_con_no_homo))
        #     print(homo_cols)

        # Use a sliding window to calculate the average homology of 10 consecutive bases starting from the right. Determine if it exceeds the threshold.
        # If it exceeds the threshold, obtain the first column with homology above the threshold within the 10bp, and consider it as the homologous boundary.
        window = homo_cols[0:0 + sliding_window_size]
        if len(window) > 0:
            # 计算窗口中的同源列的比例
            homo_col_num = 0
            for item in window:
                if item[1]:
                    homo_col_num += 1
            if float(homo_col_num) / len(window) >= homo_threshold:
                is_TE &= True
            else:
                is_TE &= False
        else:
            is_TE &= False

        return is_TE

def getReverseSequence(sequence):
    base_map = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    res = ''
    length = len(sequence)
    i = length - 1
    while i >= 0:
        base = sequence[i]
        if base not in base_map.keys():
            base = 'N'
        else:
            base = base_map[base]
        res += base
        i -= 1
    return res

def rename_reference(input, output, chr_name_map):
    names, contigs = read_fasta(input)
    chr_name_dict = {}
    ref_index = 0
    with open(output, 'w') as f_save:
        for name in names:
            seq = contigs[name]
            new_name = 'Chr'+str(ref_index)
            f_save.write('>'+new_name+'\n'+seq+'\n')
            ref_index += 1
            chr_name_dict[new_name] = name
    f_save.close()
    with open(chr_name_map, 'w') as f_save:
        for new_name in chr_name_dict.keys():
            f_save.write(new_name+'\t'+chr_name_dict[new_name]+'\n')
    f_save.close()

# Parse output of LTR_harvest
def get_LTR_seq_from_scn(genome, scn_path, ltr_terminal, ltr_internal):
    ref_names, ref_contigs = read_fasta(genome)
    LTR_terminals = {}
    LTR_ints = {}
    with open(scn_path, 'r') as f_r:
        for i, line in enumerate(f_r):
            if line.startswith('#'):
                continue
            else:
                line = line.replace('\n', '')
                parts = line.split(' ')
                LTR_start = int(parts[0])
                LTR_end = int(parts[1])
                chr_name = parts[11]
                lLTR_start = int(parts[3])
                lLTR_end = int(parts[4])
                rLTR_start = int(parts[6])
                rLTR_end = int(parts[7])
                lLTR_seq = ref_contigs[chr_name][lLTR_start-1: lLTR_end]
                rLTR_seq = ref_contigs[chr_name][rLTR_start - 1: rLTR_end]
                LTR_int_seq = ref_contigs[chr_name][lLTR_end: rLTR_start - 1]
                if len(lLTR_seq) > 0 and len(rLTR_seq) > 0 and len(LTR_int_seq) > 0 and 'NNNNNNNNNN' not in lLTR_seq and 'NNNNNNNNNN' not in rLTR_seq and 'NNNNNNNNNN' not in LTR_int_seq:
                    # LTR_seq = ref_contigs[chr_name][LTR_start-1: LTR_end]
                    # LTR_name = chr_name + '_' + str(LTR_start) + '-' + str(LTR_end) + '#LTR'
                    lLTR_name = chr_name + '_' + str(LTR_start) + '-' + str(LTR_end) + '-lLTR' + '#LTR'
                    LTR_terminals[lLTR_name] = lLTR_seq
                    LTR_int_name = chr_name + '_' + str(LTR_start) + '-' + str(LTR_end) + '-int' + '#LTR'
                    LTR_ints[LTR_int_name] = LTR_int_seq
                    rLTR_name = chr_name + '_' + str(LTR_start) + '-' + str(LTR_end) + '-rLTR' + '#LTR'
                    LTR_terminals[rLTR_name] = rLTR_seq
    f_r.close()
    store_fasta(LTR_terminals, ltr_terminal)
    store_fasta(LTR_ints, ltr_internal)

def get_overlap_len(seq1, seq2):
    """Calculate the overlap length between two sequences."""
    overlap_len = min(seq1[1], seq2[1]) - max(seq1[0], seq2[0])
    return overlap_len if overlap_len > 0 else 0

def merge_overlap_seq(seq1, seq2):
    return (min(seq1[0], seq2[0]), max(seq1[1], seq2[1]))

def multi_process_align_v1(query_path, subject_path, blastnResults_path, tmp_blast_dir, threads, chrom_length, coverage_threshold, category, is_removed_dir=True):
    tools_dir = ''
    if is_removed_dir:
        os.system('rm -rf ' + tmp_blast_dir)
    if not os.path.exists(tmp_blast_dir):
        os.makedirs(tmp_blast_dir)

    if os.path.exists(blastnResults_path):
        os.remove(blastnResults_path)

    orig_names, orig_contigs = read_fasta(query_path)

    # blast_db_command = 'makeblastdb -dbtype nucl -in ' + subject_path + ' > /dev/null 2>&1'
    # os.system(blast_db_command)

    ref_names, ref_contigs = read_fasta(subject_path)
    # Sequence alignment consumes a significant amount of memory and disk space. Therefore, we also split the target sequences into individual sequences to reduce the memory required for each alignment, avoiding out of memory errors.
    # It is important to calculate the total number of bases in the sequences, and it must meet a sufficient threshold to increase CPU utilization.
    base_threshold = 10000000  # 10Mb
    target_files = []
    file_index = 0
    base_count = 0
    cur_contigs = {}
    for name in ref_names:
        cur_seq = ref_contigs[name]
        cur_contigs[name] = cur_seq
        base_count += len(cur_seq)
        if base_count >= base_threshold:
            cur_target = tmp_blast_dir + '/' + str(file_index) + '_target.fa'
            store_fasta(cur_contigs, cur_target)
            target_files.append(cur_target)
            makedb_command = 'makeblastdb -dbtype nucl -in ' + cur_target + ' > /dev/null 2>&1'
            os.system(makedb_command)
            cur_contigs = {}
            file_index += 1
            base_count = 0
    if len(cur_contigs) > 0:
        cur_target = tmp_blast_dir + '/' + str(file_index) + '_target.fa'
        store_fasta(cur_contigs, cur_target)
        target_files.append(cur_target)
        makedb_command = 'makeblastdb -dbtype nucl -in ' + cur_target + ' > /dev/null 2>&1'
        os.system(makedb_command)


    longest_repeat_files = []
    # 为了保证处理大型library时，blastn比对结果不会过大，我们保证每个簇里的序列数量为固定值
    avg_cluster_size = 50
    cluster_num = int(len(orig_names) / avg_cluster_size) + 1
    segments_cluster = divided_array(list(orig_contigs.items()), cluster_num)
    for partition_index, cur_segments in enumerate(segments_cluster):
        if len(cur_segments) <= 0:
            continue
        single_tmp_dir = tmp_blast_dir + '/' + str(partition_index)
        #print('current partition_index: ' + str(partition_index))
        if not os.path.exists(single_tmp_dir):
            os.makedirs(single_tmp_dir)
        split_repeat_file = single_tmp_dir + '/repeats_split.fa'
        cur_contigs = {}
        for item in cur_segments:
            cur_contigs[item[0]] = item[1]
        store_fasta(cur_contigs, split_repeat_file)
        repeats_path = (split_repeat_file, target_files, single_tmp_dir + '/temp.out',
                        single_tmp_dir + '/full_length.out', single_tmp_dir + '/tmp',
                        subject_path)
        longest_repeat_files.append(repeats_path)

    ex = ProcessPoolExecutor(threads)
    jobs = []
    for file in longest_repeat_files:
        job = ex.submit(multiple_alignment_blast_v1, file, tools_dir, coverage_threshold, category, chrom_length)
        jobs.append(job)
    ex.shutdown(wait=True)

    # 合并所有进程的结果，总体去除冗余
    chr_segments_list = []
    for job in as_completed(jobs):
        cur_chr_segments = job.result()
        chr_segments_list.append(cur_chr_segments)

    # 由于可能会有多个序列比对到同一个位置，因此我们对于基因组上的某一个位置，我们只取一条比对
    segment_len = 100000  # 100K
    # chr_segments -> {chr1: {seg0: [(start, end, status)], seg1: []}}
    # Status: 0 indicates that the fragment is not marked as found, while 1 indicates that the fragment is marked as found.
    prev_chr_segments = {}
    total_chr_len = 0
    # Divide the chromosome evenly into N segments to store fragments in segments and reduce retrieval time.
    for chr_name in chrom_length.keys():
        chr_len = chrom_length[chr_name]
        total_chr_len += chr_len
        if not prev_chr_segments.__contains__(chr_name):
            prev_chr_segments[chr_name] = {}
        prev_chr_segment_list = prev_chr_segments[chr_name]
        num_segments = chr_len // segment_len
        if chr_len % segment_len != 0:
            num_segments += 1
        for i in range(num_segments):
            prev_chr_segment_list[i] = []

    for cur_chr_segments in chr_segments_list:
        # Map the fragments to the corresponding segment,
        # and check if there is an overlap of over 95% with the fragment in the segment.
        for chr_name in cur_chr_segments.keys():
            cur_chr_segment_dict = cur_chr_segments[chr_name]
            prev_chr_segment_list = prev_chr_segments[chr_name]
            for seg_index in cur_chr_segment_dict.keys():
                cur_segment_frags = cur_chr_segment_dict[seg_index]
                for cur_frag in cur_segment_frags:
                    start = cur_frag[0]
                    end = cur_frag[1]
                    seq_name = cur_frag[2]
                    seg_index = map_fragment(start, end, segment_len)

                    prev_segment_frags = prev_chr_segment_list[seg_index]
                    # Check if there is an overlap of over 95% between the fragment in the segment and the test fragment.
                    is_found = False
                    for prev_frag in prev_segment_frags:
                        overlap_len = get_overlap_len(prev_frag, cur_frag)
                        if overlap_len / abs(prev_frag[1] - prev_frag[0]) >= coverage_threshold and overlap_len / abs(
                                end - start) >= coverage_threshold:
                            is_found = True
                            break
                    if not is_found:
                        prev_segment_frags.append([start, end, seq_name])

    with open(blastnResults_path, 'w') as f_save:
        for chr_name in prev_chr_segments.keys():
            cur_chr_segments = prev_chr_segments[chr_name]
            for seg_index in cur_chr_segments.keys():
                segment_frags = cur_chr_segments[seg_index]
                for frag in segment_frags:
                    new_line = frag[2] + '\t' + chr_name + '\t' + '-1' + '\t' + '-1' + '\t' + '-1' + '\t' + '-1' + '\t' + '-1' + '\t' + '-1' + '\t' + str(frag[0]) + '\t' + str(frag[1]) + '\t' + '-1' + '\t' + '-1' + '\n'
                    f_save.write(new_line)

    if is_removed_dir:
        os.system('rm -rf ' + tmp_blast_dir)

def get_full_length_copies_RM(TE_lib, reference, tmp_output_dir, threads, divergence_threshold, full_length_threshold,
                              search_struct, tools_dir):
    if not os.path.exists(tmp_output_dir):
        os.makedirs(tmp_output_dir)
    tmp_TE_out = tmp_output_dir + '/TE_tmp.out'
    tmp_TE_gff = tmp_output_dir + '/TE_tmp.gff'

    RepeatMasker_command = 'cd ' + tmp_output_dir + ' && RepeatMasker -e ncbi -pa ' + str(threads) \
                           + ' -s -no_is -norna -nolow -div ' + str(divergence_threshold) \
                           + ' -gff -lib ' + TE_lib + ' -cutoff 225 ' + reference
    os.system(RepeatMasker_command + '> /dev/null 2>&1')

    mv_file_command = 'mv ' + reference + '.out ' + tmp_TE_out + ' && mv ' + reference + '.out.gff ' + tmp_TE_gff
    os.system(mv_file_command)

    full_length_annotations, copies_direct = get_full_length_copies_from_gff(TE_lib, reference, tmp_TE_gff,
                                                    tmp_output_dir, threads, divergence_threshold,
                                                    full_length_threshold, search_struct, tools_dir)
    return full_length_annotations, copies_direct

def get_full_length_copies_from_gff(TE_lib, reference, gff_path, tmp_output_dir, threads, divergence_threshold,
                                    full_length_threshold, search_struct, tools_dir):
    ref_names, ref_contigs = read_fasta(reference)

    query_names, query_contigs = read_fasta(TE_lib)
    new_query_contigs = {}
    for name in query_names:
        new_query_contigs[name.split('#')[0]] = query_contigs[name]
    query_contigs = new_query_contigs

    query_records = {}
    with open(gff_path, 'r') as f_r:
        for line in f_r:
            if line.startswith('#'):
                continue
            parts = line.split('\t')
            query_name = parts[8].split(' ')[1].replace('"', '').split(':')[1]
            subject_name = parts[0]
            info_parts = parts[8].split(' ')
            q_start = int(info_parts[2])
            q_end = int(info_parts[3])
            if parts[6] == '-':
                s_start = int(parts[4])
                s_end = int(parts[3])
            else:
                s_start = int(parts[3])
                s_end = int(parts[4])
            if not query_records.__contains__(query_name):
                query_records[query_name] = {}
            subject_dict = query_records[query_name]

            if not subject_dict.__contains__(subject_name):
                subject_dict[subject_name] = []
            subject_pos = subject_dict[subject_name]
            subject_pos.append((q_start, q_end, s_start, s_end))

    full_length_copies = {}
    flank_full_length_copies = {}
    copies_direct = {}

    for idx, query_name in enumerate(query_records.keys()):
        subject_dict = query_records[query_name]
        query_len = len(query_contigs[query_name])
        skip_gap = query_len * full_length_threshold
        if str(query_name).__contains__('Helitron'):
            flanking_len = 5
        else:
            flanking_len = 50

        # if there are more than one longest query overlap with the final longest query over 90%,
        # then it probably the true TE
        longest_queries = []
        for subject_name in subject_dict.keys():
            subject_pos = subject_dict[subject_name]

            # cluster all closed fragments, split forward and reverse records
            forward_pos = []
            reverse_pos = []
            for pos_item in subject_pos:
                if pos_item[2] > pos_item[3]:
                    reverse_pos.append(pos_item)
                else:
                    forward_pos.append(pos_item)
            forward_pos.sort(key=lambda x: (x[2], x[3]))
            reverse_pos.sort(key=lambda x: (-x[2], -x[3]))

            clusters = {}
            cluster_index = 0
            for k, frag in enumerate(forward_pos):
                if not clusters.__contains__(cluster_index):
                    clusters[cluster_index] = []
                cur_cluster = clusters[cluster_index]
                if k == 0:
                    cur_cluster.append(frag)
                else:
                    is_closed = False
                    for exist_frag in reversed(cur_cluster):
                        cur_subject_start = frag[2]
                        cur_query_end = frag[1]
                        prev_subject_end = exist_frag[3]
                        prev_query_end = exist_frag[1]
                        if (cur_subject_start - prev_subject_end < skip_gap and cur_query_end > prev_query_end):
                            is_closed = True
                            break
                    if is_closed:
                        cur_cluster.append(frag)
                    else:
                        cluster_index += 1
                        if not clusters.__contains__(cluster_index):
                            clusters[cluster_index] = []
                        cur_cluster = clusters[cluster_index]
                        cur_cluster.append(frag)

            cluster_index += 1
            for k, frag in enumerate(reverse_pos):
                if not clusters.__contains__(cluster_index):
                    clusters[cluster_index] = []
                cur_cluster = clusters[cluster_index]
                if k == 0:
                    cur_cluster.append(frag)
                else:
                    is_closed = False
                    for exist_frag in reversed(cur_cluster):
                        cur_subject_start = frag[2]
                        cur_query_end = frag[1]
                        prev_subject_end = exist_frag[3]
                        prev_query_end = exist_frag[1]
                        if (prev_subject_end - cur_subject_start < skip_gap and cur_query_end > prev_query_end):
                            is_closed = True
                            break
                    if is_closed:
                        cur_cluster.append(frag)
                    else:
                        cluster_index += 1
                        if not clusters.__contains__(cluster_index):
                            clusters[cluster_index] = []
                        cur_cluster = clusters[cluster_index]
                        cur_cluster.append(frag)

            for cluster_index in clusters.keys():
                cur_cluster = clusters[cluster_index]
                cur_cluster.sort(key=lambda x: (x[0], x[1]))

                # print('subject pos size: %d' %(len(cur_cluster)))
                # record visited fragments
                visited_frag = {}
                for i in range(len(cur_cluster)):
                    # keep a longest query start from each fragment
                    prev_frag = cur_cluster[i]
                    if visited_frag.__contains__(prev_frag):
                        continue
                    prev_query_start = prev_frag[0]
                    prev_query_end = prev_frag[1]
                    prev_subject_start = prev_frag[2]
                    prev_subject_end = prev_frag[3]
                    prev_query_seq = (min(prev_query_start, prev_query_end), max(prev_query_start, prev_query_end))
                    prev_subject_seq = (
                        min(prev_subject_start, prev_subject_end), max(prev_subject_start, prev_subject_end))
                    prev_query_len = abs(prev_query_end - prev_query_start)
                    prev_subject_len = abs(prev_subject_end - prev_subject_start)
                    cur_longest_query_len = prev_query_len

                    cur_extend_num = 0
                    visited_frag[prev_frag] = 1
                    # try to extend query
                    for j in range(i + 1, len(cur_cluster)):
                        cur_frag = cur_cluster[j]
                        if visited_frag.__contains__(cur_frag):
                            continue
                        cur_query_start = cur_frag[0]
                        cur_query_end = cur_frag[1]
                        cur_subject_start = cur_frag[2]
                        cur_subject_end = cur_frag[3]
                        cur_query_seq = (min(cur_query_start, cur_query_end), max(cur_query_start, cur_query_end))
                        cur_subject_seq = (
                            min(cur_subject_start, cur_subject_end), max(cur_subject_start, cur_subject_end))
                        cur_query_len = abs(cur_query_end - cur_query_start)
                        cur_subject_len = abs(cur_subject_end - cur_subject_start)

                        query_overlap_len = get_overlap_len(cur_query_seq, prev_query_seq)
                        is_same_query = float(query_overlap_len) / cur_query_len >= 0.5 or float(
                            query_overlap_len) / prev_query_len >= 0.5
                        subject_overlap_len = get_overlap_len(prev_subject_seq, cur_subject_seq)
                        is_same_subject = float(subject_overlap_len) / cur_subject_len >= 0.5 or float(
                            subject_overlap_len) / prev_subject_len >= 0.5

                        # could extend
                        # extend right
                        if cur_query_end > prev_query_end:
                            # judge subject direction
                            if prev_subject_start < prev_subject_end and cur_subject_start < cur_subject_end:
                                # +
                                if cur_subject_end > prev_subject_end:
                                    # forward extend
                                    if cur_query_start - prev_query_end < skip_gap and cur_query_end > prev_query_end \
                                            and cur_subject_start - prev_subject_end < skip_gap:  # \
                                        # and not is_same_query and not is_same_subject:
                                        # update the longest path
                                        prev_query_start = prev_query_start
                                        prev_query_end = cur_query_end
                                        prev_subject_start = prev_subject_start if prev_subject_start < cur_subject_start else cur_subject_start
                                        prev_subject_end = cur_subject_end
                                        cur_longest_query_len = prev_query_end - prev_query_start
                                        cur_extend_num += 1
                                        visited_frag[cur_frag] = 1
                                    elif cur_query_start - prev_query_end >= skip_gap:
                                        break
                            elif prev_subject_start > prev_subject_end and cur_subject_start > cur_subject_end:
                                # reverse
                                if cur_subject_end < prev_subject_end:
                                    # reverse extend
                                    if cur_query_start - prev_query_end < skip_gap and cur_query_end > prev_query_end \
                                            and prev_subject_end - cur_subject_start < skip_gap:  # \
                                        # and not is_same_query and not is_same_subject:
                                        # update the longest path
                                        prev_query_start = prev_query_start
                                        prev_query_end = cur_query_end
                                        prev_subject_start = prev_subject_start if prev_subject_start > cur_subject_start else cur_subject_start
                                        prev_subject_end = cur_subject_end
                                        cur_longest_query_len = prev_query_end - prev_query_start
                                        cur_extend_num += 1
                                        visited_frag[cur_frag] = 1
                                    elif cur_query_start - prev_query_end >= skip_gap:
                                        break
                    # keep this longest query
                    if cur_longest_query_len != -1:
                        longest_queries.append(
                            (prev_query_start, prev_query_end, cur_longest_query_len, prev_subject_start,
                             prev_subject_end, abs(prev_subject_end - prev_subject_start), subject_name,
                             cur_extend_num))

        # To determine whether each copy has a coverage exceeding the full_length_threshold with respect
        # to the consensus sequence, retaining full-length copies.
        query_copies = {}
        flank_query_copies = {}
        # query_copies[query_name] = query_contigs[query_name]
        for repeat in longest_queries:
            if repeat[2] < full_length_threshold * query_len:
                continue
            # Subject
            subject_name = repeat[6]
            subject_chr_start = 0

            if repeat[3] > repeat[4]:
                direct = '-'
                old_subject_start_pos = repeat[4] - 1
                old_subject_end_pos = repeat[3]
            else:
                direct = '+'
                old_subject_start_pos = repeat[3] - 1
                old_subject_end_pos = repeat[4]
            subject_start_pos = subject_chr_start + old_subject_start_pos
            subject_end_pos = subject_chr_start + old_subject_end_pos

            subject_pos = subject_name + ':' + str(subject_start_pos) + '-' + str(subject_end_pos)
            subject_seq = ref_contigs[subject_name][subject_start_pos: subject_end_pos]

            flank_subject_seq = ref_contigs[subject_name][
                                subject_start_pos - flanking_len: subject_end_pos + flanking_len]
            copies_direct[subject_pos] = direct
            cur_query_len = repeat[2]
            cur_subject_len = repeat[5]
            min_cur_len = min(cur_query_len, cur_subject_len)
            max_cur_len = max(cur_query_len, cur_subject_len)
            coverage = float(min_cur_len) / max_cur_len
            if coverage >= full_length_threshold:
                query_copies[subject_pos] = subject_seq
                flank_query_copies[subject_pos] = flank_subject_seq
        full_length_copies[query_name] = query_copies
        flank_full_length_copies[query_name] = flank_query_copies

    # The candidate full-length copies and the consensus are then clustered using cd-hit-est,
    # retaining copies that belong to the same cluster as the consensus.
    split_files = []
    cluster_dir = tmp_output_dir + '/cluster'
    os.system('rm -rf ' + cluster_dir)
    if not os.path.exists(cluster_dir):
        os.makedirs(cluster_dir)

    all_query_copies = {}
    for query_name in full_length_copies.keys():
        query_copies = full_length_copies[query_name]
        flank_query_copies = flank_full_length_copies[query_name]
        all_query_copies.update(query_copies)
        fc_path = cluster_dir + '/' + query_name + '.fa'
        fc_cons = cluster_dir + '/' + query_name + '.cons.fa'
        store_fasta(query_copies, fc_path)
        split_files.append((fc_path, query_name, query_copies, flank_query_copies))

    ex = ProcessPoolExecutor(threads)
    jobs = []
    for ref_index, cur_file in enumerate(split_files):
        input_file = cur_file[0]
        query_name = cur_file[1]
        query_copies = cur_file[2]
        flank_query_copies = cur_file[3]
        job = ex.submit(get_structure_info, input_file, query_name, query_copies,
                        flank_query_copies, cluster_dir, search_struct, tools_dir)
        jobs.append(job)
    ex.shutdown(wait=True)

    full_length_annotations = {}
    for job in as_completed(jobs):
        annotations = job.result()
        full_length_annotations.update(annotations)
    return full_length_annotations, copies_direct

def multi_process_align_v2(query_path, subject_path, blastnResults_path, tmp_blast_dir, threads, chrom_length, coverage_threshold, category, TRsearch_dir, is_removed_dir=True):
    if is_removed_dir:
        os.system('rm -rf ' + tmp_blast_dir)

    if not os.path.exists(tmp_blast_dir):
        os.makedirs(tmp_blast_dir)

    if os.path.exists(blastnResults_path):
        os.remove(blastnResults_path)


    # 由于 blastn 未能将一些差异性的一致性序列比对到应有的位置，因此我们调用 RepeatMasker 来进行比对
    intact_dir = tmp_blast_dir + '/intact_tmp'
    divergence_threshold = 20
    full_length_threshold = 0.8
    search_struct = False
    full_length_annotations, copies_direct = get_full_length_copies_RM(query_path, subject_path, intact_dir, threads,
                                                                       divergence_threshold,
                                                                       full_length_threshold, search_struct,
                                                                       TRsearch_dir)
    lines = []
    for seq_name in full_length_annotations.keys():
        for copy in full_length_annotations[seq_name]:
            parts = copy[0].split(':')
            chr_name = parts[0]
            pos_parts = parts[1].split('-')
            chr_start = int(pos_parts[0]) + 1
            chr_end = int(pos_parts[1])
            lines.append((seq_name, chr_name, chr_start, chr_end))


    lines = list(lines)
    sorted_lines = sorted(lines, key=lambda x: (x[1], x[2], x[3]))
    test_fragments = {}
    for line in sorted_lines:
        seq_name = line[0]
        chr_name = line[1]
        chr_start = line[2]
        chr_end = line[3]
        if chr_name not in test_fragments:
            test_fragments[chr_name] = []
        fragments = test_fragments[chr_name]
        fragments.append((chr_start, chr_end, seq_name))

    # 由于可能会有多个序列比对到同一个位置，因此我们对于基因组上的某一个位置，我们只取一条比对
    segment_len = 100000  # 100K
    # chr_segments -> {chr1: {seg0: [(start, end, status)], seg1: []}}
    # Status: 0 indicates that the fragment is not marked as found, while 1 indicates that the fragment is marked as found.
    chr_segments = {}
    total_chr_len = 0
    # Divide the chromosome evenly into N segments to store fragments in segments and reduce retrieval time.
    for chr_name in chrom_length.keys():
        chr_len = chrom_length[chr_name]
        total_chr_len += chr_len
        if not chr_segments.__contains__(chr_name):
            chr_segments[chr_name] = {}
        cur_chr_segments = chr_segments[chr_name]
        num_segments = chr_len // segment_len
        if chr_len % segment_len != 0:
            num_segments += 1
        for i in range(num_segments):
            cur_chr_segments[i] = []

    # Map the fragments to the corresponding segment,
    # and check if there is an overlap of over 95% with the fragment in the segment.
    for chr_name in test_fragments.keys():
        fragments = test_fragments[chr_name]
        cur_chr_segments = chr_segments[chr_name]
        for cur_frag in fragments:
            start = cur_frag[0]
            end = cur_frag[1]
            seq_name = cur_frag[2]
            seg_index = map_fragment(start, end, segment_len)
            segment_frags = cur_chr_segments[seg_index]
            # Check if there is an overlap of over 95% between the fragment in the segment and the test fragment.
            is_found = False
            for prev_frag in segment_frags:
                overlap_len = get_overlap_len(prev_frag, cur_frag)
                if overlap_len / abs(prev_frag[1] - prev_frag[0]) >= coverage_threshold and overlap_len / abs(
                        end - start) >= coverage_threshold:
                    is_found = True
                    break
            if not is_found:
                segment_frags.append([start, end, seq_name])

    with open(blastnResults_path, 'w') as f_save:
        for chr_name in chr_segments.keys():
            cur_chr_segments = chr_segments[chr_name]
            for seg_index in cur_chr_segments.keys():
                segment_frags = cur_chr_segments[seg_index]
                for frag in segment_frags:
                    new_line = frag[2] + '\t' + chr_name + '\t' + '-1' + '\t' + '-1' + '\t' + '-1' + '\t' + '-1' + '\t' + '-1' + '\t' + '-1' + '\t' + str(frag[0]) + '\t' + str(frag[1]) + '\t' + '-1' + '\t' + '-1' + '\n'
                    f_save.write(new_line)

    if is_removed_dir:
        os.system('rm -rf ' + tmp_blast_dir)

def map_fragment(start, end, chunk_size):
    start_chunk = start // chunk_size
    end_chunk = end // chunk_size

    if start_chunk == end_chunk:
        return start_chunk
    elif abs(end_chunk * chunk_size - start) < abs(end - end_chunk * chunk_size):
        return end_chunk
    else:
        return start_chunk

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

def multiple_alignment_blast_v1(repeats_path, tools_dir, coverage_threshold, category, chrom_length):
    split_repeats_path = repeats_path[0]
    target_files = repeats_path[1]
    blastn2Results_path = repeats_path[2]
    full_length_out = repeats_path[3]
    tmp_dir = repeats_path[4]
    genome_path = repeats_path[5]
    os.system('rm -f ' + blastn2Results_path)
    for target_file in target_files:
        align_command = 'blastn -db ' + target_file + ' -num_threads ' \
                        + str(1) + ' -query ' + split_repeats_path + ' -evalue 1e-20 -outfmt 6 >> ' + blastn2Results_path
        os.system(align_command)

    # invoke the function to retrieve the full-length copies.
    lines = generate_full_length_out(blastn2Results_path, full_length_out, split_repeats_path, genome_path, tmp_dir, tools_dir,
                             coverage_threshold, category)

    # 去除冗余的影响
    lines = list(lines)
    sorted_lines = sorted(lines, key=lambda x: (x[1], x[2], x[3]))
    test_fragments = {}
    for line in sorted_lines:
        seq_name = line[0]
        chr_name = line[1]
        chr_start = line[2]
        chr_end = line[3]
        if chr_name not in test_fragments:
            test_fragments[chr_name] = []
        fragments = test_fragments[chr_name]
        fragments.append((chr_start, chr_end, seq_name))

    # 由于可能会有多个序列比对到同一个位置，因此我们对于基因组上的某一个位置，我们只取一条比对
    segment_len = 100000  # 100K
    # chr_segments -> {chr1: {seg0: [(start, end, status)], seg1: []}}
    # Status: 0 indicates that the fragment is not marked as found, while 1 indicates that the fragment is marked as found.
    chr_segments = {}
    total_chr_len = 0
    # Divide the chromosome evenly into N segments to store fragments in segments and reduce retrieval time.
    for chr_name in chrom_length.keys():
        chr_len = chrom_length[chr_name]
        total_chr_len += chr_len
        if not chr_segments.__contains__(chr_name):
            chr_segments[chr_name] = {}
        cur_chr_segments = chr_segments[chr_name]
        num_segments = chr_len // segment_len
        if chr_len % segment_len != 0:
            num_segments += 1
        for i in range(num_segments):
            cur_chr_segments[i] = []

    # Map the fragments to the corresponding segment,
    # and check if there is an overlap of over 95% with the fragment in the segment.
    for chr_name in test_fragments.keys():
        fragments = test_fragments[chr_name]
        cur_chr_segments = chr_segments[chr_name]
        for cur_frag in fragments:
            start = cur_frag[0]
            end = cur_frag[1]
            seq_name = cur_frag[2]
            seg_index = map_fragment(start, end, segment_len)
            segment_frags = cur_chr_segments[seg_index]
            # Check if there is an overlap of over 95% between the fragment in the segment and the test fragment.
            is_found = False
            for prev_frag in segment_frags:
                overlap_len = get_overlap_len(prev_frag, cur_frag)
                if overlap_len / abs(prev_frag[1] - prev_frag[0]) >= coverage_threshold and overlap_len / abs(
                        end - start) >= coverage_threshold:
                    is_found = True
                    break
            if not is_found:
                segment_frags.append([start, end, seq_name])

    return chr_segments


def generate_full_length_out(BlastnOut, full_length_out, TE_lib, reference, tmp_output_dir, tools_dir, full_length_threshold, category):
    if not os.path.exists(tmp_output_dir):
        os.makedirs(tmp_output_dir)
    filter_tmp_out = filter_out_by_category(BlastnOut, tmp_output_dir, category)

    threads = 1
    divergence_threshold = 20
    search_struct = False
    full_length_annotations, copies_direct = get_full_length_copies_from_blastn(TE_lib, reference, filter_tmp_out,
                                                                             tmp_output_dir, threads,
                                                                             divergence_threshold,
                                                                             full_length_threshold,
                                                                             search_struct, tools_dir)

    lines = set()
    for query_name in full_length_annotations.keys():
        query_name = str(query_name)
        for copy_annotation in full_length_annotations[query_name]:
            chr_pos = copy_annotation[0]
            annotation = copy_annotation[1]
            parts = chr_pos.split(':')
            chr_name = parts[0]
            chr_pos_parts = parts[1].split('-')
            chr_start = int(chr_pos_parts[0]) + 1
            chr_end = int(chr_pos_parts[1])
            new_line = (query_name, chr_name, chr_start, chr_end)
            lines.add(new_line)

    return lines

def filter_out_by_category(TE_out, tmp_output_dir, category):
    tmp_out= tmp_output_dir + '/tmp.out'
    os.system('cp ' + TE_out + ' ' + tmp_out)
    if category == 'Total':
        return tmp_out
    else:
        lines = []
        with open(tmp_out, 'r') as f_r:
            for line in f_r:
                query_name = line.split('\t')[0]
                parts = query_name.split('#')
                type = parts[1]
                if category in type:
                    lines.append(line)
        filter_tmp_out = tmp_output_dir + '/tmp.filter.out'
        with open(filter_tmp_out, 'w') as f_save:
            for line in lines:
                f_save.write(line)
        return filter_tmp_out

def get_full_length_copies_from_blastn(TE_lib, reference, blastn_out, tmp_output_dir, threads, divergence_threshold,
                                    full_length_threshold, search_struct, tools_dir):
    ref_names, ref_contigs = read_fasta(reference)

    query_names, query_contigs = read_fasta(TE_lib)
    new_query_contigs = {}
    for name in query_names:
        new_query_contigs[name.split('#')[0]] = query_contigs[name]
    query_contigs = new_query_contigs

    query_records = {}
    with open(blastn_out, 'r') as f_r:
        for line in f_r:
            if line.startswith('#'):
                continue
            info_parts = line.split('\t')
            query_name = info_parts[0].split('#')[0]
            subject_name = info_parts[1]
            q_start = int(info_parts[6])
            q_end = int(info_parts[7])
            s_start = int(info_parts[8])
            s_end = int(info_parts[9])
            if not query_records.__contains__(query_name):
                query_records[query_name] = {}
            subject_dict = query_records[query_name]

            if not subject_dict.__contains__(subject_name):
                subject_dict[subject_name] = []
            subject_pos = subject_dict[subject_name]
            subject_pos.append((q_start, q_end, s_start, s_end))

    full_length_copies = {}
    flank_full_length_copies = {}
    copies_direct = {}
    for idx, query_name in enumerate(query_records.keys()):
        subject_dict = query_records[query_name]
        if query_name not in query_contigs:
            continue
        query_len = len(query_contigs[query_name])
        skip_gap = query_len * full_length_threshold

        if str(query_name).__contains__('Helitron'):
            flanking_len = 5
        else:
            flanking_len = 50

        # if there are more than one longest query overlap with the final longest query over 90%,
        # then it probably the true TE
        longest_queries = []
        for subject_name in subject_dict.keys():
            subject_pos = subject_dict[subject_name]

            # cluster all closed fragments, split forward and reverse records
            forward_pos = []
            reverse_pos = []
            for pos_item in subject_pos:
                if pos_item[2] > pos_item[3]:
                    reverse_pos.append(pos_item)
                else:
                    forward_pos.append(pos_item)
            forward_pos.sort(key=lambda x: (x[2], x[3]))
            reverse_pos.sort(key=lambda x: (-x[2], -x[3]))

            clusters = {}
            cluster_index = 0
            for k, frag in enumerate(forward_pos):
                if not clusters.__contains__(cluster_index):
                    clusters[cluster_index] = []
                cur_cluster = clusters[cluster_index]
                if k == 0:
                    cur_cluster.append(frag)
                else:
                    is_closed = False
                    for exist_frag in reversed(cur_cluster):
                        cur_subject_start = frag[2]
                        cur_query_end = frag[1]
                        prev_subject_end = exist_frag[3]
                        prev_query_end = exist_frag[1]
                        if (cur_subject_start - prev_subject_end < skip_gap and cur_query_end > prev_query_end):
                            is_closed = True
                            break
                    if is_closed:
                        cur_cluster.append(frag)
                    else:
                        cluster_index += 1
                        if not clusters.__contains__(cluster_index):
                            clusters[cluster_index] = []
                        cur_cluster = clusters[cluster_index]
                        cur_cluster.append(frag)

            cluster_index += 1
            for k, frag in enumerate(reverse_pos):
                if not clusters.__contains__(cluster_index):
                    clusters[cluster_index] = []
                cur_cluster = clusters[cluster_index]
                if k == 0:
                    cur_cluster.append(frag)
                else:
                    is_closed = False
                    for exist_frag in reversed(cur_cluster):
                        cur_subject_start = frag[2]
                        cur_query_end = frag[1]
                        prev_subject_end = exist_frag[3]
                        prev_query_end = exist_frag[1]
                        if (prev_subject_end - cur_subject_start < skip_gap and cur_query_end > prev_query_end):
                            is_closed = True
                            break
                    if is_closed:
                        cur_cluster.append(frag)
                    else:
                        cluster_index += 1
                        if not clusters.__contains__(cluster_index):
                            clusters[cluster_index] = []
                        cur_cluster = clusters[cluster_index]
                        cur_cluster.append(frag)

            for cluster_index in clusters.keys():
                cur_cluster = clusters[cluster_index]
                cur_cluster.sort(key=lambda x: (x[0], x[1]))

                # print('subject pos size: %d' %(len(cur_cluster)))
                # record visited fragments
                visited_frag = {}
                for i in range(len(cur_cluster)):
                    # keep a longest query start from each fragment
                    prev_frag = cur_cluster[i]
                    if visited_frag.__contains__(prev_frag):
                        continue
                    prev_query_start = prev_frag[0]
                    prev_query_end = prev_frag[1]
                    prev_subject_start = prev_frag[2]
                    prev_subject_end = prev_frag[3]
                    prev_query_seq = (min(prev_query_start, prev_query_end), max(prev_query_start, prev_query_end))
                    prev_subject_seq = (
                        min(prev_subject_start, prev_subject_end), max(prev_subject_start, prev_subject_end))
                    prev_query_len = abs(prev_query_end - prev_query_start)
                    prev_subject_len = abs(prev_subject_end - prev_subject_start)
                    cur_longest_query_len = prev_query_len

                    cur_extend_num = 0
                    visited_frag[prev_frag] = 1
                    # try to extend query
                    for j in range(i + 1, len(cur_cluster)):
                        cur_frag = cur_cluster[j]
                        if visited_frag.__contains__(cur_frag):
                            continue
                        cur_query_start = cur_frag[0]
                        cur_query_end = cur_frag[1]
                        cur_subject_start = cur_frag[2]
                        cur_subject_end = cur_frag[3]
                        cur_query_seq = (min(cur_query_start, cur_query_end), max(cur_query_start, cur_query_end))
                        cur_subject_seq = (min(cur_subject_start, cur_subject_end), max(cur_subject_start, cur_subject_end))

                        # could extend
                        # extend right
                        if cur_query_end > prev_query_end:
                            # judge subject direction
                            if prev_subject_start < prev_subject_end and cur_subject_start < cur_subject_end:
                                # +
                                if cur_subject_end > prev_subject_end:
                                    # forward extend
                                    if cur_query_start - prev_query_end < skip_gap and cur_query_end > prev_query_end \
                                            and cur_subject_start - prev_subject_end < skip_gap:  # \
                                        # and not is_same_query and not is_same_subject:
                                        # update the longest path
                                        prev_query_start = prev_query_start
                                        prev_query_end = cur_query_end
                                        prev_subject_start = prev_subject_start if prev_subject_start < cur_subject_start else cur_subject_start
                                        prev_subject_end = cur_subject_end
                                        cur_longest_query_len = prev_query_end - prev_query_start
                                        cur_extend_num += 1
                                        visited_frag[cur_frag] = 1
                                    elif cur_query_start - prev_query_end >= skip_gap:
                                        break
                            elif prev_subject_start > prev_subject_end and cur_subject_start > cur_subject_end:
                                # reverse
                                if cur_subject_end < prev_subject_end:
                                    # reverse extend
                                    if cur_query_start - prev_query_end < skip_gap and cur_query_end > prev_query_end \
                                            and prev_subject_end - cur_subject_start < skip_gap:  # \
                                        # and not is_same_query and not is_same_subject:
                                        # update the longest path
                                        prev_query_start = prev_query_start
                                        prev_query_end = cur_query_end
                                        prev_subject_start = prev_subject_start if prev_subject_start > cur_subject_start else cur_subject_start
                                        prev_subject_end = cur_subject_end
                                        cur_longest_query_len = prev_query_end - prev_query_start
                                        cur_extend_num += 1
                                        visited_frag[cur_frag] = 1
                                    elif cur_query_start - prev_query_end >= skip_gap:
                                        break
                    # keep this longest query
                    if cur_longest_query_len != -1:
                        longest_queries.append(
                            (prev_query_start, prev_query_end, cur_longest_query_len, prev_subject_start,
                             prev_subject_end, abs(prev_subject_end - prev_subject_start), subject_name,
                             cur_extend_num))

        # To determine whether each copy has a coverage exceeding the full_length_threshold with respect
        # to the consensus sequence, retaining full-length copies.
        query_copies = {}
        flank_query_copies = {}
        orig_query_len = len(query_contigs[query_name])
        # query_copies[query_name] = query_contigs[query_name]
        for repeat in longest_queries:
            if repeat[2] < full_length_threshold * query_len:
                continue
            # Subject
            subject_name = repeat[6]
            subject_chr_start = 0

            if repeat[3] > repeat[4]:
                direct = '-'
                old_subject_start_pos = repeat[4] - 1
                old_subject_end_pos = repeat[3]
            else:
                direct = '+'
                old_subject_start_pos = repeat[3] - 1
                old_subject_end_pos = repeat[4]
            subject_start_pos = subject_chr_start + old_subject_start_pos
            subject_end_pos = subject_chr_start + old_subject_end_pos

            subject_pos = subject_name + ':' + str(subject_start_pos) + '-' + str(subject_end_pos)
            subject_seq = ref_contigs[subject_name][subject_start_pos: subject_end_pos]

            flank_subject_seq = ref_contigs[subject_name][
                                subject_start_pos - flanking_len: subject_end_pos + flanking_len]
            copies_direct[subject_pos] = direct
            cur_query_len = repeat[2]
            coverage = float(cur_query_len) / orig_query_len
            if coverage >= full_length_threshold:
                query_copies[subject_pos] = subject_seq
                flank_query_copies[subject_pos] = flank_subject_seq
        full_length_copies[query_name] = query_copies
        flank_full_length_copies[query_name] = flank_query_copies

    # The candidate full-length copies and the consensus are then clustered using cd-hit-est,
    # retaining copies that belong to the same cluster as the consensus.
    split_files = []
    cluster_dir = tmp_output_dir + '/cluster'
    os.system('rm -rf ' + cluster_dir)
    if not os.path.exists(cluster_dir):
        os.makedirs(cluster_dir)

    all_query_copies = {}
    for query_name in full_length_copies.keys():
        query_copies = full_length_copies[query_name]
        flank_query_copies = flank_full_length_copies[query_name]
        all_query_copies.update(query_copies)
        fc_path = cluster_dir + '/' + query_name + '.fa'
        store_fasta(query_copies, fc_path)
        split_files.append((fc_path, query_name, query_copies, flank_query_copies))

    ex = ProcessPoolExecutor(threads)
    jobs = []
    for ref_index, cur_file in enumerate(split_files):
        input_file = cur_file[0]
        query_name = cur_file[1]
        query_copies = cur_file[2]
        flank_query_copies = cur_file[3]
        job = ex.submit(get_structure_info, input_file, query_name, query_copies,
                        flank_query_copies, cluster_dir, search_struct, tools_dir)
        jobs.append(job)
    ex.shutdown(wait=True)

    full_length_annotations = {}
    for job in as_completed(jobs):
        annotations = job.result()
        full_length_annotations.update(annotations)
    return full_length_annotations, copies_direct

def get_structure_info(input_file, query_name, query_copies, flank_query_copies, cluster_dir, search_struct, tools_dir):
    if str(query_name).__contains__('Helitron'):
        flanking_len = 5
    else:
        flanking_len = 50

    annotations = {}
    if search_struct:
        (file_dir, filename) = os.path.split(input_file)
        full_length_copies_file = input_file + '.copies.fa'
        store_fasta(query_copies, full_length_copies_file)
        flank_full_length_copies_file = input_file + '.flank.copies.fa'
        store_fasta(flank_query_copies, flank_full_length_copies_file)
        if not str(filename).__contains__('Helitron'):
            if str(filename).__contains__('TIR'):
                # get LTR/TIR length and identity for TIR transposons
                TIR_info = identify_terminals(full_length_copies_file, cluster_dir, tools_dir)
                for copy_name in query_copies:
                    TIR_str = 'tir='
                    if TIR_info.__contains__(copy_name):
                        lTIR_start, lTIR_end, rTIR_start, rTIR_end, identity = TIR_info[copy_name]
                        TIR_str += str(lTIR_start) + '-' + str(lTIR_end) + ',' + str(rTIR_start) + '-' + str(
                            rTIR_end) + ';tir_identity=' + str(identity)
                    else:
                        TIR_str += 'NA'
                    update_name = TIR_str

                    flank_seq = flank_query_copies[copy_name]
                    tir_start = flanking_len + 1
                    tir_end = len(flank_seq) - flanking_len
                    tsd_search_distance = flanking_len
                    cur_tsd, cur_tsd_len, min_distance = search_confident_tsd(flank_seq, tir_start, tir_end,
                                                                              tsd_search_distance)
                    update_name += ';tsd=' + cur_tsd + ';tsd_len=' + str(cur_tsd_len)
                    if not annotations.__contains__(query_name):
                        annotations[query_name] = []
                    annotation_list = annotations[query_name]
                    annotation_list.append((copy_name, update_name))
            elif str(filename).__contains__('Non_LTR'):
                # get TSD and polyA/T head or tail for non-ltr transposons
                for copy_name in query_copies:
                    sequence = query_copies[copy_name]
                    max_start, max_end, polyA = find_nearest_polyA_v1(sequence, min_length=6)
                    max_start, max_end, polyT = find_nearest_polyT_v1(sequence, min_length=6)
                    polyA_T = polyA if len(polyA) > len(polyT) else polyT
                    update_name = 'polya_t=' + polyA_T

                    flank_seq = flank_query_copies[copy_name]
                    tir_start = flanking_len + 1
                    tir_end = len(flank_seq) - flanking_len
                    tsd_search_distance = flanking_len
                    cur_tsd, cur_tsd_len, min_distance = search_confident_tsd(flank_seq, tir_start, tir_end,
                                                                              tsd_search_distance)
                    update_name += ';tsd=' + cur_tsd + ';tsd_len=' + str(cur_tsd_len)
                    if not annotations.__contains__(query_name):
                        annotations[query_name] = []
                    annotation_list = annotations[query_name]
                    annotation_list.append((copy_name, update_name))
        else:
            # search for hairpin loop
            EAHelitron = os.getcwd() + '/../bin/EAHelitron-master'
            copies_hairpin_loops = run_EAHelitron_v1(cluster_dir, flank_full_length_copies_file, EAHelitron, query_name)
            for copy_name in query_copies:
                if copies_hairpin_loops.__contains__(copy_name):
                    hairpin_loop = copies_hairpin_loops[copy_name]
                else:
                    hairpin_loop = 'NA'
                update_name = 'hairpin_loop=' + hairpin_loop
                if not annotations.__contains__(query_name):
                    annotations[query_name] = []
                annotation_list = annotations[query_name]
                annotation_list.append((copy_name, update_name))
    else:
        if not annotations.__contains__(query_name):
            annotations[query_name] = []
        annotation_list = annotations[query_name]
        for copy_name in query_copies.keys():
            annotation_list.append((copy_name, ''))
    return annotations

def identify_terminals(split_file, output_dir, tool_dir):
    #ltr_log = split_file + '.ltr.log'
    tir_log = split_file + '.itr.log'
    #ltrsearch_command = 'cd ' + output_dir + ' && ' + tool_dir + '/ltrsearch -l 50 ' + split_file + ' > ' + ltr_log
    itrsearch_command = 'cd ' + output_dir + ' && ' + tool_dir + '/itrsearch -i 0.7 -l 7 ' + split_file+ ' > ' + tir_log
    #run_command(ltrsearch_command)
    run_command(itrsearch_command)
    ltr_file = split_file + '.ltr'
    tir_file = split_file + '.itr'

    tir_identity_dict = {}
    sequence_id = None
    with open(tir_log, 'r') as f_r:
        for line in f_r:
            if line.startswith('load sequence'):
                sequence_id = line.split('\t')[0].split(' ')[3]
            elif line.__contains__('Identity percentage') and sequence_id is not None:
                identity = float(line.split(':')[1].strip())
                tir_identity_dict[sequence_id] = identity
                sequence_id = None

    tir_names, tir_contigs = read_fasta_v1(tir_file)
    TIR_info = {}
    for i, tir_name in enumerate(tir_names):
        parts = tir_name.split('\t')
        orig_name = parts[0].split(' ')[0]
        terminal_info = parts[-1]
        TIR_info_parts = terminal_info.split('ITR')[1].split(' ')[0].replace('(', '').replace(')', '').split('..')
        TIR_left_pos_parts = TIR_info_parts[0].split(',')
        TIR_right_pos_parts = TIR_info_parts[1].split(',')
        lTIR_start = int(TIR_left_pos_parts[0])
        lTIR_end = int(TIR_left_pos_parts[1])
        rTIR_start = int(TIR_right_pos_parts[1])
        rTIR_end = int(TIR_right_pos_parts[0])
        TIR_info[orig_name] = (lTIR_start, lTIR_end, rTIR_start, rTIR_end, tir_identity_dict[orig_name])
    return TIR_info

def run_command(command):
    subprocess.run(command, check=True, shell=True)

def search_confident_tsd(orig_seq, raw_tir_start, raw_tir_end, tsd_search_distance):
    # Change all coordinates to start from 0.
    raw_tir_start -= 1
    raw_tir_end -= 1

    orig_seq_len = len(orig_seq)
    # 1. First, take 2 * tsd_search_distance sequences near the start and end positions
    left_start = raw_tir_start - tsd_search_distance
    if left_start < 0:
        left_start = 0
    # We don’t search inwards here because we consider Repbase boundaries to be correct.
    # If we consider the boundaries to be incorrect, many abnormal TSDs may meet the requirements.
    # For simplicity, we assume that Repbase boundaries are correct.
    left_end = raw_tir_start
    left_round_seq = orig_seq[left_start: left_end]
    # Obtain the position offset of left_round_seq relative to the entire sequence to correct the subsequent TSD boundary positions.
    left_offset = left_start
    right_start = raw_tir_end + 1
    if right_start < 0:
        right_start = 0
    right_end = raw_tir_end + tsd_search_distance + 1
    right_round_seq = orig_seq[right_start: right_end]
    # Obtain the position offset of right_round_seq relative to the entire sequence to correct the subsequent TSD boundary positions.
    right_offset = right_start

    # 2. Split the left sequence into k-mers from large to small, then search for the right sequence with k-mers.
    # If found, record as a candidate TSD, and finally select the closest one to the original boundary as the TSD.
    TIR_TSDs = [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2]
    # Record the position nearest to the original boundary.
    is_found = False
    tsd_set = []
    for k_num in TIR_TSDs:
        for i in range(len(left_round_seq) - k_num, -1, -1):
            left_kmer = left_round_seq[i: i + k_num]
            left_pos = left_offset + i + k_num
            if left_pos < 0 or left_pos > orig_seq_len-1:
                continue
            found_tsd, right_pos = search_TSD_regular(left_kmer, right_round_seq)
            if found_tsd and not left_kmer.__contains__('N'):
                right_pos = right_offset + right_pos - 1
                is_found = True
                # Calculate the distance from the original boundary.
                left_distance = abs(left_pos - raw_tir_start)
                right_distance = abs(right_pos - raw_tir_end)
                distance = left_distance + right_distance
                TSD_seq = left_kmer
                TSD_len = len(TSD_seq)
                tsd_set.append((distance, TSD_len, TSD_seq))
    tsd_set = sorted(tsd_set, key=lambda x: (x[0], -x[1]))

    if not is_found:
        TSD_seq = 'NA'
        TSD_len = 'NA'
        min_distance = -1
    else:
        TSD_seq = tsd_set[0][2]
        TSD_len = tsd_set[0][1]
        min_distance = tsd_set[0][0]
    return TSD_seq, TSD_len, min_distance

def find_nearest_polyA_v1(sequence, search_range=30, min_length=6):
    max_length = 0
    max_start = -1
    max_end = -1

    # 在序列开头处查找多聚A结构
    current_length = 0
    start = 0
    for i, base in enumerate(sequence):
        if i >= search_range:
            break
        if base == 'A':
            current_length += 1
            if current_length == 1:
                start = i
        else:
            if current_length >= min_length and current_length > max_length:
                max_length = current_length
                max_start = start
                max_end = i
            current_length = 0

    # 更新最长多聚A结构的起始和结束位置
    if current_length >= min_length and current_length > max_length:
        max_start = start
        max_end = len(sequence)
    seq1 = sequence[max_start:max_end]

    # 在序列结尾处查找多聚A结构
    current_length = 0
    start = 0
    for i in range(len(sequence) - 1, -1, -1):
        if len(sequence) - i >= search_range:
            break
        if sequence[i] == 'A':
            current_length += 1
            if current_length == 1:
                start = i
        else:
            if current_length >= min_length and current_length > max_length:
                max_length = current_length
                max_start = start
                max_end = i + 1
            current_length = 0
    seq2 = sequence[max_end: max_start+1]

    seq = seq1 if len(seq1) > len(seq2) else seq2
    return max_start, max_end, seq

def find_nearest_polyT_v1(sequence, search_range=30, min_length=6):
    max_length = 0
    max_start = -1
    max_end = -1

    # 在序列开头处查找多聚T结构
    current_length = 0
    start = 0
    for i, base in enumerate(sequence):
        if i >= search_range:
            break
        if base == 'T':
            current_length += 1
            if current_length == 1:
                start = i
        else:
            if current_length >= min_length and current_length > max_length:
                max_length = current_length
                max_start = start
                max_end = i
            current_length = 0

    # 更新最长多聚A结构的起始和结束位置
    if current_length >= min_length and current_length > max_length:
        max_start = start
        max_end = len(sequence)
    seq1 = sequence[max_start:max_end]

    # 在序列结尾处查找多聚A结构
    current_length = 0
    start = 0
    for i in range(len(sequence) - 1, -1, -1):
        if len(sequence) - i >= search_range:
            break
        if sequence[i] == 'T':
            current_length += 1
            if current_length == 1:
                start = i
        else:
            if current_length >= min_length and current_length > max_length:
                max_length = current_length
                max_start = start
                max_end = i + 1
            current_length = 0
    seq2 = sequence[max_end: max_start+1]

    seq = seq1 if len(seq1) > len(seq2) else seq2
    return max_start, max_end, seq

def run_EAHelitron_v1(temp_dir, all_candidate_helitron_path, EAHelitron, partition_index):
    # 输入是Helitron序列，输出是hairpin loop序列
    all_candidate_helitron_contigs = {}
    contigNames, contigs = read_fasta(all_candidate_helitron_path)
    for query_name in contigNames:
        seq = contigs[query_name]
        all_candidate_helitron_contigs[query_name] = seq
    store_fasta(all_candidate_helitron_contigs, all_candidate_helitron_path)
    EAHelitron_command = 'cd ' + temp_dir + ' && ' + 'perl ' + EAHelitron + '/EAHelitron -o ' + str(partition_index) + ' -u 20000 -T "TC" -r 3 ' + all_candidate_helitron_path
    os.system(EAHelitron_command + '> /dev/null 2>&1')

    all_EAHelitron_res = temp_dir + '/' + str(partition_index) + '.3.txt'
    all_copies_out_names, all_copies_out_contigs = read_fasta_v1(all_EAHelitron_res)
    # search for hairpin loop sequence
    copies_hairpin_loops = {}
    for cur_name in all_copies_out_contigs.keys():
        name_parts = cur_name.split(' ')
        raw_name = name_parts[1]
        parts = raw_name.split(':')
        query_name = ':'.join(parts[:-1])
        forward_loop = name_parts[3]
        mid_loop = name_parts[4]
        reverse_loop = getReverseSequence(forward_loop)
        hairpin_loop_seq = forward_loop + mid_loop + reverse_loop
        r_hairpin_loop_seq = getReverseSequence(hairpin_loop_seq)
        cur_tail_seq = all_copies_out_contigs[cur_name]
        if cur_tail_seq.__contains__(hairpin_loop_seq):
            final_hairpin_loop_seq = hairpin_loop_seq
        elif cur_tail_seq.__contains__(r_hairpin_loop_seq):
            final_hairpin_loop_seq = r_hairpin_loop_seq
        else:
            final_hairpin_loop_seq = 'None'
        copies_hairpin_loops[query_name] = final_hairpin_loop_seq
    return copies_hairpin_loops

def search_TSD_regular(motif, sequence):
    motif_length = len(motif)
    pattern = ''

    # Build a regular expression pattern based on motif length.
    if motif_length >= 8:
        for i in range(motif_length):
            pattern += f"{motif[:i]}[ACGT]{motif[i + 1:]}" if i < motif_length - 1 else motif[:i] + "[ACGT]"
            if i < motif_length - 1:
                pattern += "|"
    else:
        pattern = motif

    matches = re.finditer(pattern, sequence)

    found = False
    pos = None
    for match in matches:
        #print(f"Found motif at position {match.start()}: {match.group()}")
        found = True
        pos = match.start()
        break
    return found, pos

def is_recombination(query_seq, subject_seq, candidate_index):
    exec_command = f"blastn -subject <(echo -e '{subject_seq}') -query <(echo -e '{query_seq}') -outfmt 6"
    query_len = len(query_seq)
    result = subprocess.run(exec_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            executable='/bin/bash')
    if result.returncode == 0:
        lines = result.stdout.split('\n')
        for line in lines:
            parts = line.split('\t')
            if len(parts) != 12:
                continue
            query_start = int(parts[6])
            query_end = int(parts[7])
            if abs(query_end - query_start) / query_len >= 0.95:
                return True, candidate_index
    return False, candidate_index

def get_recombination_ltr(ltr_candidates, ref_contigs, threads, log):
    ex = ProcessPoolExecutor(threads)
    jobs = []
    for candidate_index in ltr_candidates.keys():
        (chr_name, left_ltr_start, left_ltr_end, right_ltr_start, right_ltr_end) = ltr_candidates[candidate_index]
        if chr_name not in ref_contigs:
            log.logger.error(
                'Error: Chromosome names in the SCN file do not match the input genome names. Please correct this and rerun.')
            exit(-1)
        ref_seq = ref_contigs[chr_name]
        left_ltr_name = chr_name + ':' + str(left_ltr_start) + '-' + str(left_ltr_end)
        left_ltr_seq = ref_seq[left_ltr_start - 1: left_ltr_end]

        int_ltr_name = chr_name + ':' + str(left_ltr_end) + '-' + str(right_ltr_start)
        int_ltr_seq = ref_seq[left_ltr_end: right_ltr_start - 1]
        job = ex.submit(is_recombination, left_ltr_seq, int_ltr_seq, candidate_index)
        jobs.append(job)
    ex.shutdown(wait=True)

    recombination_candidates = []
    for job in as_completed(jobs):
        cur_is_recombination, cur_candidate_index = job.result()
        if cur_is_recombination:
            recombination_candidates.append(cur_candidate_index)
    return recombination_candidates


def deredundant_for_LTR(redundant_ltr, work_dir, threads):
    # We found that performing a direct mafft alignment on the redundant LTR library was too slow.
    # Therefore, we first need to use Blastn for alignment clustering, and then proceed with mafft processing.
    tmp_blast_dir = work_dir + '/LTR_blastn'
    blastnResults_path = work_dir + '/LTR_blastn.out'
    # 1. Start by performing an all-vs-all comparison using blastn.
    multi_process_align(redundant_ltr, redundant_ltr, blastnResults_path, tmp_blast_dir, threads, is_removed_dir=True)
    if not os.path.exists(blastnResults_path):
        return redundant_ltr
    # 2. Next, using the FMEA algorithm, bridge across the gaps and link together sequences that can be connected.
    full_length_threshold = 0.95
    longest_repeats = FMEA_new(redundant_ltr, blastnResults_path, full_length_threshold)
    # 3. If the combined sequence length constitutes 95% or more of the original individual sequence lengths, we place these two sequences into a cluster.
    contigNames, contigs = read_fasta(redundant_ltr)
    keep_clusters = []
    relations = set()
    for query_name in longest_repeats.keys():
        longest_repeats_list = longest_repeats[query_name]
        for cur_longest_repeat in longest_repeats_list:
            query_name = cur_longest_repeat[0]
            query_len = len(contigs[query_name])
            q_len = abs(cur_longest_repeat[2] - cur_longest_repeat[1])
            subject_name = cur_longest_repeat[3]
            subject_len = len(contigs[subject_name])
            s_len = abs(cur_longest_repeat[4] - cur_longest_repeat[5])
            # 我们这里先将跨过 gap 之后的全长拷贝先聚类在一起，后续再使用 cd-hit 将碎片化合并到全长拷贝中
            if float(q_len) / query_len >= 0.95 and float(s_len) / subject_len >= 0.95:
                # we consider the query and subject to be from the same family.
                if (query_name, subject_name) in relations:
                    continue
                relations.add((query_name, subject_name))
                relations.add((subject_name, query_name))
                is_new_cluster = True
                for cluster in keep_clusters:
                    if query_name in cluster or subject_name in cluster:
                        is_new_cluster = False
                        cluster.add(query_name)
                        cluster.add(subject_name)
                        break
                if is_new_cluster:
                    new_cluster = set()
                    new_cluster.add(query_name)
                    new_cluster.add(subject_name)
                    keep_clusters.append(new_cluster)
                    # print(keep_clusters)
    # Iterate through each cluster, if any element in the cluster overlaps with elements in other clusters, merge the clusters.
    merged_clusters = []
    while keep_clusters:
        current_cluster = keep_clusters.pop(0)
        for other_cluster in keep_clusters[:]:
            if current_cluster.intersection(other_cluster):
                current_cluster.update(other_cluster)
                keep_clusters.remove(other_cluster)
        merged_clusters.append(current_cluster)
    keep_clusters = merged_clusters
    # store cluster
    all_unique_name = set()
    raw_cluster_files = []
    cluster_dir = work_dir + '/raw_ltr_cluster'
    if not os.path.exists(cluster_dir):
        os.makedirs(cluster_dir)
    for cluster_id, cur_cluster in enumerate(keep_clusters):
        cur_cluster_path = cluster_dir + '/' + str(cluster_id) + '.fa'
        cur_cluster_contigs = {}
        for ltr_name in cur_cluster:
            cur_cluster_contigs[ltr_name] = contigs[ltr_name]
            all_unique_name.add(ltr_name)
        store_fasta(cur_cluster_contigs, cur_cluster_path)
        raw_cluster_files.append((cluster_id, cur_cluster_path))
    # We save the sequences that did not appear in any clusters separately. These sequences do not require clustering.
    uncluster_path = work_dir + '/uncluster_ltr.fa'
    uncluster_contigs = {}
    for name in contigNames:
        if name not in all_unique_name:
            uncluster_contigs[name] = contigs[name]
    store_fasta(uncluster_contigs, uncluster_path)

    # 4. The final cluster should encompass all instances from the same family.
    # We use Ninja to cluster families precisely, and
    # We then use the mafft+majority principle to generate a consensus sequence for each cluster.
    ex = ProcessPoolExecutor(threads)
    jobs = []
    for cluster_id, cur_cluster_path in raw_cluster_files:
        job = ex.submit(generate_cons, cluster_id, cur_cluster_path, cluster_dir)
        jobs.append(job)
    ex.shutdown(wait=True)
    all_cons = {}
    name_maps = {}
    for job in as_completed(jobs):
        cur_cons_contigs, cur_name_maps = job.result()
        all_cons.update(cur_cons_contigs)
        name_maps.update(cur_name_maps)
    all_cons.update(uncluster_contigs)
    ltr_cons_path = redundant_ltr + '.tmp.cons'
    store_fasta(all_cons, ltr_cons_path)

    cons_map_path = work_dir + '/cons_name.map'
    with open(cons_map_path, 'w', encoding='utf-8') as f:
        json.dump(name_maps, f, ensure_ascii=False, indent=4)

    ltr_cons_cons = redundant_ltr + '.cons'
    # 调用 cd-hit-est 合并碎片化序列
    cd_hit_command = 'cd-hit-est -aS ' + str(0.95) + ' -aL ' + str(0.95) + ' -c ' + str(0.8) \
                     + ' -G 0 -g 1 -A 80 -i ' + ltr_cons_path + ' -o ' + ltr_cons_cons + ' -T 0 -M 0'
    os.system(cd_hit_command + ' > /dev/null 2>&1')

    #rename_fasta(ltr_cons_path, ltr_cons_path, 'LTR')
    return ltr_cons_path

def cons_from_mafft(align_file):
    align_names, align_contigs = read_fasta(align_file)
    if len(align_names) <= 0:
        return None

    # Generate a consensus sequence using full-length copies.
    first_seq = align_contigs[align_names[0]]
    col_num = len(first_seq)
    row_num = len(align_names)
    matrix = [[''] * col_num for i in range(row_num)]
    for row, name in enumerate(align_names):
        seq = align_contigs[name]
        for col in range(len(seq)):
            matrix[row][col] = seq[col]
    # Record the base composition of each column.
    col_base_map = {}
    for col_index in range(col_num):
        if not col_base_map.__contains__(col_index):
            col_base_map[col_index] = {}
        base_map = col_base_map[col_index]
        # Calculate the percentage of each base in the current column.
        if len(base_map) == 0:
            for row in range(row_num):
                cur_base = matrix[row][col_index]
                if not base_map.__contains__(cur_base):
                    base_map[cur_base] = 0
                cur_count = base_map[cur_base]
                cur_count += 1
                base_map[cur_base] = cur_count
        if not base_map.__contains__('-'):
            base_map['-'] = 0

    ## Generate a consensus sequence.
    model_seq = ''
    for col_index in range(col_num):
        base_map = col_base_map[col_index]
        # Identify the most frequently occurring base if it exceeds the threshold valid_col_threshold.
        max_base_count = 0
        max_base = ''
        for cur_base in base_map.keys():
            if cur_base == '-':
                continue
            cur_count = base_map[cur_base]
            if cur_count > max_base_count:
                max_base_count = cur_count
                max_base = cur_base
        if max_base_count >= int(row_num / 2):
            if max_base != '-':
                model_seq += max_base
            else:
                continue
        # else:
        #     # Here, we do not use 'N' because it can make it difficult to find the boundary. Therefore, we take the base with the highest non-empty count.
        #     max_base_count = 0
        #     max_base = ''
        #     for cur_base in base_map.keys():
        #         if cur_base == '-':
        #             continue
        #         cur_count = base_map[cur_base]
        #         if cur_count > max_base_count:
        #             max_base_count = cur_count
        #             max_base = cur_base
        #     model_seq += max_base
    return model_seq

def generate_cons(cluster_id, cur_cluster_path, cluster_dir):
    ltr_internal_names, ltr_internal_contigs = read_fasta(cur_cluster_path)
    temp_cluster_dir = cluster_dir
    # 记录下cons name 与 raw name的对应关系
    name_maps = {}
    cons_contigs = {}
    ltr_prefix = 'ltr_cons-' + str(cluster_id)
    if len(ltr_internal_contigs) >= 1:
        align_file = cur_cluster_path + '.maf.fa'
        align_command = 'cd ' + cluster_dir + ' && mafft --preservecase --quiet --thread -1 ' + cur_cluster_path + ' > ' + align_file
        os.system(align_command)

        # 调用 Ninja 对多序列比对再次聚类
        cluster_file = align_file + '.dat'
        Ninja_command = 'Ninja --in ' + align_file + ' --out ' + cluster_file + ' --out_type c --corr_type m --cluster_cutoff 0.2 --threads 1'
        os.system(Ninja_command + ' > /dev/null 2>&1')

        # 解析聚类文件，生成不同簇
        Ninja_cluster_dir = temp_cluster_dir + '/Ninja_' + str(cluster_id)
        if not os.path.exists(Ninja_cluster_dir):
            os.makedirs(Ninja_cluster_dir)
        clusters = read_Ninja_clusters(cluster_file)
        for cur_cluster_id in clusters.keys():
            cur_cluster_file = Ninja_cluster_dir + '/' + str(cur_cluster_id) + '.fa'
            cur_cluster_contigs = {}
            for name in clusters[cur_cluster_id]:
                seq = ltr_internal_contigs[name]
                cur_cluster_contigs[name] = seq
            store_fasta(cur_cluster_contigs, cur_cluster_file)
            cur_ltr_name = ltr_prefix + '-' + str(cur_cluster_id)
            cur_ltr_name += '-' + str(random.randint(1, 1000))
            name_maps[cur_ltr_name] = list(clusters[cur_cluster_id])
            cur_align_file = cur_cluster_file + '.maf.fa'
            if len(cur_cluster_contigs) >= 1:
                align_command = 'cd ' + Ninja_cluster_dir + ' && mafft --preservecase --quiet --thread -1 ' + cur_cluster_file + ' > ' + cur_align_file
                os.system(align_command)
                cons_seq = cons_from_mafft(cur_align_file)
                cons_contigs[cur_ltr_name] = cons_seq

    return cons_contigs, name_maps

def file_exist(resut_file):
    if os.path.exists(resut_file) and os.path.getsize(resut_file) > 0:
        if resut_file.endswith('.fa') or resut_file.endswith('.fasta'):
            names, contigs = read_fasta(resut_file)
            if len(contigs) > 0:
                return True
            else:
                return False
        else:
            line_count = 0
            with open(resut_file, 'r') as f_r:
                for line in f_r:
                    if line.startswith('#'):
                        continue
                    line_count += 1
                    if line_count > 0:
                        return True
            f_r.close()
            return False
    else:
        return False

def FMEA_LTR(blastn2Results_path, fixed_extend_base_threshold):
    # parse blastn output, determine the repeat boundary
    # query_records = {query_name: {subject_name: [(q_start, q_end, s_start, s_end), (q_start, q_end, s_start, s_end), (q_start, q_end, s_start, s_end)] }}
    query_records = {}
    with open(blastn2Results_path, 'r') as f_r:
        for idx, line in enumerate(f_r):
            # print('current line idx: %d' % (idx))
            parts = line.split('\t')
            query_name = parts[0]
            subject_name = parts[1]
            identity = float(parts[2])
            alignment_len = int(parts[3])
            q_start = int(parts[6])
            q_end = int(parts[7])
            s_start = int(parts[8])
            s_end = int(parts[9])
            if query_name == subject_name:
                continue
            if not query_records.__contains__(query_name):
                query_records[query_name] = {}
            subject_dict = query_records[query_name]

            if not subject_dict.__contains__(subject_name):
                subject_dict[subject_name] = []
            subject_pos = subject_dict[subject_name]
            subject_pos.append((q_start, q_end, s_start, s_end))
    f_r.close()

    skip_gap = fixed_extend_base_threshold
    longest_repeats = {}
    for idx, query_name in enumerate(query_records.keys()):
        subject_dict = query_records[query_name]

        # if there are more than one longest query overlap with the final longest query over 90%,
        # then it probably the true TE
        longest_queries = []
        for subject_name in subject_dict.keys():
            subject_pos = subject_dict[subject_name]

            # cluster all closed fragments, split forward and reverse records
            forward_pos = []
            reverse_pos = []
            for pos_item in subject_pos:
                if pos_item[2] > pos_item[3]:
                    reverse_pos.append(pos_item)
                else:
                    forward_pos.append(pos_item)
            forward_pos.sort(key=lambda x: (x[2], x[3]))
            reverse_pos.sort(key=lambda x: (-x[2], -x[3]))

            clusters = {}
            cluster_index = 0
            for k, frag in enumerate(forward_pos):
                if not clusters.__contains__(cluster_index):
                    clusters[cluster_index] = []
                cur_cluster = clusters[cluster_index]
                if k == 0:
                    cur_cluster.append(frag)
                else:
                    is_closed = False
                    for exist_frag in reversed(cur_cluster):
                        cur_subject_start = frag[2]
                        cur_query_end = frag[1]
                        prev_subject_end = exist_frag[3]
                        prev_query_end = exist_frag[1]
                        if (cur_subject_start - prev_subject_end < skip_gap and cur_query_end > prev_query_end):
                            is_closed = True
                            break
                    if is_closed:
                        cur_cluster.append(frag)
                    else:
                        cluster_index += 1
                        if not clusters.__contains__(cluster_index):
                            clusters[cluster_index] = []
                        cur_cluster = clusters[cluster_index]
                        cur_cluster.append(frag)

            cluster_index += 1
            for k, frag in enumerate(reverse_pos):
                if not clusters.__contains__(cluster_index):
                    clusters[cluster_index] = []
                cur_cluster = clusters[cluster_index]
                if k == 0:
                    cur_cluster.append(frag)
                else:
                    is_closed = False
                    for exist_frag in reversed(cur_cluster):
                        cur_subject_start = frag[2]
                        cur_query_end = frag[1]
                        prev_subject_end = exist_frag[3]
                        prev_query_end = exist_frag[1]
                        if (prev_subject_end - cur_subject_start < skip_gap and cur_query_end > prev_query_end):
                            is_closed = True
                            break
                    if is_closed:
                        cur_cluster.append(frag)
                    else:
                        cluster_index += 1
                        if not clusters.__contains__(cluster_index):
                            clusters[cluster_index] = []
                        cur_cluster = clusters[cluster_index]
                        cur_cluster.append(frag)


            for cluster_index in clusters.keys():
                cur_cluster = clusters[cluster_index]
                cur_cluster.sort(key=lambda x: (x[0], x[1]))

                # print('subject pos size: %d' %(len(cur_cluster)))
                # record visited fragments
                visited_frag = {}
                for i in range(len(cur_cluster)):
                    # keep a longest query start from each fragment
                    prev_frag = cur_cluster[i]
                    if visited_frag.__contains__(prev_frag):
                        continue
                    prev_query_start = prev_frag[0]
                    prev_query_end = prev_frag[1]
                    prev_subject_start = prev_frag[2]
                    prev_subject_end = prev_frag[3]
                    prev_query_seq = (min(prev_query_start, prev_query_end), max(prev_query_start, prev_query_end))
                    prev_subject_seq = (
                        min(prev_subject_start, prev_subject_end), max(prev_subject_start, prev_subject_end))
                    prev_query_len = abs(prev_query_end - prev_query_start)
                    prev_subject_len = abs(prev_subject_end - prev_subject_start)
                    cur_longest_query_len = prev_query_len

                    cur_extend_num = 0
                    visited_frag[prev_frag] = 1
                    # try to extend query
                    for j in range(i + 1, len(cur_cluster)):
                        cur_frag = cur_cluster[j]
                        if visited_frag.__contains__(cur_frag):
                            continue
                        cur_query_start = cur_frag[0]
                        cur_query_end = cur_frag[1]
                        cur_subject_start = cur_frag[2]
                        cur_subject_end = cur_frag[3]
                        cur_query_seq = (min(cur_query_start, cur_query_end), max(cur_query_start, cur_query_end))
                        cur_subject_seq = (
                            min(cur_subject_start, cur_subject_end), max(cur_subject_start, cur_subject_end))
                        cur_query_len = abs(cur_query_end - cur_query_start)
                        cur_subject_len = abs(cur_subject_end - cur_subject_start)

                        query_overlap_len = get_overlap_len(cur_query_seq, prev_query_seq)
                        is_same_query = float(query_overlap_len) / cur_query_len >= 0.5 or float(
                            query_overlap_len) / prev_query_len >= 0.5
                        subject_overlap_len = get_overlap_len(prev_subject_seq, cur_subject_seq)
                        is_same_subject = float(subject_overlap_len) / cur_subject_len >= 0.5 or float(
                            subject_overlap_len) / prev_subject_len >= 0.5

                        # could extend
                        # extend right
                        if cur_query_end > prev_query_end:
                            # judge subject direction
                            if prev_subject_start < prev_subject_end and cur_subject_start < cur_subject_end:
                                # +
                                if cur_subject_end > prev_subject_end:
                                    # forward extend
                                    if cur_query_start - prev_query_end < skip_gap and cur_query_end > prev_query_end \
                                            and cur_subject_start - prev_subject_end < skip_gap:  # \
                                        # and not is_same_query and not is_same_subject:
                                        # update the longest path
                                        prev_query_start = prev_query_start
                                        prev_query_end = cur_query_end
                                        prev_subject_start = prev_subject_start if prev_subject_start < cur_subject_start else cur_subject_start
                                        prev_subject_end = cur_subject_end
                                        cur_longest_query_len = prev_query_end - prev_query_start
                                        cur_extend_num += 1
                                        visited_frag[cur_frag] = 1
                                    elif cur_query_start - prev_query_end >= skip_gap:
                                        break
                            elif prev_subject_start > prev_subject_end and cur_subject_start > cur_subject_end:
                                # reverse
                                if cur_subject_end < prev_subject_end:
                                    # reverse extend
                                    if cur_query_start - prev_query_end < skip_gap and cur_query_end > prev_query_end \
                                            and prev_subject_end - cur_subject_start < skip_gap:  # \
                                        # and not is_same_query and not is_same_subject:
                                        # update the longest path
                                        prev_query_start = prev_query_start
                                        prev_query_end = cur_query_end
                                        prev_subject_start = prev_subject_start if prev_subject_start > cur_subject_start else cur_subject_start
                                        prev_subject_end = cur_subject_end
                                        cur_longest_query_len = prev_query_end - prev_query_start
                                        cur_extend_num += 1
                                        visited_frag[cur_frag] = 1
                                    elif cur_query_start - prev_query_end >= skip_gap:
                                        break
                    # keep this longest query
                    if cur_longest_query_len != -1:
                        longest_queries.append(
                            (prev_query_start, prev_query_end, cur_longest_query_len, prev_subject_start,
                             prev_subject_end, abs(prev_subject_end - prev_subject_start), subject_name,
                             cur_extend_num))
        if not longest_repeats.__contains__(query_name):
            longest_repeats[query_name] = []
        cur_longest_repeats = longest_repeats[query_name]
        for repeat in longest_queries:
            # Subject序列处理流程
            subject_name = repeat[6]
            old_subject_start_pos = repeat[3] - 1
            old_subject_end_pos = repeat[4]
            # Query序列处理流程
            old_query_start_pos = repeat[0] - 1
            old_query_end_pos = repeat[1]
            cur_query_seq_len = abs(old_query_end_pos - old_query_start_pos)
            cur_longest_repeats.append((query_name, old_query_start_pos, old_query_end_pos, subject_name, old_subject_start_pos, old_subject_end_pos))

    return longest_repeats

def FMEA_new(query_path, blastn2Results_path, full_length_threshold):
    # parse blastn output, determine the repeat boundary
    # query_records = {query_name: {subject_name: [(q_start, q_end, s_start, s_end), (q_start, q_end, s_start, s_end), (q_start, q_end, s_start, s_end)] }}
    query_records = {}
    with open(blastn2Results_path, 'r') as f_r:
        for idx, line in enumerate(f_r):
            # print('current line idx: %d' % (idx))
            parts = line.split('\t')
            query_name = parts[0]
            subject_name = parts[1]
            identity = float(parts[2])
            alignment_len = int(parts[3])
            q_start = int(parts[6])
            q_end = int(parts[7])
            s_start = int(parts[8])
            s_end = int(parts[9])
            if query_name == subject_name and q_start == s_start and q_end == s_end:
                continue
            if not query_records.__contains__(query_name):
                query_records[query_name] = {}
            subject_dict = query_records[query_name]

            if not subject_dict.__contains__(subject_name):
                subject_dict[subject_name] = []
            subject_pos = subject_dict[subject_name]
            subject_pos.append((q_start, q_end, s_start, s_end))
    f_r.close()

    query_names, query_contigs = read_fasta(query_path)

    # 我们现在尝试新的策略，直接在生成簇的时候进行扩展，同时新的比对片段和所有的扩展片段进行比较，判断是否可以扩展
    longest_repeats = {}
    for idx, query_name in enumerate(query_records.keys()):
        subject_dict = query_records[query_name]

        if query_name not in query_contigs:
            continue
        query_len = len(query_contigs[query_name])
        skip_gap = query_len * full_length_threshold

        longest_queries = []
        for subject_name in subject_dict.keys():
            subject_pos = subject_dict[subject_name]

            # if query_name == 'Chr1_4990099-4997254-int#LTR' and subject_name == 'Chr451_1655236-1664338-int#LTR':
            #     print('h')

            # cluster all closed fragments, split forward and reverse records
            forward_pos = []
            reverse_pos = []
            for pos_item in subject_pos:
                if pos_item[2] > pos_item[3]:
                    reverse_pos.append(pos_item)
                else:
                    forward_pos.append(pos_item)
            forward_pos.sort(key=lambda x: (x[2], x[3]))
            reverse_pos.sort(key=lambda x: (-x[2], -x[3]))

            forward_long_frags = {}
            frag_index = 0
            for k, frag in enumerate(forward_pos):
                is_update = False
                cur_subject_start = frag[2]
                cur_subject_end = frag[3]
                cur_query_start = frag[0]
                cur_query_end = frag[1]
                for cur_frag_index in forward_long_frags.keys():
                    cur_frag = forward_long_frags[cur_frag_index]
                    prev_subject_start = cur_frag[2]
                    prev_subject_end = cur_frag[3]
                    prev_query_start = cur_frag[0]
                    prev_query_end = cur_frag[1]

                    if cur_subject_end > prev_subject_end:
                        # forward extend
                        if cur_query_start - prev_query_end < skip_gap and cur_query_end > prev_query_end \
                                and cur_subject_start - prev_subject_end < skip_gap:  # \
                            # extend frag
                            prev_query_start = prev_query_start
                            prev_query_end = cur_query_end
                            prev_subject_start = prev_subject_start if prev_subject_start < cur_subject_start else cur_subject_start
                            prev_subject_end = cur_subject_end
                            extend_frag = (prev_query_start, prev_query_end, prev_subject_start, prev_subject_end, subject_name)
                            forward_long_frags[cur_frag_index] = extend_frag
                            is_update = True
                if not is_update:
                    forward_long_frags[frag_index] = (cur_query_start, cur_query_end, cur_subject_start, cur_subject_end, subject_name)
                    frag_index += 1
            longest_queries += list(forward_long_frags.values())

            reverse_long_frags = {}
            frag_index = 0
            for k, frag in enumerate(reverse_pos):
                is_update = False
                cur_subject_start = frag[2]
                cur_subject_end = frag[3]
                cur_query_start = frag[0]
                cur_query_end = frag[1]
                for cur_frag_index in reverse_long_frags.keys():
                    cur_frag = reverse_long_frags[cur_frag_index]
                    prev_subject_start = cur_frag[2]
                    prev_subject_end = cur_frag[3]
                    prev_query_start = cur_frag[0]
                    prev_query_end = cur_frag[1]

                    # reverse
                    if cur_subject_end < prev_subject_end:
                        # reverse extend
                        if cur_query_start - prev_query_end < skip_gap and cur_query_end > prev_query_end \
                                and prev_subject_end - cur_subject_start < skip_gap:  # \
                            # extend frag
                            prev_query_start = prev_query_start
                            prev_query_end = cur_query_end
                            prev_subject_start = prev_subject_start if prev_subject_start > cur_subject_start else cur_subject_start
                            prev_subject_end = cur_subject_end
                            extend_frag = (prev_query_start, prev_query_end, prev_subject_start, prev_subject_end, subject_name)
                            reverse_long_frags[cur_frag_index] = extend_frag
                            is_update = True
                if not is_update:
                    reverse_long_frags[frag_index] = (cur_query_start, cur_query_end, cur_subject_start, cur_subject_end, subject_name)
                    frag_index += 1
            longest_queries += list(reverse_long_frags.values())

        if not longest_repeats.__contains__(query_name):
            longest_repeats[query_name] = []
        cur_longest_repeats = longest_repeats[query_name]
        for repeat in longest_queries:
            # Subject序列处理流程
            subject_name = repeat[4]
            old_subject_start_pos = repeat[2] - 1
            old_subject_end_pos = repeat[3]
            # Query序列处理流程
            old_query_start_pos = repeat[0] - 1
            old_query_end_pos = repeat[1]
            cur_query_seq_len = abs(old_query_end_pos - old_query_start_pos)
            cur_longest_repeats.append((query_name, old_query_start_pos, old_query_end_pos, subject_name, old_subject_start_pos, old_subject_end_pos))

    return longest_repeats


def multi_process_align(query_path, subject_path, blastnResults_path, tmp_blast_dir, threads, is_removed_dir=True):
    tools_dir = ''
    if is_removed_dir:
        os.system('rm -rf ' + tmp_blast_dir)
    if not os.path.exists(tmp_blast_dir):
        os.makedirs(tmp_blast_dir)

    if os.path.exists(blastnResults_path):
        os.remove(blastnResults_path)

    orig_names, orig_contigs = read_fasta(query_path)

    blast_db_command = 'makeblastdb -dbtype nucl -in ' + subject_path + ' > /dev/null 2>&1'
    os.system(blast_db_command)

    longest_repeat_files = []
    segments_cluster = divided_array(list(orig_contigs.items()), threads)
    for partition_index, cur_segments in enumerate(segments_cluster):
        if len(cur_segments) <= 0:
            continue
        single_tmp_dir = tmp_blast_dir + '/' + str(partition_index)
        #print('current partition_index: ' + str(partition_index))
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
        job = ex.submit(multiple_alignment_blast, file, tools_dir)
        jobs.append(job)
    ex.shutdown(wait=True)

    for job in as_completed(jobs):
        cur_blastn2Results_path = job.result()
        os.system('cat ' + cur_blastn2Results_path + ' >> ' + blastnResults_path)

def multi_process_align_blastx(query_path, subject_path, blastxResults_path, tmp_blast_dir, threads, is_removed_dir=True):
    tools_dir = ''
    if is_removed_dir:
        os.system('rm -rf ' + tmp_blast_dir)
    if not os.path.exists(tmp_blast_dir):
        os.makedirs(tmp_blast_dir)

    if os.path.exists(blastxResults_path):
        os.remove(blastxResults_path)

    orig_names, orig_contigs = read_fasta(query_path)

    blast_db_command = 'makeblastdb -dbtype prot -in ' + subject_path + ' > /dev/null 2>&1'
    os.system(blast_db_command)

    longest_repeat_files = []
    segments_cluster = divided_array(list(orig_contigs.items()), threads)
    for partition_index, cur_segments in enumerate(segments_cluster):
        if len(cur_segments) <= 0:
            continue
        single_tmp_dir = tmp_blast_dir + '/' + str(partition_index)
        #print('current partition_index: ' + str(partition_index))
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
        job = ex.submit(multiple_alignment_blastx, file, tools_dir)
        jobs.append(job)
    ex.shutdown(wait=True)

    for job in as_completed(jobs):
        cur_blastn2Results_path = job.result()
        os.system('cat ' + cur_blastn2Results_path + ' >> ' + blastxResults_path)

def multiple_alignment_blast(repeats_path, tools_dir):
    split_repeats_path = repeats_path[0]
    ref_db_path = repeats_path[1]
    blastn2Results_path = repeats_path[2]

    align_command = 'blastn -db ' + ref_db_path + ' -num_threads ' \
                    + str(1) + ' -query ' + split_repeats_path + ' -evalue 1e-20 -outfmt 6 > ' + blastn2Results_path
    os.system(align_command)

    return blastn2Results_path

def multiple_alignment_blastx(repeats_path, tools_dir):
    split_repeats_path = repeats_path[0]
    ref_db_path = repeats_path[1]
    blastn2Results_path = repeats_path[2]

    align_command = 'blastx -word_size 3 -max_target_seqs 10 -db ' + ref_db_path + ' -num_threads ' \
                    + str(1) + ' -query ' + split_repeats_path + ' -evalue 1e-3 -outfmt 6 > ' + blastn2Results_path
    os.system(align_command)

    return blastn2Results_path

def read_Ninja_clusters(cluster_file):
    clusters = {}
    with open(cluster_file, 'r') as f_r:
        for line in f_r:
            line = line.replace('\n', '')
            parts = line.split('\t')
            cluster_id = int(parts[0])
            seq_name = parts[1]
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            cur_cluster = clusters[cluster_id]
            cur_cluster.append(seq_name)

    return clusters

def convert_LtrDetector_scn(LtrDetector_output, scn_file):
    with open(scn_file, 'w') as f_save:
        f_save.write('# LtrDetector \n')
        with open(LtrDetector_output, 'r') as f_r:
            for i, line in enumerate(f_r):
                line = str(line).replace('\n', '')
                parts = line.split('\t')
                # 由于多线程运行的LtrDetector结果会出现多个header，因此我们排除掉这些无用的信息
                if len(parts) != 18 or parts[0] == '':
                    continue
                ltr_start = int(parts[1]) + 1
                ltr_end = int(parts[2])
                ltr_len = ltr_end - ltr_start + 1
                left_ltr_start = int(parts[3]) + 1
                left_ltr_end = int(parts[4])
                left_ltr_len = left_ltr_end - left_ltr_start + 1
                right_ltr_start = int(parts[5]) + 1
                right_ltr_end = int(parts[6])
                right_ltr_len = right_ltr_end - right_ltr_start + 1
                ltr_identity = float(parts[7])
                seq_id = 'NA'
                chr_name = parts[0]
                new_line = str(ltr_start) + ' ' + str(ltr_end) + ' ' + str(ltr_len) + ' ' + \
                           str(left_ltr_start) + ' ' + str(left_ltr_end) + ' ' + str(left_ltr_len) + ' ' + \
                           str(right_ltr_start) + ' ' + str(right_ltr_end) + ' ' + str(right_ltr_len) + ' ' + \
                           str(ltr_identity) + ' ' + str(seq_id) + ' ' + chr_name + '\n'
                f_save.write(new_line)

def judge_both_ends_frame_v1(maxtrix_file, debug=1):
    # 我现在想的假阳性过滤方法：
    # 1. 对matrix file 搜索同源边界，如果不存在，则说明是真实LTR，否则为假阳性

    is_ltr = True
    seq_name = os.path.basename(maxtrix_file).split('.')[0]
    # Step3. 对候选随机序列 搜索同源边界。我们将窗口设置为40.
    is_left_ltr, new_boundary_start = judge_left_frame_LTR(maxtrix_file)
    if debug:
        print(maxtrix_file, is_left_ltr, new_boundary_start)
    is_right_ltr, new_boundary_end = judge_right_frame_LTR(maxtrix_file)
    if debug:
        print(maxtrix_file, is_right_ltr, new_boundary_end)
    is_ltr &= is_left_ltr and is_right_ltr
    return seq_name, is_ltr

def judge_right_frame_LTR(matrix_file):
    pos = 0
    debug = 0
    col_num = -1
    row_num = 0
    lines = []
    no_empty_row = 0
    with open(matrix_file, 'r') as f_r:
        for line in f_r:
            line = line.replace('\n', '').split('\t')[1]
            col_num = len(line)
            row_num += 1
            lines.append(line)
            if line != '-' * col_num:
                no_empty_row += 1

    # 过滤掉单拷贝的LTR，因为我们没办法判断它是否是LTR
    if row_num <= 1:
        return True, -1

    matrix = [[''] * col_num for i in range(row_num)]
    for row, line in enumerate(lines):
        for col in range(len(line)):
            matrix[row][col] = line[col]

    col_base_map = {}
    for col_index in range(col_num):
        if not col_base_map.__contains__(col_index):
            col_base_map[col_index] = {}
        base_map = col_base_map[col_index]
        # Calculate the base composition ratio in the current column.
        if len(base_map) == 0:
            for row in range(row_num):
                cur_base = matrix[row][col_index]
                if not base_map.__contains__(cur_base):
                    base_map[cur_base] = 0
                cur_count = base_map[cur_base]
                cur_count += 1
                base_map[cur_base] = cur_count
        if not base_map.__contains__('-'):
            base_map['-'] = 0

    search_len = 100
    sliding_window_size = 20
    valid_col_threshold = int(row_num / 2)

    if row_num <= 5:
        homo_threshold = 0.95
    elif row_num <= 10:
        homo_threshold = 0.9
    elif row_num <= 50:
        homo_threshold = 0.8
    else:
        homo_threshold = 0.75

    valid_col_count = 0
    homo_col_count = 0

    max_con_homo = 0
    con_homo = 0
    prev_homo = False

    max_con_no_homo = 0
    con_no_homo = 0
    prev_non_homo = False

    col_index = pos
    homo_cols = []
    while valid_col_count < search_len and col_index < col_num:
        # Starting from position 'pos', search for 15 effective columns to the right.
        # Determine if the current column is effective.
        is_homo_col = False
        base_map = col_base_map[col_index]
        # If the number of non-empty rows exceeds the threshold, then it is an effective row.
        no_gap_num = row_num - base_map['-']
        if no_gap_num <= 1:
            col_index += 1
            continue
        max_homo_ratio = 0
        gap_num = base_map['-']
        # If the number of gaps in the current column is <= half of the copy count, then it is an effective column.
        if gap_num <= valid_col_threshold:
            valid_col_count += 1
            # Determine if the effective column is homologous.
            for base in base_map.keys():
                if base == '-':
                    continue
                # 修正bug，row_num 替换成 no_gap_num
                cur_homo_ratio = float(base_map[base]) / row_num
                if cur_homo_ratio > max_homo_ratio:
                    max_homo_ratio = cur_homo_ratio
                if cur_homo_ratio >= homo_threshold:
                    homo_col_count += 1
                    # Check for consecutive homologous columns.
                    if prev_homo:
                        con_homo += 1
                    is_homo_col = True
                    break
            if not is_homo_col:
                max_con_homo = con_homo if con_homo > max_con_homo else max_con_homo
                con_homo = 0

                if prev_non_homo:
                    con_no_homo += 1
                else:
                    max_con_no_homo = con_no_homo if con_no_homo > max_con_no_homo else max_con_no_homo
                    con_no_homo = 0
                is_no_homo_col = True
                prev_non_homo = True
                prev_homo = False
            else:
                max_con_no_homo = con_no_homo if con_no_homo > max_con_no_homo else max_con_no_homo
                prev_homo = True
                prev_non_homo = False
                con_no_homo = 0
                is_no_homo_col = False
            homo_cols.append(
                (col_index, is_homo_col, con_homo, is_no_homo_col, con_no_homo, max_homo_ratio))
        col_index += 1
    max_con_homo = con_homo if con_homo > max_con_homo else max_con_homo
    max_con_no_homo = con_no_homo if con_no_homo > max_con_no_homo else max_con_no_homo

    # Use a sliding window to calculate the average homology of 10 consecutive bases starting from the right. Determine if it exceeds the threshold.
    # If it exceeds the threshold, obtain the first column with homology above the threshold within the 10bp, and consider it as the homologous boundary.
    cur_boundary = pos
    homo_cols.reverse()
    new_boundary_end = -1
    for i in range(len(homo_cols) - sliding_window_size + 1):
        window = homo_cols[i:i + sliding_window_size]
        avg_homo_ratio = 0
        first_candidate_boundary = -1
        for item in window:
            cur_homo_ratio = item[5]
            if cur_homo_ratio >= homo_threshold - 0.1 and first_candidate_boundary == -1:
                first_candidate_boundary = item[0]
            avg_homo_ratio += cur_homo_ratio
        avg_homo_ratio = float(avg_homo_ratio) / sliding_window_size
        if avg_homo_ratio >= homo_threshold:
            # If homology in the sliding window exceeds the threshold, find the boundary.
            new_boundary_end = first_candidate_boundary
            break
    if new_boundary_end != cur_boundary and new_boundary_end != -1:
        if debug:
            print('align end right homology, new boundary: ' + str(new_boundary_end))
        cur_boundary = new_boundary_end

    if new_boundary_end != -1 and abs(new_boundary_end - pos) > 20:
        return False, new_boundary_end
    else:
        return True, new_boundary_end

def judge_left_frame_LTR(matrix_file):
    pos = 99
    debug = 0
    col_num = -1
    row_num = 0
    lines = []
    no_empty_row = 0
    with open(matrix_file, 'r') as f_r:
        for line in f_r:
            line = line.replace('\n', '').split('\t')[0]
            col_num = len(line)
            row_num += 1
            lines.append(line)
            if line != '-' * col_num:
                no_empty_row += 1

    # 过滤掉单拷贝的LTR，因为我们没办法判断它是否是LTR
    if row_num <= 1:
        return True, -1

    matrix = [[''] * col_num for i in range(row_num)]
    for row, line in enumerate(lines):
        for col in range(len(line)):
            matrix[row][col] = line[col]

    col_base_map = {}
    for col_index in range(col_num):
        if not col_base_map.__contains__(col_index):
            col_base_map[col_index] = {}
        base_map = col_base_map[col_index]
        # Calculate the base composition ratio in the current column.
        if len(base_map) == 0:
            for row in range(row_num):
                cur_base = matrix[row][col_index]
                if not base_map.__contains__(cur_base):
                    base_map[cur_base] = 0
                cur_count = base_map[cur_base]
                cur_count += 1
                base_map[cur_base] = cur_count
        if not base_map.__contains__('-'):
            base_map['-'] = 0

    search_len = 100
    sliding_window_size = 10
    valid_col_threshold = int(row_num / 2)

    if row_num <= 5:
        homo_threshold = 0.95
    elif row_num <= 10:
        homo_threshold = 0.9
    elif row_num <= 50:
        homo_threshold = 0.8
    else:
        homo_threshold = 0.75

    col_index = pos
    valid_col_count = 0
    homo_col_count = 0

    max_con_homo = 0
    con_homo = 0
    prev_homo = False

    max_con_no_homo = 0
    con_no_homo = 0
    prev_non_homo = False

    homo_cols = []
    while valid_col_count < search_len and col_index >= 0:
        # Starting from position 'pos', search for 15 effective columns to the left.
        # Determine if the current column is effective.
        is_homo_col = False
        base_map = col_base_map[col_index]
        max_homo_ratio = 0
        no_gap_num = row_num - base_map['-']
        if no_gap_num <= 1:
            col_index -= 1
            continue
        gap_num = base_map['-']
        # If the number of gaps in the current column is <= half of the copy count, then it is an effective column.
        if gap_num <= valid_col_threshold:
            valid_col_count += 1
            # Determine if the effective column is homologous.
            for base in base_map.keys():
                if base == '-':
                    continue
                # 修正bug，row_num 替换成 no_gap_num
                cur_homo_ratio = float(base_map[base]) / row_num
                if cur_homo_ratio > max_homo_ratio:
                    max_homo_ratio = cur_homo_ratio
                if cur_homo_ratio >= homo_threshold:
                    homo_col_count += 1
                    # Check for consecutive homologous columns.
                    if prev_homo:
                        con_homo += 1
                    is_homo_col = True
                    break
            if not is_homo_col:
                max_con_homo = con_homo if con_homo > max_con_homo else max_con_homo
                con_homo = 0

                if prev_non_homo:
                    con_no_homo += 1
                else:
                    max_con_no_homo = con_no_homo if con_no_homo > max_con_no_homo else max_con_no_homo
                    con_no_homo = 0
                is_no_homo_col = True
                prev_non_homo = True
                prev_homo = False
            else:
                max_con_no_homo = con_no_homo if con_no_homo > max_con_no_homo else max_con_no_homo
                prev_homo = True
                prev_non_homo = False
                con_no_homo = 0
                is_no_homo_col = False
            homo_cols.append(
                (col_index, is_homo_col, con_homo, is_no_homo_col, con_no_homo, max_homo_ratio))
        col_index -= 1
    max_con_homo = con_homo if con_homo > max_con_homo else max_con_homo
    max_con_no_homo = con_no_homo if con_no_homo > max_con_no_homo else max_con_no_homo


    # Use a sliding window to calculate the average homology of 10 consecutive bases starting from the left. Determine if it exceeds the threshold.
    # If it exceeds the threshold, obtain the first column with homology above the threshold within the 10bp, and consider it as the homologous boundary.
    homo_cols.reverse()
    new_boundary_start = -1
    for i in range(len(homo_cols) - sliding_window_size + 1):
        window = homo_cols[i:i + sliding_window_size]
        avg_homo_ratio = 0
        first_candidate_boundary = -1
        for item in window:
            cur_homo_ratio = item[5]
            if cur_homo_ratio >= homo_threshold-0.1 and first_candidate_boundary == -1:
                first_candidate_boundary = item[0]
            avg_homo_ratio += cur_homo_ratio
        avg_homo_ratio = float(avg_homo_ratio)/sliding_window_size
        if avg_homo_ratio >= homo_threshold:
            # If homology in the sliding window exceeds the threshold, find the boundary.
            new_boundary_start = first_candidate_boundary
            break
    if new_boundary_start != pos and new_boundary_start != -1:
        if debug:
            print('align start left homology, new boundary: ' + str(new_boundary_start))
        cur_boundary = new_boundary_start

    if new_boundary_start != -1 and abs(new_boundary_start - pos) > 20:
        return False, new_boundary_start
    else:
        return True, new_boundary_start

def filter_ltr_by_homo(dl_output_path, homo_output_path, matrix_dir, threads):
    true_ltr_names = []
    ltr_dict = {}
    with open(dl_output_path, 'r') as f_r:
        for line in f_r:
            line = line.replace('\n', '')
            parts = line.split('\t')
            ltr_name = parts[0]
            is_ltr = int(parts[1])
            ltr_dict[ltr_name] = is_ltr
            if is_ltr:
                true_ltr_names.append(ltr_name)

    ex = ProcessPoolExecutor(threads)
    jobs = []
    for ltr_name in true_ltr_names:
        cur_matrix_file = matrix_dir + '/' + ltr_name + '.matrix'
        job = ex.submit(judge_both_ends_frame_v1, cur_matrix_file, debug=0)
        jobs.append(job)
    ex.shutdown(wait=True)
    filter_ltr_names = []
    for job in as_completed(jobs):
        cur_seq_name, cur_is_ltr = job.result()
        if cur_is_ltr:
            cur_is_ltr = 1
        else:
            cur_is_ltr = 0
            filter_ltr_names.append(cur_seq_name)
        ltr_dict[cur_seq_name] = cur_is_ltr
    print('Deep Learning LTR num: ' + str(len(true_ltr_names)) + ', Homology filter LTR num: ' + str(len(filter_ltr_names)))
    print(filter_ltr_names)
    with open(homo_output_path, 'w') as f_save:
        for ltr_name in ltr_dict.keys():
            f_save.write(ltr_name+'\t'+str(ltr_dict[ltr_name])+'\n')


def allow_mismatch(left_tsd, right_tsd, allow_mismatch_num):
    mismatch_num = 0
    for i in range(len(left_tsd)):
        if left_tsd[i] == right_tsd[i]:
            continue
        else:
            mismatch_num += 1
    if mismatch_num <= allow_mismatch_num:
        return True
    else:
        return False

def get_non_empty_seq(raw_align_seq):
    align_seq = ''
    for i, base in enumerate(raw_align_seq):
        if base == '-':
            continue
        else:
            align_seq += base
    return align_seq


def is_rich_in_ta(sequence, threshold=0.8):
    """
    判断给定的DNA序列是否富含TA
    :param sequence: DNA序列，只包含ATCG
    :param threshold: 富集阈值，默认为0.8（即TA的比例超过80%）
    :return: 如果序列富含TA，返回True；否则返回False
    """
    # 统计TA的数量
    ta_count = sequence.count('T') + sequence.count('A')

    # 计算TA的比例
    ta_ratio = ta_count / len(sequence)

    # 判断是否富含TA
    return ta_ratio > threshold

# 判断窗口的拷贝是否具有TSD特征
def is_TIR_frame(matrix_file, ltr_name):
    # 我们只过滤 4-6bp 以外的tsd
    TIR_TSDs = [15, 14, 13, 12, 11, 10, 9, 8, 7, 3, 2]
    has_tsd_copy_count = 0
    copy_count = 0
    with open(matrix_file, 'r') as f_r:
        for i, line in enumerate(f_r):
            parts = line.split('\t')
            left_line = parts[0]
            right_line = parts[1]
            copy_count += 1
            left_non_empty_seq = get_non_empty_seq(left_line)
            right_non_empty_seq = get_non_empty_seq(right_line)
            for tsd_len in TIR_TSDs:
                left_tsd = left_non_empty_seq[-tsd_len:]
                right_tsd = right_non_empty_seq[:tsd_len]
                if len(left_tsd) != tsd_len or len(right_tsd) != tsd_len:
                    continue
                allow_mismatch_num = 0
                if len(left_tsd) >= 8:
                    allow_mismatch_num = 1
                if allow_mismatch(left_tsd, right_tsd, allow_mismatch_num):
                    # 规定了几种特定的短TSD
                    if (tsd_len == 2 and left_tsd != 'TA') \
                            or (tsd_len == 3 and (left_tsd != 'TTA' or left_tsd != 'TAA')) \
                            or (tsd_len == 4 and left_tsd != 'TTAA'):
                        continue
                    # if is_rich_in_ta(left_tsd):
                    #     continue
                    has_tsd_copy_count += 1
                    # print(i, left_tsd, right_tsd)
                    break
    # print(has_tsd_copy_count)
    is_TIR = False
    if has_tsd_copy_count >= 10:
        is_TIR = True
    return ltr_name, is_TIR


def get_confident_TIR(candidate_tir_path, tool_dir):
    output_dir = os.path.dirname(candidate_tir_path)
    itrsearch_command = 'cd ' + output_dir + ' && ' + tool_dir + '/itrsearch -i 0.85 -l 10 ' + candidate_tir_path + ' > /dev/null 2>&1'
    run_command(itrsearch_command)
    tir_file = candidate_tir_path + '.itr'
    tir_names, tir_contigs = read_fasta_v1(tir_file)
    TIR_info = {}
    for tir_name in tir_names:
        parts = tir_name.split('\t')
        orig_name = parts[0]
        terminal_info = parts[-1]
        TIR_info_parts = terminal_info.split('ITR')[1].split(' ')[0].replace('(', '').replace(')', '').split('..')
        TIR_left_pos_parts = TIR_info_parts[0].split(',')
        TIR_right_pos_parts = TIR_info_parts[1].split(',')
        lTIR_start = int(TIR_left_pos_parts[0])
        lTIR_end = int(TIR_left_pos_parts[1])
        rTIR_start = int(TIR_right_pos_parts[1])
        rTIR_end = int(TIR_right_pos_parts[0])
        TIR_info[orig_name.split(' ')[0]] = (lTIR_start, lTIR_end, rTIR_start, rTIR_end)
    return TIR_info

def search_ltr_structure(ltr_name, left_seq, right_seq, left_LTR_seq):
    # 搜索左右两侧是否存在TG...CA 或 TSD
    has_tgca = False
    if 'TG' in left_seq and 'CA' in right_seq:
        has_tgca = True

    has_tsd = False
    tsd_lens = [6, 5, 4]
    tsd_seq = ''
    exist_tsd = set()
    for k_num in tsd_lens:
        for i in range(len(left_seq) - k_num + 1):
            left_kmer = left_seq[i: i + k_num]
            if 'N' not in left_kmer:
                exist_tsd.add(left_kmer)
    for k_num in tsd_lens:
        if has_tsd:
            break
        for i in range(len(right_seq) - k_num + 1):
            right_kmer = right_seq[i: i + k_num]
            if 'N' not in right_kmer and right_kmer in exist_tsd:
                has_tsd = True
                tsd_seq = right_kmer
                break
    # print(ltr_name, left_seq, right_seq, has_tgca, has_tsd, tsd_seq)

    # 判断LTR终端是否由 SINE元素 构成
    has_sine_tail = identify_SINE_tail(left_LTR_seq)
    return ltr_name, (has_tgca or has_tsd) and not has_sine_tail

def has_consecutive_repeats_near_tail(sequence, min_repeat_length=2, min_repeats=3, max_distance_from_tail=5):
    """
    检查序列尾部是否存在至少 min_repeats 次串联重复子序列，
    并且这些重复子序列离序列尾部的距离不超过 max_distance_from_tail 个碱基对。

    :param sequence: 要检查的序列（字符串或列表）
    :param min_repeat_length: 最小重复单元长度，默认为2
    :param min_repeats: 至少重复次数，默认为3
    :param max_distance_from_tail: 串联重复离尾部的最大距离，默认为5
    :return: 如果尾部存在至少 min_repeats 次串联重复且距离不超过 max_distance_from_tail 返回True，否则返回False
    """
    sequence_length = len(sequence)

    # 检查尾部区域
    for repeat_length in range(min_repeat_length, sequence_length // min_repeats + 1):
        for i in range(max(0, sequence_length - repeat_length * min_repeats - max_distance_from_tail),
                       sequence_length - repeat_length * min_repeats + 1):
            repeat = sequence[i:i + repeat_length]
            match = True
            for j in range(1, min_repeats):
                if sequence[i + j * repeat_length:i + (j + 1) * repeat_length] != repeat:
                    match = False
                    break
            if match:
                return True
    return False


def has_polyA_or_polyT_near_tail(sequence, poly_length=6, max_mismatches=1, max_distance_from_tail=5):
    """
    检查序列尾部是否存在 polyA 或 polyT 序列，并且这些序列离序列尾部的距离不超过 max_distance_from_tail，
    容许最多 max_mismatches 个错配。

    :param sequence: 要检查的序列（字符串）
    :param poly_length: polyA 或 polyT 序列的长度，默认为5
    :param max_mismatches: 允许的最大错配数，默认为1
    :param max_distance_from_tail: polyA 或 polyT 离尾部的最大距离，默认为5
    :return: 如果尾部存在符合条件的 polyA 或 polyT 序列返回True，否则返回False
    """
    sequence_length = len(sequence)
    tail_start = max(0, sequence_length - poly_length - max_distance_from_tail)
    polyA = 'A' * poly_length
    polyT = 'T' * poly_length

    for i in range(tail_start, sequence_length - poly_length + 1):
        segment = sequence[i:i + poly_length]
        mismatches_A = sum(1 for a, b in zip(segment, polyA) if a != b)
        mismatches_T = sum(1 for a, b in zip(segment, polyT) if a != b)
        if mismatches_A <= max_mismatches or mismatches_T <= max_mismatches:
            return True
    return False

# 识别 polyA 和 simple repeat
def identify_SINE_tail(sequence, tail_length=20):
    tail_sequence = sequence[-tail_length:]
    if has_polyA_or_polyT_near_tail(tail_sequence) or has_consecutive_repeats_near_tail(tail_sequence):
        return True
    else:
        return False

def filter_ltr_by_structure(output_path, structure_output_path, leftLtr2Candidates, ltr_lines, reference, threads, log):
    ref_names, ref_contigs = read_fasta(reference)

    true_ltrs = {}
    ltr_dict = {}
    with open(output_path, 'r') as f_r:
        for line in f_r:
            line = line.replace('\n', '')
            parts = line.split('\t')
            ltr_name = parts[0]
            is_ltr = int(parts[1])
            if is_ltr:
                true_ltrs[ltr_name] = is_ltr
            ltr_dict[ltr_name] = is_ltr

    confident_lines = []
    for name in leftLtr2Candidates.keys():
        # if name not in FP_ltrs:
        if name in true_ltrs:
            candidate_index = leftLtr2Candidates[name]
            line = ltr_lines[candidate_index]
            confident_lines.append((name, line))

    left_size = 8
    internal_size = 3
    ex = ProcessPoolExecutor(threads)
    jobs = []
    for ltr_name, line in confident_lines:
        parts = line.split(' ')
        LTR_start = int(parts[0])
        LTR_end = int(parts[1])
        chr_name = parts[11]
        ref_seq = ref_contigs[chr_name]

        lLTR_start = int(parts[3])
        lLTR_end = int(parts[4])
        left_LTR_seq = ref_seq[lLTR_start: lLTR_end]

        # 取左/右侧 8bp + 3bp
        # 计算左LTR的切片索引，并确保它们在范围内
        left_start = max(LTR_start - 1 - left_size, 0)
        left_end = min(LTR_start + internal_size, len(ref_seq))
        left_seq = ref_seq[left_start: left_end]

        # 计算右LTR的切片索引，并确保它们在范围内
        right_start = max(LTR_end - 1 - internal_size, 0)
        right_end = min(LTR_end + left_size, len(ref_seq))
        right_seq = ref_seq[right_start: right_end]

        job = ex.submit(search_ltr_structure, ltr_name, left_seq, right_seq, left_LTR_seq)
        jobs.append(job)
    ex.shutdown(wait=True)

    FP_ltrs = {}
    for job in as_completed(jobs):
        cur_seq_name, cur_is_tp = job.result()
        if not cur_is_tp:
            FP_ltrs[cur_seq_name] = 1

    log.logger.debug('LTR num: ' + str(len(true_ltrs)) + ', LTR structure filter num: ' + str(len(FP_ltrs)) + ', remaining LTR num: ' + str(len(true_ltrs) - len(FP_ltrs)))
    with open(structure_output_path, 'w') as f_save:
        for ltr_name in ltr_dict.keys():
            if ltr_name in FP_ltrs:
                ltr_dict[ltr_name] = 0
            f_save.write(ltr_name + '\t' + str(ltr_dict[ltr_name]) + '\n')



def filter_tir(dl_output_path, tsd_output_path, matrix_dir, threads, left_LTR_contigs, tmp_output_dir, tool_dir, log):
    true_ltr_names = []
    ltr_dict = {}
    all_ltr_contigs = {}
    with open(dl_output_path, 'r') as f_r:
        for line in f_r:
            line = line.replace('\n', '')
            parts = line.split('\t')
            ltr_name = parts[0]
            is_ltr = int(parts[1])
            ltr_dict[ltr_name] = is_ltr
            if is_ltr:
                true_ltr_names.append(ltr_name)
                all_ltr_contigs[ltr_name] = left_LTR_contigs[ltr_name]
    # 检查 LTR 终端序列是否具有 terminal inverted repeats 结构，如果有，过滤掉
    all_ltr_path = tmp_output_dir + '/ltr_temp.fa'
    store_fasta(all_ltr_contigs, all_ltr_path)
    all_confident_tirs = get_confident_TIR(all_ltr_path, tool_dir)

    # 存储候选的LTR/TIR序列
    candidate_ltrs = {}
    candidate_tirs = {}
    for ltr_name in all_ltr_contigs.keys():
        seq = str(all_ltr_contigs[ltr_name])
        if ltr_name not in all_confident_tirs or (seq.startswith('TG') and seq.endswith('CA')):
            candidate_ltrs[ltr_name] = seq
        else:
            candidate_tirs[ltr_name] = seq
    candidate_ltr_path = tmp_output_dir + '/candidate_ltr.fa'
    candidate_tir_path = tmp_output_dir + '/candidate_tir.fa'
    store_fasta(candidate_ltrs, candidate_ltr_path)
    store_fasta(candidate_tirs, candidate_tir_path)

    if file_exist(candidate_ltr_path) and file_exist(candidate_tir_path):
        # 如果候选LTR中存在TIR，说明是假阳性，应过滤
        blastnResults_path = tmp_output_dir + '/rm_tir.out'
        temp_blast_dir = tmp_output_dir + '/rm_tir_blast'
        multi_process_align(candidate_ltr_path, candidate_tir_path, blastnResults_path, temp_blast_dir, threads)
        remain_candidate_tirs = find_tir_in_ltr(blastnResults_path, candidate_ltr_path, candidate_tir_path)
        candidate_tirs.update(remain_candidate_tirs)

    filter_ltr_names = []
    for ltr_name in true_ltr_names:
        if ltr_name in candidate_tirs:
            cur_is_ltr = 0
            filter_ltr_names.append(ltr_name)
        else:
            cur_is_ltr = 1
        ltr_dict[ltr_name] = cur_is_ltr

    log.logger.debug('LTR num: ' + str(len(true_ltr_names)) + ', TIR filter LTR num: ' + str(len(filter_ltr_names)) + ', remaining LTR num: ' + str(len(true_ltr_names)-len(filter_ltr_names)))
    log.logger.debug(filter_ltr_names)
    with open(tsd_output_path, 'w') as f_save:
        for ltr_name in ltr_dict.keys():
            f_save.write(ltr_name+'\t'+str(ltr_dict[ltr_name])+'\n')


def filter_tir_by_tsd(dl_output_path, tsd_output_path, matrix_dir, threads, left_LTR_contigs, tmp_output_dir, tool_dir,
                      log):
    true_ltr_names = []
    ltr_dict = {}
    with open(dl_output_path, 'r') as f_r:
        for line in f_r:
            line = line.replace('\n', '')
            parts = line.split('\t')
            ltr_name = parts[0]
            is_ltr = int(parts[1])
            ltr_dict[ltr_name] = is_ltr
            if is_ltr:
                true_ltr_names.append(ltr_name)

    # 检查窗口，是否具有可靠TSD的TIR
    ex = ProcessPoolExecutor(threads)
    jobs = []
    for ltr_name in true_ltr_names:
        cur_matrix_file = matrix_dir + '/' + ltr_name + '.matrix'
        job = ex.submit(is_TIR_frame, cur_matrix_file, ltr_name)
        jobs.append(job)
    ex.shutdown(wait=True)
    # 收集候选的 TIR 序列
    candidate_tirs = {}
    candidate_ltrs = {}
    for job in as_completed(jobs):
        cur_seq_name, cur_is_tir = job.result()
        if cur_is_tir:
            candidate_tirs[cur_seq_name] = left_LTR_contigs[cur_seq_name]
        else:
            candidate_ltrs[cur_seq_name] = left_LTR_contigs[cur_seq_name]

    candidate_ltr_path = tmp_output_dir + '/candidate_ltr.fa'
    store_fasta(candidate_ltrs, candidate_ltr_path)

    # 判断序列中是否有 terminal inverted repeats 来判断是否有 TIR
    candidate_tir_path = tmp_output_dir + '/candidate_tir.fa'
    store_fasta(candidate_tirs, candidate_tir_path)
    confident_tir_path = tmp_output_dir + '/confident_tir.fa'
    confident_tirs = get_confident_TIR(candidate_tir_path, tool_dir)
    confident_tir_contigs = {}
    for name in candidate_tirs.keys():
        if name in confident_tirs:
            confident_tir_contigs[name] = candidate_tirs[name]
    store_fasta(confident_tir_contigs, confident_tir_path)

    # 如果候选LTR中存在TIR，说明是假阳性，应过滤
    blastnResults_path = tmp_output_dir + '/rm_tir.out'
    temp_blast_dir = tmp_output_dir + '/rm_tir_blast'
    multi_process_align(candidate_ltr_path, confident_tir_path, blastnResults_path, temp_blast_dir, threads)
    candidate_tirs = find_tir_in_ltr(blastnResults_path, candidate_ltr_path, confident_tir_path)
    confident_tirs.update(candidate_tirs)

    filter_ltr_names = []
    for ltr_name in true_ltr_names:
        if ltr_name in confident_tirs:
            cur_is_ltr = 0
            filter_ltr_names.append(ltr_name)
        else:
            cur_is_ltr = 1
        ltr_dict[ltr_name] = cur_is_ltr

    log.logger.debug('LTR num: ' + str(len(true_ltr_names)) + ', TIR TSD filter LTR num: ' + str(
        len(filter_ltr_names)) + ', remaining LTR num: ' + str(len(true_ltr_names) - len(filter_ltr_names)))
    log.logger.debug(filter_ltr_names)
    with open(tsd_output_path, 'w') as f_save:
        for ltr_name in ltr_dict.keys():
            f_save.write(ltr_name + '\t' + str(ltr_dict[ltr_name]) + '\n')

def find_tir_in_ltr(blastnResults_path, query_path, subject_path, coverage = 0.95):
    full_length_threshold = coverage
    longest_repeats = FMEA_new(query_path, blastnResults_path, full_length_threshold)

    query_names, query_contigs = read_fasta(query_path)
    subject_names, subject_contigs = read_fasta(subject_path)

    candidate_tirs = {}
    for query_name in longest_repeats.keys():
        for item in longest_repeats[query_name]:
            query_name = item[0]
            subject_name = item[3]

            query_len = len(query_contigs[query_name])
            subject_len = len(subject_contigs[subject_name])
            q_start = int(item[1])
            q_end = int(item[2])
            q_len = abs(q_end - q_start)
            s_start = int(item[4])
            s_end = int(item[5])
            s_len = abs(s_end - s_start)
            # 我们认为 比对上的 query 和 subject 部分应该占它自身的95%以上
            if float(q_len) / query_len >= coverage and float(s_len) / subject_len >= coverage:
                candidate_tirs[query_name] = 1
                break
    return candidate_tirs

def get_low_copy_LTR(output_dir, low_copy_output_dir, copy_num_threshold=3):
    if os.path.exists(low_copy_output_dir):
        os.system('rm -rf ' + low_copy_output_dir)
    if not os.path.exists(low_copy_output_dir):
        os.makedirs(low_copy_output_dir)

    for name in os.listdir(output_dir):
        seq_name = name.split('.')[0]
        matrix_file = output_dir + '/' + name
        cur_copy_num = 0
        with open(matrix_file, 'r') as f_r:
            for line in f_r:
                cur_copy_num += 1
        if cur_copy_num <= copy_num_threshold:
            os.system('cp ' + matrix_file + ' ' + low_copy_output_dir)

def get_high_copy_LTR(output_dir, high_copy_output_dir, copy_num_threshold=3):
    if os.path.exists(high_copy_output_dir):
        os.system('rm -rf ' + high_copy_output_dir)
    if not os.path.exists(high_copy_output_dir):
        os.makedirs(high_copy_output_dir)

    for name in os.listdir(output_dir):
        seq_name = name.split('.')[0]
        matrix_file = output_dir + '/' + name
        cur_copy_num = 0
        with open(matrix_file, 'r') as f_r:
            for line in f_r:
                cur_copy_num += 1
        if cur_copy_num > copy_num_threshold:
            os.system('cp ' + matrix_file + ' ' + high_copy_output_dir)

def find_files_recursively(root_dir, file_extension=''):
    """
    递归搜索指定目录及其子目录中的文件，并返回文件路径列表。

    参数:
    root_dir (str): 根目录路径。
    file_extension (str): 可选的文件扩展名，例如 '.txt'。如果不指定，则搜索所有文件。

    返回:
    files (list): 匹配的文件路径列表。
    """
    files = []

    # 遍历目录中的每一个条目
    for root, dirs, file_names in os.walk(root_dir):
        # 过滤出具有指定扩展名的文件
        if file_extension:
            filtered_file_names = [f for f in file_names if f.endswith(file_extension)]
        else:
            filtered_file_names = file_names

            # 为每个匹配的文件构建完整的文件路径，并添加到列表中
        for file_name in filtered_file_names:
            files.append(os.path.join(root, file_name))

    return files

def judge_ltr_from_both_ends_frame(output_dir, output_path, threads, type, log):
    file_extension = '.matrix'
    all_matrix_files = find_files_recursively(output_dir, file_extension)

    ex = ProcessPoolExecutor(threads)
    jobs = []
    for matrix_file in all_matrix_files:
        job = ex.submit(judge_both_ends_frame, matrix_file, debug=1)
        jobs.append(job)
    ex.shutdown(wait=True)

    FP_num = 0
    true_ltrs = {}
    for job in as_completed(jobs):
        cur_name, cur_is_ltr = job.result()
        # print(cur_name, cur_is_ltr)
        if cur_is_ltr:
            cur_is_ltr = 1
        else:
            cur_is_ltr = 0
            FP_num += 1
        true_ltrs[cur_name] = cur_is_ltr

    log.logger.debug(type + ' LTR num: ' + str(len(all_matrix_files)) + ', LTR Homo filter LTR num: ' + str(FP_num) + ', remain LTR num: ' + str(len(all_matrix_files) - FP_num))

    with open(output_path, 'w') as f_save:
        for cur_name in true_ltrs.keys():
            f_save.write(cur_name + '\t' + str(true_ltrs[cur_name]) + '\n')

def judge_ltr_from_both_ends_frame_v1(output_dir, output_path, left_LTR_contigs, tmp_output_dir, threads, type, log):
    file_extension = '.matrix'
    all_matrix_files = find_files_recursively(output_dir, file_extension)

    ex = ProcessPoolExecutor(threads)
    jobs = []
    for matrix_file in all_matrix_files:
        job = ex.submit(judge_both_ends_frame, matrix_file, debug=1)
        jobs.append(job)
    ex.shutdown(wait=True)

    TP_contigs = {}
    FP_contigs = {}
    ltr_dict = {}
    for job in as_completed(jobs):
        cur_name, cur_is_ltr = job.result()
        if cur_is_ltr:
            cur_is_ltr = 1
            TP_contigs[cur_name] = left_LTR_contigs[cur_name]
        else:
            cur_is_ltr = 0
            FP_contigs[cur_name] = left_LTR_contigs[cur_name]
        ltr_dict[cur_name] = cur_is_ltr
    confident_fp_path = tmp_output_dir + '/FP_terminal.fa'
    candidate_tp_path = tmp_output_dir + '/candidate_TP_terminal.fa'
    store_fasta(TP_contigs, candidate_tp_path)
    store_fasta(FP_contigs, confident_fp_path)

    # 根据规则的过滤方法很可能获取许多假阳性的终端序列
    # 如果候选终端与这些假阳性终端高度相似，则大概率也是假阳性
    blastnResults_path = tmp_output_dir + '/rm_fp_terminal.out'
    temp_blast_dir = tmp_output_dir + '/rm_fp_terminal_blast'
    multi_process_align(candidate_tp_path, confident_fp_path, blastnResults_path, temp_blast_dir, threads)
    candidate_FP_contigs = find_tir_in_ltr(blastnResults_path, candidate_tp_path, confident_fp_path)
    FP_contigs.update(candidate_FP_contigs)

    log.logger.debug(type + ' LTR num: ' + str(len(all_matrix_files)) + ', LTR Homo filter LTR num: ' + str(len(FP_contigs)) + ', remain LTR num: ' + str(len(all_matrix_files) - len(FP_contigs)))

    with open(output_path, 'w') as f_save:
        for cur_name in ltr_dict.keys():
            if cur_name not in FP_contigs:
                f_save.write(cur_name + '\t' + str(1) + '\n')
            else:
                f_save.write(cur_name + '\t' + str(0) + '\n')

def judge_ltr_has_structure(lc_output_path, structure_output_path, leftLtr2Candidates, ltr_lines, reference, log):
    ref_names, ref_contigs = read_fasta(reference)

    true_ltr_names = []
    ltr_dict = {}
    with open(lc_output_path, 'r') as f_r:
        for line in f_r:
            line = line.replace('\n', '')
            parts = line.split('\t')
            ltr_name = parts[0]
            is_ltr = int(parts[1])
            ltr_dict[ltr_name] = is_ltr
            if is_ltr:
                true_ltr_names.append(ltr_name)

    filter_ltr_names = []
    for ltr_name in true_ltr_names:
        candidate_index = leftLtr2Candidates[ltr_name]
        ltr_line = ltr_lines[candidate_index]
        parts = ltr_line.split(' ')
        chr_name = parts[11]
        ref_seq = ref_contigs[chr_name]
        left_ltr_start = int(parts[3])
        left_ltr_end = int(parts[4])
        right_ltr_start = int(parts[6])
        right_ltr_end = int(parts[7])

        left_ltr_seq = ref_seq[left_ltr_start - 1: left_ltr_end]
        right_ltr_seq = ref_seq[right_ltr_start - 1: right_ltr_end]

        tsd_lens = [6, 5, 4]
        allow_mismatch_num = 0
        has_structure = False
        for tsd_len in tsd_lens:
            left_tsd = ref_seq[left_ltr_start - 1 - tsd_len: left_ltr_start - 1]
            right_tsd = ref_seq[right_ltr_end: right_ltr_end + tsd_len]
            if allow_mismatch(left_tsd, right_tsd, allow_mismatch_num):
                has_structure = True
                break
        if has_structure:
            ltr_dict[ltr_name] = 1
        else:
            ltr_dict[ltr_name] = 0
            filter_ltr_names.append(ltr_name)

    log.logger.debug('Low copy LTR after HomoFilter num: ' + str(len(true_ltr_names)) + ', LTR structure filter LTR num: ' + str(len(filter_ltr_names)) + ', remain LTR num: ' + str(len(true_ltr_names)-len(filter_ltr_names)))

    with open(structure_output_path, 'w') as f_save:
        for ltr_name in ltr_dict.keys():
            f_save.write(ltr_name + '\t' + str(ltr_dict[ltr_name]) + '\n')

def judge_both_ends_frame(maxtrix_file, debug=1):
    # 我现在想的假阳性过滤方法：
    # 1. 对matrix file 搜索同源边界，如果不存在，则说明是真实LTR，否则为假阳性
    is_ltr = True
    seq_name = os.path.basename(maxtrix_file).split('.')[0]
    # Step3. 对候选随机序列 搜索同源边界。我们将窗口设置为20.
    is_left_ltr, new_boundary_start = judge_left_frame_LTR(maxtrix_file)
    # if debug:
    #     print('left', maxtrix_file, is_left_ltr, new_boundary_start)
    is_right_ltr, new_boundary_end = judge_right_frame_LTR(maxtrix_file)
    # if debug:
    #     print('right', maxtrix_file, is_right_ltr, new_boundary_end)
    is_ltr &= is_left_ltr and is_right_ltr
    return seq_name, is_ltr

def alter_deep_learning_results(dl_output_path, hc_output_path, alter_dl_output_path):
    ltr_dict = {}
    with open(dl_output_path, 'r') as f_r:
        for line in f_r:
            line = line.replace('\n', '')
            parts = line.split('\t')
            ltr_name = parts[0]
            is_ltr = int(parts[1])
            ltr_dict[ltr_name] = is_ltr

    with open(hc_output_path, 'r') as f_r:
        for line in f_r:
            line = line.replace('\n', '')
            parts = line.split('\t')
            ltr_name = parts[0]
            is_ltr = int(parts[1])
            # 当同源方法预测为 0，即非LTR时，以其预测结果为准
            if not is_ltr:
                ltr_dict[ltr_name] = is_ltr

    with open(alter_dl_output_path, 'w') as f_save:
        for ltr_name in ltr_dict.keys():
            f_save.write(ltr_name + '\t' + str(ltr_dict[ltr_name]) + '\n')