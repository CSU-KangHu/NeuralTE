if __name__ == '__main__':
    data_dir = '/home/hukang/TE_Classification/DeepTE-master/HiTE_util/results'
    result_path = data_dir + '/test_results.txt'
    # 搜索所有预测错误的分类，并记录它们的数量
    error_predict = {}
    error_predict_names = {}
    with open(result_path, 'r') as f_r:
        for line in f_r:
            line = line.replace('\n', '')
            parts = line.split(',')
            seq_name = parts[0]
            label = parts[1]
            predict_label = parts[2]
            if label != predict_label:
                if not error_predict_names.__contains__(label):
                    error_predict_names[label] = []
                error_predict_list = error_predict_names[label]
                error_predict_list.append(predict_label)

                if not error_predict.__contains__(label):
                    error_predict[label] = 0
                error_predict_num = error_predict[label]
                error_predict[label] = error_predict_num + 1
    error_predict = sorted(error_predict.items(), key=lambda x: -x[1])
    print(error_predict)
    print(error_predict_names['hAT'])
    print(error_predict_names['Gypsy'])
    print(error_predict_names['Copia'])
    print(error_predict_names['PIF-Harbinger'])
    print(error_predict_names['Mutator'])