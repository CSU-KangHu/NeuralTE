#注：这里模型只是简单的利用CNN进行分类

#from keras.layers import Conv2D, MaxPooling2D

from HiTE_util.Util import load_repbase_with_TSD, generate_mats

all_wicker_class = ('Tc1-Mariner', 'hAT', 'Mutator', 'Merlin', 'Transib', 'P', 'PiggyBac', 'PIF-Harbinger', 'CACTA', 'Crypton', 'Helitron', 'Maverick', 'Copia', 'Gypsy', 'Bel-Pao', 'Retrovirus', 'DIRS', 'Ngaro', 'VIPER', 'Penelope', 'R2', 'RTE', 'Jockey', 'L1', 'I', 'tRNA', '7SL', '5S')
class_num = len(all_wicker_class)
print(class_num)

work_dir = '/home/hukang/TE_Classification/DeepTE-master/example_data'
repbase_train = work_dir + '/repbase_train_part.ref'
repbase_test = work_dir + '/repbase_test_part.ref'
X, Y = load_repbase_with_TSD(repbase_train)

X = generate_mats(X)
print(X)