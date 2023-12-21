# NeuralTE
[![GitHub](https://img.shields.io/badge/python-3-blue)](https://www.python.org/)
[![GitHub](https://img.shields.io/badge/license-GPL--3.0-green)](https://github.com/CSU-KangHu/HiTE/blob/master/LICENSE)
[![DockerHub](https://img.shields.io/badge/Singularity-support-blue)](https://docs.sylabs.io/guides/3.5/user-guide/introduction.html)
[![DockerHub](https://img.shields.io/badge/Docker-support-orange)](https://hub.docker.com/repository/docker/kanghu/hite/general)
[![Conda](https://img.shields.io/badge/Conda-support-yellow)](https://docs.conda.io/en/latest/)

`NeuralTE` uses a Convolutional Neural Network (CNN) to classify the TE library at the **superfamily** level, based on the **sequence** and **structural features** of transposable elements (TEs).


## Table of Contents
- [Installation](#install)
- [Trained models](#models)
- [Demo data](#demo)
- [Train a new model](#train_model)
- [Usage](#cmd)
- [Experiment reproduction](#ER)
- [More tutorials](#QA)

## <a name="install"></a>Installation
NeuralTE is built on [Python3](https://www.python.org/) and [Keras](https://keras.io/).
   - Prerequisites: \
       [Python3](https://www.python.org/) (version=3.8.16)\
       [CUDA Toolkit](https://anaconda.org/anaconda/cudatoolkit) (version>=11.2, for GPU only)
   - Dependencies: \
       [tensorflow](https://www.tensorflow.org/) (version=2.6.0) \
       [Keras](https://keras.io/) (version=2.6.0) \
       [numpy](http://www.numpy.org/) \
       [biopython](https://biopython.org/) \
       [scikit-learn](https://scikit-learn.org/stable/) \
       [matplotlib](https://matplotlib.org/) \
       [seaborn](https://seaborn.pydata.org/) \
       [rmblast](https://www.repeatmasker.org/rmblast/) \
       [openpyxl](https://openpyxl.readthedocs.io/)

#### System Requirements
`NeuralTE` requires a standard computer to use the Convolutional Neural Network (CNN). Using GPU could acceralate the process of TE classification.

Recommended Hardware requirements: 40 CPU processors, 128 GB RAM.

Recommended OS: (Ubuntu 16.04, CentOS 7, etc.)

#### 运行前须知
`NeuralTE` 使用了structure features如long terminal repeats, terminal inverted repeats, and target site duplications来帮助分类，
因此我们默认输入到NeuralTE中的TE library为full-length TEs，即具有完整的TE结构，我们推荐使用[HiTE](https://github.com/CSU-KangHu/HiTE)来生成TE libraries。
如果输入的TE不具备任何的domain特征，同时是结构不完整的碎片化TE，我们推荐使用配置了[完整Dfam library](https://www.repeatmasker.org/RepeatMasker/)的[RepeatClassifier](https://github.com/Dfam-consortium/RepeatModeler/blob/master/RepeatClassifier)利用同源搜索进行分类。

```sh
git clone https://github.com/CSU-KangHu/NeuralTE.git
# Alternatively, you can download the zip file directly from the repository.
cd NeuralTE
chmod +x tools/*

conda env create --name NeuralTE -f environment.yml
```

## <a name="models"></a>Pre-trained models
#### Model Differences:
* NeuralTE Model: \
The [NeuralTE model]() uses sequence k-mer, terminal, TE domain, and 5bp terminal features, trained using Repbase version 28.06. 
* NeuralTE-TSDs Model: \
The [NeuralTE-TSDs model]() uses sequence k-mer, terminal, TE domain, 5bp terminal, and **target site duplications (TSDs)** features, trained using partial species data (463 species) from Repbase version 28.06. This model needs to be used with the corresponding genome of the species.

## <a name="reformat"></a>Input Format Conversion

To reduce redundancy, many tools for TE identification separate the LTR termini and internal sequences of LTR retrotransposons in their output, which can affect the accurate classification of LTR retrotransposons.

We recommend using a complete TE library or reconstructing full-length LTR sequences based on LTR termini and internal sequences for proper classification.

Regarding the output results from three popular TE identification tools (EDTA, RepeatModeler2, HiTE), we categorize the TE library into two types:

1. TE libraries whose headers do not include the correspondence between LTR termini and internal sequences:
   1. The TE library output by EDTA is named $genome.mod.EDTA.TElib.fa, lacking the information in the sequence headers to determine the correspondence between LTR termini and internal sequences. Thus, we recommend using the complete TE library outputted by EDTA for classification, $genome.mod.EDTA.final/$genome.mod.EDTA.intact.fa.

2. TE libraries whose headers include the correspondence between LTR termini and internal sequences:
   1. The TE library output by RepeatModeler2 is named ${species}-families.fa, containing headers with information about the correspondence between LTR termini and internal sequences, such as `>ltr-1_family-1#LTR/Gypsy [ Type=LTR, Final Multiple Alignment Size = 68 ]` and `>ltr-1_family-2#LTR/Gypsy [ Type=INT, Final Multiple Alignment Size = 22 ]`.
   For the TE library output from RepeatModeler2, we recommend using `NeuralTE/utils/reName_RM2.py` to rename the headers of the input TE library, formatted as `>Gypsy-171_OS-I` and `>Gypsy-171_OS-LTR`.
   2. The TE library output by HiTE is named confident_TE.cons.fa, containing headers with information about the correspondence between LTR termini and internal sequences, such as `>ltr_1241-INT` and `>ltr_1241-LTR`.
   The TE library output by HiTE does not require additional processing.

```sh
# 1. Rename the LTR retrotransposons in the TE library output from RepeatModeler2, format: `RM2_intactLTR_114-LTR` and `RM2_intactLTR_114-I`
python ${pathTo}/NeuralTE/utils/reName_RM2.py \
 -i ${pathTo}/${species}-families.fa \
 -o ${pathTo}/${species}-families.rename.fa
```

## <a name="reformat"></a>输入格式转换
为了减少冗余，目前许多 TEs 识别的工具输出都将LTR转座子的LTR终端和LTR内部序列分开存储，这会影响对LTR转座子正确地分类。
我们推荐使用完整的TE library，或者根据LTR终端和内部序列恢复完整的LTR序列，然后进行分类。
针对当前热门的三种TE识别工具(EDTA, RepeatModeler2, HiTE)的输出结果，我们将TE library划分为两类:
1. TE library的header中不包含LTR终端和内部序列对应关系
   1. EDTA输出的TE library为$genome.mod.EDTA.TElib.fa，无法根据序列的header来判断LTR终端和内部序列的对应关系，因此我们推荐使用EDTA输出的完整TE library 进行分类，$genome.mod.EDTA.final/$genome.mod.EDTA.intact.fa。
2. TE library的header中包含LTR终端和内部序列对应关系
   1. RepeatModeler2输出的TE library为${species}-families.fa，它的header中包含了LTR终端和内部序列的对应关系，如`>ltr-1_family-1#LTR/Gypsy [ Type=LTR, Final Multiple Alignment Size = 68 ]`和 `>ltr-1_family-2#LTR/Gypsy [ Type=INT, Final Multiple Alignment Size = 22 ]`。
   针对RepeatModeler2输出的TE library，我们推荐使用`NeuralTE/utils/reName_RM2.py` 将输入的TE library的header重命名，格式如`>Gypsy-171_OS-I`和`>Gypsy-171_OS-LTR`。
   2. HiTE输出的TE library为confident_TE.cons.fa，它的header中包含了LTR终端和内部序列的对应关系，如`>ltr_1241-INT`和`>ltr_1241-LTR`。
   HiTE输出的TE library不需要额外处理。

```sh
# 1.将RepeatModeler2输出的TE library中的LTR转座子重命名，格式为: `RM2_intactLTR_114-LTR` 和 `RM2_intactLTR_114-I`
python ${pathTo}/NeuralTE/utils/reName_RM2.py \
 -i ${pathTo}/${species}-families.fa \
 -o ${pathTo}/${species}-families.rename.fa
```

## <a name="demo"></a>Demo data

Please refer to `NeuralTE/demo` for some demo data to play with:
* _test.fa_: demo TE library.
* _genome.fa_: demo genome sequence.

```sh
# 1.Classify TE library without genome
# inputs: 
#       --data: TE library to be classified.
#       --model_path: Pre-trained Neural TE model without using TSDs features.
#       --outdir: Output directory. The `--outdir` should not be the same as the directory where the `--data` file is located.
# outputs: 
#       classified.info: Classification labels corresponding to TE names.
#       classified_TE.fa: Classified TE library.
python ${pathTo}/NeuralTE/src/Classifier.py \
 --data ${pathTo}/NeuralTE/demo/test.fa \
 --model_path ${pathTo}/NeuralTE/models/NeuralTE_model.h5 \
 --outdir ${outdir} \
 --thread ${threads_num}

 # e.g., my command: python /home/hukang/NeuralTE/src/Classifier.py \
 # --data /home/hukang/NeuralTE/demo/test.fa \
 # --model_path /home/hukang/NeuralTE/models/NeuralTE_model.h5 \
 # --outdir /home/hukang/NeuralTE/work \
 # --thread 40
 
 
 # 2.Classify the TE library with genome
 #       test.fa: TE library to be classified 
 #       genome.fa: The genome corresponding to TE library
 #       NeuralTE-TSDs_model.h5: Pre-trained Neural TE model using TSDs features
 # outputs: 
 #       classified.info: Classification labels corresponding to TE names
 #       classified_TE.fa: Classified TE library
python ${pathTo}/NeuralTE/src/Classifier.py \
 --data ${pathTo}/NeuralTE/demo/test.fa \
 --genome ${pathTo}/NeuralTE/demo/genome.fa \
 --is_plant 1 \
 --use_TSD 1 \
 --model_path ${pathTo}/NeuralTE/models/NeuralTE-TSDs_model.h5 \
 --outdir ${outdir} \
 --thread ${threads_num}

 # e.g., my command: python /home/hukang/NeuralTE/src/Classifier.py \
 # --data /home/hukang/NeuralTE/demo/test.fa \
 # --genome /home/hukang/NeuralTE/demo/genome.fa \
 # --is_plant 1 \
 # --use_TSD 1 \
 # --model_path /home/hukang/NeuralTE/models/NeuralTE-TSDs_model.h5 \
 # --outdir /home/hukang/NeuralTE/work \
 # --thread 40
```

## <a name="demo"></a>使用某个物种的Repbase library评估NeuralTE性能
这一块后面不需要，要去掉。
```sh
 # 2.评估NeuralTE with genome
 #       test.fa: TE library to be classified 
 #       genome.fa: The genome corresponding to TE library
 #       NeuralTE-TSDs_model.h5: Pre-trained Neural TE model using TSDs features
 # outputs: 
 #       classified.info: Classification labels corresponding to TE names
 #       classified_TE.fa: Classified TE library
python ${pathTo}/NeuralTE/src/Classifier.py \
 --data ${pathTo}/repbase.ref \
 --genome ${pathTo}/genome.fa \
 --species '${species_name}' \
 --is_plant 1 \
 --use_TSD 1 \
 --is_predict 0 \
 --model_path ${pathTo}/NeuralTE/models/NeuralTE-TSDs_model.h5 \
 --outdir ${outdir} \
 --thread ${threads_num} \
 --is_wicker 1
 # e.g., my command: python /home/hukang/NeuralTE/src/Classifier.py \
 # --data /home/hukang/Repbase/oryrep.ref \
 # --genome /home/hukang/Genome/GCF_001433935.1_IRGSP-1.0_genomic.fna \
 # --species 'Oryza sativa' \
 # --is_plant 1 \
 # --use_TSD 1 \
 # --is_predict 0 \
 # --model_path /home/hukang/NeuralTE/models/NeuralTE-TSDs_model.h5 \
 # --outdir /home/hukang/NeuralTE/work \
 # --thread 40 \
 # --is_wicker 1
```

## <a name="train_model"></a>Train a new model
- Prerequisites: \
       [Repbase*.fasta.tar.gz](https://www.girinst.org/server/RepBase/index.php) (version>=28.06)\
       [Genomes](https://www.ncbi.nlm.nih.gov/) (Optional, for TSDs model only)

```sh
 # 0. preprocess repbase database （包含合并repbase子文件，连接LTR终端和内部序列，过滤不完整的LTR转座子等功能）
 # inputs: 
 #        repbase_dir: 包含所有Repbase文件的目录 
 #        out_dir: 包含预处理之后结果的输出目录
 # outputs: 
 #        all_repbase.ref: 所有Repbase数据库序列
python ${pathTo}/utils/preprocess_repbase.py \
 --repbase_dir ${pathTo}/RepBase*.fasta \
 --out_dir ${out_dir}
 # e.g., my command: python /home/hukang/NeuralTE/utils/preprocess_repbase.py \
 # --repbase_dir /home/hukang/RepBase28.06.fasta/ \
 # --out_dir /home/hukang/test/
 
 
 # 1. split train test datasets
 # inputs: 
 #        data_path: 所有Repbase数据库序列
 #        out_dir: 数据集划分后的输出目录
 #        ratio: Ratio of training set to test set
 # outputs: 
 #        train.ref: 所有Repbase数据库序列的80%形成训练集
 #        test.ref:所有Repbase数据库序列的20%形成测试集
python ${pathTo}/utils/split_train_test.py \
 --data_path ${pathTo}/all_repbase.ref \
 --out_dir ${out_dir} \
 --ratio 0.8
 # e.g., my command: python /home/hukang/NeuralTE/utils/split_train_test.py \
 # --data_path /home/hukang/test/all_repbase.ref \
 # --out_dir /home/hukang/test/ \
 # --ratio 0.8


 # 2.Train a new NeuralTE Model
 # inputs: 
 #        train.ref: 训练集
 # outputs: 
 #        model_${time}.h5: Generate h5 format file in the ${pathTo}/NeuralTE/models directory
python ${pathTo}/NeuralTE/src/Trainer.py \
 --data ${pathTo}/train.ref \
 --is_train 1 \
 --is_predict 0 \
 --outdir ${outdir} \
 --thread ${threads_num}
 # e.g., my command: python /home/hukang/NeuralTE/src/Trainer.py \
 # --data /home/hukang/NeuralTE/data/train.ref \
 # --is_train 1 \
 # --is_predict 0 \
 # --outdir /home/hukang/NeuralTE/work \
 # --thread 40
 # 替换原模型
 cd ${pathTo}/NeuralTE/models && mv model_${time}.h5 NeuralTE_model.h5
 
 
 # 3.Train a new NeuralTE-TSDs Model
 # inputs: 
 #        train.ref: 训练集
 #        genome.info: Modify the 'genome.info' file in the ${pathTo}/NeuralTE/data directory. Ensure that 'Scientific Name' corresponds to the species names in `train.ref`, and 'Genome Path' should be an absolute path.
 # outputs: 
 #        model_${time}.h5: Generate h5 format file in the ${pathTo}/NeuralTE/models directory
python ${pathTo}/NeuralTE/src/Trainer.py \
 --data ${pathTo}/train.ref \
 --genome ${pathTo}/NeuralTE/data/genome.info \
 --is_train 1 \
 --is_predict 0 \
 --use_TSD 1 \
 --outdir ${outdir} \
 --thread ${threads_num}
 # e.g., my command: python /home/hukang/NeuralTE/src/Trainer.py \
 # --data /home/hukang/NeuralTE/data/train.ref \
 # --genome /home/hukang/NeuralTE/data/genome.info \
 # --is_train 1 \
 # --is_predict 0 \
 # --use_TSD 1 \
 # --outdir /home/hukang/NeuralTE/work \
 # --thread 40
 # 替换原模型
cd ${pathTo}/NeuralTE/models && mv model_${time}.h5 NeuralTE-TSDs_model.h5


 # 4.CrossValidator 进行五折交叉验证
 # inputs: 
 #        all_repbase.ref: 所有具有TSD特征的Repbase数据
 # outputs: 
 #        model_fold_{fold}.h5: Generate h5 format file in the ${pathTo}/NeuralTE/models directory
python ${pathTo}/NeuralTE/src/CrossValidator.py \
 --data ${pathTo}/all_repbase.ref \
 --use_TSD 1 \
 --use_terminal 1 \
 --use_domain 1 \
 --use_ends 1 \
 --use_kmers 1 \
 --is_predict 0 \
 --thread ${threads_num}
 # e.g., my command: python /home/hukang/NeuralTE/src/CrossValidator.py \
 # --data /home/hukang/NeuralTE/data/TSD_data/all_repbase.ref \
 # --use_TSD 1 \
 # --use_terminal 1 \
 # --use_domain 1 \
 # --use_ends 1 \
 # --use_kmers 1 \
 # --is_predict 0 \
 # --thread 40
```

## <a name="reproduction"></a>Experiment reproduction
- Prerequisites: \
       [Repbase*.fasta.tar.gz](https://www.girinst.org/server/RepBase/index.php) (version>=28.06)\
       [Genomes](https://www.ncbi.nlm.nih.gov/) (Optional, for TSDs model only)

```sh
 # 1.重现Dataset1对应的实验结果
 # 1.1 不使用TSDs特征
 # inputs: 
 #        repbase_test.fa: Merge all subfiles within the Repbase directory to generate repbase.fa, and extract 20% of the sequences to create repbase_test.ref.
 #        NeuralTE_data1_model.h5: 
 # outputs: 
 #        classified.info: Classification labels corresponding to TE names
 #        classified_TE.fa: Classified TE library 
 #        confusion_matrix.png: Confusion matrix of prediction results.
python ${pathTo}/NeuralTE/src/Classifier.py \
 --data ${pathTo}/repbase_test.fa \
 --model_path ${pathTo}/NeuralTE/models/NeuralTE_data1_model.h5
 --use_TSD 0 \
 --is_predict 0 \
 --outdir ${outdir} \
 --thread ${threads_num}
 # e.g., my command: python /home/hukang/NeuralTE/src/Classifier.py \
 # --data /home/hukang/NeuralTE/data/repbase_test.fa \
 # --model_path /home/hukang/NeuralTE/models/NeuralTE_data1_model.h5 \
 # --use_TSD 0 \
 # --is_predict 0 \
 # --outdir /home/hukang/NeuralTE/work \
 # --thread 40
 
 
 # 1.2 使用TSDs特征
 # inputs: 
 #        repbase_test.fa: Merge all subfiles within the Repbase directory to generate repbase.fa, and extract 20% of the sequences to create repbase_test.ref.
 #        NeuralTE_data1_model.h5: 
 # outputs: 
 #        classified.info: Classification labels corresponding to TE names
 #        classified_TE.fa: Classified TE library 
 #        confusion_matrix.png: Confusion matrix of prediction results.
python ${pathTo}/NeuralTE/src/Classifier.py \
 --data ${pathTo}/repbase_test.fa \
 --genome ${pathTo}/NeuralTE/data/genome.info
 --model_path ${pathTo}/NeuralTE/models/NeuralTE_data1-TSDs_model.h5
 --use_TSD 1 \
 --is_predict 0 \
 --outdir ${outdir} \
 --thread ${threads_num}
 # e.g., my command: python /home/hukang/NeuralTE/src/Classifier.py \
 # --data /home/hukang/NeuralTE/data/repbase_test.fa \
 # --model_path /home/hukang/NeuralTE/models/NeuralTE_data1-TSDs_model.h5 \
 # --use_TSD 1 \
 # --is_predict 0 \
 # --outdir /home/hukang/NeuralTE/work \
 # --thread 40
```

```sh
# 重现消融实验 (所有消融实验结果都基于五折交叉验证)
## Dataset2 (所有转座子)
## 1. 使用所有特征
python /home/hukang/NeuralTE/src/CrossValidator.py --data /home/hukang/NeuralTE_dataset/Dataset2/all_repbase.ref --is_predict 0 --use_TSD 1 --use_kmers 1 --use_terminal 1 --use_domain 1 --use_ends 1 --outdir /home/hukang/NeuralTE/work/cross_validate
## 2. 使用除domain特征外的所有其余特征
python /home/hukang/NeuralTE/src/CrossValidator.py --data /home/hukang/NeuralTE_dataset/Dataset2/all_repbase.ref --is_predict 0 --use_TSD 1 --use_kmers 1 --use_terminal 1 --use_domain 0 --use_ends 1 --outdir /home/hukang/NeuralTE/work/cross_validate
## 3. 使用除k-mer特征外的所有其余特征
python /home/hukang/NeuralTE/src/CrossValidator.py --data /home/hukang/NeuralTE_dataset/Dataset2/all_repbase.ref --is_predict 0 --use_TSD 1 --use_kmers 0 --use_terminal 1 --use_domain 1 --use_ends 1 --outdir /home/hukang/NeuralTE/work/cross_validate
## 4. 使用除TSDs特征外的所有其余特征
python /home/hukang/NeuralTE/src/CrossValidator.py --data /home/hukang/NeuralTE_dataset/Dataset2/all_repbase.ref --is_predict 0 --use_TSD 0 --use_kmers 1 --use_terminal 1 --use_domain 1 --use_ends 1 --outdir /home/hukang/NeuralTE/work/cross_validate
## 5. 使用除terminal特征外的所有其余特征
python /home/hukang/NeuralTE/src/CrossValidator.py --data /home/hukang/NeuralTE_dataset/Dataset2/all_repbase.ref --is_predict 0 --use_TSD 1 --use_kmers 1 --use_terminal 0 --use_domain 1 --use_ends 1 --outdir /home/hukang/NeuralTE/work/cross_validate
## 6. 使用除终端5-bp特征外的所有其余特征
python /home/hukang/NeuralTE/src/CrossValidator.py --data /home/hukang/NeuralTE_dataset/Dataset2/all_repbase.ref --is_predict 0 --use_TSD 1 --use_kmers 1 --use_terminal 1 --use_domain 1 --use_ends 0 --outdir /home/hukang/NeuralTE/work/cross_validate

## Dataset3 (非自治DNA转座子)
## 1. 使用所有特征
python /home/hukang/NeuralTE/src/CrossValidator.py --data /home/hukang/NeuralTE_dataset/Dataset3/all_repbase.non_auto.ref --is_predict 0 --use_TSD 1 --use_kmers 1 --use_terminal 1 --use_domain 1 --use_ends 1 --outdir /home/hukang/NeuralTE/work/cross_validate_non_auto
## 2. 使用除domain特征外的所有其余特征
python /home/hukang/NeuralTE/src/CrossValidator.py --data /home/hukang/NeuralTE_dataset/Dataset3/all_repbase.non_auto.ref --is_predict 0 --use_TSD 1 --use_kmers 1 --use_terminal 1 --use_domain 0 --use_ends 1 --outdir /home/hukang/NeuralTE/work/cross_validate_non_auto
## 3. 使用除k-mer特征外的所有其余特征
python /home/hukang/NeuralTE/src/CrossValidator.py --data /home/hukang/NeuralTE_dataset/Dataset3/all_repbase.non_auto.ref --is_predict 0 --use_TSD 1 --use_kmers 0 --use_terminal 1 --use_domain 1 --use_ends 1 --outdir /home/hukang/NeuralTE/work/cross_validate_non_auto
## 4. 使用除TSDs特征外的所有其余特征
python /home/hukang/NeuralTE/src/CrossValidator.py --data /home/hukang/NeuralTE_dataset/Dataset3/all_repbase.non_auto.ref --is_predict 0 --use_TSD 0 --use_kmers 1 --use_terminal 1 --use_domain 1 --use_ends 1 --outdir /home/hukang/NeuralTE/work/cross_validate_non_auto
## 5. 使用除terminal特征外的所有其余特征
python /home/hukang/NeuralTE/src/CrossValidator.py --data /home/hukang/NeuralTE_dataset/Dataset3/all_repbase.non_auto.ref --is_predict 0 --use_TSD 1 --use_kmers 1 --use_terminal 0 --use_domain 1 --use_ends 1 --outdir /home/hukang/NeuralTE/work/cross_validate_non_auto
## 6. 使用除终端5-bp特征外的所有其余特征
python /home/hukang/NeuralTE/src/CrossValidator.py --data /home/hukang/NeuralTE_dataset/Dataset3/all_repbase.non_auto.ref --is_predict 0 --use_TSD 1 --use_kmers 1 --use_terminal 1 --use_domain 1 --use_ends 0 --outdir /home/hukang/NeuralTE/work/cross_validate_non_auto
```

```sh
# 重现Dataset6水稻实验
## 1. 使用所有特征
python /home/hukang/NeuralTE/src/Trainer.py --data /home/hukang/NeuralTE_dataset/Dataset6/train.ref --is_train 1 --is_predict 0 --use_TSD 1 --outdir /home/hukang/NeuralTE/work/dataset6
python /home/hukang/NeuralTE/src/Classifier.py --data /home/hukang/NeuralTE_dataset/Dataset6/test.ref --use_TSD 1 --is_predict 0 --model_path /home/hukang/NeuralTE/models/model_2023-12-21.10-25-26.h5 --outdir /home/hukang/NeuralTE/work/dataset6 --threads 40
```

# Usage
#### 1. Classify TE library
```shell
usage: Classifier.py [-h] [--data data] [--outdir output_dir] [--genome genome] [--model_path model_path] [--use_terminal use_terminal] [--use_TSD use_TSD] [--use_domain use_domain] [--use_ends use_ends]
                     [--is_predict is_predict] [--is_wicker is_wicker] [--is_plant is_plant] [--threads thread_num] [--internal_kmer_sizes internal_kmer_sizes] [--terminal_kmer_sizes terminal_kmer_sizes]

########################## NeuralTE, version 1.0.0 ##########################

optional arguments:
  -h, --help            show this help message and exit
  --data data           Input fasta file used to predict, header format: seq_name label species_name, refer to "data/test.example.fa" for example.
  --outdir output_dir   Output directory, store temporary files
  --genome genome       Genome path, use to search for TSDs
  --model_path model_path
                        Input the path of trained model, absolute path.
  --use_terminal use_terminal
                        Whether to use LTR, TIR terminal features, 1: true, 0: false. default = [ 1 ]
  --use_TSD use_TSD     Whether to use TSD features, 1: true, 0: false. default = [ 0 ]
  --use_domain use_domain
                        Whether to use domain features, 1: true, 0: false. default = [ 1 ]
  --use_ends use_ends   Whether to use 5-bp terminal ends features, 1: true, 0: false. default = [ 1 ]
  --is_predict is_predict
                        Enable prediction mode, 1: true, 0: false. default = [ 1 ]
  --is_wicker is_wicker
                        Use Wicker or RepeatMasker classification labels, 1: Wicker, 0: RepeatMasker. default = [ 1 ]
  --is_plant is_plant   Is the input genome of a plant? 0 represents non-plant, while 1 represents plant. default = [ 0 ]
  --threads thread_num  Input thread num, default = [ 104 ]
  --internal_kmer_sizes internal_kmer_sizes
                        The k-mer size used to convert internal sequences to k-mer frequency features, default = [ [1, 2, 4] MB ]
  --terminal_kmer_sizes terminal_kmer_sizes
                        The k-mer size used to convert terminal sequences to k-mer frequency features, default = [ [1, 2, 4] ]
```
#### 2. train a new model
```shell
usage: Trainer.py [-h] [--data data] [--outdir output_dir] [--use_terminal use_terminal] [--use_TSD use_TSD] [--use_domain use_domain] [--use_ends use_ends] [--threads thread_num]
                  [--internal_kmer_sizes internal_kmer_sizes] [--terminal_kmer_sizes terminal_kmer_sizes] [--cnn_num_convs cnn_num_convs] [--cnn_filters_array cnn_filters_array]
                  [--cnn_kernel_sizes_array cnn_kernel_sizes_array] [--cnn_dropout cnn_dropout] [--batch_size batch_size] [--epochs epochs] [--use_checkpoint use_checkpoint]

########################## NeuralTE, version 1.0.0 ##########################

optional arguments:
  -h, --help            show this help message and exit
  --data data           Input fasta file used to train model, header format: seq_name label species_name, refer to "data/train.example.fa" for example.
  --outdir output_dir   Output directory, store temporary files
  --use_terminal use_terminal
                        Whether to use LTR, TIR terminal features, 1: true, 0: false. default = [ 1 ]
  --use_TSD use_TSD     Whether to use TSD features, 1: true, 0: false. default = [ 0 ]
  --use_domain use_domain
                        Whether to use domain features, 1: true, 0: false. default = [ 1 ]
  --use_ends use_ends   Whether to use 5-bp terminal ends features, 1: true, 0: false. default = [ 1 ]
  --threads thread_num  Input thread num, default = [ 104 ]
  --internal_kmer_sizes internal_kmer_sizes
                        The k-mer size used to convert internal sequences to k-mer frequency features, default = [ [1, 2, 4] MB ]
  --terminal_kmer_sizes terminal_kmer_sizes
                        The k-mer size used to convert terminal sequences to k-mer frequency features, default = [ [1, 2, 4] ]
  --cnn_num_convs cnn_num_convs
                        The number of CNN convolutional layers. default = [ 3 ]
  --cnn_filters_array cnn_filters_array
                        The number of filters in each CNN convolutional layer. default = [ [32, 32, 32] ]
  --cnn_kernel_sizes_array cnn_kernel_sizes_array
                        The kernel size in each of CNN convolutional layer. default = [ [3, 3, 3] ]
  --cnn_dropout cnn_dropout
                        The threshold of CNN Dropout. default = [ 0.5 ]
  --batch_size batch_size
                        The batch size in training model. default = [ 32 ]
  --epochs epochs       The number of epochs in training model. default = [ 50 ]
  --use_checkpoint use_checkpoint
                        Whether to use breakpoint training. 1: true, 0: false. The model will continue training from the last failed parameters to avoid training from head. default = [ 0 ]
```


