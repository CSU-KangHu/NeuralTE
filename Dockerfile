FROM continuumio/miniconda3

# Author and maintainer
MAINTAINER Kang Hu <kanghu@csu.edu.cn>
LABEL description="NeuralTE: an accurate approach for Transposable Element superfamily classification with multi-feature fusion" \
      author="kanghu@csu.edu.cn"

ARG DNAME="NeuralTE"

RUN apt-get update && apt-get install unzip --yes && apt-get install less --yes && apt-get install curl --yes

# Download NeuralTE from Github
# RUN git clone https://github.com/CSU-KangHu/NeuralTE.git
# Download NeuralTE from Zenodo
RUN curl -LJO https://zenodo.org/records/10538960/files/CSU-KangHu/NeuralTE-v1.0.0.zip?download=1 &&  \
    unzip NeuralTE-v1.0.0.zip && mv CSU-KangHu-NeuralTE-* /NeuralTE

RUN conda install mamba -c conda-forge -y
RUN cd /NeuralTE && chmod +x tools/* && mamba env create --name ${DNAME} --file=environment.yml && conda clean -a

# Make RUN commands use the new environment
# name need to be the same with the above ${DNAME}
SHELL ["conda", "run", "-n", "NeuralTE", "/bin/bash", "-c"]

# avoid different perl version conflict
ENV PERL5LIB /
ENV PATH /opt/conda/envs/${DNAME}/bin:$PATH
USER root

WORKDIR /NeuralTE
RUN cd /NeuralTE

CMD ["bash"]