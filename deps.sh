#!/bin/bash

# Install SCSPKG
git clone https://github.com/grc-iit/scspkg.git
pushd scspkg
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
popd
scspkg init False

# Install jarvis CD
git clone https://github.com/grc-iit/jarvis-cd
pushd jarvis-cd
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
popd

# Install apache thrift
#scspkg create thrift
#pushd $(scspkg pkg src thrift)
#wget https://dlcdn.apache.org/thrift/0.20.0/thrift-0.20.0.tar.gz
#tar -xzf thrift-0.20.0.tar.gz
#pushd thrift-0.20.0
#./bootstrap.sh
#./configure --prefix=$(scspkg pkg root thrift) --with-python=false
#make -j32 install
#module load thrift
#popd
#popd
#
## Install apache arrow
#scspkg create arrow
#pushd $(scspkg pkg src arrow)
#git clone https://github.com/apache/arrow.git -b apache-arrow-15.0.1
#pushd arrow/cpp
#mkdir build
#pushd build
#cmake ../ -DARROW_PARQUET=ON -DCMAKE_INSTALL_PREFIX=$(scspkg pkg root arrow)
#make -j32 install
#module load arrow
#popd
#popd
#popd

# Install arrow
spack install arrow +parquet^thrift -python

# Install spark
spack install openjdk@11
spack load openjdk@11
scspkg create spark
pushd $(scspkg pkg src spark)
wget https://dlcdn.apache.org/spark/spark-3.5.1/spark-3.5.1.tgz
tar -xzf spark-3.5.1.tgz
pushd spark-3.5.1
./build/mvn -T 16 -DskipTests clean package
scspkg env set spark SPARK_SCRIPTS=${PWD}
scspkg env prepend spark PATH "${PWD}/bin"
module load spark
popd
popd

# Download SCSREPO
git clone https://github.com/lukemartinlogan/scs-repo.git
spack repo add scs-repo

# Install hermes
spack install hermes@master
spack load hermes@master

# Install mega_mmap
scspkg create mega_mmap
export MM_PATH=$(scspkg pkg root mega_mmap)
scspkg env set mega_mmap MM_PATH=${MM_PATH}
pushd build
cmake ../ \
-DCMAKE_BUILD_TYPE=Release \
-DCMAKE_INSTALL_PREFIX=$(scspkg pkg root mega_mmap)
make -j32 install
module load mega_mmap
popd
