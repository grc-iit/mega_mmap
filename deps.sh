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

# Install apache arrow
scspkg create arrow
cd $(scspkg pkg src arrow)
git clone https://github.com/apache/arrow.git -b apache-arrow-15.0.1
cd arrow/cpp
mkdir build
cd build
cmake ../ -DARROW_PARQUET=ON -DCMAKE_INSTALL_PREFIX=$(scspkg pkg root arrow)
make -j32 install

# Install spark
spack install openjdk@11
spack load openjdk@11
scspkg create spark
cd `scspkg pkg src spark`
wget https://dlcdn.apache.org/spark/spark-3.5.1/spark-3.5.1.tgz
tar -xzf spark-3.5.1.tgz
cd spark-3.5.1
./build/mvn -T 16 -DskipTests clean package
scspkg env set spark SPARK_SCRIPTS=${PWD}
scspkg env prepend spark PATH "${PWD}/bin"
module load spark

# Install megammap
spack install mega_mmap

