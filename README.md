# MegaMmap

MegaMmap is a software distributed shared memory which can abstract over both memory
and storage.

# Dependencies

```
scspkg create arrow
cd $(scspkg pkg src arrow)
git clone https://github.com/apache/arrow.git -b apache-arrow-15.0.1
cd arrow/cpp
mkdir build
cd build
cmake ../ -DARROW_PARQUET=ON -DCMAKE_INSTALL_PREFIX=$(scspkg pkg root arrow)
make -j32 install
```

## Spark
```
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
```

## Hermes

```
spack install hermes@master
```

# Install

```
module load hermes_run mega_mmap arrow
spack load hermes_shm
```

```
scspkg create mega_mmap
cd $(scspkg pkg src mega_mmap)
git clone https://github.com/lukemartinlogan/mega_mmap.git
cd mega_mmap
export MM_PATH=$(scspkg pkg root mega_mmap)
scspkg env set mega_mmap MM_PATH=${MM_PATH}
mkdir build
cd build
cmake ../ \
-DCMAKE_BUILD_TYPE=Release \
-DCMAKE_INSTALL_PREFIX=$(scspkg pkg root mega_mmap)
make -j8
```

# Build environment

```
module load hermes_run mega_mmap spark arrow
spack load hermes_shm  
jarvis env build mega_mmap +MM_PATH +SPARK_SCRIPTS
```