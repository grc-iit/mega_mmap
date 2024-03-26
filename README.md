# MegaMmap

MegaMmap is a software distributed shared memory which can abstract over both memory
and storage.

# Dependencies

## Apache Arrow
https://arrow.apache.org/docs/developers/cpp/building.html
```
spack install arrow@15 +parquet
```

## Spark
```
spack install spark
scspkg create spark
scspkg env set spark SPARK_SCRIPTS=$(spack find --format {PREFIX} spark)
```

## Hermes

```
spack install hermes@master
```

# Install

```
module load hermes_run
module load mega_mmap
spack load hermes_shm arrow
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
module load hermes_run
module load mega_mmap
spack load hermes_shm arrow
module load spark
jarvis env build mega_mmap +MM_PATH +SPARK_SCRIPTS
```