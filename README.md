# MegaMmap

MegaMmap is a software distributed shared memory which can abstract over both memory
and storage.

# Dependencies

## Apache Arrow
https://arrow.apache.org/docs/developers/cpp/building.html
```
scspkg create arrow
cd $(scspkg pkg src arrow)
git clone https://github.com/apache/arrow.git
cd arrow/cpp
mkdir build
cd build
cmake ../ -DCMAKE_INSTALL_PREFIX=`scspkg pkg root arrow` -DARROW_PARQUET=ON
make -j8
make install
```

## Hermes

```
spack install hermes@dev-1.1
```

# Install

```
module load arrow
module load hermes_run
module load mega_mmap
spack load hermes_shm
```

```
scspkg create mega_mmap
cd $(scspkg pkg src mega_mmap)
git clone https://github.com/lukemartinlogan/mega_mmap.git
cd mega_mmap
export MM_PATH=${PWD}
scspkg env set mega_mmap MM_PATH ${MM_PATH}
mkdir build
cd build
cmake ../ -DCMAKE_INSTALL_PREFIX=`scspkg pkg src mega_mmap`
make -j8
```

# Build environment

```
module load arrow
module load hermes_run
module load mega_mmap
spack load hermes_shm
jarvis env build mega_mmap +MM_PATH
```