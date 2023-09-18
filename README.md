# MegaMmap

MegaMmap is a software distributed shared memory which can abstract over both memory
and storage.

# Dependencies

## Apache Arrow
https://arrow.apache.org/docs/developers/cpp/building.html
```
scspkg create arrow
cd `scspkg pkg src arrow`
git clone https://github.com/apache/arrow.git
cd arrow/cpp
mkdir build
cd build
cmake ../ -DCMAKE_INSTALL_PREFIX=`scspkg pkg root arrow` -DARROW_PARQUET=ON
make -j8
make install
```

