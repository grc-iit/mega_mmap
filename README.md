# MegaMmap

MegaMmap is a software distributed shared memory which can abstract over both memory
and storage.

# Dockerfile

We have a dockerfile to install MegaMmap and its dependencies. This is more
of a unit test of the installation process.

```
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
docker build -t lukemartinlogan/mega_mmap . -f docker/deps.Dockerfile
docker run -it --mount src=${PWD},target=/hermes,type=bind \
--name hermes_deps_c \
--network host \
--memory=8G \
--shm-size=8G \
-p 4000:4000 \
-p 4001:4001 \
lukemartinlogan/mega_mmap
```

# Dependencies

* scs-repo
* Hermes
* Apache arrow (check deps.sh)
* Apache spark (only for evaluation)

deps.sh installs these dependencies. However, we find that the
installation of Apache Arrow is not very reliable. Reviewers should
install Apache Arrow manually.

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