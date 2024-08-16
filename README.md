# MegaMmap

MegaMmap is a software distributed shared memory which can abstract over both memory
and storage.

For anyone interested in our code and experiments, feel free to email:
llogan@hawk.iit.edu
lukemartinlogan@gmail.com

We are happy to help with any questions (or issues) 
regarding the code or experiments. We also keep a lookout for
github issues.

# Dockerfile

We have a dockerfile to install MegaMmap and its dependencies. This is more
of a unit test of the installation process and deps.sh.

```
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
docker build -t lukemartinlogan/mega_mmap . -f docker/deps.Dockerfile
docker run -it --mount src=${PWD},target=/mega_mmap,type=bind \
--name mega_mmap_c \
--network host \
--memory=8G \
--shm-size=8G \
-p 4000:4000 \
-p 4001:4001 \
lukemartinlogan/mega_mmap
```

NOTE: Hermes uses shared memory. Shared memory goes against docker's security
philosophy. To get around this, we add shm-size. You may need to change
this parameter depending on your system.

# Dependencies

* scs-repo
* Hermes
* Apache arrow (check deps.sh)
* Apache spark (only for evaluation)

deps.sh installs these dependencies. However, Hermes is a complex pacakge.
One should change the parameters for libfabric in particular. Currently,
Hermes only installs with TCP/sockets. To get verbs, you will have to know
your machine's architecture. For example, a mellanox network may need to
do ``spack install hermes^libfabric fabrics=tcp,sockets,verbs,mlx``.

# Install

```
spack load hermes arrow
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

# Final environment

```
spack load hermes arrow
module load mega_mmap spark
```