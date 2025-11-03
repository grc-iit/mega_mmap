# MegaMmap: Blurring the Boundary Between Memory and Storage

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B17)
[![HPC](https://img.shields.io/badge/Domain-HPC-green.svg)](https://en.wikipedia.org/wiki/High-performance_computing)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.1109/SC41406.2024.00114)

**MegaMmap** is a software distributed shared memory (DSM) system that eliminates the traditional boundary between memory and storage for data-intensive HPC workloads. By providing a unified, byte-addressable interface that spans DRAM, NVMe, SSD, and HDD tiers, MegaMmap enables applications to work with datasets larger than memory capacity while maintaining competitive performance.

## Key Features

- **Infinite Memory Abstraction**: Present massive datasets as if they were in main memory
- **Intelligent Tiering**: Automatically manage data across DRAM, NVMe, SSD, and HDD tiers
- **Transactional Memory API**: Declare access patterns to optimize prefetching and coherence
- **Intent-Aware Coherence**: Reduce communication overhead with workload-specific optimizations
- **Persistent Integration**: Transparently stage data to/from HDF5, Parquet, and other formats
- **HPC-Optimized**: Designed for scientific simulations, machine learning, and data analytics

## Performance Highlights

- **2.6x DRAM Reduction**: Maintain performance with 60% less memory usage
- **2x Faster than Spark**: Outperform cloud-based solutions for memory-intensive workloads
- **Unbounded Dataset Size**: Process datasets 2x larger than available memory
- **45% Code Reduction**: Simpler development compared to traditional out-of-core approaches

## Architecture

MegaMmap consists of several key components:

- **Private Cache (pcache)**: Per-process DRAM cache for low-latency access
- **Shared Cache (scache)**: Distributed, tiered cache across all processes
- **Data Organizer**: Intelligent placement based on access patterns and scores
- **Prefetcher**: Overlaps computation with data movement
- **Transaction System**: Declares intent for optimized coherence

## Quick Start

### Prerequisites

- C++17 compliant compiler (GCC 9.4.1+)
- MPI implementation (MPICH 3.4.3+ recommended)
- [Hermes](https://github.com/HDFGroup/hermes) buffering system
- CMake 3.12+

### Installation

```bash
# Clone the repository
git clone https://github.com/grc-iit/mega_mmap.git
cd mega_mmap

# Install dependencies (requires Spack)
./deps.sh

# Build MegaMmap
mkdir build && cd build
cmake ../ -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$(scspkg pkg root mega_mmap)
make -j8

# Load environment
module load mega_mmap
```

### Basic Usage

```cpp
#include <mega_mmap/vector.h>

void KMeansInertia(std::vector<Point3D> &ks) {
    int rank = mpi::get_rank();
    int nprocs = mpi::get_comm_size();
    
    // Create a shared vector from a Parquet file
    mm::Vector<Point3D> pts("/points.parquet");
    pts.BoundMemory(MEGABYTES(1));  // Limit to 1MB DRAM
    pts.Pgas(rank, nprocs);          // Partition across processes
    
    // Begin read-only transaction
    auto tx = pts.SeqTxBegin(pts.local_off(), pts.local_size(), MM_READ_ONLY);
    
    float distance = 0;
    for (Point3D p : tx) {
        distance += pow(NearestCentroid(p, ks), 2);
    }
    
    pts.TxEnd();
}
```
### MegaMmap AI Guidelines (if visiting using a coding Agent)

## Build & Test Commands
- **Build**: `mkdir build && cd build && cmake ../ -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$(scspkg pkg root mega_mmap) && make -j8`
- **Debug Build**: Use `-DCMAKE_BUILD_TYPE=Debug` for debug symbols
- **Single Test**: `jarvis pipeline run yaml test/unit/pipelines/{test_name}.yaml` (e.g., `mm_kmeans_mega.yaml`)
- **All Tests**: Tests defined in `test/unit/CMakeLists.txt` using `jarvis_test()` function

## Code Style Guidelines
- **Language**: C++17 standard
- **Namespaces**: Use `mm` namespace for core library code
- **Naming**: 
  - Classes: PascalCase (e.g., `Vector`, `Bounds`)
  - Variables: snake_case with trailing underscore (e.g., `window_size_`, `elmts_per_page_`)
  - Functions: PascalCase for public methods
  - Constants: UPPER_CASE with prefix (e.g., `MM_READ_ONLY`, `MM_PAGE_SIZE`)
- **Headers**: Include guards format: `MEGAMMAP_INCLUDE_{PATH}_{FILE}_H_`
- **File Organization**: 
  - Headers in `include/mega_mmap/`
  - Implementations in `benchmark/` for executables
  - Tests in `test/unit/`
- **Dependencies**: Uses Hermes, MPI, Arrow, Parquet, YAML-CPP, Catch2, OpenMP
- **Macros**: Use `BIT_OPT(u32, n)` for bit flags, `KILOBYTES()`, `MEGABYTES()` for sizes
- **Error Handling**: Hermes logging via `hermes_shm/util/logging.h`

## 📈 Supported Workloads

MegaMmap has been validated on production HPC applications:

- **Machine Learning**: KMeans clustering, Random Forest classification
- **Scientific Simulation**: Gray-Scott reaction-diffusion models
- **Data Analytics**: DBSCAN clustering on cosmological datasets
- **Signal Processing**: Gadget2 cosmological simulation conversion

## 🧪 Running Benchmarks

```bash
# Run single benchmark
jarvis pipeline run yaml test/unit/pipelines/mm_kmeans_mega.yaml

# Run all benchmarks
cd test/unit && make -j8
```

## 📚 Documentation

- [Paper (SC24)](megammap.pdf) - Complete research paper with evaluation
- [API Reference](include/mega_mmap/) - API documentation
- [Examples](benchmark/) - Application examples
- [Configuration Guide](docs/configuration.md) - Configuration parameters


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- National Science Foundation (NSF) Grants CSSI-2104013 and Core-2313154
- U.S. Department of Energy (DOE) Contract DE-SC0024593
- Chameleon Cloud Testbed for development environment

## 📞 Contact

For questions, support, or collaborations:

- **Luke Logan** (Author & Maintainer): llogan@illinoistech.edu
- **Anthony Kougkas**: akougkas@illinoistech.edu  
- **Xian-He Sun**: sun@illinoistech.edu

[Gnosis Research Center](https://grc.iit.edu) - Illinois Institute of Technology

---

**Citation**: If you use MegaMmap in your research, please cite our [SC24 paper](https://doi.org/10.1109/SC41406.2024.00114).

```bibtex
@inproceedings{logan2024megammap,
  title={MegaMmap: Blurring the Boundary Between Memory and Storage for Data-Intensive Workloads},
  author={Logan, Luke and Kougkas, Anthony and Sun, Xian-He},
  booktitle={Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC)},
  year={2024}
}
```
