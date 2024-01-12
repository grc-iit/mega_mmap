//
// Created by lukemartinlogan on 1/1/24.
//

#ifndef MEGAMMAP_INCLUDE_MEGA_MMAP_VECTOR_MMAP_MPI_H_
#define MEGAMMAP_INCLUDE_MEGA_MMAP_VECTOR_MMAP_MPI_H_

#include <string>
#include <mpi.h>
#include <sys/mman.h>
#include <fcntl.h>
#include "hermes_shm/util/logging.h"
#include "hermes_shm/util/config_parse.h"
#include "hermes_shm/data_structures/data_structure.h"
#include <filesystem>
#include <cereal/types/memory.hpp>
#include <sys/resource.h>


namespace stdfs = std::filesystem;

namespace mm {

template<typename T>
struct VectorMmapEntry {
  bool modified_ = false;
  T data_;
};

/** Forward declaration */
template<typename T, bool USE_REAL_CACHE=false>
class VectorMmapMpiIterator;

/** A wrapper for mmap-based vectors */
template<typename T, bool USE_REAL_CACHE=false>
class VectorMmapMpi {
 public:
  T* data_ = nullptr;
  size_t size_ = 0;
  size_t elmt_size_ = 0;
  std::vector<VectorMmapEntry<T>> real_data_;
  std::vector<T> back_data_;
  std::string path_;
  std::string dir_;
  int rank_;

 public:
  VectorMmapMpi() = default;
  ~VectorMmapMpi() = default;

  /** Constructor */
  VectorMmapMpi(const std::string &path, size_t count) {
    Init(path, count);
  }

  /** Construct from pointer */
  VectorMmapMpi(T *data, size_t size) {
    data_ = data;
    size_ = size;
    elmt_size_ = sizeof(T);
  }

  /** Copy constructor */
  VectorMmapMpi(const VectorMmapMpi<T> &other) {
    data_ = other.data_;
    size_ = other.size_;
    elmt_size_ = other.elmt_size_;
  }

  /** Explicit initializer */
  void Init(const std::string &path) {
    size_t data_size = stdfs::file_size(path);
    size_t size = data_size / sizeof(T);
    Init(path, size);
  }

  /** Explicit initializer */
  void Init(const std::string &path, size_t count) {
    Init(path, count, sizeof(T));
  }

  /** Explicit initializer */
  void Init(const std::string &path, size_t count, size_t elmt_size) {
    if (data_ != nullptr) {
      return;
    }
    path_ = path;
    dir_ = stdfs::path(path).parent_path();
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    int fd = open64(path.c_str(), O_RDWR | O_CREAT, 0666);
    if (fd < 0) {
      HELOG(kFatal, "Failed to open file {}: {}",
            path.c_str(), strerror(errno));
    }
    size_t file_size = count * elmt_size;
    if (ftruncate64(fd, (ssize_t)(file_size)) < 0) {
      HELOG(kFatal, "Failed to truncate file {}: {}",
              path.c_str(), strerror(errno));
    }
    data_ = (T*)mmap64(NULL, file_size, PROT_READ | PROT_WRITE,
                       MAP_SHARED, fd, 0);
    if (data_ == MAP_FAILED || data_ == nullptr) {
      data_ = nullptr;
      HELOG(kFatal, "Failed to mmap file {}: {}",
            path.c_str(), strerror(errno));
    }
    if constexpr (USE_REAL_CACHE) {
      real_data_.resize(count);
    }
    size_ = count;
    elmt_size_ = elmt_size;
    MaximizeFds();
  }

  void MaximizeFds() {
    struct rlimit rlim;
    if (getrlimit(RLIMIT_NOFILE, &rlim) == 0) {
      rlim.rlim_cur = rlim.rlim_max;
    }
    if (setrlimit(RLIMIT_NOFILE, &rlim) != 0) {
      HILOG(kInfo, "Cannot set maximum fd limit to {}", rlim.rlim_max);
    }
  }

  void PrintNumOpenFds() {
    struct rlimit rlim;
    if (getrlimit(RLIMIT_NOFILE, &rlim) == 0) {
      HILOG(kInfo, "Number of open files: {} / {}",
            CountFds(), (long)rlim.rlim_cur)
    }
  }

  int CountFds() {
    int count = 0;

    // Iterate through file descriptors
    for (int fd = 0; fd < getdtablesize(); ++fd) {
      if (fcntl(fd, F_GETFD) != -1) {
        // Descriptor is valid
        count++;
      }
    }

    return count;
  }

  /** Serialize the in-memory cache back to backend */
  void _SerializeToBackend() {
    if constexpr (USE_REAL_CACHE) {
      for (size_t i = 0; i < size_; ++i) {
        const VectorMmapEntry<T> &elmt = real_data_[i];
        if (elmt.modified_) {
          std::stringstream ss;
          cereal::BinaryOutputArchive ar(ss);
          ar(elmt.data_);
          std::string srl = ss.str();
          size_t off = i * elmt_size_;
          if (srl.size() > elmt_size_) {
            HELOG(kFatal, "Serialization size {} is larger than element size {}",
                  srl.size(), elmt_size_);
          }
          memcpy((char*)data_ + off, srl.c_str(), srl.size());
        }
      }
    }
  }

  /** Deserialize in-memory cache from backend */
  void _DeserializeFromBackend() {
    if constexpr (USE_REAL_CACHE) {
      for (size_t i = 0; i < size_; ++i) {
        char *elmt_data = (char*)data_ + i * elmt_size_;
        VectorMmapEntry<T> &elmt = real_data_[i];
        std::stringstream ss(std::string(elmt_data, elmt_size_));
        cereal::BinaryInputArchive ar(ss);
        ar(elmt.data_);
        elmt.modified_ = false;
      }
    }
  }

  /** Lock a region */
  void Barrier() {
    _SerializeToBackend();
    MPI_Barrier(MPI_COMM_WORLD);
    _DeserializeFromBackend();
  }

  /** Lock a region for subset of nodes */
  void Barrier(int proc_off, int nprocs) {
    MPI_Comm subset_comm;
    bool color = proc_off <= rank_ && rank_ < proc_off + nprocs;
    MPI_Comm_split(MPI_COMM_WORLD, color, rank_ - proc_off, &subset_comm);
    _SerializeToBackend();
    MPI_Barrier(subset_comm);
    _DeserializeFromBackend();
    MPI_Comm_free(&subset_comm);
  }

  /** Index operator */
  T& operator[](int idx) {
    if constexpr (!USE_REAL_CACHE) {
      return data_[idx];
    } else {
      VectorMmapEntry<T> &entry = real_data_[idx];
      entry.modified_ = true;
      return entry.data_;
    }
  }

  /** Addition operator */
  VectorMmapMpi<T> operator+(int idx) {
    return VectorMmapMpi<T>(data_ + idx, size_ - idx);
  }

  /** Addition assign operator */
  VectorMmapMpi<T>& operator+=(int idx) {
    data_ += idx;
    size_ -= idx;
    return *this;
  }

  /** Subtraction operator */
  VectorMmapMpi<T> operator-(int idx) {
    return VectorMmapMpi<T>(data_ - idx, size_ + idx);
  }

  /** Subtraction assign operator */
  VectorMmapMpi<T>& operator-=(int idx) {
    data_ -= idx;
    size_ += idx;
    return *this;
  }

  /** Memcpy operator */
  void Memcpy(T *src, size_t size, size_t off = 0) {
    memcpy(data_ + off, src, size);
  }

  /** Memcpy operator (II) */
  void Memcpy(VectorMmapMpi<T> &src, size_t size, size_t off = 0) {
    memcpy(data_ + off, src.data_, size);
  }

  /** check if sorted */
  bool IsSorted(size_t off, size_t size) {
    return std::is_sorted(data_ + off, data_ + off + size);
  }

  /** Sort a subset */
  void Sort(size_t off, size_t size) {
      std::sort(data_ + off, data_ + off + size);
  }

  /** Size */
  size_t size() const {
    return size_;
  }

  /** Begin iterator */
  VectorMmapMpiIterator<T, USE_REAL_CACHE> begin() {
    return VectorMmapMpiIterator<T, USE_REAL_CACHE>(this, 0);
  }

  /** End iterator */
  VectorMmapMpiIterator<T, USE_REAL_CACHE> end() {
    return VectorMmapMpiIterator<T, USE_REAL_CACHE>(this, size());
  }

  /** Close region */
  void Close() {
    munmap(data_, size_ * sizeof(T));
  }

  /** Destroy region */
  void Destroy() {
    Close();
    remove(path_.c_str());
  }

  /** Emplace back */
  void emplace_back(const T &elmt) {
    back_data_.reserve(MEGABYTES(1));
    back_data_.emplace_back(elmt);
  }

  /** Flush emplace */
  void flush_emplace(int proc_off, int nprocs) {
    VectorMmapMpi<size_t, false> back(
        hshm::Formatter::format("{}_flusher_{}_{}", path_, proc_off, nprocs),
        nprocs);
    back[rank_ - proc_off] = back_data_.size();
    back.Barrier(proc_off, nprocs);
    size_t my_off = 0, new_size = 0;
    for (size_t i = 0; i < rank_ - proc_off; ++i) {
      my_off += back[i];
    }
    for (size_t i = 0; i < nprocs; ++i) {
      new_size += back[i];
    }
    for (size_t i = 0; i < back_data_.size(); ++i) {
      data_[my_off + i] = back_data_[i];
    }
    back.Barrier(proc_off, nprocs);
    back.Destroy();
    back_data_.clear();
    size_ = new_size;
  }
};

/** Iterator for VectorMmapMpi */
template<typename T, bool USE_REAL_CACHE>
class VectorMmapMpiIterator {
 public:
  VectorMmapMpi<T, USE_REAL_CACHE> *ptr_;
  size_t idx_;

 public:
  VectorMmapMpiIterator() : idx_(0) {}
  ~VectorMmapMpiIterator() = default;

  /** Constructor */
  VectorMmapMpiIterator(VectorMmapMpi<T, USE_REAL_CACHE> *ptr, size_t idx) {
    ptr_ = ptr;
    idx_ = idx;
  }

  /** Copy constructor */
  VectorMmapMpiIterator(const VectorMmapMpiIterator &other) {
    ptr_ = other.ptr_;
    idx_ = other.idx_;
  }

  /** Assignment operator */
  VectorMmapMpiIterator& operator=(const VectorMmapMpiIterator &other) {
    ptr_ = other.ptr_;
    idx_ = other.idx_;
    return *this;
  }

  /** Equality operator */
  bool operator==(const VectorMmapMpiIterator &other) const {
    return ptr_ == other.ptr_ && idx_ == other.idx_;
  }

  /** Inequality operator */
  bool operator!=(const VectorMmapMpiIterator &other) const {
    return ptr_ != other.ptr_ || idx_ != other.idx_;
  }

  /** Dereference operator */
  T& operator*() {
    return (*ptr_)[idx_];
  }

  /** Prefix increment operator */
  VectorMmapMpiIterator& operator++() {
    ++idx_;
    return *this;
  }

  /** Postfix increment operator */
  VectorMmapMpiIterator operator++(int) {
    VectorMmapMpiIterator tmp(*this);
    ++idx_;
    return tmp;
  }

  /** Prefix decrement operator */
  VectorMmapMpiIterator& operator--() {
    --idx_;
    return *this;
  }

  /** Postfix decrement operator */
  VectorMmapMpiIterator operator--(int) {
    VectorMmapMpiIterator tmp(*this);
    --idx_;
    return tmp;
  }

  /** Addition operator */
  VectorMmapMpiIterator operator+(int idx) {
    return VectorMmapMpiIterator(ptr_, idx_ + idx);
  }

  /** Addition assign operator */
  VectorMmapMpiIterator& operator+=(int idx) {
    idx_ += idx;
    return *this;
  }

  /** Subtraction operator */
  VectorMmapMpiIterator operator-(int idx) {
    return VectorMmapMpiIterator(ptr_, idx_ - idx);
  }

  /** Subtraction assign operator */
  VectorMmapMpiIterator& operator-=(int idx) {
    idx_ -= idx;
    return *this;
  }
};

}  // namespace mm

#endif  // MEGAMMAP_INCLUDE_MEGA_MMAP_VECTOR_MMAP_MPI_H_
