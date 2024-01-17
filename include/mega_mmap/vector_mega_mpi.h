//
// Created by lukemartinlogan on 1/1/24.
//

#ifndef MEGAMMAP_INCLUDE_MEGA_MMAP_VECTOR_MEGA_MPI_H_
#define MEGAMMAP_INCLUDE_MEGA_MMAP_VECTOR_MEGA_MPI_H_

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
#include "hermes/hermes.h"

namespace stdfs = std::filesystem;

namespace mm {

/** Forward declaration */
template<typename T, bool USE_REAL_CACHE=false>
class VectorMegaMpiIterator;

template<typename T>
struct Page {
  std::vector<T> elmts_;
  bool modified_;

  Page() : modified_(false) {}

  template<typename Ar>
  void serialize(Ar &ar) {
    ar & elmts_;
  }
};

/** A wrapper for mmap-based vectors */
template<typename T, bool USE_REAL_CACHE=false>
class VectorMegaMpi {
 public:
  std::vector<Page<T>> data_;
  std::vector<T> back_data_;
  hermes::Bucket bkt_;
  size_t off_ = 0;
  size_t size_ = 0;
  size_t max_size_ = 0;
  size_t elmt_size_ = 0;
  size_t elmts_per_page_ = 0;
  size_t num_pages_ = 0;
  std::string path_;
  std::string dir_;
  int rank_, nprocs_;
  size_t window_size_ = 0;
  size_t emplace_elts_ = 0;

 public:
  VectorMegaMpi() = default;
  ~VectorMegaMpi() = default;

  /** Constructor */
  VectorMegaMpi(const std::string &path, size_t count) {
    Init(path, count);
  }

  /** Copy constructor */
  VectorMegaMpi(const VectorMegaMpi<T> &other) {
    data_ = other.data_;
    off_ = other.off_;
    size_ = other.size_;
    max_size_ = other.max_size_;
    elmt_size_ = other.elmt_size_;
    rank_ = other.rank_;
    nprocs_ = other.nprocs_;
    path_ = other.path_;
    dir_ = other.dir_;
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
    if (data_.size()) {
      return;
    }
    path_ = path;
    dir_ = stdfs::path(path).parent_path();
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs_);
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
    elmts_per_page_ = KILOBYTES(256) / elmt_size;
    if (elmts_per_page_ == 0) {
      elmts_per_page_ = 1;
    }
    num_pages_ = (count + elmts_per_page_ - 1) / elmts_per_page_;
    data_.resize(num_pages_);
    bkt_ = HERMES->GetBucket(path);
    off_ = 0;
    size_ = count;
    max_size_ = count;
    elmt_size_ = elmt_size;
  }

  void Resize(size_t count) {
    size_t num_pages = (count + elmts_per_page_ - 1) / elmts_per_page_;
    if (num_pages > num_pages_) {
      data_.resize(num_pages);
      num_pages_ = num_pages;
    }
  }

  void BoundMemory(size_t window_size) {
    window_size_ = window_size;
  }

  VectorMegaMpi Subset(size_t off, size_t size) {
    VectorMegaMpi mmap(*this);
    mmap.off_ = off;
    mmap.size_ = size;
    return mmap;
  }

  /** Serialize the in-memory cache back to backend */
  void _SerializeToBackend() {
    if constexpr (USE_REAL_CACHE) {
      hermes::Context ctx;
      for (size_t i = 0; i < num_pages_; ++i) {
        Page<T> &page = data_[i];
        if (page.modified_) {
          bkt_.Put<Page<T>>(std::to_string(i), page, ctx);
          page.modified_ = false;
        }
      }
    }
  }

  /** Deserialize in-memory cache from backend */
  void _DeserializeFromBackend() {
  }

  /** Lock a region */
  void Barrier(MPI_Comm comm = MPI_COMM_WORLD) {
    _SerializeToBackend();
    MPI_Barrier(comm);
    _DeserializeFromBackend();
  }

  /** Index operator */
  T& operator[](size_t idx) {
    size_t page_idx = idx / elmts_per_page_;
    size_t page_off = idx % elmts_per_page_;
    Page<T> &page = data_[page_idx];
    if (page.elmts_.size() == 0) {
      hermes::Context ctx;
      page.elmts_.resize(elmts_per_page_);
      std::string page_name = std::to_string(page_idx);
      bkt_.Get<Page<T>>(page_name, page, ctx);
      page.modified_ = true;
    }
    return page.elmts_[page_off];
  }

  /** Addition operator */
  VectorMegaMpi<T> operator+(size_t idx) {
    return Subset(off_ + idx, size_ - idx);
  }

  /** Addition assign operator */
  VectorMegaMpi<T>& operator+=(size_t idx) {
    off_ += idx;
    size_ -= idx;
    return *this;
  }

  /** Subtraction operator */
  VectorMegaMpi<T> operator-(size_t idx) {
    return Subset(off_ - idx, size_ + idx);
  }

  /** Subtraction assign operator */
  VectorMegaMpi<T>& operator-=(size_t idx) {
    off_ -= idx;
    size_ += idx;
    return *this;
  }

  /** check if sorted */
  bool IsSorted(size_t off, size_t size) {
    return std::is_sorted(begin() + off,
                          begin() + off + size);
  }

  /** Sort a subset */
  void Sort(size_t off, size_t size) {
      std::sort(begin() + off,
                begin() + off + size);
  }

  /** Size */
  size_t size() const {
    return size_;
  }

  /** Begin iterator */
  VectorMegaMpiIterator<T, USE_REAL_CACHE> begin() {
    return VectorMegaMpiIterator<T, USE_REAL_CACHE>(this, 0);
  }

  /** End iterator */
  VectorMegaMpiIterator<T, USE_REAL_CACHE> end() {
    return VectorMegaMpiIterator<T, USE_REAL_CACHE>(this, size());
  }

  /** Close region */
  void Close() {}

  /** Destroy region */
  void Destroy() {
    Close();
    bkt_.Destroy();
  }

  /** Emplace back */
  void emplace_back(const T &elmt) {
    back_data_.reserve(MEGABYTES(1));
    back_data_.emplace_back(elmt);
  }

  /** Flush emplace */
  void flush_emplace(MPI_Comm comm, int proc_off, int nprocs) {
    std::string flush_map = hshm::Formatter::format(
        "{}_flusher_{}_{}", path_, proc_off, nprocs);
    VectorMegaMpi<size_t, false> back(
        flush_map,
        nprocs);
    back[rank_ - proc_off] = back_data_.size();
    back.Barrier(comm);
    size_t my_off = 0, new_size = 0;
    for (size_t i = 0; i < rank_ - proc_off; ++i) {
      my_off += back[i];
    }
    for (size_t i = 0; i < nprocs; ++i) {
      new_size += back[i];
    }
    Resize(new_size);
    for (size_t i = 0; i < back_data_.size(); ++i) {
      (*this)[my_off + i] = back_data_[i];
    }
    back.Barrier(comm);
    back.Destroy();
    back_data_.clear();
    size_ = new_size;
  }
};

/** Iterator for VectorMegaMpi */
template<typename T, bool USE_REAL_CACHE>
class VectorMegaMpiIterator {
 public:
  VectorMegaMpi<T, USE_REAL_CACHE> *ptr_;
  size_t idx_;

 public:
  VectorMegaMpiIterator() : idx_(0) {}
  ~VectorMegaMpiIterator() = default;

  /** Constructor */
  VectorMegaMpiIterator(VectorMegaMpi<T, USE_REAL_CACHE> *ptr, size_t idx) {
    ptr_ = ptr;
    idx_ = idx;
  }

  /** Copy constructor */
  VectorMegaMpiIterator(const VectorMegaMpiIterator &other) {
    ptr_ = other.ptr_;
    idx_ = other.idx_;
  }

  /** Assignment operator */
  VectorMegaMpiIterator& operator=(const VectorMegaMpiIterator &other) {
    ptr_ = other.ptr_;
    idx_ = other.idx_;
    return *this;
  }

  /** Equality operator */
  bool operator==(const VectorMegaMpiIterator &other) const {
    return ptr_ == other.ptr_ && idx_ == other.idx_;
  }

  /** Inequality operator */
  bool operator!=(const VectorMegaMpiIterator &other) const {
    return ptr_ != other.ptr_ || idx_ != other.idx_;
  }

  /** Dereference operator */
  T& operator*() {
    return (*ptr_)[idx_];
  }

  /** Prefix increment operator */
  VectorMegaMpiIterator& operator++() {
    ++idx_;
    return *this;
  }

  /** Postfix increment operator */
  VectorMegaMpiIterator operator++(int) {
    VectorMegaMpiIterator tmp(*this);
    ++idx_;
    return tmp;
  }

  /** Prefix decrement operator */
  VectorMegaMpiIterator& operator--() {
    --idx_;
    return *this;
  }

  /** Postfix decrement operator */
  VectorMegaMpiIterator operator--(int) {
    VectorMegaMpiIterator tmp(*this);
    --idx_;
    return tmp;
  }

  /** Addition operator */
  VectorMegaMpiIterator operator+(size_t idx) {
    return VectorMegaMpiIterator(ptr_, idx_ + idx);
  }

  /** Addition assign operator */
  VectorMegaMpiIterator& operator+=(size_t idx) {
    idx_ += idx;
    return *this;
  }

  /** Subtraction operator */
  VectorMegaMpiIterator operator-(size_t idx) {
    return VectorMegaMpiIterator(ptr_, idx_ - idx);
  }

  /** Subtraction assign operator */
  VectorMegaMpiIterator& operator-=(size_t idx) {
    idx_ -= idx;
    return *this;
  }
};

}  // namespace mm

#endif  // MEGAMMAP_INCLUDE_MEGA_MMAP_VECTOR_MEGA_MPI_H_
