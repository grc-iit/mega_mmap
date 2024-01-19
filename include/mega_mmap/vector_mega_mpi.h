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
#include "data_stager/factory/stager_factory.h"
#include "macros.h"

namespace stdfs = std::filesystem;

namespace mm {

/** Forward declaration */
template<typename T, bool IS_COMPLEX_TYPE=false>
class VectorMegaMpiIterator;

template<typename T>
struct Page {
  std::vector<T> elmts_;
  u32 id_;

  Page() = default;

  Page(u32 id) : id_(id) {}
};

/** A wrapper for mmap-based vectors */
template<typename T, bool IS_COMPLEX_TYPE=false>
class VectorMegaMpi {
 public:
  std::unordered_map<size_t, Page<T>> data_;
  std::vector<T> append_data_;
  Page<T> *cur_page_ = nullptr;
  hermes::Bucket bkt_;
  hermes::Bucket append_;
  size_t off_ = 0;
  size_t size_ = 0;
  size_t max_size_ = 0;
  size_t elmt_size_ = 0;
  size_t elmts_per_page_ = 0;
  size_t page_size_ = 0;
  size_t page_mem_ = 0;
  std::string path_;
  std::string dir_;
  int rank_, nprocs_;
  size_t window_size_ = 0;
  size_t cur_memory_ = 0;
  size_t min_page_ = -1, max_page_ = -1;
  bitfield32_t flags_;
  PGAS pgas_;

 public:
  VectorMegaMpi() = default;
  ~VectorMegaMpi() = default;

  /** Constructor */
  VectorMegaMpi(const std::string &path,
                size_t count,
                u32 flags) {
    Init(path, count, flags);
  }

  /** Copy constructor */
  VectorMegaMpi(const VectorMegaMpi<T> &other) {
    data_ = other.data_;
    cur_page_ = other.cur_page_;
    bkt_ = other.bkt_;
    off_ = other.off_;
    size_ = other.size_;
    max_size_ = other.max_size_;
    elmt_size_ = other.elmt_size_;
    elmts_per_page_ = other.elmts_per_page_;
    page_size_ = other.page_size_;
    path_ = other.path_;
    dir_ = other.dir_;
    rank_ = other.rank_;
    nprocs_ = other.nprocs_;
    window_size_ = other.window_size_;
    cur_memory_ = other.cur_memory_;
    min_page_ = other.min_page_;
    max_page_ = other.max_page_;
    flags_ = other.flags_;
    pgas_ = other.pgas_;
  }

  /** Explicit initializer */
  void Init(const std::string &path,
            u32 flags) {
    size_t data_size = 0;
    if (stdfs::exists(path)) {
      data_size = stdfs::file_size(path);
    }
    size_t size = data_size / sizeof(T);
    Init(path, size, flags);
  }

  /** Explicit initializer */
  void Init(const std::string &path, size_t count,
            u32 flags) {
    Init(path, count, sizeof(T), flags);
  }

  /** Explicit initializer */
  void Init(const std::string &path, size_t count, size_t elmt_size,
            u32 flags) {
    TRANSPARENT_HERMES();
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
    if (!IS_COMPLEX_TYPE) {
      elmts_per_page_ = KILOBYTES(256) / elmt_size;
      if (elmts_per_page_ == 0) {
        elmts_per_page_ = 1;
      }
    } else {
      elmts_per_page_ = 1;
    }
    page_size_ = elmts_per_page_ * elmt_size;
    page_mem_ = page_size_ + sizeof(Page<T>);
    hermes::Context ctx;
    if constexpr(!IS_COMPLEX_TYPE) {
      ctx = hermes::data_stager::BinaryFileStager::BuildContext(
          page_size_, elmt_size);
    }
    bkt_ = HERMES->GetBucket(path, ctx);
    off_ = 0;
    size_ = count;
    max_size_ = count;
    elmt_size_ = elmt_size;
    flags_ = bitfield32_t(flags);
    if (flags_.Any(MM_APPEND_ONLY)) {
      size_ = 0;
    }
    pgas_.off_ = 0;
    pgas_.size_ = 0;
    append_data_.reserve(elmts_per_page_);
  }

  void Resize(size_t count) {
  }

  void BoundMemory(size_t window_size) {
    window_size_ = window_size;
  }

  void Pgas(size_t off, size_t count) {
    pgas_.Init(off * elmt_size_,
               count * elmt_size_,
               page_size_);
  }

  VectorMegaMpi Subset(size_t off, size_t size) {
    VectorMegaMpi mmap(*this);
    mmap.off_ = off;
    mmap.size_ = size;
    return mmap;
  }

  /** Flush data to backend */
  void _Flush() {
    if (min_page_ == -1) {
      return;
    }
    if (flags_.Any(MM_WRITE_ONLY | MM_READ_WRITE) && pgas_.size_ > 0) {
      hermes::Context ctx;
      for (size_t page_idx = min_page_; page_idx <= max_page_; ++page_idx) {
        auto it = data_.find(page_idx);
        if (it == data_.end()) {
          continue;
        }
        Page<T> &page = it->second;
        size_t page_off, page_size;
        if constexpr (!IS_COMPLEX_TYPE) {
          pgas_.GetModBounds(page_idx, page_off, page_size);
          std::string page_name =
              hermes::adapter::BlobPlacement::CreateBlobName(page_idx).str();
          ctx.flags_.SetBits(HERMES_SHOULD_STAGE);
          hermes::Blob blob((char*)page.elmts_.data() + page_off,
                            page_size);
          bkt_.PartialPut(page_name, blob, page_off, ctx);
        } else {
          std::string page_name =
              hermes::adapter::BlobPlacement::CreateBlobName(page_idx).str();
          bkt_.Put<T>(page_name, page.elmts_[0], ctx);
        }
      }
    }
  }

  /** Serialize the in-memory cache back to backend */
  void _Evict() {
    _Flush();
    if (min_page_ == -1) {
      return;
    }
    for (size_t page_idx = min_page_; page_idx <= max_page_; ++page_idx) {
      if (cur_memory_ - 2 * page_mem_ < window_size_) {
        break;
      }
      auto it = data_.find(page_idx);
      if (it == data_.end()) {
        continue;
      }
      if (cur_page_ == &it->second) {
        cur_page_ = nullptr;
      }
      data_.erase(it);
      cur_memory_ -= elmts_per_page_ * elmt_size_;
    }
  }

  /** Lock a region */
  void Barrier(u32 flags = 0, MPI_Comm comm = MPI_COMM_WORLD) {
    _Flush();
    MPI_Barrier(comm);
    flags_.SetBits(flags);
  }

  /** Hint access pattern */
  void Hint(u32 flags) {
    flags_.SetBits(flags);
  }

  /** Induct a page */
  Page<T>* _Fault(size_t page_idx) {
    if (window_size_ > page_mem_ && cur_memory_ > window_size_ - page_mem_) {
      _Evict();
    }
    hermes::Context ctx;
    data_.emplace(page_idx, Page<T>(page_idx));
    Page<T> &page = data_[page_idx];
    page.elmts_.resize(elmts_per_page_);
    std::string page_name =
        hermes::adapter::BlobPlacement::CreateBlobName(page_idx).str();
    if (flags_.Any(MM_READ_ONLY | MM_READ_WRITE)) {
      if constexpr (!IS_COMPLEX_TYPE) {
        ctx.flags_.SetBits(HERMES_SHOULD_STAGE);
        hermes::Blob blob((char*)page.elmts_.data(), page_size_);
        bkt_.Get(page_name, blob, ctx);
      } else {
        bkt_.Get<T>(page_name, page.elmts_[0], ctx);
      }
    }
    cur_memory_ += page_mem_;
    if (min_page_ == -1) {
      min_page_ = page_idx;
      max_page_ = page_idx;
    }
    if (page_idx < min_page_) {
      min_page_ = page_idx;
    }
    if (page_idx > max_page_) {
      max_page_ = page_idx;
    }
    return &page;
  }

  /** Index operator */
  T& operator[](size_t idx) {
    size_t page_idx = idx / elmts_per_page_;
    size_t page_off = idx % elmts_per_page_;
    Page<T> *page_ptr;
    if (cur_page_ && cur_page_->id_ == page_idx) {
      page_ptr = cur_page_;
    } else {
      auto it = data_.find(page_idx);
      if (it == data_.end()) {
        page_ptr = _Fault(page_idx);
      } else {
        page_ptr = &it->second;
      }
    }
    Page<T> &page = *page_ptr;
    cur_page_ = page_ptr;
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
  VectorMegaMpiIterator<T, IS_COMPLEX_TYPE> begin() {
    return VectorMegaMpiIterator<T, IS_COMPLEX_TYPE>(this, 0);
  }

  /** End iterator */
  VectorMegaMpiIterator<T, IS_COMPLEX_TYPE> end() {
    return VectorMegaMpiIterator<T, IS_COMPLEX_TYPE>(this, size());
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
    append_data_.emplace_back(elmt);
    if (append_data_.size() == elmts_per_page_) {
      _flush_emplace();
    }
    ++size_;
  }

  /** Flush append buffer */
  void _flush_emplace() {
    if constexpr(!IS_COMPLEX_TYPE) {
      hermes::Context ctx;
      ctx.flags_.SetBits(HERMES_SHOULD_STAGE);
      hermes::Blob blob((char*)append_data_.data(),
                        append_data_.size() * elmt_size_);
      bkt_.Append(blob, page_size_, ctx);
      append_data_.clear();
    } else {
      throw std::runtime_error("Complex types not supported");
    }
  }

  /** Flush emplace */
  void flush_emplace(MPI_Comm comm, int proc_off, int nprocs) {
    if (append_data_.size()) {
      _flush_emplace();
    }
    Barrier(comm);
    size_t new_size = bkt_.GetSize();
    new_size = new_size / elmt_size_;
    Resize(new_size);
    size_ = new_size;
  }
};

/** Iterator for VectorMegaMpi */
template<typename T, bool IS_COMPLEX_TYPE>
class VectorMegaMpiIterator {
 public:
  VectorMegaMpi<T, IS_COMPLEX_TYPE> *ptr_;
  size_t idx_;

 public:
  VectorMegaMpiIterator() : idx_(0) {}
  ~VectorMegaMpiIterator() = default;

  /** Constructor */
  VectorMegaMpiIterator(VectorMegaMpi<T, IS_COMPLEX_TYPE> *ptr, size_t idx) {
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
