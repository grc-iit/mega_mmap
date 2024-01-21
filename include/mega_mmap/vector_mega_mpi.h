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
  LPointer<hrunpq::TypedPushTask<hermes::GetBlobTask>> async_;
  std::vector<T> elmts_;
  u32 id_;

  Page() = default;

  Page(u32 id) : id_(id) {
    async_.ptr_ = nullptr;
  }
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
  size_t size_ = 0;
  size_t max_size_ = 0;
  size_t elmt_size_ = 0;
  size_t elmts_per_page_ = 0;
  size_t page_size_ = 0;
  size_t page_mem_ = 0;
  std::string path_;
  size_t window_size_ = 0;
  size_t cur_memory_ = 0;
  size_t min_page_ = -1, max_page_ = -1;
  bitfield32_t flags_;
  PGAS pgas_;
  PGAS pgas_elmt_;

 public:
  VectorMegaMpi() = default;
  ~VectorMegaMpi() = default;

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
  void Init(const std::string &path, size_t count,
            size_t elmt_size, u32 flags) {
    if (data_.size()) {
      return;
    }
    path_ = path;
    size_ = count;
    max_size_ = count;
    elmt_size_ = elmt_size;
    flags_ = bitfield32_t(flags);
    if (flags_.Any(MM_APPEND_ONLY)) {
      size_ = 0;
    }
    pgas_.off_ = 0;
    pgas_.size_ = 0;
    pgas_elmt_.off_ = 0;
    pgas_elmt_.size_ = 0;
    SetPageSize(MM_PAGE_SIZE);
  }

  /** Resize this DSM */
  void Resize(size_t count) {
    size_ = count;
    max_size_ = count;
  }

  /** Ensure this DSM doesn't exceed DRAM capacity */
  void BoundMemory(size_t window_size) {
    window_size_ = window_size;
  }

  /** Set the exact size in bytes of a DSM page */
  void SetPageSize(size_t page_size) {
    if (!IS_COMPLEX_TYPE) {
      elmts_per_page_ = page_size / elmt_size_;
      if (elmts_per_page_ == 0) {
        elmts_per_page_ = 1;
      }
    } else {
      elmts_per_page_ = 1;
    }
    page_size_ = elmts_per_page_ * elmt_size_;
    page_mem_ = page_size_ + sizeof(Page<T>);
  }

  /** Set the expected number of elements stored in a DSM page */
  void SetElmtsPerPage(size_t count) {
    elmts_per_page_ = count;
    page_size_ = elmts_per_page_ * elmt_size_;
    page_mem_ = page_size_ + sizeof(Page<T>);
  }

  /** Split DSM among processes */
  void Pgas(size_t off, size_t count, size_t count_per_page = 0) {
    if (count_per_page != 0) {
      SetElmtsPerPage(count_per_page);
    }
    pgas_.Init(off * elmt_size_,
               count * elmt_size_,
               page_size_);
    pgas_elmt_.Init(off,
                    count,
                    elmts_per_page_);
  }

  /** Evenly split DSM among processes */
  void EvenPgas(int rank, int nprocs, size_t max_count,
                size_t count_per_page = 0) {
    Bounds bounds(rank, nprocs, max_count);
    Pgas(bounds.off_, bounds.size_, count_per_page);
  }

  /** Allocate the DSM */
  void Allocate() {
    hermes::Context ctx;
    if constexpr(!IS_COMPLEX_TYPE) {
      bitfield32_t flags;
      if (!flags_.Any(MM_STAGE_READ_FROM_BACKEND)) {
        flags.SetBits(HERMES_STAGE_NO_READ);
      }
      ctx = hermes::data_stager::BinaryFileStager::BuildContext(
          page_size_, flags.bits_, elmt_size_);
    }
    bkt_ = HERMES->GetBucket(path_, ctx);
    append_data_.reserve(elmts_per_page_);
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
  void _Evict(size_t bytes = 0) {
    _Flush();
    if (min_page_ == -1) {
      return;
    }
    if (bytes == 0) {
      bytes = 2 * page_mem_;
    }
    for (size_t page_idx = min_page_; page_idx <= max_page_; ++page_idx) {
      if (cur_memory_ + bytes < window_size_) {
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
      cur_memory_ -= page_size_;
    }
  }

  /** Lock a region */
  void Barrier(u32 flags, MPI_Comm comm) {
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
    if (page_idx > size_ / elmts_per_page_) {
      int rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      HELOG(kFatal, "{}: Cannot seek past size of {}: {} / {} ",
            rank, path_, page_idx, size_ / elmts_per_page_);
    }
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
    if (min_page_ == -1 || max_page_ == -1) {
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

  /** Create an async fault */
  void _AsyncFaultBegin(size_t page_idx) {
    if (page_idx > size_ / elmts_per_page_) {
      int rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      HELOG(kFatal, "{}: Cannot seek past size of {}: {} / {} ",
            rank, path_, page_idx, size_ / elmts_per_page_);
    }
    hermes::Context ctx;
    data_.emplace(page_idx, Page<T>(page_idx));
    Page<T> &page = data_[page_idx];
    page.elmts_.resize(elmts_per_page_);
    std::string page_name =
        hermes::adapter::BlobPlacement::CreateBlobName(page_idx).str();
    if constexpr (!IS_COMPLEX_TYPE) {
      ctx.flags_.SetBits(HERMES_SHOULD_STAGE);
      hermes::Blob blob((char*)page.elmts_.data(), page_size_);
      page.async_ = bkt_.AsyncGet(page_name, blob, ctx);
    } else {
      throw std::runtime_error(
          "Async not supported for serialized types at this time.");
    }
    cur_memory_ += page_mem_;
    return &page;
  }

  /** Finish inducting a page */
  void _AsyncFaultEnd(Page<T> &page) {
    if (window_size_ > page_mem_ && cur_memory_ > window_size_ - page_mem_) {
      _Evict();
    }
    size_t page_idx = page.id_;
    page.async_->Wait();
    HRUN_CLIENT->DelTask(page.async_);
    page.async_.ptr_ = nullptr;
    if (min_page_ == -1 || max_page_ == -1) {
      min_page_ = page_idx;
      max_page_ = page_idx;
    }
    if (page_idx < min_page_) {
      min_page_ = page_idx;
    }
    if (page_idx > max_page_) {
      max_page_ = page_idx;
    }
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
      } else if (page_ptr->async_.ptr_ == nullptr) {
        page_ptr = &it->second;
      } else {
        page_ptr = &it->second;
        _AsyncFaultEnd(*page_ptr);
      }
    }
    Page<T> &page = *page_ptr;
    cur_page_ = page_ptr;
    return page.elmts_[page_off];
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

  /** Size of PGAS region */
  size_t local_size() const {
    return pgas_elmt_.size_;
  }

  /** Offset of PGAS region */
  size_t local_off() const {
    return pgas_elmt_.off_;
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
    HRUN_ADMIN->FlushRoot(DomainId::GetLocal());
    MPI_Barrier(comm);
    size_t new_size = bkt_.GetSize();
    new_size = new_size / elmt_size_;
    Resize(new_size);
    size_ = new_size;
    Hint(MM_READ_ONLY);
  }

  void Prefetch(const std::vector<size_t> &next_pages) {
    _Evict(next_pages.size() * page_size_);
    for (size_t page_idx : next_pages) {
      _AsyncFaultBegin(page_idx);
    }
  }
};

}  // namespace mm

#endif  // MEGAMMAP_INCLUDE_MEGA_MMAP_VECTOR_MEGA_MPI_H_
