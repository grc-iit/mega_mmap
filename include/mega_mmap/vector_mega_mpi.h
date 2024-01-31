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

template<typename T>
struct Page {
  std::vector<T> elmts_;
  u32 id_;

  Page() = default;

  Page(u32 id) : id_(id) {}
};

struct PageLog {
  size_t page_idx_;
  size_t mod_start_;
  size_t mod_size_;

  // Default constructor
  PageLog() = default;

  // Constructor
  PageLog(size_t page_idx, size_t mod_start, size_t mod_size)
      : page_idx_(page_idx), mod_start_(mod_start), mod_size_(mod_size) {}

  // Copy constructor
  PageLog(const PageLog &other) = default;
};

class SequentialIterator {
 public:
  size_t off_, size_;
  size_t elmts_per_page_;

 public:
  explicit SequentialIterator(size_t off, size_t size, size_t elmts_per_page)
      : off_(off), size_(size), elmts_per_page_(elmts_per_page) {}

  size_t get(size_t iter_cur) {
    return iter_cur + off_;
  }

  std::vector<PageLog> next_pages(size_t iter_cur, int max_n,
                                  size_t &iter_max) {
    std::vector<PageLog> pages;
    pages.reserve(max_n);
    size_t off = iter_cur + off_;
    if (iter_cur + elmts_per_page_ > size_) {
      PageLog only;
      only.page_idx_ = off / elmts_per_page_;
      only.mod_start_ = off % elmts_per_page_;
      only.mod_size_ = size_ - iter_cur;
      pages.emplace_back(only);
      iter_max += pages.back().mod_size_;
      return pages;
    }

    // Get first page
    PageLog first;
    first.page_idx_ = off / elmts_per_page_;
    first.mod_start_ = off % elmts_per_page_;
    first.mod_size_ = elmts_per_page_ - first.mod_start_;
    size_t page_idx = first.page_idx_ + 1;
    pages.emplace_back(first);

    // Get maximum number of pages
    size_t max_count = max_n * elmts_per_page_;
    size_t last_idx = off + max_count;
    if (max_count > size_) {
      last_idx = off + size_ - iter_cur;
      max_n = (size_ - iter_cur) / elmts_per_page_;
    }

    // Get middle pages
    for (int i = 1; i < max_n - 1; ++i) {
      pages.emplace_back(PageLog(page_idx++, 0, elmts_per_page_));
      iter_max += pages.back().mod_size_;
    }

    // Get last page
    PageLog last;
    last.page_idx_ = last_idx / elmts_per_page_;
    last.mod_start_ = 0;
    last.mod_size_ = last_idx % elmts_per_page_;
    if (last.mod_size_ == 0) {
      last.mod_size_ = elmts_per_page_;
    }
    pages.emplace_back(last);
    iter_max += pages.back().mod_size_;
    return pages;
  }
};

class RandomIterator {
 public:
  size_t seed_, size_;
  size_t elmts_per_page_;
  hshm::UniformDistribution dist_;
  size_t page_ = 0;

 public:
  explicit RandomIterator(size_t seed,
                          size_t rand_left, size_t rand_size,
                          size_t size, size_t elmts_per_page)
      : seed_(seed), size_(size), elmts_per_page_(elmts_per_page) {
    dist_.Seed(seed);
    size_t page_left = rand_left / elmts_per_page_;
    size_t page_right = (rand_left + rand_size) / elmts_per_page_;
    if (page_right == page_left) {
      dist_.Shape(page_left, page_right);
    } else {
      dist_.Shape(page_left, page_right - 1);
    }
  }

  size_t get(size_t iter_cur) {
    if (iter_cur % elmts_per_page_ == 0) {
      page_ = dist_.GetSize() * elmts_per_page_;
      return page_;
    } else {
      return page_ + iter_cur % elmts_per_page_;
    }
  }

  std::vector<PageLog> next_pages(size_t iter_cur, int max_n,
                                  size_t &iter_max) {
    hshm::UniformDistribution dist = dist_;
    std::vector<PageLog> pages;
    size_t max_count = max_n * elmts_per_page_;
    if (iter_cur + max_count > size_) {
      max_n = (size_ - iter_cur) / elmts_per_page_;
    }
    for (int i = 0; i < max_n; ++i) {
      size_t page_idx = dist.GetSize();
      pages.emplace_back(PageLog(page_idx, 0, elmts_per_page_));
      iter_max += pages.back().mod_size_;
    }
    return pages;
  }
};

template<typename AssignT>
class IndexIterator {
 public:
  AssignT &assign_;
  size_t size_, elmts_per_page_;

 public:
  IndexIterator(size_t size, size_t elmts_per_page,
                AssignT &assign)
      : size_(size), elmts_per_page_(elmts_per_page), assign_(assign) {}

  size_t get(size_t iter_cur) {
    return assign_[iter_cur];
  }

  std::vector<PageLog> next_pages(size_t iter_cur, int max_n,
                                  size_t &iter_max) {
    std::vector<PageLog> pages;
    size_t max_count = max_n * elmts_per_page_;
    if (iter_cur + max_count > size_) {
      max_count = size_ - iter_cur;
    }

    for (size_t i = 0; i < max_count; ++i) {
      if (iter_max >= assign_.size() || pages.size() >= max_n) {
        break;
      }
      size_t page_idx = assign_[iter_max] / elmts_per_page_;
      if (pages.size() && pages.back().page_idx_ != page_idx) {
        pages.emplace_back(PageLog(page_idx, 0, elmts_per_page_));
      }
      ++iter_max;
    }
    return pages;
  }
};

template<typename IterT, typename VectorT, typename T>
class Tx {
 public:
  MPI_Comm comm_;
  IterT iter_;
  size_t off_ = 0;      /**< The offset in the data buffer */
  size_t iter_cur_ = 0;    /**< The current iteration we are at */
  size_t iter_max_ = 0;    /**< The maximum iteration before evicts */
  std::vector<PageLog> pages_;
  VectorT &mm_vec_;
  bitfield32_t flags_;
  size_t mm_lookahead_;

 public:
  // begin
  Tx begin() {
    return Tx(iter_, mm_vec_, flags_, mm_lookahead_);
  }

  // end
  Tx end() {
    return Tx(iter_.size_);
  }

  // Copy constructor
  Tx(const Tx &other) {
    comm_ = other.comm_;
    iter_ = other.iter_;
    off_ = other.off_;
    iter_cur_ = other.iter_cur_;
    iter_max_ = other.iter_max_;
    pages_ = other.pages_;
    mm_vec_ = other.mm_vec_;
    flags_ = other.flags_;
    mm_lookahead_ = other.mm_lookahead_;
  }

  // Begin iterator
  explicit Tx(MPI_Comm comm, const IterT &iter, VectorT *mm_vec,
              const bitfield32_t &flags, size_t mm_lookahead)
      : comm_(comm), iter_(iter), mm_vec_(*mm_vec),
        flags_(flags), mm_lookahead_(mm_lookahead) {
    if (mm_lookahead < 1) {
      mm_lookahead = 1;
    }
    iter_cur_ = 0;
    iter_max_ = 0;
    off_ = iter_.get(iter_cur_);
    pages_ = iter_.next_pages(iter_cur_, mm_lookahead_, iter_max_);
  }

  // End iterator
  explicit Tx(size_t end) {
    iter_cur_ = end;
  }

  // Const dereference operator
  const T& operator*() const {
    return mm_vec_[off_];
  }

  // Dereference operator
  T& operator*() {
    return mm_vec_[off_];
  }

  // Prefix increment operator
  Tx& operator++() {
    ++iter_cur_;
    off_ = iter_.get(iter_cur_);
    if (iter_cur_ == iter_max_) {
      ConsistencyAndEviction();
    }
    return *this;
  }

  // Consistency guarantees
  void ConsistencyAndEviction() {
    // Flush all modifications
    if (flags_.Any(MM_WRITE_ONLY)) {
      for (size_t i = 0; i < pages_.size(); ++i) {
        mm_vec_._Flush(pages_[i].page_idx_,
                       pages_[i].mod_start_,
                       pages_[i].mod_size_);
      }
    }
    // Evict all pages
    for (size_t i = 0; i < pages_.size(); ++i) {
      mm_vec_._Evict(pages_[i].page_idx_);
    }
    // Prefetch next pages
    pages_ = iter_.next_pages(iter_cur_, mm_lookahead_, iter_max_);
    if (flags_.Any(MM_READ_ONLY)) {
      for (size_t i = 0; i < pages_.size(); ++i) {
        mm_vec_.Prefetch(pages_[i].page_idx_, 1.0);
      }
    }
    if (iter_cur_ >= iter_.size_) {
      pages_.clear();
    }
  }

  // Equality operator
  bool operator==(const Tx& rhs) const {
    return iter_cur_ == rhs.iter_cur_;
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
  size_t elmts_per_window_ = 0;
  size_t cur_memory_ = 0;
  bitfield32_t flags_;
  PGAS pgas_;

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
    elmts_per_window_ = window_size / elmt_size_;
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

  /** Evenly split DSM among processes */
  void EvenPgas(int rank, int nprocs, size_t max_count,
                size_t count_per_page = 0) {
    Bounds bounds(rank, nprocs, max_count);
    if (count_per_page != 0) {
      SetElmtsPerPage(count_per_page);
    }
    pgas_.Init(bounds.off_,
               bounds.size_,
               elmts_per_page_);
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

  /** Create a sequential transaction */
  Tx<SequentialIterator, VectorMegaMpi<T, IS_COMPLEX_TYPE>, T>
  SeqTxBegin(size_t off, size_t size, uint32_t flags,
             MPI_Comm comm = MPI_COMM_WORLD) {
    SequentialIterator iter(off, size, elmts_per_page_);
    return Tx<SequentialIterator, VectorMegaMpi<T, IS_COMPLEX_TYPE>, T>(
        comm, iter, this,
        bitfield32_t(flags),
        elmts_per_window_ / elmts_per_page_);
  }

  /** Create a random transaction */
  Tx<RandomIterator, VectorMegaMpi<T, IS_COMPLEX_TYPE>, T>
  RandTxBegin(size_t seed, size_t rand_left, size_t rand_size,
              size_t size, uint32_t flags,
              MPI_Comm comm = MPI_COMM_WORLD) {
    RandomIterator iter(seed, rand_left, rand_size, size, elmts_per_page_);
    return Tx<RandomIterator, VectorMegaMpi<T, IS_COMPLEX_TYPE>, T>(
        comm, iter, this,
        bitfield32_t(flags),
        elmts_per_window_ / elmts_per_page_);
  }

  /** Create a transaction */
  template<typename GenT>
  Tx<GenT, VectorMegaMpi<T, IS_COMPLEX_TYPE>, T>
  TxBegin(GenT &iter, uint32_t flags,
          MPI_Comm comm = MPI_COMM_WORLD) {
    return Tx<GenT, VectorMegaMpi<T, IS_COMPLEX_TYPE>, T>(
        comm, iter, *this,
        bitfield32_t(flags),
        elmts_per_window_ / elmts_per_page_);
  }

  /** End a transaction */
  template<typename GenT>
  void TxEnd(Tx<GenT, VectorMegaMpi<T, IS_COMPLEX_TYPE>, T> &tx) {
    tx.ConsistencyAndEviction();
    if (tx.flags_.Any(MM_APPEND_ONLY)) {
      FlushEmplace(tx.comm_, 0, 1);
    }
  }

  /** Flush data to backend */
  void _Flush(size_t page_idx, size_t mod_start, size_t mod_count) {
    if (flags_.Any(MM_WRITE_ONLY | MM_READ_WRITE) && pgas_.size_ > 0) {
      hermes::Context ctx;
      auto it = data_.find(page_idx);
      if (it == data_.end()) {
        return;
      }
      Page<T> &page = it->second;
      if constexpr (!IS_COMPLEX_TYPE) {
        std::string page_name =
            hermes::adapter::BlobPlacement::CreateBlobName(page_idx).str();
        ctx.flags_.SetBits(HERMES_SHOULD_STAGE);
        hermes::Blob blob((char*)page.elmts_.data() + mod_start * elmt_size_,
                          mod_count * elmt_size_);
        bkt_.PartialPut(page_name, blob, mod_start * elmt_size_, ctx);
      } else {
        std::string page_name =
            hermes::adapter::BlobPlacement::CreateBlobName(page_idx).str();
        bkt_.Put<T>(page_name, page.elmts_[0], ctx);
      }
    }
  }

  /** Serialize the in-memory cache back to backend */
  void _Evict(size_t page_idx) {
    auto it = data_.find(page_idx);
    if (it == data_.end()) {
      return;
    }
    if (cur_page_ == &it->second) {
      cur_page_ = nullptr;
    }
    data_.erase(it);
    cur_memory_ -= page_size_;
  }

  /** Lock a region */
  void Barrier(u32 flags, MPI_Comm comm) {
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

  /** Size */
  size_t size() const {
    return size_;
  }

  /** Size of PGAS region */
  size_t local_size() const {
    return pgas_.size_;
  }

  /** Offset of PGAS region */
  size_t local_off() const {
    return pgas_.off_;
  }

  /** Index of last element + 1 */
  size_t local_last() const {
    return pgas_.off_ + pgas_.size_;
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
      _FlushEmplace();
    }
    ++size_;
  }

  /** Flush append buffer */
  void _FlushEmplace() {
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
  void FlushEmplace(MPI_Comm comm, int proc_off, int nprocs) {
    if (append_data_.size()) {
      _FlushEmplace();
    }
    HRUN_ADMIN->FlushRoot(DomainId::GetLocal());
    MPI_Barrier(comm);
    size_t new_size = bkt_.GetSize();
    new_size = new_size / elmt_size_;
    Resize(new_size);
    size_ = new_size;
    Hint(MM_READ_ONLY);
  }

  void Prefetch(size_t page_idx, float score) {
    if (flags_.Any(MM_READ_ONLY | MM_READ_WRITE)) {
//      hermes::Context ctx;
//      std::string page_name =
//          hermes::adapter::BlobPlacement::CreateBlobName(page_idx).str();
//      ctx.flags_.SetBits(HERMES_SHOULD_STAGE);
//      bkt_.ReorganizeBlob(page_name, score, ctx);
    }
  }
};

}  // namespace mm

#endif  // MEGAMMAP_INCLUDE_MEGA_MMAP_VECTOR_MEGA_MPI_H_
