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
#include "vector.h"

#include "transaction/transaction.h"
#include "transaction/seq_iter_tx.h"
#include "transaction/rand_iter_tx.h"
#include "transaction/pgas_tx.h"

namespace stdfs = std::filesystem;

namespace mm {

template<typename T>
struct Page {
  std::vector<T> elmts_;
  u32 id_;

  Page() = default;

  Page(u32 id) : id_(id) {}
};

/** A wrapper for mmap-based vectors */
template<typename T, bool IS_COMPLEX_TYPE=false>
class VectorMegaMpi : public Vector {
 public:
  std::unordered_map<size_t, Page<T>> data_;   /**< Map page index to page */
  std::vector<T> append_data_;   /**< Buffer containing data to append to vector */
  Page<T> *cur_page_ = nullptr;  /**< The last page accessed by this thread */
  hermes::Bucket bkt_;     /**< The Hermes bucket */
  std::string path_;       /**< The path being mapped into memory */
  std::shared_ptr<Tx> cur_tx_ = nullptr;   /**< The current access pattern transaction */

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
  void SeqTxBegin(size_t off, size_t size, uint32_t flags) {
    cur_tx_ = std::make_shared<SeqIterTx>(
        this, off, size, flags);
  }

  /** Create a PGAS transaction */
  void PgasTxBegin(size_t off, size_t size, uint32_t flags) {
    cur_tx_ = std::make_shared<PgasTx>(
        this, off, size, flags);
  }

  /** Create a random transaction */
  void RandTxBegin(size_t seed, size_t rand_left, size_t rand_size,
                   size_t size, uint32_t flags) {
    cur_tx_ = std::make_shared<RandIterTx>(
        this, seed, rand_left, rand_size, size, flags);
  }

  /** Begin an arbitrary transaction */
  template<typename TxT, typename ...Args>
  void TxBegin(Args&& ...args) {
    cur_tx_ = std::make_shared<TxT>(
        this, std::forward<Args>(args)...);
  }

  /** Get the current point in iterator */
  template<typename TxT>
  size_t TxGetIdx() const {
    return reinterpret_cast<TxT*>(cur_tx_.get())->Get();
  }

  /** Get the value at the current iterator point */
  template<typename TxT>
  T& TxGet() {
    size_t idx = TxGetIdx<TxT>();
    return (*this)[idx];
  }

  /** Get the value at the current iterator point */
  template<typename TxT>
  const T& TxGet() const {
    size_t idx = TxGetIdx<TxT>();
    return (*this)[idx];
  }

  /** End a transaction */
  void TxEnd() {
    cur_tx_->ProcessLog(true);
    cur_tx_ = nullptr;
  }

  /** Flush data to backend */
  void _Flush(size_t page_idx, size_t mod_start, size_t mod_count) {
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
    // Ensure that the page_idx makes sense
    if (page_idx > size_ / elmts_per_page_) {
      int rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      HELOG(kFatal, "{}: Cannot seek past size of {}: {} / {} ",
            rank, path_, page_idx, size_ / elmts_per_page_);
    }

    // Add page to page table
    hermes::Context ctx;
    data_.emplace(page_idx, Page<T>(page_idx));
    Page<T> &page = data_[page_idx];
    page.elmts_.resize(elmts_per_page_);
    std::string page_name =
        hermes::adapter::BlobPlacement::CreateBlobName(page_idx).str();

    // If we need to read data from the page, ensure we read it from Hermes
    if (flags_.Any(MM_READ_ONLY | MM_READ_WRITE)) {
      if constexpr (!IS_COMPLEX_TYPE) {
        ctx.flags_.SetBits(HERMES_SHOULD_STAGE);
        hermes::Blob blob((char*)page.elmts_.data(), page_size_);
        bkt_.Get(page_name, blob, ctx);
      } else {
        bkt_.Get<T>(page_name, page.elmts_[0], ctx);
      }
    }

    // Increment the current memory counter
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
    if (cur_tx_) {
      if ((cur_tx_->tail_ % elmts_per_page_) == 0) {
        cur_tx_->ProcessLog(false);
      }
      ++cur_tx_->tail_;
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
    HRUN_ADMIN->FlushRoot(DomainId::GetLocal());
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
  void FlushEmplace(MPI_Comm comm) {
    if (append_data_.size()) {
      _FlushEmplace();
    }
    HRUN_ADMIN->FlushRoot(DomainId::GetLocal());
    MPI_Barrier(comm);
    size_t new_size = bkt_.GetSize();
    HILOG(kInfo, "New bucket ({}) size: {}", bkt_.id_, new_size);
    new_size = new_size / elmt_size_;
    Resize(new_size);
    size_ = new_size;
    Hint(MM_READ_ONLY);
  }

  void Rescore(size_t page_idx, size_t mod_start, size_t mod_count,
               float score, bitfield32_t flags) override {
    // Flush and evict modified data
    if (score < 1) {
      if (flags.Any(MM_READ_WRITE | MM_WRITE_ONLY)) {
        _Flush(page_idx, mod_start, mod_count);
      }
      _Evict(page_idx);
    }

    // Stage data to be read from storage
    hermes::Context ctx;
    if (flags.Any(MM_STAGE_READ_FROM_BACKEND)) {
      ctx.flags_.SetBits(HERMES_SHOULD_STAGE);
    }
    std::string page_name =
        hermes::adapter::BlobPlacement::CreateBlobName(page_idx).str();

    // Reorganize the blob
    bkt_.ReorganizeBlob(page_name, score, ctx);

    // Async fault the data
    if (flags.Any(MM_READ_ONLY | MM_READ_WRITE | MM_STAGE_READ_FROM_BACKEND)) {
    }
  }
};

}  // namespace mm

#endif  // MEGAMMAP_INCLUDE_MEGA_MMAP_VECTOR_MEGA_MPI_H_
