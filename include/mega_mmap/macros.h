//
// Created by lukemartinlogan on 1/17/24.
//

#ifndef MEGAMMAP_INCLUDE_MEGA_MMAP_MACROS_H_
#define MEGAMMAP_INCLUDE_MEGA_MMAP_MACROS_H_

#include <hermes_shm/data_structures/data_structure.h>
#include <hrun/hrun_types.h>

#define MM_READ_ONLY BIT_OPT(u32, 0)
#define MM_WRITE_ONLY BIT_OPT(u32, 1)
#define MM_APPEND_ONLY BIT_OPT(u32, 2)
#define MM_READ_WRITE BIT_OPT(u32, 3)

namespace mm {

using hshm::bitfield32_t;

struct PGAS {
  size_t off_;
  size_t size_;
  size_t min_page_idx_;
  size_t min_page_off_;
  size_t max_page_idx_;
  size_t max_page_off_;
  size_t page_size_;

  void Init(size_t off, size_t size, size_t page_size) {
    off_ = off;
    size_ = size;
    min_page_idx_ = off / page_size;
    min_page_off_ = off % page_size;
    max_page_idx_ = (off + size) / page_size;
    max_page_off_ = (off + size) % page_size;
    page_size_ = page_size;
  }

  void GetModBounds(size_t page_idx, size_t &page_off, size_t &page_size) {
    if (page_idx == min_page_idx_) {
      page_off = min_page_off_;
      page_size = page_size_ - min_page_off_;
    } else if (page_idx == max_page_idx_) {
      page_off = 0;
      page_size = max_page_off_;
    } else {
      page_off = 0;
      page_size = page_size_;
    }
  }
};

#define MM_PAGE_SIZE KILOBYTES(64)

struct UniformSampler {
 public:
  hshm::UniformDistribution dist_;
  size_t num_pages_;
  size_t page_size_;
  size_t sample_size_;
  size_t last_page_size_;

 public:
  UniformSampler(size_t page_size, size_t max_size, size_t seed) {
    sample_size_ = max_size;
    page_size_ = page_size;
    num_pages_ = max_size / page_size;
    last_page_size_ = max_size % page_size;
    if (last_page_size_) {
      num_pages_ += 1;
    } else {
      last_page_size_ = page_size;
    }
    dist_.Seed(seed);
    dist_.Shape(0, num_pages_ - 1);
  }

  template<typename AssignT>
  void SamplePage(AssignT &assign, size_t &off) {
    size_t page_idx = dist_.GetSize();
    size_t page_size = page_size_;
    if (page_idx == num_pages_ - 1) {
      page_size = last_page_size_;
    }
    if (off + page_size > assign.size()) {
      page_size = assign.size() - off;
    }
    for (size_t i = 0; i < page_size; ++i) {
      assign[off++] = page_idx * page_size_ + i;
    }
  }

  template<typename AssignT>
  void SamplePage(AssignT &pool,
                  AssignT &assign,
                  size_t &off) {
    size_t page_idx = dist_.GetSize();
    size_t page_size = page_size_;
    if (page_idx == num_pages_ - 1) {
      page_size = last_page_size_;
    }
    if (off + page_size > assign.size()) {
      page_size = assign.size() - off;
    }
    for (size_t i = 0; i < page_size; ++i) {
      assign[off++] = pool[page_idx * page_size_ + i];
    }
  }
};

}  // namespace mm

#endif //MEGAMMAP_INCLUDE_MEGA_MMAP_MACROS_H_
