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

}  // namespace mm

#endif //MEGAMMAP_INCLUDE_MEGA_MMAP_MACROS_H_
