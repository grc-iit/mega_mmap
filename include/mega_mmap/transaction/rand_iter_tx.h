//
// Created by llogan on 3/11/24.
//

#ifndef MEGAMMAP_INCLUDE_MEGA_MMAP_TRANSACTION_RAND_TX_H_
#define MEGAMMAP_INCLUDE_MEGA_MMAP_TRANSACTION_RAND_TX_H_

namespace mm {

class RandIterTx : public Tx {
 public:
  hshm::UniformDistribution gen_;
  hshm::UniformDistribution log_gen_;
  size_t size_;
  size_t base_ = 0;
  bitfield32_t flags_;

 public:
  RandIterTx(Vector *vec, size_t seed, size_t rand_left, size_t rand_size,
             size_t size, uint32_t flags) : Tx(vec) {
    gen_.Seed(seed);
    gen_.Shape((double)rand_left,
               (double)rand_left + (double)rand_size);
    size_ = size;
    flags_.SetBits(flags);
    log_gen_ = gen_;
  }

  virtual ~RandIterTx() = default;

  void _ProcessLog(bool end) override {
    // Get number of pages iterated over
    size_t num_pages = (tail_ - head_) / vec_->elmts_per_page_;
    if (num_pages == 0) {
      return;
    }

    // Evict processed pages
    for (size_t i = 0; i < num_pages; ++i) {
      size_t page_idx = log_gen_.GetSize();
      vec_->Rescore(page_idx, 0, vec_->elmts_per_page_,
                    0, flags_);
    }

    // Prefetch future pages
  }

  size_t Get() {
    size_t off = tail_ % vec_->elmts_per_page_;
    if (off == 0) {
      base_ = gen_.GetSize();
    }
    return base_ + off;
  }
};

}  // namespace mm

#endif //MEGAMMAP_INCLUDE_MEGA_MMAP_TRANSACTION_RAND_TX_H_
