//
// Created by lukemartinlogan on 8/23/23.
//

#ifndef MEGA_MMAP_INCLUDE_MEGA_MMAP_Array_H_
#define MEGA_MMAP_INCLUDE_MEGA_MMAP_Array_H_

#include <string>
#include <unordered_map>
#include <vector>
#include "hermes.h"
#include "bucket.h"

namespace mm {

template<typename T>
struct CacheLine {
  bool dirty_;
  std::vector<T> data_;
};

template<typename T>
class Array {
 public:
  hapi::Bucket bkt_;
  std::vector<CacheLine<T>> cache_;
  std::unordered_map<int, CacheLine<T>*> map_;
  size_t page_size_;

 public:
  Array() = default;
  ~Array() = default;

  /** Constructor */
  Array(const std::string &name,
        size_t page_size,
        size_t page_cache_count,
        size_t array_size) {
    bkt_ = HERMES->GetBucket(name);
    page_size_ = page_size;
    cache_.resize(page_cache_count * page_size / sizeof(T));
  }

  /** Index operator */
  T& operator[](int idx) {
    auto it = map_.find(idx);
    if (it == map_.end()) {
      std::string blob_name = std::to_string(idx);
      hermes::BlobId blob_id = bkt_.GetBlobId(blob_name);
      if (!blob_id.IsNull()) {
        hapi::Blob blob();
        hermes::Context ctx;
        bkt_.Get(blob_id, blob, ctx);
      } else {
      }
    } else {
      CacheLine<T> *data = it->second;
      size_t off = idx % page_size_;
      return ()
    }
  }
};

}  // namespace mm

#endif  // MEGA_MMAP_INCLUDE_MEGA_MMAP_Array_H_
