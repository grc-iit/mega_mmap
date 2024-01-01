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
#include <filesystem>
namespace stdfs = std::filesystem;

namespace mm {

/** Forward declaration */
template<typename T>
class VectorMmapMpiIterator;

/** A wrapper for mmap-based vectors */
template<typename T>
class VectorMmapMpi {
 public:
  T *data_;
  size_t size_;

 public:
  VectorMmapMpi() = default;
  ~VectorMmapMpi() = default;

  /** Constructor */
  VectorMmapMpi(const std::string &path, size_t data_size) {
    Init(path, data_size);
  }

  /** Construct from pointer */
  VectorMmapMpi(T *data, size_t size) {
    data_ = data;
    size_ = size;
  }

  /** Copy constructor */
  VectorMmapMpi(const VectorMmapMpi<T> &other) {
    data_ = other.data_;
    size_ = other.size_;
  }

  /** Explicit initializer */
  void Init(const std::string &path) {
    size_t data_size = stdfs::file_size(path);
    size_t size = data_size / sizeof(T);
    Init(path, size);
  }

  /** Explicit initializer */
  void Init(const std::string &path, size_t size) {
    int fd = open(path.c_str(), O_RDWR | O_CREAT, 0666);
    if (fd < 0) {
      HELOG(kFatal, "Failed to open file {}: {}",
            path.c_str(), strerror(errno));
    }
    data_ = (T*)mmap(NULL, size * sizeof(T), PROT_READ | PROT_WRITE,
                     MAP_SHARED, fd, 0);
    if (data_ == nullptr) {
      HELOG(kFatal, "Failed to mmap file {}: {}",
            path.c_str(), strerror(errno));
    }
    size_ = size;
  }

  /** Lock a region */
  void Barrier() {
    MPI_Barrier(MPI_COMM_WORLD);
  }

  /** Index operator */
  T& operator[](int idx) {
    return data_[idx];
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
  VectorMmapMpiIterator<T> begin() {
    return VectorMmapMpiIterator<T>(data_);
  }

  /** End iterator */
  VectorMmapMpiIterator<T> end() {
    return VectorMmapMpiIterator<T>(data_ + size());
  }
};

/** Iterator for VectorMmapMpi */
template<typename T>
class VectorMmapMpiIterator {
 public:
  T *data_;

 public:
  VectorMmapMpiIterator() = default;
  ~VectorMmapMpiIterator() = default;

  /** Constructor */
  VectorMmapMpiIterator(T *data) {
    data_ = data;
  }

  /** Copy constructor */
  VectorMmapMpiIterator(const VectorMmapMpiIterator<T> &other) {
    data_ = other.data_;
  }

  /** Addition operator */
  VectorMmapMpiIterator<T> operator+(int idx) {
    return VectorMmapMpiIterator<T>(data_ + idx);
  }

  /** Addition assign operator */
  VectorMmapMpiIterator<T> &operator+=(int idx) {
    data_ += idx;
    return *this;
  }

  /** Subtraction operator */
  VectorMmapMpiIterator<T> operator-(int idx) {
    return VectorMmapMpiIterator<T>(data_ - idx);
  }

  /** Subtraction assign operator */
  VectorMmapMpiIterator<T> &operator-=(int idx) {
    data_ -= idx;
    return *this;
  }

  /** Index operator */
  T &operator[](int idx) {
    return data_[idx];
  }

  /** Dereference operator */
  T &operator*() {
    return *data_;
  }

  /** Prefix increment operator */
  VectorMmapMpiIterator<T> &operator++() {
    ++data_;
    return *this;
  }

  /** Postfix increment operator */
  VectorMmapMpiIterator<T> operator++(int) {
    VectorMmapMpiIterator<T> tmp(*this);
    operator++();
    return tmp;
  }

  /** Prefix decrement operator */
  VectorMmapMpiIterator<T> &operator--() {
    --data_;
    return *this;
  }

  /** Postfix decrement operator */
  VectorMmapMpiIterator<T> operator--(int) {
    VectorMmapMpiIterator<T> tmp(*this);
    operator--();
    return tmp;
  }

  /** Addition operator */
  VectorMmapMpiIterator<T> operator+(const VectorMmapMpiIterator<T> &other) {
    return VectorMmapMpiIterator<T>(data_ + other.data_);
  }

  /** Addition assign operator */
  VectorMmapMpiIterator<T> &operator+=(const VectorMmapMpiIterator<T> &other) {
    data_ += other.data_;
    return *this;
  }

  /** Subtraction operator */
  VectorMmapMpiIterator<T> operator-(const VectorMmapMpiIterator<T> &other) {
    return VectorMmapMpiIterator<T>(data_ - other.data_);
  }

  /** Subtraction assign operator */
  VectorMmapMpiIterator<T> &operator-=(const VectorMmapMpiIterator<T> &other) {
    data_ -= other.data_;
    return *this;
  }
};

}  // namespace mm

#endif  // MEGAMMAP_INCLUDE_MEGA_MMAP_VECTOR_MMAP_MPI_H_
