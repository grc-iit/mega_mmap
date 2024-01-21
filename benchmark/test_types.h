//
// Created by lukemartinlogan on 1/2/24.
//

#ifndef MEGAMMAP_BENCHMARK_TEST_TYPES_H_
#define MEGAMMAP_BENCHMARK_TEST_TYPES_H_

#include "mega_mmap/macros.h"

#define MM_VEC mm::VectorMegaMpi
#define MM_VEC_2 mm::VectorMmapMpi
#define SEED 23425323

using mm::Bounds;

class MpiComm {
 public:
  MPI_Comm comm_;

 public:
  MpiComm(MPI_Comm comm, int proc_off, int nprocs) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    bool color = rank >= proc_off && rank < proc_off + nprocs;
    MPI_Comm_split(comm, color, rank, &comm_);
  }

  ~MpiComm() {
    MPI_Comm_free(&comm_);
  }
};

struct Row {
  float x_;
  float y_;

  Row() = default;

  Row(float x, float y) : x_(x), y_(y) {}

  Row(float x) : x_(x), y_(x) {}

  Row(const Row &other) {
    x_ = other.x_;
    y_ = other.y_;
  }

  Row &operator=(const Row &other) {
    x_ = other.x_;
    y_ = other.y_;
    return *this;
  }

  Row &operator+=(const Row &other) {
    x_ += other.x_;
    y_ += other.y_;
    return *this;
  }

  Row &operator/=(const size_t &other) {
    x_ /= other;
    y_ /= other;
    return *this;
  }

  double Distance(const Row &other) const {
    return sqrt((x_ - other.x_) * (x_ - other.x_) +
        (y_ - other.y_) * (y_ - other.y_));
  }

  void Zero() {
    x_ = 0;
    y_ = 0;
  }

  template<typename Ar>
  void serialize(Ar &ar) {
    ar(x_, y_);
  }

  bool LessThan(const Row &other, int feature) const {
    if (feature == 0) {
      return x_ < other.x_;
    } else if (feature == 1) {
      return y_ < other.y_;
    } else {
      HILOG(kFatal, "Invalid feature: {}", feature);
      exit(1);
    }
  }

  size_t GetNumFeatures() {
    return 2;
  }

  size_t operator()(const Row &row) const {
    return (size_t) (row.x_ + row.y_);
  }

  bool operator==(const Row &other) const {
    return x_ == other.x_ && y_ == other.y_;
  }

  std::string ToString() const {
    return hshm::Formatter::format("({}, {})", x_, y_);
  }
};

template<int N>
struct RowND {
  float p_[N];

  RowND() = default;

  RowND(const RowND &other) {
    memcpy(p_, other.p_, N * sizeof(float));
  }

  RowND &operator=(const RowND &other) {
    memcpy(p_, other.p_, N * sizeof(float));
    return *this;
  }

  RowND &operator+=(const RowND &other) {
    for (int i = 0; i < N; ++i) {
      p_[i] += other.p_[i];
    }
    return *this;
  }

  RowND &operator/=(const size_t &other) {
    for (int i = 0; i < N; ++i) {
      p_[i] /= other;
    }
    return *this;
  }

  double Distance(const RowND &other) const {
    size_t sum = 0;
    for (int i = 0; i < N; ++i) {
      sum += sqrt((p_[i] - other.p_[i]) * (p_[i] - other.p_[i]));
    }
    return sum;
  }

  void Zero() {
    for (int i = 0; i < N; ++i) {
      p_[i] = 0;
    }
  }

  template<typename Ar>
  void serialize(Ar &ar) {
    for (int i = 0; i < N; ++i) {
      ar(p_[i]);
    }
  }

  bool LessThan(const RowND &other, int feature) const {
    if (feature < 0 || feature >= N) {
      HILOG(kFatal, "Invalid feature: {}", feature);
      exit(1);
    }
    return p_[feature] < other.p_[feature];
  }

  size_t GetNumFeatures() {
    return N;
  }

  size_t operator()(const RowND &row) const {
    size_t hash = 0;
    for (int i = 0; i < N; ++i) {
      hash += (size_t) row.p_[i];
    }
    return hash;
  }

  bool operator==(const RowND &other) const {
    bool eq = true;
    for (int i = 0; i < N; ++i) {
      eq &= p_[i] == other.p_[i];
    }
    return eq;
  }

  std::string ToString() const {
    std::string str = "(";
    for (int i = 0; i < N - 1; ++i) {
      str += hshm::Formatter::format("{}, ", p_[i]);
    }
    str += hshm::Formatter::format("{})", p_[N - 1]);
    return str;
  }
};

struct ClassRow {
  float x_;
  float y_;
  float class_;

  ClassRow() = default;

  ClassRow(float x, float y) : x_(x), y_(y) {}

  ClassRow(float x) : x_(x), y_(x) {}

  ClassRow(const ClassRow &other) {
    x_ = other.x_;
    y_ = other.y_;
    class_ = other.class_;
  }

  ClassRow &operator=(const ClassRow &other) {
    x_ = other.x_;
    y_ = other.y_;
    class_ = other.class_;
    return *this;
  }

  ClassRow &operator+=(const ClassRow &other) {
    x_ += other.x_;
    y_ += other.y_;
    return *this;
  }

  ClassRow &operator/=(const size_t &other) {
    x_ /= other;
    y_ /= other;
    return *this;
  }

  bool LessThan(const ClassRow &other, int feature) const {
    if (feature == 0) {
      return x_ < other.x_;
    } else if (feature == 1) {
      return y_ < other.y_;
    } else {
      HILOG(kFatal, "Invalid feature");
      exit(1);
    }
  }

  const float &last() const {
    return class_;
  }

  template<typename Ar>
  void serialize(Ar &ar) {
    ar(x_, y_, class_);
  }
};

struct GiniSum {
 public:
  size_t left_;
  size_t right_;
};

template<typename T>
struct Gini {
  std::unordered_map<int, GiniSum> count_;

  void Induct(T &row, int part) {
    if (count_.find(row.last()) == count_.end()) {
      count_[row.last()] = GiniSum();
    }
    if (part == 0) {
      count_[row.last()].left_++;
    } else {
      count_[row.last()].right_++;
    }
  }

  float Get() {
    float gini = 0;
    for (const std::pair<int, GiniSum> &gpair : count_) {
      GiniSum sum = gpair.second;
      float total = sum.left_ + sum.right_;
      float left_prob = sum.left_ / total;
      float right_prob = sum.right_ / total;
      gini += left_prob * (1 - left_prob) +
          right_prob * (1 - right_prob);
    }
    return gini;
  }
};

#endif //MEGAMMAP_BENCHMARK_TEST_TYPES_H_
