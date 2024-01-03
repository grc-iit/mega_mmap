//
// Created by lukemartinlogan on 1/2/24.
//

#ifndef MEGAMMAP_BENCHMARK_TEST_TYPES_H_
#define MEGAMMAP_BENCHMARK_TEST_TYPES_H_

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

  double Distance(const Row &other) {
    return (x_ - other.x_) * (x_ - other.x_) +
        (y_ - other.y_) * (y_ - other.y_);
  }

  void Zero() {
    x_ = 0;
    y_ = 0;
  }

  float& operator[](size_t idx) {
    if (idx == 0) {
      return x_;
    } else if (idx == 1) {
      return y_;
    } else {
      HILOG(kFatal, "Invalid index");
    }
  }

  const float& operator[](size_t idx) const {
    if (idx == 0) {
      return x_;
    } else if (idx == 1) {
      return y_;
    } else {
      HILOG(kFatal, "Invalid index");
    }
  }

  const float &last() const {
    return y_;
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

  float& operator[](size_t idx) {
    if (idx == 0) {
      return x_;
    } else if (idx == 1) {
      return y_;
    } else if (idx == 2) {
      return class_;
    } else {
      HILOG(kFatal, "Invalid index");
      exit(1);
    }
  }

  const float& operator[](size_t idx) const {
    if (idx == 0) {
      return x_;
    } else if (idx == 1) {
      return y_;
    } else if (idx == 2) {
      return class_;
    } else {
      HILOG(kFatal, "Invalid index");
      exit(1);
    }
  }

  const float &last() const {
    return class_;
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
