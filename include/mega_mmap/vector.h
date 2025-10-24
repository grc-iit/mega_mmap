/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Copyright (c) 2024-2025 Illinois Institute of Technology.
 * Gnosis Research Center.
 * All rights reserved.
 *
 * This file is part of MegaMmap.
 * Project website: https://github.com/grc-iit/megammap
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the conditions in the
 * LICENSE file are met. See the LICENSE file at the root of this
 * repository for details.
 *
 * Developed by: Gnosis Research Center
 *               Illinois Institute of Technology
 *               https://grc.iit.edu
 *
 * Contact: grc@iit.edu
 */

/**
 * @file vector.h
 * @brief Main header file for MegaMmap vector classes.
 *
 * This header defines the core vector classes and interfaces for the MegaMmap library, providing memory-mapped vector functionality.
 *
 * This project is part of the IoWarp project, a collaborative NSF-funded effort. It is also part of the production code in the github.com/iowarp codebase.
 *
 * @author Anthony Kougkas <akougkas@iit.edu>
 * @date 2025-10-24
 * @version 1.0
 */

#ifndef MEGAMMAP_INCLUDE_MEGA_MMAP_VECTOR_H_
#define MEGAMMAP_INCLUDE_MEGA_MMAP_VECTOR_H_

namespace mm {

/**
 * A larger-than-memory vector
 * window: maximum amount of main memory this vector can absorb
 * */
class Vector {
 public:
  size_t window_size_ = 0;        /**< bytes in a window */
  size_t elmts_per_window_ = 0;   /**< number of elements in a window */
  size_t cur_memory_ = 0;         /**< Bytes currently occupied by the vector */

  size_t size_ = 0;            /** Number of elements in the vector */
  size_t max_size_ = 0;        /** Maximum number of elements in the vector */
  size_t elmt_size_ = 0;       /** Average size of elements in the vector */
  size_t elmts_per_page_ = 0;  /** Number of elements in a single page */
  size_t page_size_ = 0;       /** Number of data bytes a single page can hold */
  size_t page_mem_ = 0;        /** Page Size + Page Header Size */
  PGAS pgas_;                  /** PGAS mapping of vector elements */
  Bounds bounds_;              /** Bounds of the vector */
  bitfield32_t flags_;         /** Access flags for this vector */

 public:
  virtual void Rescore(size_t page_idx, size_t mod_start, size_t mod_count,
                       float score, bitfield32_t flags) = 0;
};

}  // namespace mm

#endif  // MEGAMMAP_INCLUDE_MEGA_MMAP_VECTOR_H_
