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

}  // namespace mm

#endif //MEGAMMAP_INCLUDE_MEGA_MMAP_MACROS_H_
