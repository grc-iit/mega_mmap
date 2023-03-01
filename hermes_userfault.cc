//
// Created by lukemartinlogan on 3/1/23.
//

#include <linux/userfaultfd.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <pthread.h>
#include <poll.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>

void *fault_handler_thread(void *arg) {
  int uffd = *(int*)arg;
  struct uffd_msg fault_msg;
  struct iovec iov = {
    .iov_base = &fault_msg,
    .iov_len = sizeof(fault_msg)
  };
  struct pollfd poll_fd = {
    .fd = uffd,
    .events = POLLIN
  };

  while (poll(&poll_fd, 1, -1) > 0) {
    if (poll_fd.revents & POLLERR || poll_fd.revents & POLLHUP || poll_fd.revents & POLLNVAL) {
      printf("poll error\n");
      break;
    }

    switch (fault_msg.event) {
      case UFFD_EVENT_PAGEFAULT: {
        printf("pagefault occurred\n");
        break;
      }
    }

    if (fault_msg.event & UFFD_EVENT_PAGEFAULT) {
      printf("pagefault occurred\n");
      // handle pagefault
      // ...

      // signal completion of handling
      struct uffdio_copy copy = {
        .dst = (unsigned long)fault_msg.arg.pagefault.address,
        .src = (unsigned long)NULL,
        .len = 0
      };
      ioctl(uffd, UFFDIO_COPY, &copy);
    }
  }

  return NULL;
}

int main() {
  int uffd = syscall(__NR_userfaultfd, O_CLOEXEC | O_NONBLOCK);
  if (uffd == -1) {
    perror("userfaultfd");
    exit(EXIT_FAILURE);
  }

  // create a shared memory region to be used for handling pagefaults
  size_t region_size = 4096;
  void *region = mmap(NULL, region_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (region == MAP_FAILED) {
    perror("mmap");
    exit(EXIT_FAILURE);
  }

  // register the shared memory region with userfaultfd
  struct uffdio_register reg = {
    .mode = UFFDIO_REGISTER_MODE_MISSING,
    .range = {
      .start = (unsigned long)region,
      .len = region_size
    }
  };
  if (ioctl(uffd, UFFDIO_REGISTER, &reg) == -1) {
    perror("ioctl(UFFDIO_REGISTER)");
    exit(EXIT_FAILURE);
  }

  // create a thread to handle pagefaults
  pthread_t thread;
  if (pthread_create(&thread, NULL, fault_handler_thread, &uffd) == -1) {
    perror("pthread_create");
    exit(EXIT_FAILURE);
  }

  // generate a pagefault by accessing an invalid address
  int *p = (int*)(region + region_size + 1);
  printf("%d\n", *p);  // this should trigger a pagefault

  // wait for the fault handler thread to complete
  if (pthread_join(thread, NULL) == -1) {
    perror("pthread_join");
    exit(EXIT_FAILURE);
  }

  return EXIT_SUCCESS;
}