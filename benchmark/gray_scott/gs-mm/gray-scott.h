#ifndef __GRAY_SCOTT_H__
#define __GRAY_SCOTT_H__

#include <random>
#include <vector>

#include <mpi.h>

#include "../settings.h"
#include "mega_mmap/vector_mega_mpi.h"

class GrayScott {
 public:
  // Dimension of process grid
  size_t npx, npy, npz;
  // Coordinate of this rank in process grid
  size_t px, py, pz;
  // Dimension of local array
  size_t size_x, size_y, size_z;
  // Offset of local array in the global array
  size_t offset_x, offset_y, offset_z;

  GrayScott(const Settings &settings, MPI_Comm comm);
  ~GrayScott();

  void init();
  void iterate();

 protected:
  Settings settings;

  mm::VectorMegaMpi<double> u, v, u2, v2;

  int rank, procs;
  int west, east, up, down, north, south;
  MPI_Comm comm;
  MPI_Comm cart_comm;

  // MPI datatypes for halo exchange
  MPI_Datatype xy_face_type;
  MPI_Datatype xz_face_type;
  MPI_Datatype yz_face_type;

  std::random_device rand_dev;
  std::mt19937 mt_gen;
  std::uniform_real_distribution<double> uniform_dist;

  // Setup cartesian communicator data types
  void init_mm();
  // Setup initial conditions
  void init_field();

  // Progess simulation for one timestep
  void calc(mm::VectorMegaMpi<double> &u,
            mm::VectorMegaMpi<double> &v,
            mm::VectorMegaMpi<double> &u2,
            mm::VectorMegaMpi<double> &v2);
  // Compute reaction term for U
  double calcU(double tu, double tv) const;
  // Compute reaction term for V
  double calcV(double tu, double tv) const;
  // Compute laplacian of field s at (ix, iy, iz)
  double laplacian(int ix, int iy, int iz,
                   mm::VectorMegaMpi<double> &s) const;

  // Check if point is included in my subdomain
  inline bool is_inside(int x, int y, int z) const
  {
    if (x < offset_x)
      return false;
    if (x >= offset_x + size_x)
      return false;
    if (y < offset_y)
      return false;
    if (y >= offset_y + size_y)
      return false;
    if (z < offset_z)
      return false;
    if (z >= offset_z + size_z)
      return false;

    return true;
  }
  // Convert global coordinate to local index
  inline int g2i(int gx, int gy, int gz) const
  {
    int x = gx - offset_x;
    int y = gy - offset_y;
    int z = gz - offset_z;

    return l2i(x + 1, y + 1, z + 1);
  }
  // Convert local coordinate to local index
  inline int l2i(int x, int y, int z) const
  {
    return x + y * (size_x + 2) + z * (size_x + 2) * (size_y + 2);
  }
};

#endif
