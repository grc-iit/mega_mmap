#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

// #IO# include IO library
#include "gray-scott.h"

void print_settings(const Settings &s)
{
  std::cout << "grid:             " << s.L << "x" << s.L << "x" << s.L
            << std::endl;
  std::cout << "steps:            " << s.steps << std::endl;
  std::cout << "plotgap:          " << s.plotgap << std::endl;
  std::cout << "F:                " << s.F << std::endl;
  std::cout << "k:                " << s.k << std::endl;
  std::cout << "dt:               " << s.dt << std::endl;
  std::cout << "Du:               " << s.Du << std::endl;
  std::cout << "Dv:               " << s.Dv << std::endl;
  std::cout << "noise:            " << s.noise << std::endl;
  std::cout << "output:           " << s.output << std::endl;
}

void print_simulator_settings(const GrayScott &s)
{
  std::cout << "process layout:   " << s.npx << "x" << s.npy << "x" << s.npz
            << std::endl;
  std::cout << "local grid size:  " << s.size_x << "x" << s.size_y << "x"
            << s.size_z << std::endl;
}

int main(int argc, char **argv)
{
  upcxx::init();
  int rank, procs;
  rank = upcxx::rank_me();
  procs = upcxx::rank_n();

  if (argc < 2) {
    if (rank == 0) {
      std::cerr << "Too few arguments" << std::endl;
      std::cerr << "Usage: gray-scott settings.yaml" << std::endl;
    }
    exit(1);
  }

  Settings settings;
  settings.load(argv[1]);

  GrayScott sim(settings);
  sim.init();

  // #IO# Need to initialize IO library
  //

  if (rank == 0) {
    std::cout << "========================================" << std::endl;
    print_settings(settings);
    print_simulator_settings(sim);
    std::cout << "========================================" << std::endl;
  }

  for (int i = 0; i < settings.steps;) {
    for (int j = 0; j < settings.plotgap; j++) {
      sim.iterate();
      i++;
    }
    if (rank == 0) {
      std::cout << "Simulation at step " << i
                << " publishing output step     " << i / settings.plotgap
                << std::endl;
    }
  }

  upcxx::finalize();
}
