#ifndef __SETTINGS_H__
#define __SETTINGS_H__

#include <string>
#include <yaml-cpp/yaml.h>

struct Settings {
  size_t L;
  int steps;
  int plotgap;
  double F;
  double k;
  double dt;
  double Du;
  double Dv;
  double noise;
  std::string output;

  Settings() {
    L = 128;
    steps = 20000;
    plotgap = 200;
    F = 0.04;
    k = 0.06075;
    dt = 0.2;
    Du = 0.05;
    Dv = 0.1;
    noise = 0.0;
    output = "foo.bp";
  }

  void load(const std::string &fname) {
    YAML::Node config = YAML::LoadFile(fname);
    L = config["L"].as<size_t>();
    steps = config["steps"].as<int>();
    plotgap = config["plotgap"].as<int>();
    F = config["F"].as<double>();
    k = config["k"].as<double>();
    dt = config["dt"].as<double>();
    Du = config["Du"].as<double>();
    Dv = config["Dv"].as<double>();
    noise = config["noise"].as<double>();
    output = config["output"].as<std::string>();
  }
};

#endif
