
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <ctime>
#include "cpg_module.hpp"

extern "C" {
    #include "include/cpg_workspace.h"
    #include "include/cpg_solve.h"
}

namespace py = pybind11;

static int i;

