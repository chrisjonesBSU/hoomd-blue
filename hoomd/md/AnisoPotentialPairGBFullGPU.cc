// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "AnisoPotentialPairGPU.h"
#include "EvaluatorPairGBFull.h"

namespace hoomd
    {
namespace md
    {
namespace detail
    {
void export_AnisoPotentialPairGBFullGPU(pybind11::module& m)
    {
    export_AnisoPotentialPairGPU<EvaluatorPairGBFull>(m, "AnisoPotentialPairGBFullGPU");
    }
    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
