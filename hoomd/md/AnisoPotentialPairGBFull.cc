// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "AnisoPotentialPair.h"
#include "EvaluatorPairGBFull.h"

namespace hoomd
    {
namespace md
    {
namespace detail
    {
template void export_AnisoPotentialPair<EvaluatorPairGBFull>(pybind11::module& m,
                                                              const std::string& name);

void export_AnisoPotentialPairGBFull(pybind11::module& m)
    {
    export_AnisoPotentialPair<EvaluatorPairGBFull>(m, "AnisoPotentialPairGBFull");
    }
    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
