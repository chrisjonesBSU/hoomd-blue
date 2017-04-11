// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/CellListGPU.h
 * \brief Declaration of mpcd::CellListGPU
 */

#ifndef MPCD_CELL_LIST_GPU_H_
#define MPCD_CELL_LIST_GPU_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "CellList.h"
#include "hoomd/Autotuner.h"

namespace mpcd
{

//! Computes the MPCD cell list on the GPU
class CellListGPU : public mpcd::CellList
    {
    public:
    //! Constructor
        CellListGPU(std::shared_ptr<SystemDefinition> sysdef,
                    std::shared_ptr<mpcd::ParticleData> mpcd_pdata);

        virtual ~CellListGPU();

        //! Set autotuner parameters
        /*!
         * \param enable Enable/disable autotuning
         * \param period period (approximate) in time steps when returning occurs
         */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            mpcd::CellList::setAutotunerParams(enable, period);

            m_tuner_cell->setPeriod(period);
            m_tuner_cell->setEnabled(enable);
            }

    protected:
        //! Compute the cell list of particles on the GPU
        virtual void buildCellList();

    private:
        std::unique_ptr<Autotuner> m_tuner_cell;    //!< Autotuner for the cell list calculation
    };

namespace detail
{
//! Export the CellListGPU class to python
void export_CellListGPU(pybind11::module& m);
} // end namespace detail

} // end namespace mpcd

#endif // MPCD_CELL_LIST_GPU_H_
