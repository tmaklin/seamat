// seamat: templatized matrix library
// https://github.com/tmaklin/seamat
//
// Copyright (C) 2021 Tommi Mäklin (tommi@maklin.fi)
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
#ifndef SEAMAT_SPARSE_MATRIX_CPP
#define SEAMAT_SPARSE_MATRIX_CPP
#include "Matrix.hpp"

#include <cmath>
#include <stdexcept>

#include "openmp_config.hpp"

namespace seamat {
// Resize a matrix
template<typename T, typename U>
void SparseMatrix<T,U>::resize(const uint32_t new_rows, const uint32_t new_cols, const T initial) {
    throw std::runtime_error("Resizing a sparse matrix is not supported.");
}

// Access individual elements
template <typename T, typename U>
T& SparseMatrix<T,U>::operator()(uint32_t row, uint32_t col) {
    uint32_t col_start = col_ptr[col];
    uint32_t col_end = col_ptr[col + 1];
    uint32_t nnz_col = col_end - col_start; // Number of non-zero elements in row `row`.
    if (nnz_col > 0) {
	for (uint32_t i = 0; i < nnz_col; ++i) {
	    uint32_t index = col_start + i;
	    if (this->col_ptr[index] == row) {
		return this->vals[index];
	    }
	}
    }
    return this->zero_val;
}

// Access individual elements (const)
template <typename T, typename U>
const T& SparseMatrix<T,U>::operator()(uint32_t row, uint32_t col) const {
    uint32_t col_start = col_ptr[col];
    uint32_t col_end = col_ptr[col + 1];
    uint32_t nnz_col = col_end - col_start; // Number of non-zero elements in row `row`.
    if (nnz_col > 0) {
	for (uint32_t i = 0; i < nnz_col; ++i) {
	    uint32_t index = col_start + i;
	    if (this->col_ptr[index] == row) {
		return this->vals[index];
	    }
	}
    }
    return this->zero_val;
}
}

#endif
