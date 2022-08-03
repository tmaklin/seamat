// seamat: templatized matrix library
// https://github.com/tmaklin/seamat
//
// Copyright (C) 2021 Tommi Mäklin (tommi@maklin.fi)
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
#ifndef SEAMAT_INDEX_MATRIX_CPP
#define SEAMAT_INDEX_MATRIX_CPP
#include "Matrix.hpp"

#include <cmath>
#include <stdexcept>

#include "openmp_config.hpp"

namespace seamat {
// Resize a matrix
template<typename T, typename U>
void IndexMatrix<T,U>::resize(const uint32_t new_rows, const uint32_t new_cols, const T initial) {
    throw std::runtime_error("Resizing an IndexMatrix is not supported.");
}

// Access individual elements
template <typename T, typename U>
T& IndexMatrix<T,U>::operator()(uint32_t row, uint32_t col) {
    uint32_t out_col = (*this->indices)(row, col);
    return (*this->vals)(row, out_col);
}

// Access individual elements (const)
template <typename T, typename U>
const T& IndexMatrix<T,U>::operator()(uint32_t row, uint32_t col) const {
    uint32_t out_col = (*this->indices)(row, col);
    return (*this->vals)(row, out_col);
}

// Initialize from vals and indices
template<typename T, typename U>
IndexMatrix<T, U>::IndexMatrix(const Matrix<T> &_vals, const Matrix<U> &_indices, const bool store_as_sparse) {
    this->resize_rows(_indices.get_rows());
    this->resize_cols(_indices.get_cols());

    this->n_rows_vals = _vals.get_rows();
    this->n_cols_vals = _vals.get_cols();
    if (store_as_sparse) {
	this->indices.reset(new SparseMatrix<U>(_indices, 0));
	this->vals.reset(new SparseMatrix<T>(_vals, 0.0));
    } else {
	this->indices.reset(new DenseMatrix<U>(_indices));
	this->vals.reset(new DenseMatrix<T>(_vals));
    }
}
}

#endif
