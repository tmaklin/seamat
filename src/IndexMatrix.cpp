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
// Access individual elements
template <typename T, typename U>
T& IndexMatrix<T,U>::operator()(size_t row, size_t col) {
    size_t out_col = (*this->indices)(row, col);
    return (*this->vals)(row, out_col);
}

// Access individual elements (const)
template <typename T, typename U>
const T& IndexMatrix<T,U>::operator()(size_t row, size_t col) const {
    size_t out_col = (*this->indices)(row, col);
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

// TODO implement index matrix operators

// In-place matrix-matrix addition
template<typename T, typename U>
IndexMatrix<T, U>& IndexMatrix<T, U>::operator+=(const Matrix<T>& rhs) {
    // seamat::IndexMatrix<T, U>::operator+=
    //
    // Add values from rhs to the calling matrix in-place.
    //
    //   Input:
    //     `rhs`: Matrix to add, must have the same dimensions as the caller.
    //
    throw std::runtime_error("Index matrix operators have not been implemented.");
    return *this;
}

// In-place matrix-matrix subtraction
template<typename T, typename U>
IndexMatrix<T, U>& IndexMatrix<T, U>::operator-=(const Matrix<T>& rhs) {
    // seamat::IndexMatrix<T, U>::operator-=
    //
    // Subtract values of rhs from the calling matrix in-place.
    //
    //   Input:
    //     `rhs`: Matrix to subtract, must have the same dimensions as the caller.
    //
    throw std::runtime_error("Index matrix operators have not been implemented.");

    return *this;
}

// In-place right multiplication
template<typename T, typename U>
IndexMatrix<T, U>& IndexMatrix<T, U>::operator*=(const Matrix<T>& rhs) {
    // seamat::IndexMatrix<T, U>::operator*=
    //
    // Matrix right-multiplication of the caller with rhs in-place.
    //
    //   Input:
    //     `rhs`: Matrix to right multiply with,
    //            must have the same number of rows as the caller has columns.
    //
    throw std::runtime_error("Index matrix operators have not been implemented.");
    return *this;
}

// In-place left multiplication
template<typename T, typename U>
IndexMatrix<T, U>& IndexMatrix<T, U>::operator%=(const Matrix<T>& lhs) {
    // seamat::IndexMatrix<T, U>::operator%=
    //
    // Matrix left-multiplication of the caller with lhs in-place.
    //
    //   Input:
    //     `lhs`: Matrix to left multiply with,
    //            must have the same number of columns as the caller has rows.
    //
    throw std::runtime_error("Index matrix operators have not been implemented.");
    return *this;
}

// In-place matrix-scalar addition
template<typename T, typename U>
IndexMatrix<T, U>& IndexMatrix<T, U>::operator+=(const T& scalar) {
    // seamat::IndexMatrix<T, U>::operator+=
    //
    // In-place addition of a scalar to caller.
    //
    //   Input:
    //     `scalar`: Scalar value to add to all caller values.
    //
    throw std::runtime_error("Index matrix operators have not been implemented.");
    return *this;
}

// In-place matrix-scalar subtraction
template<typename T, typename U>
IndexMatrix<T, U>& IndexMatrix<T, U>::operator-=(const T& scalar) {
    // seamat::IndexMatrix<T, U>::operator-=
    //
    // In-place subtraction of a scalar from the caller.
    //
    //   Input:
    //     `scalar`: Scalar value to subtract from all caller values.
    //
    throw std::runtime_error("Index matrix operators have not been implemented.");
    return *this;
}

// In-place matrix-scalar multiplication
template<typename T, typename U>
IndexMatrix<T, U>& IndexMatrix<T, U>::operator*=(const T& scalar) {
    // seamat::IndexMatrix<T, U>::operator*=
    //
    // In-place multiplication of the caller with a scalar.
    //
    //   Input:
    //     `scalar`: Scalar value to multiply all caller values with.
    //
    throw std::runtime_error("Index matrix operators have not been implemented.");
    return *this;
}

// In-place matrix-scalar division
template<typename T, typename U>
IndexMatrix<T, U>& IndexMatrix<T, U>::operator/=(const T& scalar) {
    // seamat::IndexMatrix<T, U>::operator/=
    //
    // In-place division of the caller with a scalar.
    //
    //   Input:
    //     `scalar`: Scalar value to divide all caller values with.
    //
    throw std::runtime_error("Index matrix operators have not been implemented.");
    return *this;
}
}

#endif
