// seamat: templatized matrix library
// https://github.com/tmaklin/seamat
//
// Copyright (C) 2021 Tommi Mäklin (tommi@maklin.fi)
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
#ifndef SEAMAT_SPARSE_INTEGER_TYPE_MATRIX_CPP
#define SEAMAT_SPARSE_INTEGER_TYPE_MATRIX_CPP

#include <cmath>
#include <stdexcept>

#include "bm.h"
#include "bmsparsevec.h"

#include "Matrix.hpp"
#include "openmp_config.hpp"
#include "math_util.hpp"

namespace seamat {
template <typename T> class SparseIntegerTypeMatrix : public Matrix<T> {
private:
    // Specialized sparse matrix for integer types using BitMagic's
    // sparse integer vectors.
    //
    bm::sparse_vector<T, bm::bvector<>> vals;
    T zero_val;

public:
    SparseIntegerTypeMatrix() = default;
    ~SparseIntegerTypeMatrix() = default;
    // Parameter constructor
    SparseIntegerTypeMatrix(size_t _rows, size_t _cols, const T _zero_val);
    // Initialize from a generic Matrix
    SparseIntegerTypeMatrix(const Matrix<T> &_vals, const T _zero_val);
    // Initialize from a DenseMatrix
    SparseIntegerTypeMatrix(const DenseMatrix<T> &_vals, const T _zero_val);
    // Initialize from a 2D vector
    SparseIntegerTypeMatrix(const std::vector<std::vector<T>> &rhs, const T _zero_val);
    // Copy constructor from contiguous 2D vector
    SparseIntegerTypeMatrix(const std::vector<T> &rhs, const size_t _rows, const size_t _cols, const T _zero_val);

    // Access individual elements
    T& operator()(size_t row, size_t col) override;
    const T& operator()(size_t row, size_t col) const override;

    // Mathematical operators
    // Matrix-matrix in-place summation and subtraction
    SparseIntegerTypeMatrix<T>& operator+=(const Matrix<T>& rhs) override;
    SparseIntegerTypeMatrix<T>& operator-=(const Matrix<T>& rhs) override;

    // In-place right multiplication
    SparseIntegerTypeMatrix<T>& operator*=(const Matrix<T>& rhs) override;
    // In-place left multiplication
    SparseIntegerTypeMatrix<T>& operator%=(const Matrix<T>& rhs) override;

    // Matrix-scalar, in-place
    SparseIntegerTypeMatrix<T>& operator+=(const T& rhs) override;
    SparseIntegerTypeMatrix<T>& operator-=(const T& rhs) override;
    SparseIntegerTypeMatrix<T>& operator*=(const T& rhs) override;
    SparseIntegerTypeMatrix<T>& operator/=(const T& rhs) override;
};

// Parameter constructor
template<typename T>
SparseIntegerTypeMatrix<T>::SparseIntegerTypeMatrix(size_t _rows, size_t _cols, const T _zero_val) {
    // Initializes an empty SparseIntegerTypeMatrix.
    //
    this->resize_rows(_rows);
    this->resize_cols(_cols);
    this->zero_val = _zero_val;
    this->vals.resize((size_t)(_rows*_cols));
}

// Copy constructor from a generic Matrix
template<typename T>
SparseIntegerTypeMatrix<T>::SparseIntegerTypeMatrix(const Matrix<T> &_vals, const T _zero_val) {
    // Initializes a SparseIntegerTypeMatrix from a generic Matrix _vals.
    //
    size_t _rows = _vals.get_rows();
    size_t _cols = _vals.get_cols();

    this->resize_rows(_rows);
    this->resize_cols(_cols);
    this->zero_val = _zero_val;

    this->vals.resize((size_t)(_rows*_cols));

    for (size_t i = 0; i < _rows; ++i) {
	for (size_t j = 0; j < _cols; ++j) {
	    if (!nearly_equal<T>(_vals(i, j), _zero_val)) {
		this->vals.set(i*_cols + j, _vals(i, j));
	    }
	}
    }
}

// Copy constructor from a DenseMatrix
template<typename T>
SparseIntegerTypeMatrix<T>::SparseIntegerTypeMatrix(const DenseMatrix<T> &_vals, const T _zero_val) {
    // Initializes a SparseMatrix from DenseMatrix _vals.
    //
    size_t _rows = _vals.get_rows();
    size_t _cols = _vals.get_cols();

    this->resize_rows(_rows);
    this->resize_cols(_cols);
    this->zero_val = _zero_val;

    // Construct the bm::sparse_vector from _vals.mat's internal array
    T* arr_start = const_cast<T*>(&_vals.mat[0]);
    this->vals.import(arr_start, (size_t)(_rows*_cols));
}

// Copy constructor from 2D vector
template<typename T>
SparseIntegerTypeMatrix<T>::SparseIntegerTypeMatrix(const std::vector<std::vector<T>> &_vals, const T _zero_val) {
    // Initialize from a 2D vector
    // NOTE: can be slow, use one of the other constructors if possible
    //
    size_t _rows = _vals.size();
    size_t _cols = _vals.at(0).size();

    this->resize_rows(_rows);
    this->resize_cols(_cols);
    this->zero_val = _zero_val;

    // Construct the bm::sparse_vector from _vals's internal arrays
    this->vals.resize((size_t)(_rows*_cols));
    for (size_t i = 0; i < _rows; ++i) {
	for (size_t j = 0; j < _cols; ++j) {
	    if (!nearly_equal<T>(_vals[i][j], _zero_val)) {
		this->vals.set(i*_cols + j, _vals[i][j]);
	    }
	}
    }
}

// Copy constructor from contiguous 2D vector
template <typename T>
SparseIntegerTypeMatrix<T>::SparseIntegerTypeMatrix(const std::vector<T> &_vals, const size_t _rows, const size_t _cols, const T _zero_val) {
    // Initializes a SparseMatrix from a contiguous vector _vals.
    //
    this->resize_rows(_rows);
    this->resize_cols(_cols);
    this->zero_val = _zero_val;

    // Construct the bm::sparse_vector from _vals's internal array
    T* arr_start = const_cast<T*>(&_vals[0]);
    this->vals.import(arr_start, (size_t)(_rows*_cols));
}

// Access individual elements
template <typename T>
T& SparseIntegerTypeMatrix<T>::operator()(size_t row, size_t col) {
    size_t address = row*this->get_cols() + col;
    const T& out = (this->vals[address] != this->zero_val ? this->vals[address] : this->zero_val);
    return const_cast<T&>(out);
}

// Access individual elements (const)
template <typename T>
const T& SparseIntegerTypeMatrix<T>::operator()(size_t row, size_t col) const {
    size_t address = row*this->get_cols() + col;
    const T& out = (this->vals[address] != this->zero_val ? this->vals[address] : this->zero_val);
    return out;
}

// TODO implement sparse matrix operators
// see https://www.geeksforgeeks.org/operations-sparse-matrices/ for reference

// In-place matrix-matrix addition
template<typename T>
SparseIntegerTypeMatrix<T>& SparseIntegerTypeMatrix<T>::operator+=(const Matrix<T>& rhs) {
    // seamat::SparseIntegerTypeMatrix<T>::operator+=
    //
    // Add values from rhs to the calling matrix in-place.
    //
    //   Input:
    //     `rhs`: Matrix to add, must have the same dimensions as the caller.
    //
    throw std::runtime_error("Sparse matrix operators have not been implemented.");
    return *this;
}

// In-place matrix-matrix subtraction
template<typename T>
SparseIntegerTypeMatrix<T>& SparseIntegerTypeMatrix<T>::operator-=(const Matrix<T>& rhs) {
    // seamat::SparseIntegerTypeMatrix<T>::operator-=
    //
    // Subtract values of rhs from the calling matrix in-place.
    //
    //   Input:
    //     `rhs`: Matrix to subtract, must have the same dimensions as the caller.
    //
    throw std::runtime_error("Sparse matrix operators have not been implemented.");

    return *this;
}

// In-place right multiplication
template<typename T>
SparseIntegerTypeMatrix<T>& SparseIntegerTypeMatrix<T>::operator*=(const Matrix<T>& rhs) {
    // seamat::SparseIntegerTypeMatrix<T>::operator*=
    //
    // Matrix right-multiplication of the caller with rhs in-place.
    //
    //   Input:
    //     `rhs`: Matrix to right multiply with,
    //            must have the same number of rows as the caller has columns.
    //
    throw std::runtime_error("Sparse matrix operators have not been implemented.");
    return *this;
}

// In-place left multiplication
template<typename T>
SparseIntegerTypeMatrix<T>& SparseIntegerTypeMatrix<T>::operator%=(const Matrix<T>& lhs) {
    // seamat::SparseIntegerTypeMatrix<T>::operator%=
    //
    // Matrix left-multiplication of the caller with lhs in-place.
    //
    //   Input:
    //     `lhs`: Matrix to left multiply with,
    //            must have the same number of columns as the caller has rows.
    //
    throw std::runtime_error("Sparse matrix operators have not been implemented.");
    return *this;
}

// In-place matrix-scalar addition
template<typename T>
SparseIntegerTypeMatrix<T>& SparseIntegerTypeMatrix<T>::operator+=(const T& scalar) {
    // seamat::SparseIntegerTypeMatrix<T>::operator+=
    //
    // In-place addition of a scalar to caller.
    //
    //   Input:
    //     `scalar`: Scalar value to add to all caller values.
    //
    throw std::runtime_error("Sparse matrix operators have not been implemented.");
    return *this;
}

// In-place matrix-scalar subtraction
template<typename T>
SparseIntegerTypeMatrix<T>& SparseIntegerTypeMatrix<T>::operator-=(const T& scalar) {
    // seamat::SparseIntegerTypeMatrix<T>::operator-=
    //
    // In-place subtraction of a scalar from the caller.
    //
    //   Input:
    //     `scalar`: Scalar value to subtract from all caller values.
    //
    throw std::runtime_error("Sparse matrix operators have not been implemented.");
    return *this;
}

// In-place matrix-scalar multiplication
template<typename T>
SparseIntegerTypeMatrix<T>& SparseIntegerTypeMatrix<T>::operator*=(const T& scalar) {
    // seamat::SparseIntegerTypeMatrix<T>::operator*=
    //
    // In-place multiplication of the caller with a scalar.
    //
    //   Input:
    //     `scalar`: Scalar value to multiply all caller values with.
    //
    // Handle special case where the whole matrix is multiplied by zero and becomes sparse.
    throw std::runtime_error("Sparse matrix operators have not been implemented.");
    return *this;
}

// In-place matrix-scalar division
template<typename T>
SparseIntegerTypeMatrix<T>& SparseIntegerTypeMatrix<T>::operator/=(const T& scalar) {
    // seamat::SparseIntegerTypeMatrix<T>::operator/=
    //
    // In-place division of the caller with a scalar.
    //
    //   Input:
    //     `scalar`: Scalar value to divide all caller values with.
    //
    throw std::runtime_error("Sparse matrix operators have not been implemented.");
    return *this;
}
}

#endif
