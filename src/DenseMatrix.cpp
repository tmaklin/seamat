// seamat: templatized matrix library
// https://github.com/tmaklin/seamat
//
// Copyright (C) 2021 Tommi Mäklin (tommi@maklin.fi)
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
#ifndef SEAMAT_DENSE_MATRIX_CPP
#define SEAMAT_DENSE_MATRIX_CPP
#include "Matrix.hpp"

#include <cmath>

#include "openmp_config.hpp"

namespace seamat {
// Parameter Constructor
template<typename T>
DenseMatrix<T>::DenseMatrix(uint32_t _rows, uint32_t _cols, const T& _initial) {
    mat.resize(_rows*_cols, _initial);
    this->resize_rows(_rows);
    this->resize_cols(_cols);
}

// Copy constructor from contiguous 2D vector
template <typename T>
DenseMatrix<T>::DenseMatrix(const std::vector<T> &rhs, const uint32_t _rows, const uint32_t _cols) {
    mat = rhs;
    this->resize_rows(_rows);
    this->resize_cols(_cols);
}

// Copy constructor from 2D vector
template<typename T>
DenseMatrix<T>::DenseMatrix(const std::vector<std::vector<T>> &rhs) {
    this->resize_rows(rhs.size());
    this->resize_cols(rhs.at(0).size());
    mat.resize(this->get_rows()*this->get_cols());
#pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < this->get_rows(); ++i) {
	for (uint32_t j = 0; j < this->get_cols(); ++j) {
	    this->operator()(i, j) = rhs[i][j];
	}
    }
}

// Copy constructor from another matrix
template<typename T>
DenseMatrix<T>::DenseMatrix(const Matrix<T> &rhs) {
    this->resize_rows(rhs.get_rows());
    this->resize_cols(rhs.get_cols());
    mat.resize(this->get_rows()*this->get_cols());
#pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < this->get_rows(); ++i) {
	for (uint32_t j = 0; j < this->get_cols(); ++j) {
	    this->operator()(i, j) = rhs(i, j);
	}
    }
}

// Assignment Operator
template<typename T>
DenseMatrix<T>& DenseMatrix<T>::operator=(const Matrix<T>& rhs) {
    if (&rhs == this)
	return *this;

    uint32_t new_rows = rhs.get_rows();
    uint32_t new_cols = rhs.get_cols();
    if (new_rows != this->get_rows() || new_cols != this->get_cols()) {
	resize(new_rows, new_cols, (T)0);
    }
#pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < new_rows; i++) {
	for (uint32_t j = 0; j < new_cols; j++) {
	    this->operator()(i, j) = rhs(i, j);
	}
    }
    return *this;
}

// Resize a matrix
template<typename T>
void DenseMatrix<T>::resize(const uint32_t new_rows, const uint32_t new_cols, const T initial) {
    if (new_rows != this->get_rows() || new_cols != this->get_cols()) {
	mat.resize(new_rows*new_cols, initial);
	this->resize_rows(new_rows);
	this->resize_cols(new_cols);
    }
}

// Access individual elements
template <typename T>
T& DenseMatrix<T>::operator()(uint32_t row, uint32_t col) {
    return this->mat[row*this->get_cols() + col];
}

// Access individual elements (const)
template <typename T>
const T& DenseMatrix<T>::operator()(uint32_t row, uint32_t col) const {
    return this->mat[row*this->get_cols() + col];
}
}

#endif
