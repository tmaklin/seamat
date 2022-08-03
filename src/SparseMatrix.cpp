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
template<typename T>
T* SparseMatrix<T>::get_address(uint32_t row, uint32_t col) {
    // Returns the position of element (i, j) in this->vals
    uint32_t col_start = this->col_ptr[col];
    uint32_t col_end = this->col_ptr[col + 1];
    uint32_t nnz_col = col_end - col_start; // Number of non-zero elements in row `row`.
    if (nnz_col > 0) {
	for (uint32_t i = 0; i < nnz_col; ++i) {
	    uint32_t index = col_start + i;
	    if (this->col_ptr[index] == row) {
		return &this->vals[index];
	    }
	}
    }
    return NULL;
}

template<typename T>
const T* SparseMatrix<T>::get_address(uint32_t row, uint32_t col) const {
    // Returns the position of element (i, j) in this->vals
    uint32_t col_start = this->col_ptr[col];
    uint32_t col_end = this->col_ptr[col + 1];
    uint32_t nnz_col = col_end - col_start; // Number of non-zero elements in row `row`.
    if (nnz_col > 0) {
	for (uint32_t i = 0; i < nnz_col; ++i) {
	    uint32_t index = col_start + i;
	    if (this->col_ptr[index] == row) {
		return &this->vals[index];
	    }
	}
    }
    return NULL;
}

// Parameter constructor
template<typename T>
SparseMatrix<T>::SparseMatrix(uint32_t _rows, uint32_t _cols, const T& _initial) {
    // Initializes a dense SparseMatrix; use
    // SparseMatrix<T>::remove_nonzeros to sparsify the matrix after
    // filling it.
    this->resize_rows(_rows);
    this->resize_cols(_cols);

    uint64_t n_elements = _rows*_cols;
    this->vals.resize(n_elements, _initial);
    this->col_ptr.resize(n_elements, 0);
    this->row_ind.resize(_rows, 0);
    for (uint32_t i = 0; i < _rows; ++i) {
	this->row_ind[i + 1] = this->row_ind[i] + _cols;
	for (uint32_t j = 0; j < _cols; ++j) {
	    this->col_ptr[i + j] = j;
	}
    }
}
    
// Initialize from a DenseMatrix
template<typename T>
SparseMatrix<T>::SparseMatrix(const Matrix<T> &_vals, const T& _zero_val) {
    this->zero_val = _zero_val;
    this->resize_rows(_vals.get_rows());
    this->resize_cols(_vals.get_cols());

    this->row_ind.resize(this->get_rows() + 1, 0);
    this->col_ptr.resize(this->get_rows()*this->get_cols(), 0);
    this->vals.resize(this->get_rows()*this->get_cols(), 0);

    for (uint32_t i = 0; i < this->get_rows(); ++i) {
	this->row_ind[i + 1] = this->row_ind[i];
	for (uint32_t j = 0; j < this->get_cols(); ++j) {
	    if (_vals(i, j) != this->zero_val) { // todo: use std::nextafter for floating point comparisons
		++this->row_ind[i + 1];
		this->col_ptr[i*this->get_cols() + j] = j;
		this->vals[i*this->get_cols() + j] = _vals(i, j);
	    }
	}
    }
}

// Initialize from a 2D vector
template<typename T>
SparseMatrix<T>::SparseMatrix(const std::vector<std::vector<T>> &rhs, const T& _zero_val) {
    this->resize_rows(rhs.size());
    this->resize_cols(rhs.at(0).size());

    uint64_t n_nonzero_elem = 0;
    for (uint32_t i = 0; i < this->get_rows(); ++i) {
	for (uint32_t j = 0; j < this->get_cols(); ++j) {
	    if (rhs[i][j] != this->zero_val) { // todo: floating point comparisons
		++n_nonzero_elem;
	    }
	}
    }

    this->row_ind.resize(this->get_rows() + 1, 0);
    this->col_ptr.resize(n_nonzero_elem, 0);
    this->vals.resize(n_nonzero_elem, _zero_val);

    uint32_t index = 0;
    for (uint32_t i = 0; i < this->get_rows(); ++i) {
	this->row_ind[i + 1] = this->row_ind[i];
	for (uint32_t j = 0; j < this->get_cols(); ++j) {
	    if (rhs[i][j] != this->zero_val) { // todo: use std::nextafter for floating point comparisons
		++this->row_ind[i + 1];
		this->col_ptr[index] = j;
		this->vals[index] = rhs[i][j];
		++index;
	    }
	}
    }
}

// Copy constructor from contiguous 2D vector
template <typename T>
SparseMatrix<T>::SparseMatrix(const std::vector<T> &rhs, const uint32_t _rows, const uint32_t _cols, const T& _zero_val) {
    this->resize_rows(_rows);
    this->resize_cols(_cols);
    this->zero_val = _zero_val;

    uint64_t n_nonzero_elem = 0;
    for (uint32_t i = 0; i < _rows; ++i) {
	for (uint32_t j = 0; j < _cols; ++j) {
	    if (!this->nearly_equal(rhs[i*_cols + j], this->zero_val)) {
		++n_nonzero_elem;
	    }
	}
    }

    this->row_ind.resize(_rows + 1, 0);
    this->col_ptr.resize(n_nonzero_elem, 0);
    this->vals.resize(n_nonzero_elem, _zero_val);

    uint32_t index = 0;
    for (uint32_t i = 0; i < this->get_rows(); ++i) {
	this->row_ind[i + 1] = this->row_ind[i];
	for (uint32_t j = 0; j < this->get_cols(); ++j) {
	    const T &rhs_val = rhs[i*_cols + j];
	    if (!this->nearly_equal(rhs_val, this->zero_val)) {
		++this->row_ind[i + 1];
		this->col_ptr[index] = j;
		this->vals[index] = rhs_val;
		++index;
	    }
	}
    }
}

// Resize a matrix
template<typename T>
void SparseMatrix<T>::resize(const uint32_t new_rows, const uint32_t new_cols, const T initial) {
    throw std::runtime_error("Resizing a sparse matrix is not supported.");
}

// Access individual elements
template <typename T>
T& SparseMatrix<T>::operator()(uint32_t row, uint32_t col) {
    T* address = this->get_address(row, col);
    if (address == NULL) {
	return this->zero_val;
    }
    return *address;
}

// Access individual elements (const)
template <typename T>
const T& SparseMatrix<T>::operator()(uint32_t row, uint32_t col) const {
    const T* address = this->get_address(row, col);
    if (address == NULL) {
	return this->zero_val;
    }
    return *address;
}
}

#endif
