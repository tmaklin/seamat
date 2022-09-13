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

#include <cmath>
#include <stdexcept>

#include "Matrix.hpp"
#include "openmp_config.hpp"
#include "math_util.hpp"

namespace seamat {
    template <typename T> class SparseMatrix : public Matrix<T> {
private:
    // Sparse matrix implemented in the compressed row storage (CRS) format.
    // See link below for reference.
    // https://netlib.org/linalg/html_templates/node91.html#SECTION00931100000000000000
    //
    std::vector<T> vals;
    std::vector<size_t> row_ptr;
    std::vector<size_t> col_ind;
    T zero_val;

    // Internal helper functions
    //
    // Get pointer to an element
    T* get_address(size_t row, size_t col);
    const T* get_address(size_t row, size_t col) const;
    // For constructors:
    // Insert a value to the COO list if it's not a zero
    void coo_insert(const T& _val, const size_t row, const size_t col, std::vector<size_t> *row_ind);
    void row_ind_to_row_ptr(const std::vector<size_t> &row_ind);

public:
    SparseMatrix() = default;
    ~SparseMatrix() = default;

    // Parameter constructor, initialize an empty SparseMatrix
    SparseMatrix(const size_t _rows, const size_t _cols, const T _zero_val);

    // (Copy) constructors from existing objects, these use the coordinate list (COO)
    // format internally to build the matrix and then convert back to CRS.
    //
    // Construct from another matrix object
    SparseMatrix(const Matrix<T> &_vals, const T _zero_val = (T)0);
    // Construct from a 2D vector (vector of vectors)
    SparseMatrix(const std::vector<std::vector<T>> &rhs, const T _zero_val = (T)0);
    // Construct from a contiguously stored 2D vector (vector with known dimensions)
    SparseMatrix(const std::vector<T> &rhs, const size_t _rows, const size_t _cols, const T _zero_val = (T)0);

    // Access individual elements
    T& operator()(size_t row, size_t col) override;
    const T& operator()(size_t row, size_t col) const override;

    // Mathematical operators
    // Matrix-matrix in-place summation and subtraction
    SparseMatrix<T>& operator+=(const Matrix<T>& rhs) override;
    SparseMatrix<T>& operator-=(const Matrix<T>& rhs) override;

    // In-place right multiplication
    SparseMatrix<T>& operator*=(const Matrix<T>& rhs) override;
    // In-place left multiplication
    SparseMatrix<T>& operator%=(const Matrix<T>& rhs) override;

    // Matrix-scalar, in-place
    SparseMatrix<T>& operator+=(const T& rhs) override;
    SparseMatrix<T>& operator-=(const T& rhs) override;
    SparseMatrix<T>& operator*=(const T& rhs) override;
    SparseMatrix<T>& operator/=(const T& rhs) override;
};

template<typename T>
T* SparseMatrix<T>::get_address(size_t row, size_t col) {
    // Returns the position of element (i, j) in this->vals
    size_t row_start = this->row_ptr[row];
    size_t row_end = this->row_ptr[row + 1];
    size_t nnz_row = row_end - row_start; // Number of non-zero elements in row `row`.
    if (nnz_row > 0) {
	for (size_t i = 0; i < nnz_row; ++i) {
	    size_t index = row_start + i;
	    if (this->col_ind[index] == col) {
		return &this->vals[index];
	    }
	}
    }
    return NULL;
}

template<typename T>
const T* SparseMatrix<T>::get_address(size_t row, size_t col) const {
    // Returns the position of element (i, j) in this->vals
    size_t row_start = this->row_ptr[row];
    size_t row_end = this->row_ptr[row + 1];
    size_t nnz_row = row_end - row_start; // Number of non-zero elements in row `row`.
    if (nnz_row > 0) {
	for (size_t i = 0; i < nnz_row; ++i) {
	    size_t index = row_start + i;
	    if (this->col_ind[index] == col) {
		return &this->vals[index];
	    }
	}
    }
    return NULL;
}

template <typename T>
void SparseMatrix<T>::coo_insert(const T& _val, const size_t row, const size_t col, std::vector<size_t> *row_ind) {
    if (!nearly_equal<T>(_val, this->zero_val)) {
	this->vals.emplace_back(_val);
	row_ind->emplace_back(row);
	this->col_ind.emplace_back(col);
    }
}

template <typename T>
void SparseMatrix<T>::row_ind_to_row_ptr(const std::vector<size_t> &row_ind) {
    this->row_ptr.clear(); // Clear the row_ptr just in case.
    this->row_ptr.resize(this->get_rows() + 1, 0);
    size_t current_row_id = 0;
    for (size_t i = 0; i < row_ind.size(); ++i) {
      if (row_ind[i] != current_row_id) {
	++current_row_id;
	row_ptr[current_row_id + 1] = row_ptr[current_row_id];
      }
      ++row_ptr[current_row_id + 1];
    }
}


// Parameter constructor, initialize an empty SparseMatrix
template<typename T>
SparseMatrix<T>::SparseMatrix(const size_t _rows, const size_t _cols, const T _zero_val) {
    // Initializes an empty SparseMatrix (ie filled with _zero_val).
    // Input:
    //   _rows: Number of rows in the matrix.
    //   _col: Number of cols in the matrix.
    //   _zero_val: The "zero-value" to return when the matrix does
    //              not contain an inserted value at the position.
    //
    this->resize_rows(_rows);
    this->resize_cols(_cols);
    this->zero_val = _zero_val;

    this->vals = std::vector<T>(0, _zero_val);
    this->col_ind.resize(0, 0);
    this->row_ptr.resize(_rows + 1, 0); // _rows but they contain no elements
}

// Construct from another matrix object
template<typename T>
SparseMatrix<T>::SparseMatrix(const Matrix<T> &_vals, const T _zero_val) {
    // Initializes a SparseMatrix containing the values in the input Matrix
    // Input:
    //   _vals: an arbitrary Matrix<T> object that supports operator() for accessing values.
    //   _zero_val: which value to consider as the "zero" in _vals.
    //
    size_t _rows = _vals.get_rows();
    size_t _cols = _vals.get_cols();

    this->resize_rows(_rows);
    this->resize_cols(_cols);
    this->zero_val = _zero_val;

    // Construct in COO format
    std::vector<size_t> row_ind; // Temporary for storing the row indices
    for (size_t i = 0; i < _rows; ++i) {
	for (size_t j = 0; j < _cols; ++j) {
	    this->coo_insert(_vals(i, j), i, j, &row_ind);
	}
    }

    // Convert to CRS format by creating the row_ptr vector
    this->row_ind_to_row_ptr(row_ind);
}

// Construct from a 2D vector (vector of vectors)
template<typename T>
SparseMatrix<T>::SparseMatrix(const std::vector<std::vector<T>> &_vals, const T _zero_val) {
    // Initializes a SparseMatrix containing the values in the input 2D vector
    // Input:
    //   _vals: a 2D vector containing the values (first dimension rows, second cols).
    //   _zero_val: which value to consider as the "zero" in _vals.
    // TODO:
    //   - Implement bounds checking (see DenseMatrix.hpp for how-to).
    //
    size_t _rows = _vals.size();
    size_t _cols = _vals.at(0).size(); // Use at() here to throw a descriptive error message if _vals is uninitialized

    this->resize_rows(_rows);
    this->resize_cols(_cols);
    this->zero_val = _zero_val;

    // Construct in COO format
    std::vector<size_t> row_ind; // Temporary for storing the row indices
    for (size_t i = 0; i < _rows; ++i) {
	for (size_t j = 0; j < _cols; ++j) {
	    this->coo_insert(_vals[i][j], i, j, &row_ind); // Use [][] and don't check bounds (faster).
	}
    }

    // Convert to CRS format by creating the row_ptr vector
    this->row_ind_to_row_ptr(row_ind);
}

// Construct from a contiguously stored 2D vector (vector with known dimensions)
template <typename T>
SparseMatrix<T>::SparseMatrix(const std::vector<T> &_vals, const size_t _rows, const size_t _cols, const T _zero_val) {
    // Initializes a SparseMatrix containing the values in the input vector
    // Input:
    //   _vals: a _rows*_cols vector containing the values.
    //   _rows: number of rows in the contiguously stored 2D object.
    //   _cols: number of columns in the 2D object.
    //   _zero_val: which value to consider as the "zero" in _vals.
    // TODO:
    //   - Implement bounds checking (see DenseMatrix.hpp for how-to).
    //   - Check if the parallel implementation is faster.
    //
    this->resize_rows(_rows);
    this->resize_cols(_cols);
    this->zero_val = _zero_val;

    // Construct in COO format
    std::vector<size_t> row_ind; // Temporary for storing the row indices
    for (size_t i = 0; i < _rows; ++i) {
	for (size_t j = 0; j < _cols; ++j) {
	    size_t address = i*_cols + j;
	    this->coo_insert(_vals[address], i, j, &row_ind); // Use [] and don't check bounds (faster).
	}
    }

    // Convert to CRS format by creating the row_ptr vector
    this->row_ind_to_row_ptr(row_ind);

    // Parallel implementation below, TODO test if its faster.
    //     // TODO wrap with OpenMP support guard and use globals
    //     size_t n_threads;
    // #pragma omp parallel
    //     {
    //     n_threads = omp_get_num_threads();
    //     }

    //     std::vector<std::vector<size_t>> my_row_ind(n_threads, std::vector<size_t>());
    //     std::vector<std::vector<size_t>> my_col_ind(n_threads, std::vector<size_t>());
    //     std::vector<std::vector<T>> my_vals(n_threads, std::vector<T>());
    // #pragma omp parallel for schedule(static)
    //     for (size_t i = 0; i < _rows; ++i) {
    //       size_t rank = omp_get_thread_num(); // TODO wrap for support checking
    //       for (size_t j = 0; j < _cols; ++j) {
    // 	  size_t address = i*_cols + j;
    // 	    if (!nearly_equal<T>(rhs[address], this->zero_val)) {
    // 		my_vals[rank].emplace_back(rhs[address]);
    // 		my_row_ind[rank].emplace_back(i);
    // 		my_col_ind[rank].emplace_back(j);
    // 	    }
    // 	}
    //     }

    //     vals = std::move(my_vals[0]);
    //     col_ind = std::move(my_col_ind[0]);
    //     std::vector<size_t> row_ind = std::move(my_row_ind[0]);
    //     for (size_t t = 1; t < n_threads; ++t) {
    //       vals.insert(vals.end(), std::make_move_iterator(my_vals[t].begin()), std::make_move_iterator(my_vals[t].end()));
    //       col_ind.insert(col_ind.end(), std::make_move_iterator(my_col_ind[t].begin()), std::make_move_iterator(my_col_ind[t].end()));
    //       row_ind.insert(row_ind.end(), std::make_move_iterator(my_row_ind[t].begin()), std::make_move_iterator(my_row_ind[t].end()));
    //     }

    //     row_ptr.resize(_rows + 1, 0);
    //     size_t current_row_id = 0;
    //     for (size_t i = 0; i < row_ind.size(); ++i) {
    //       if (row_ind[i] != current_row_id) {
    // 	++current_row_id;
    // 	row_ptr[current_row_id + 1] = row_ptr[current_row_id];
    //       }
    //       ++row_ptr[current_row_id + 1];
    //     }
}

// Access individual elements
template <typename T>
T& SparseMatrix<T>::operator()(size_t row, size_t col) {
    T* address = this->get_address(row, col);
    if (address == NULL) {
	return this->zero_val;
    }
    return *address;
}

// Access individual elements (const)
template <typename T>
const T& SparseMatrix<T>::operator()(size_t row, size_t col) const {
    const T* address = this->get_address(row, col);
    if (address == NULL) {
	return this->zero_val;
    }
    return *address;
}


// TODO implement sparse matrix operators
// see https://www.geeksforgeeks.org/operations-sparse-matrices/ for reference

// In-place SparseMatrix-SparseMatrix addition
template<typename T>
SparseMatrix<T>& SparseMatrix<T>::operator+=(const Matrix<T>& rhs) {
    // seamat::SparseMatrix<T>::operator+=
    //
    // Add values from rhs to the calling matrix in-place.
    //
    //   Input:
    //     `rhs`: Matrix to add, must have the same dimensions as the caller.
    //   TODO:
    //     tests
    //
    std::vector<T> new_vals;
    std::vector<size_t> new_row_ind;
    std::vector<size_t> new_col_ind;

    for (size_t i = 0; i < this->get_rows(); ++i) {
	for (size_t j = 0; j < this->get_cols(); ++j) {
	    T new_val;
	    bool lhs_is_zero = nearly_equal<T>(this->operator()(i, j), this->zero_val);
	    bool rhs_is_zero = nearly_equal<T>(rhs(i, j), this->zero_val);
	    if (!lhs_is_zero && !rhs_is_zero) {
		new_val = this->operator()(i, j) + rhs(i, j);
	    } else if (!rhs_is_zero) {
		new_val = rhs(i, j);
	    } else {
		new_val = this->operator()(i, j);
	    }
	    if (!lhs_is_zero || !rhs_is_zero) {
		new_vals.emplace_back(new_val);
		new_row_ind.emplace_back(i);
		new_col_ind.emplace_back(j);
	    }
	}
    }
    this->vals = std::move(new_vals);
    this->col_ind = std::move(new_col_ind);
    this->row_ind_to_row_ptr(new_row_ind);

    return *this;
}

// In-place matrix-matrix subtraction
template<typename T>
SparseMatrix<T>& SparseMatrix<T>::operator-=(const Matrix<T>& rhs) {
    // seamat::SparseMatrix<T>::operator-=
    //
    // Subtract values of rhs from the calling matrix in-place.
    //
    //   Input:
    //     `rhs`: Matrix to subtract, must have the same dimensions as the caller.
    //
    //   TODO:
    //     tests
    //
    std::vector<T> new_vals;
    std::vector<size_t> new_row_ind;
    std::vector<size_t> new_col_ind;

    for (size_t i = 0; i < this->get_rows(); ++i) {
	for (size_t j = 0; j < this->get_cols(); ++j) {
	    T new_val;
	    bool lhs_is_zero = nearly_equal<T>(this->operator()(i, j), this->zero_val);
	    bool rhs_is_zero = nearly_equal<T>(rhs(i, j), this->zero_val);
	    if (!lhs_is_zero && !rhs_is_zero) {
		new_val = this->operator()(i, j) - rhs(i, j);
	    } else if (!rhs_is_zero) {
		new_val = -rhs(i, j);
	    } else {
		new_val = this->operator()(i, j);
	    }
	    if (!lhs_is_zero || !rhs_is_zero) {
		new_vals.emplace_back(new_val);
		new_row_ind.emplace_back(i);
		new_col_ind.emplace_back(j);
	    }
	}
    }
    this->vals = std::move(new_vals);
    this->col_ind = std::move(new_col_ind);
    this->row_ind_to_row_ptr(new_row_ind);

    return *this;
}

// In-place right multiplication
template<typename T>
SparseMatrix<T>& SparseMatrix<T>::operator*=(const Matrix<T>& rhs) {
    // seamat::SparseMatrix<T>::operator*=
    //
    // Matrix right-multiplication of the caller with rhs in-place.
    //
    //   Input:
    //     `rhs`: Matrix to right multiply with,
    //            must have the same number of rows as the caller has columns.
    //
    throw std::runtime_error("SparseMatrix-Matrix in-place multiplication is not implemented.");

    return *this;
}

// In-place left multiplication
template<typename T>
SparseMatrix<T>& SparseMatrix<T>::operator%=(const Matrix<T>& lhs) {
    // seamat::SparseMatrix<T>::operator%=
    //
    // Matrix left-multiplication of the caller with lhs in-place.
    //
    //   Input:
    //     `lhs`: Matrix to left multiply with,
    //            must have the same number of columns as the caller has rows.
    //
    throw std::runtime_error("Matrix-SparseMatrix in-place multiplication is not implemented.");

    return *this;
}

// In-place matrix-scalar addition
template<typename T>
SparseMatrix<T>& SparseMatrix<T>::operator+=(const T& scalar) {
    // seamat::SparseMatrix<T>::operator+=
    //
    // In-place addition of a scalar to caller.
    //
    //   Input:
    //     `scalar`: Scalar value to add to all caller values.
    //
    this->zero_val += scalar;
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < this->vals.size(); ++i) {
	this->vals[i] += scalar;
    }

    return *this;
}

// In-place matrix-scalar subtraction
template<typename T>
SparseMatrix<T>& SparseMatrix<T>::operator-=(const T& scalar) {
    // seamat::SparseMatrix<T>::operator-=
    //
    // In-place subtraction of a scalar from the caller.
    //
    //   Input:
    //     `scalar`: Scalar value to subtract from all caller values.
    //
    this->zero_val -= scalar;
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < this->vals.size(); ++i) {
	this->vals[i] -= scalar;
    }

    return *this;
}

// In-place matrix-scalar multiplication
template<typename T>
SparseMatrix<T>& SparseMatrix<T>::operator*=(const T& scalar) {
    // seamat::SparseMatrix<T>::operator*=
    //
    // In-place multiplication of the caller with a scalar.
    //
    //   Input:
    //     `scalar`: Scalar value to multiply all caller values with.
    //
    // Handle special case where the whole matrix is multiplied by zero and becomes sparse.
    if (nearly_equal<T>(scalar, (T)0)) {
	this->vals.clear();
	this->vals.shrink_to_fit();
	this->row_ptr.clear(); // Remove contents
	this->row_ptr.resize(this->get_rows() + 1, 0); // This matrix now contains zero non-zero elements.
	this->col_ind.clear();
	this->col_ind.shrink_to_fit();
	this->zero_val = (T)0;
    } else {
	this->zero_val *= scalar;
#pragma omp parallel for schedule(static)
	for (size_t i = 0; i < this->vals.size(); ++i) {
	    this->vals[i] *= scalar;
	}
    }

    return *this;
}

// In-place matrix-scalar division
template<typename T>
SparseMatrix<T>& SparseMatrix<T>::operator/=(const T& scalar) {
    // seamat::SparseMatrix<T>::operator/=
    //
    // In-place division of the caller with a scalar.
    //
    //   Input:
    //     `scalar`: Scalar value to divide all caller values with.
    //
    if (nearly_equal<T>(scalar, (T)0))
	throw std::runtime_error("Math error: attempt to divide by zero.");

    this->zero_val /= scalar;
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < this->vals.size(); ++i) {
	this->vals[i] /= scalar;
    }

    return *this;
}
}

#endif
