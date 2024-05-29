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

#include <cmath>
#include <stdexcept>
#include <memory>
#include <typeinfo>

#include "Matrix.hpp"
#include "seamat_openmp_config.hpp"
#include "math_util.hpp"

namespace seamat {
template <typename T, typename U,
	  template <typename> class ValuesMatrix, // Should be of type Matrix<T>
	  template <typename> class IndicesMatrix> // Matrix<U>
class IndexMatrix : public Matrix<T> {
private:
    // Dimensions of the `vals` member variable.
    //
    size_t n_rows_vals;
    size_t n_cols_vals;

    // Store the vals and indices as a pointer so they can be sparse or
    // dense as needed. Shared_ptr used here because multiple
    // IndexMatrices may share the same vals and/or indices.
    //
    std::shared_ptr<ValuesMatrix<T>> vals;
    std::shared_ptr<IndicesMatrix<U>> indices;

    // Helper for setting the dimensions in constructors
    void resize_self(const size_t _rows, const size_t _cols, const size_t _rows_vals, const size_t _cols_vals);

public:
    // Default constructor
    IndexMatrix();
    ~IndexMatrix() = default;

    // Copy constructor from Matrix _vals and Matrix _indices
    IndexMatrix(const ValuesMatrix<T> &_vals, const IndicesMatrix<U> &_indices);

    // Move constructor from Matrix _vals and Matrix _indices
    IndexMatrix(ValuesMatrix<T> &&_vals, IndicesMatrix<U> &&_indices);

    // Copy constructor from Matrix _vals and contiguously stored matrix _indices
    IndexMatrix(const ValuesMatrix<T> &_vals, const std::vector<U> &_indices, const size_t _n_rows, const size_t _n_cols);
    // Copy constructor from contiguously stored matrix _vals and Matrix _indices
    IndexMatrix(const std::vector<T> &_vals, const IndicesMatrix<U> &_indices, const size_t _n_rows_vals, const size_t _n_cols_vals);
    // Copy constructor from contiguously stored matrix _vals and _indices
    IndexMatrix(const std::vector<T> &_vals, const std::vector<U> &_indices,
		const size_t _n_rows_vals, const size_t _n_cols_vals,
		const size_t _n_rows, const size_t _n_cols);

    // Access individual elements
    T& operator()(size_t row, size_t col) override;
    const T& operator()(size_t row, size_t col) const override;

    // Mathematical operators
    // Matrix-matrix in-place summation and subtraction
    IndexMatrix<T, U, ValuesMatrix, IndicesMatrix>& operator+=(const Matrix<T>& rhs) override;
    IndexMatrix<T, U, ValuesMatrix, IndicesMatrix>& operator-=(const Matrix<T>& rhs) override;

    // In-place right multiplication
    IndexMatrix<T, U, ValuesMatrix, IndicesMatrix>& operator*=(const Matrix<T>& rhs) override;
    // In-place left multiplication
    IndexMatrix<T, U, ValuesMatrix, IndicesMatrix>& operator%=(const Matrix<T>& rhs) override;

    // Matrix-scalar, in-place
    IndexMatrix<T, U, ValuesMatrix, IndicesMatrix>& operator+=(const T& rhs) override;
    IndexMatrix<T, U, ValuesMatrix, IndicesMatrix>& operator-=(const T& rhs) override;
    IndexMatrix<T, U, ValuesMatrix, IndicesMatrix>& operator*=(const T& rhs) override;
    IndexMatrix<T, U, ValuesMatrix, IndicesMatrix>& operator/=(const T& rhs) override;

};

// Resize the dimensions of the matrix (helper function for constructors)
template <typename T, typename U, template <typename> class ValuesMatrix, template <typename> class IndicesMatrix>
void IndexMatrix<T, U, ValuesMatrix, IndicesMatrix>::resize_self(const size_t _rows, const size_t _cols, const size_t _rows_vals, const size_t _cols_vals) {
    // Resizes the dimensions of both vals and indices (private)
    // Helper function for constructors
    //
    this->resize_rows(_rows);
    this->resize_cols(_cols);
    this->n_rows_vals = _rows_vals;
    this->n_cols_vals = _cols_vals;
}

// Access individual elements
template <typename T, typename U, template <typename> class ValuesMatrix, template <typename> class IndicesMatrix>
T& IndexMatrix<T, U, ValuesMatrix, IndicesMatrix>::operator()(size_t row, size_t col) {
    size_t out_col = this->indices->operator()(row, col);
    return this->vals->operator()(row, out_col);
}

// Access individual elements (const)
template <typename T, typename U, template <typename> class ValuesMatrix, template <typename> class IndicesMatrix>
const T& IndexMatrix<T, U, ValuesMatrix, IndicesMatrix>::operator()(size_t row, size_t col) const {
    size_t out_col = this->indices->operator()(row, col);
    return this->vals->operator()(row, out_col);
}

//////
// Constructor definitions
//
/// Default constructor
template <typename T, typename U, template <typename> class ValuesMatrix, template <typename> class IndicesMatrix>
IndexMatrix<T, U, ValuesMatrix, IndicesMatrix>::IndexMatrix() {
    // Default constructor initializes both vals and indices as empty 0x0 matrices.
    //
    this->resize_self(0, 0, 0, 0);

    this->vals = std::make_shared<ValuesMatrix<T>>(0, 0, (T)0);
    this->indices = std::make_shared<IndicesMatrix<U>>(0, 0, (U)0);
}

////
// Copy constructors
// (There's got to be a better way to do this than a separate constructor for every derived Matrix class??
//  but make_shared doesn't work for abstract type and reset doesn't copy the object...)
//
/// Copy constructor from Matrix _vals and Matrix _indices
template <typename T, typename U, template <typename> class ValuesMatrix, template <typename> class IndicesMatrix>
IndexMatrix<T, U, ValuesMatrix, IndicesMatrix>::IndexMatrix(const ValuesMatrix<T> &_vals, const IndicesMatrix<U> &_indices) {
    this->resize_self(_indices.get_rows(), _indices.get_cols(), _vals.get_rows(), _vals.get_cols());

    this->vals = std::make_shared<ValuesMatrix<T>>(_vals);
    this->indices = std::make_shared<IndicesMatrix<U>>(_indices);
}

/// Copy constructor from Matrix _vals, contiguously stored matrix _indices
template <typename T, typename U, template <typename> class ValuesMatrix, template <typename> class IndicesMatrix>
IndexMatrix<T, U, ValuesMatrix, IndicesMatrix>::IndexMatrix(const ValuesMatrix<T> &_vals, const std::vector<U> &_indices, size_t _n_rows, size_t _n_cols) {
    this->resize_self(_n_rows, _n_cols, _vals.get_rows(), _vals.get_cols());

    this->vals = std::make_shared<ValuesMatrix<T>>(_vals);
    this->indices = std::make_shared<IndicesMatrix<U>>(_indices, _n_rows, _n_cols);
}

/// Copy constructor from contiguously stored matrix _vals, Matrix _indices
template <typename T, typename U, template <typename> class ValuesMatrix, template <typename> class IndicesMatrix>
IndexMatrix<T, U, ValuesMatrix, IndicesMatrix>::IndexMatrix(const std::vector<T> &_vals, const IndicesMatrix<U> &_indices, size_t _n_rows_vals, size_t _n_cols_vals) {
    this->resize_self(_indices.get_rows(), _indices.get_cols(), _n_rows_vals, _n_cols_vals);

    this->vals = std::make_shared<ValuesMatrix<T>>(_vals, _n_rows_vals, _n_cols_vals);
    this->indices = std::make_shared<IndicesMatrix<U>>(_indices);
}

/// Copy constructor from contiguously stored matrices _vals, _indices
template <typename T, typename U, template <typename> class ValuesMatrix, template <typename> class IndicesMatrix>
IndexMatrix<T, U, ValuesMatrix, IndicesMatrix>::IndexMatrix(const std::vector<T> &_vals, const std::vector<U> &_indices,
				   const size_t _n_rows_vals, const size_t _n_cols_vals,
				   const size_t _n_rows, const size_t _n_cols) {
    this->resize_self(_n_rows, _n_cols, _n_rows_vals, _n_cols_vals);

    this->vals = std::make_shared<ValuesMatrix<T>>(_vals, _n_rows_vals, _n_cols_vals);
    this->indices = std::make_shared<IndicesMatrix<U>>(_indices, _n_rows, _n_cols);
}

////
// Move constructors
/// Move constructor from Matrix _vals and Matrix _indices
template <typename T, typename U, template <typename> class ValuesMatrix, template <typename> class IndicesMatrix>
IndexMatrix<T, U, ValuesMatrix, IndicesMatrix>::IndexMatrix(ValuesMatrix<T> &&_vals, IndicesMatrix<U> &&_indices) {
    this->resize_self(_indices.get_rows(), _indices.get_cols(), _vals.get_rows(), _vals.get_cols());

    this->vals = std::make_shared<ValuesMatrix<T>>(_vals);
    this->indices = std::make_shared<IndicesMatrix<U>>(_indices);
}

//
// End constructors
//////

// TODO implement index matrix operators

// In-place matrix-matrix addition
template <typename T, typename U, template <typename> class ValuesMatrix, template <typename> class IndicesMatrix>
IndexMatrix<T, U, ValuesMatrix, IndicesMatrix>& IndexMatrix<T, U, ValuesMatrix, IndicesMatrix>::operator+=(const Matrix<T>& rhs) {
    // seamat::IndexMatrix<T, U, ValuesMatrix, IndicesMatrix>::operator+=
    //
    // Add values from rhs to the calling matrix in-place.
    //
    //   Input:
    //     `rhs`: Matrix to add, must have the same dimensions as the caller.
    //
    // Can only in-place sum IndexMatrices together
    if (typeid(rhs) != typeid(*this)) {
	throw std::runtime_error("In-place addition of IndexMatrix `this` with Matrix `rhs` is only defined for `rhs` of type IndexMatrix");
    }

    const IndexMatrix<T, U, ValuesMatrix, IndicesMatrix> *rhs_ptr = static_cast<const IndexMatrix<T, U, ValuesMatrix, IndicesMatrix>*>(&rhs);
    // TODO check that the values set have the same "sizes" in each col
    if (compare_shared_ptr(this->indices, rhs_ptr->indices)) {
	// If the matrices have the same index set and value set with same dimensions (TODO), sum up the value set.
	this->vals = std::make_shared<ValuesMatrix<T>>(*(this->vals) + *(rhs_ptr->vals));
    } else {
	throw std::runtime_error("Index matrix in-place operators are only implemented for matrices sharing the same index set.");
    }
    return *this;
}

// In-place matrix-matrix subtraction
template <typename T, typename U, template <typename> class ValuesMatrix, template <typename> class IndicesMatrix>
IndexMatrix<T, U, ValuesMatrix, IndicesMatrix>& IndexMatrix<T, U, ValuesMatrix, IndicesMatrix>::operator-=(const Matrix<T>& rhs) {
    // seamat::IndexMatrix<T, U, ValuesMatrix, IndicesMatrix>::operator-=
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
template <typename T, typename U, template <typename> class ValuesMatrix, template <typename> class IndicesMatrix>
IndexMatrix<T, U, ValuesMatrix, IndicesMatrix>& IndexMatrix<T, U, ValuesMatrix, IndicesMatrix>::operator*=(const Matrix<T>& rhs) {
    // seamat::IndexMatrix<T, U, ValuesMatrix, IndicesMatrix>::operator*=
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
template <typename T, typename U, template <typename> class ValuesMatrix, template <typename> class IndicesMatrix>
IndexMatrix<T, U, ValuesMatrix, IndicesMatrix>& IndexMatrix<T, U, ValuesMatrix, IndicesMatrix>::operator%=(const Matrix<T>& lhs) {
    // seamat::IndexMatrix<T, U, ValuesMatrix, IndicesMatrix>::operator%=
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
template <typename T, typename U, template <typename> class ValuesMatrix, template <typename> class IndicesMatrix>
IndexMatrix<T, U, ValuesMatrix, IndicesMatrix>& IndexMatrix<T, U, ValuesMatrix, IndicesMatrix>::operator+=(const T& scalar) {
    // seamat::IndexMatrix<T, U, ValuesMatrix, IndicesMatrix>::operator+=
    //
    // In-place addition of a scalar to caller.
    //
    //   Input:
    //     `scalar`: Scalar value to add to all caller values.
    //
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < this->n_rows_vals; ++i) {
	for (size_t j = 0; j < this->n_cols_vals; ++j) {
	    this->vals->operator()(i, j) += scalar;
	}
    }

    return *this;
}

// In-place matrix-scalar subtraction
template <typename T, typename U, template <typename> class ValuesMatrix, template <typename> class IndicesMatrix>
IndexMatrix<T, U, ValuesMatrix, IndicesMatrix>& IndexMatrix<T, U, ValuesMatrix, IndicesMatrix>::operator-=(const T& scalar) {
    // seamat::IndexMatrix<T, U, ValuesMatrix, IndicesMatrix>::operator-=
    //
    // In-place subtraction of a scalar from the caller.
    //
    //   Input:
    //     `scalar`: Scalar value to subtract from all caller values.
    //
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < this->n_rows_vals; ++i) {
	for (size_t j = 0; j < this->n_cols_vals; ++j) {
	    this->vals->operator()(i, j) -= scalar;
	}
    }

    return *this;
}

// In-place matrix-scalar multiplication
template <typename T, typename U, template <typename> class ValuesMatrix, template <typename> class IndicesMatrix>
IndexMatrix<T, U, ValuesMatrix, IndicesMatrix>& IndexMatrix<T, U, ValuesMatrix, IndicesMatrix>::operator*=(const T& scalar) {
    // seamat::IndexMatrix<T, U, ValuesMatrix, IndicesMatrix>::operator*=
    //
    // In-place multiplication of the caller with a scalar.
    //
    //   Input:
    //     `scalar`: Scalar value to multiply all caller values with.
    //
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < this->n_rows_vals; ++i) {
	for (size_t j = 0; j < this->n_cols_vals; ++j) {
	    this->vals->operator()(i, j) *= scalar;
	}
    }

    return *this;
}

// In-place matrix-scalar division
template <typename T, typename U, template <typename> class ValuesMatrix, template <typename> class IndicesMatrix>
IndexMatrix<T, U, ValuesMatrix, IndicesMatrix>& IndexMatrix<T, U, ValuesMatrix, IndicesMatrix>::operator/=(const T& scalar) {
    // seamat::IndexMatrix<T, U, ValuesMatrix, IndicesMatrix>::operator/=
    //
    // In-place division of the caller with a scalar.
    //
    //   Input:
    //     `scalar`: Scalar value to divide all caller values with.
    //
    if (nearly_equal<T>(scalar, (T)0))
	throw std::runtime_error("Math error: attempt to divide by zero.");

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < this->n_rows_vals; ++i) {
	for (size_t j = 0; j < this->n_cols_vals; ++j) {
	    this->vals->operator()(i, j) /= scalar;
	}
    }

    return *this;
}
}

#endif
