// seamat: templatized matrix library
// https://github.com/tmaklin/seamat
//
// Copyright (C) 2021 Tommi Mäklin (tommi@maklin.fi)
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
#ifndef SEAMAT_MATRIX_CPP
#define SEAMAT_MATRIX_CPP
#include "Matrix.hpp"

#include <cmath>

#include "openmp_config.hpp"
#include "seamat_exceptions.hpp"

namespace seamat {
// Matrix-matrix addition
template<typename T>
DenseMatrix<T>& Matrix<T>::operator+(const Matrix<T>& rhs) const {
    // seamat::Matrix<T>::operator+
    //
    // Creates a new DenseMatrix<T> that contains the result of
    // summing two instances of Matrix<T>.
    //
    //   Input:
    //     `rhs`: Matrix to add,
    //            must have the same dimensions as the caller.
    //
    //   Output:
    //   `result`: A new matrix containing the sum of the caller and `rhs`.
    //
#if defined(SEAMAT_CHECK_BOUNDS) && (SEAMAT_CHECK_BOUNDS) == 1
    try {
	MatrixSizesAreEqual(*this, rhs);
    } catch (const std::exception &e) {
	throw e;
    }
#endif

    DenseMatrix<T> result(this->get_rows(), this->get_cols(), (T)0);

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < this->get_rows(); ++i) {
	for (size_t j = 0; j < this->get_cols(); ++j) {
	    result(i, j) = this->operator()(i, j) + rhs(i, j);
	}
    }

    return result;
}

// In-place matrix-matrix addition
template<typename T>
Matrix<T>& Matrix<T>::operator+=(const Matrix<T>& rhs) {
    // seamat::Matrix<T>::operator+=
    //
    // Add values from rhs to the calling matrix in-place.
    //
    //   Input:
    //     `rhs`: Matrix to add, must have the same dimensions as the caller.
    //
#if defined(SEAMAT_CHECK_BOUNDS) && (SEAMAT_CHECK_BOUNDS) == 1
    try {
	MatrixSizesAreEqual(*this, rhs);
    } catch (const std::exception &e) {
	throw e;
    }
#endif

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < this->get_rows(); i++) {
	for (size_t j = 0; j < this->get_cols(); j++) {
	    this->operator()(i, j) += rhs(i, j);
	}
    }

    return *this;
}

// Matrix-matrix subtraction
template<typename T>
DenseMatrix<T>& Matrix<T>::operator-(const Matrix<T>& rhs) const {
    // seamat::Matrix<T>::operator-
    //
    // Creates a new DenseMatrix<T> that contains the result of
    // subtracting rhs from the caller.
    //
    //   Input:
    //     `rhs`: Matrix to subtract,
    //            must have the same dimensions as the caller.
    //
    //   Output:
    //   `result`: A new matrix containing the result.
    //
#if defined(SEAMAT_CHECK_BOUNDS) && (SEAMAT_CHECK_BOUNDS) == 1
    try {
	MatrixSizesAreEqual(*this, rhs);
    } catch (const std::exception &e) {
	throw e;
    }
#endif

    DenseMatrix<T> result(this->get_rows(), this->get_cols(), (T)0);

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < this->get_rows(); i++) {
	for (size_t j = 0; j < this->get_cols(); j++) {
	    result(i, j) = this->operator()(i, j) - rhs(i, j);
	}
    }

    return result;
}

// In-place matrix-matrix subtraction
template<typename T>
Matrix<T>& Matrix<T>::operator-=(const Matrix<T>& rhs) {
    // seamat::Matrix<T>::operator-=
    //
    // Subtract values of rhs from the calling matrix in-place.
    //
    //   Input:
    //     `rhs`: Matrix to subtract, must have the same dimensions as the caller.
    //
#if defined(SEAMAT_CHECK_BOUNDS) && (SEAMAT_CHECK_BOUNDS) == 1
    try {
	MatrixSizesAreEqual(*this, rhs);
    } catch (const std::exception &e) {
	throw e;
    }
#endif

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < this->get_rows(); i++) {
	for (size_t j = 0; j < this->get_cols(); j++) {
	    this->operator()(i, j) -= rhs(i, j);
	}
    }

    return *this;
}

// Matrix-matrix right multiplication
template<typename T>
DenseMatrix<T>& Matrix<T>::operator*(const Matrix<T>& rhs) const {
    // seamat::Matrix<T>::operator*
    //
    // Creates a new DenseMatrix<T> that contains the result of
    // right multiplying the caller with rhs.
    //
    //   Input:
    //     `rhs`: Matrix to right multiply with,
    //            must have the same number of rows as the caller has columns.
    //
    //   Output:
    //   `result`: A new matrix containing the result.
    //
#if defined(SEAMAT_CHECK_BOUNDS) && (SEAMAT_CHECK_BOUNDS) == 1
    try {
	MatricesCanBeMultiplied(*this, rhs);
    } catch (const std::exception &e) {
	throw e;
    }
#endif

    DenseMatrix<T> result(this->get_rows(), rhs.get_cols(), (T)0);

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < this->get_rows(); i++) {
	for (size_t j = 0; j < this->get_cols(); j++) {
	    for (size_t k = 0; k < rhs.get_cols(); k++) {
		result(i, k) += this->operator()(i, j) * rhs(j, k);
	    }
	}
    }

    return result;
}

// Matrix-matrix left multiplication
template<typename T>
DenseMatrix<T>& Matrix<T>::operator%(const Matrix<T>& lhs) const {
    // seamat::Matrix<T>::operator%
    //
    // Creates a new DenseMatrix<T> that contains the result of
    // left multiplying the caller with lhs.
    //
    //   Input:
    //     `lhs`: Matrix to right multiply with,
    //            must have the same number of columns as the caller has rows.
    //
    //   Output:
    //   `result`: A new matrix containing the result.
    //
    DenseMatrix<T> &result = lhs * (*this); // operator* checks bounds
    return result;
}



// In-place right multiplication
template<typename T>
Matrix<T>& Matrix<T>::operator*=(const Matrix<T>& rhs) {
    // seamat::Matrix<T>::operator*=
    //
    // Matrix right-multiplication of the caller with rhs in-place.
    //
    //   Input:
    //     `rhs`: Matrix to right multiply with,
    //            must have the same number of rows as the caller has columns.
    //
    const DenseMatrix<T> &result = (*this) * rhs;
    (*this) = result;
    return *this;
}

// In-place left multiplication
template<typename T>
Matrix<T>& Matrix<T>::operator%=(const Matrix<T>& lhs) {
    // seamat::Matrix<T>::operator%=
    //
    // Matrix left-multiplication of the caller with lhs in-place.
    //
    //   Input:
    //     `lhs`: Matrix to left multiply with,
    //            must have the same number of columns as the caller has rows.
    //
    const DenseMatrix<T> &result = lhs * (*this);
    (*this) = result;
    return *this;
}


// Fill matrix with sum of two matrices
template <typename T>
void Matrix<T>::sum_fill(const Matrix<T>& rhs1, const Matrix<T>& rhs2) {
#pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < this->rows; ++i) {
	for (uint32_t j = 0; j < this->cols; ++j) {
	    this->operator()(i, j) = rhs1(i, j) + rhs2(i, j);
	}
    }
}

// Transpose matrix
template<typename T>
Matrix<T> Matrix<T>::transpose() const {
    Matrix result(this->rows, this->cols, 0.0);

#pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < this->rows; i++) {
	for (uint32_t j = 0; j < this->cols; j++) {
	    result(i, j) = this->operator()(j, i);
	}
    }

    return result;
}

// In-place matrix-scalar addition
template<typename T>
Matrix<T>& Matrix<T>::operator+=(const T& rhs) {
#pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < this->rows; i++) {
	for (uint32_t j = 0; j < this->cols; j++) {
	    this->operator()(i, j) += rhs;
	}
    }

    return *this;
}

// In-place matrix-scalar subtraction
template<typename T>
Matrix<T>& Matrix<T>::operator-=(const T& rhs) {
#pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < this->rows; i++) {
	for (uint32_t j=0; j < this->cols; j++) {
	    this->operator()(i, j) -= rhs;
	}
    }

    return *this;
}

// In-place matrix-scalar multiplication
template<typename T>
Matrix<T>& Matrix<T>::operator*=(const T& rhs) {
#pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < this->rows; ++i) {
	for (uint32_t j = 0; j < this->cols; ++j) {
	    this->operator()(i, j) *= rhs;
	}
    }

    return *this;
}

// In-place matrix-scalar division
template<typename T>
Matrix<T>& Matrix<T>::operator/=(const T& rhs) {
#pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < this->rows; ++i) {
	for (uint32_t j = 0; j < this->cols; ++j) {
	    this->operator()(i, j) /= rhs;
	}
    }

    return *this;
}

// Matrix-matrix comparison
template<typename T>
bool Matrix<T>::operator==(const Matrix<double>& rhs) const {
    bool all_equal = this->rows == rhs.get_rows();
    all_equal &= (this->cols == rhs.get_cols());
    double tol = 1e-4;
#pragma omp parallel for schedule(static) reduction (&:all_equal)
    for (uint32_t i = 0; i < this->rows; ++i) {
	for (uint32_t j = 0; j < this->cols; ++j) {
	    all_equal &= (std::abs(this->operator()(i, j) - rhs(i, j)) < tol);
	}
    }
    return all_equal;
}

// Matrix-vector right multiplication
template<typename T>
std::vector<T> Matrix<T>::operator*(const std::vector<T>& rhs) const {
    std::vector<T> result(rhs.size(), 0.0);

#pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < rows; i++) {
	for (uint32_t j = 0; j < cols; j++) {
	    result[i] += this->operator()(i, j) * rhs[j];
	}
    }

    return result;
}

// Matrix-vector right multiplication, store result in arg
template<typename T>
void Matrix<T>::right_multiply(const std::vector<long unsigned>& rhs, std::vector<T>& result) const {
#pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < this->rows; i++) {
	result[i] = 0.0;
	for (uint32_t j = 0; j < this->cols; j++) {
	    result[i] += this->operator()(i, j) * rhs[j];
	}
    }
}

// log-space Matrix-vector right multiplication, store result in arg
template<typename T>
void Matrix<T>::exp_right_multiply(const std::vector<T>& rhs, std::vector<T>& result) const {
    std::fill(result.begin(), result.end(), 0.0);
#pragma omp parallel for schedule(static) reduction(vec_double_plus:result)
    for (uint32_t i = 0; i < this->rows; i++) {
	for (uint32_t j = 0; j < this->cols; j++) {
	    result[i] += std::exp(this->operator()(i, j) + rhs[j]);
	}
    }
}

template<typename T>
T Matrix<T>::log_sum_exp_col(uint32_t col_id) const {
    // Note: this function accesses the elements rather inefficiently so
    // it shouldn't be parallellised here. However, the caller can
    // parallellize logsumexping multiple cols.
    T max_elem = 0;
    T sum = 0;
    for (uint32_t i = 0; i < this->rows; ++i) {
	max_elem = (this->operator()(i, col_id) > max_elem ? this->operator()(i, col_id) : max_elem);
    }

    for (uint32_t i = 0; i < this->rows; ++i) {
	sum += std::exp(this->operator()(i, col_id) - max_elem);
    }
    return max_elem + std::log(sum);
}

// Specialized matrix-vector right multiplication
template<typename T>
std::vector<double> Matrix<T>::operator*(const std::vector<long unsigned>& rhs) const {
    std::vector<double> result(this->rows, 0.0);

#pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < this->rows; i++) {
	for (uint32_t j = 0; j < this->cols; j++) {
	    result[i] += this->operator()(i, j) * rhs[j];
	}
    }

    return result;
}
}

#endif
