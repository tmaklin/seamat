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

namespace seamat {
// Matrix-matrix addition
template<typename T>
Matrix<T> Matrix<T>::operator+(const Matrix<T>& rhs) const {
    Matrix result(this->rows, this->cols, 0.0);

#pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < this->rows; i++) {
	for (uint32_t j = 0; j < this->cols; j++) {
	    result(i, j) = this->operator()(i, j) + rhs(i,j);
	}
    }

    return result;
}

// In-place matrix-matrix addition
template<typename T>
Matrix<T>& Matrix<T>::operator+=(const Matrix<T>& rhs) {
#pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < this->rows; i++) {
	for (uint32_t j = 0; j < this->cols; j++) {
	    this->operator()(i, j) += rhs(i, j);
	}
    }

    return *this;
}

// In-place left multiplication
template<typename T>
Matrix<T>& Matrix<T>::operator*=(const Matrix<T>& rhs) {
    Matrix result = (*this) * rhs;
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

// Matrix-matrix subtraction
template<typename T>
Matrix<T> Matrix<T>::operator-(const Matrix<T>& rhs) const {
    Matrix result(this->rows, this->cols, 0.0);

#pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < this->rows; i++) {
	for (uint32_t j = 0; j < this->cols; j++) {
	    result(i, j) = this->operator()(i, j) - rhs(i, j);
	}
    }

    return result;
}

// In-place matrix-matrix subtraction
template<typename T>
Matrix<T>& Matrix<T>::operator-=(const Matrix<T>& rhs) {
#pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < this->rows; i++) {
	for (uint32_t j = 0; j < this->cols; j++) {
	    this->operator()(i, j) -= rhs(i, j);
	}
    }

    return *this;
}

// Matrix-matrix left multiplication
template<typename T>
Matrix<T> Matrix<T>::operator*(const Matrix<T>& rhs) const {
    Matrix result(this->rows, this->cols, 0.0);

#pragma omp parallel for schedule(static)
    for (uint32_t i = 0; i < this->rows; i++) {
	for (uint32_t j = 0; j < this->cols; j++) {
	    for (uint32_t k = 0; k < this->rows; k++) {
		result(i, j) += this->operator()(i, k) * rhs(k, j);
	    }
	}
    }

    return result;
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
