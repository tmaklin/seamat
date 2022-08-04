// seamat: templatized matrix library
// https://github.com/tmaklin/seamat
//
// Copyright (C) 2021 Tommi Mäklin (tommi@maklin.fi)
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
#ifndef SEAMAT_DENSE_MATRIX_HPP
#define SEAMAT_DENSE_MATRIX_HPP

#include <vector>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>

// Basic matrix structure and operations
// Implementation was done following the instructions at
// https://www.quantstart.com/articles/Matrix-Classes-in-C-The-Header-File
//
// **None of the operations validate the matrix sizes**
//
// This file provides implementations for the functions that are not
// parallellized.  Implementations for the functions that are
// parallellized should be provided in the .cpp file that is included
// at the very end of this file.

namespace seamat {
// Forward declare SparseMatrix
template <typename T> class SparseMatrix;

template <typename T> class Matrix {
private:
    uint32_t rows;
    uint32_t cols;

protected:
    // Derived classes can use these to resize the base class
    void resize_rows(const uint32_t new_rows) { this->rows = new_rows; };
    void resize_cols(const uint32_t new_cols) { this->cols = new_cols; };

public:
    //////
    // Pure virtuals - must override in derived classes.
    ////
    // Resize a matrix
    virtual void resize(const uint32_t new_rows, const uint32_t new_cols, const T initial) =0;

    // Access individual elements
    virtual T& operator()(uint32_t row, uint32_t col) =0;
    virtual const T& operator()(uint32_t row, uint32_t col) const =0;

    //////
    // Implemented in Matrix.cpp - these functions only use the
    // pure virtual functions and the private member variables.
    ////
    // Mathematical operators
    // Matrix-matrix summation and subtraction
    Matrix<T>& operator+(const Matrix<T>& rhs) const;
    Matrix<T>& operator+=(const Matrix<T>& rhs);
    Matrix<T> operator-(const Matrix<T>& rhs) const;
    Matrix<T>& operator-=(const Matrix<T>& rhs);

    // Matrix product
    Matrix<T> operator*(const Matrix<T>& rhs) const;
    // In-place left multiplication
    Matrix<T>& operator*=(const Matrix<T>& rhs);

    // Matrix-scalar, only in-place
    Matrix<T>& operator+=(const T& rhs);
    Matrix<T>& operator-=(const T& rhs);
    Matrix<T>& operator*=(const T& rhs);
    Matrix<T>& operator/=(const T& rhs);

    // Matrix-matrix comparison
    bool operator==(const Matrix<double>& rhs) const;

    // Matrix-vector multiplication
    std::vector<T> operator*(const std::vector<T>& rhs) const;
    std::vector<double> operator*(const std::vector<long unsigned>& rhs) const;

    // Matrix-vector right multiplication, store result in arg
    void right_multiply(const std::vector<long unsigned>& rhs, std::vector<T>& result) const;
    void exp_right_multiply(const std::vector<T>& rhs, std::vector<T>& result) const;

    // Turn an arbitrary matrix into a column sparse matrix
    SparseMatrix<T> sparsify(const T &zero_val) const { return SparseMatrix<T>(this, zero_val); }

    // LogSumExp a Matrix column
    T log_sum_exp_col(uint32_t col_id) const;

    // Fill a matrix with the sum of two matrices
    void sum_fill(const Matrix<T>& rhs1, const Matrix<T>& rhs2);

    // Transpose
    Matrix<T> transpose() const;

    // Get the number of rows of the matrix
    uint32_t get_rows() const { return this->rows; }
    // Get the number of columns of the matrix
    uint32_t get_cols() const { return this->cols; }
};

template <typename T> class DenseMatrix : public Matrix<T> {
private:
    std::vector<T> mat;

public:
    DenseMatrix() = default;
    ~DenseMatrix() = default;
    // Parameter constructor
    DenseMatrix(uint32_t _rows, uint32_t _cols, const T& _initial);
    // Copy constructor from contiguous 2D vector
    DenseMatrix(const std::vector<T> &rhs, const uint32_t _rows, const uint32_t _cols);
    // Copy constructor from 2D vector
    DenseMatrix(const std::vector<std::vector<T>> &rhs);
    // Copy constructor from another matrix
    DenseMatrix(const Matrix<T> &rhs);

    // Assignment operator
    DenseMatrix<T>& operator=(const Matrix<T>& rhs);

   // Resize a matrix
    void resize(const uint32_t new_rows, const uint32_t new_cols, const T initial) override;

    // Access individual elements
    T& operator()(uint32_t row, uint32_t col) override;
    const T& operator()(uint32_t row, uint32_t col) const override;
};

template <typename T, typename U> class IndexMatrix : public Matrix<T> {
private:
    uint32_t n_rows_vals;
    uint32_t n_cols_vals;
    std::unique_ptr<Matrix<T>> vals;
    std::unique_ptr<Matrix<U>> indices;

public:
    IndexMatrix() = default;
    ~IndexMatrix() = default;

    // Initialize from vals and indices
    IndexMatrix(const Matrix<T> &_vals, const Matrix<U> &_indices, const bool store_as_sparse);

    // Resize a matrix
    void resize(const uint32_t new_rows, const uint32_t new_cols, const T initial) override;

    // Access individual elements
    T& operator()(uint32_t row, uint32_t col) override;
    const T& operator()(uint32_t row, uint32_t col) const override;
};

template <typename T> class SparseMatrix : public Matrix<T> {
private:
    // Sparse matrix implemented in the compressed row storage (CRS) format.
    // See link below for reference.
    // https://netlib.org/linalg/html_templates/node91.html#SECTION00931100000000000000
    //
    std::vector<T> vals;
    std::vector<uint32_t> row_ptr;
    std::vector<uint32_t> col_ind;
    T zero_val;

    T* get_address(uint32_t row, uint32_t col);
    const T* get_address(uint32_t row, uint32_t col) const;

    bool nearly_equal(double a, double b)
    {
	return std::nextafter(a, std::numeric_limits<double>::lowest()) <= b
      && std::nextafter(a, std::numeric_limits<double>::max()) >= b;
    }

    bool nearly_equal(double a, double b, int factor /* a factor of epsilon */)
    {
	double min_a = a - (a - std::nextafter(a, std::numeric_limits<double>::lowest())) * factor;
	double max_a = a + (std::nextafter(a, std::numeric_limits<double>::max()) - a) * factor;

	return min_a <= b && max_a >= b;
    }
public:
    SparseMatrix() = default;
    ~SparseMatrix() = default;
    // Parameter constructor
    SparseMatrix(uint32_t _rows, uint32_t _cols, const T& _initial);
    // Initialize from a DenseMatrix
    SparseMatrix(const Matrix<T> &_vals, const T& _zero_val);
    // Initialize from a 2D vector
    SparseMatrix(const std::vector<std::vector<T>> &rhs, const T& _zero_val);
    // Copy constructor from contiguous 2D vector
    SparseMatrix(const std::vector<T> &rhs, const uint32_t _rows, const uint32_t _cols, const T& _zero_val);

    // Resize a matrix
    void resize(const uint32_t new_rows, const uint32_t new_cols, const T initial) override;

    // Access individual elements
    T& operator()(uint32_t row, uint32_t col) override;
    const T& operator()(uint32_t row, uint32_t col) const override;
};
}

#include "../src/Matrix.cpp"
#include "../src/DenseMatrix.cpp"
#include "../src/IndexMatrix.cpp"
#include "../src/SparseMatrix.cpp"

#endif
