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

    // Get position of first element in this->mat
    virtual T& front() =0;
    virtual const T& front() const =0;

    //////
    // Implemented in Matrix.cpp - these functions only use the
    // pure virtual functions and the private member variables.
    ////
    // Mathematical operators
    // Matrix-matrix summation and subtraction
    Matrix<T> operator+(const Matrix<T>& rhs) const;
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
    // Copy constructor
    DenseMatrix(const DenseMatrix<T>& rhs);
    // Copy constructor from contiguous 2D vector
    DenseMatrix(const std::vector<T> &rhs, const uint32_t _rows, const uint32_t _cols);
    // Copy constructor from 2D vector
    DenseMatrix(const std::vector<std::vector<T>> &rhs);

    // Assignment operator
    DenseMatrix<T>& operator=(const Matrix<T>& rhs);

   // Resize a matrix
    void resize(const uint32_t new_rows, const uint32_t new_cols, const T initial) override;

    // Access individual elements
    T& operator()(uint32_t row, uint32_t col) override;
    const T& operator()(uint32_t row, uint32_t col) const override;

    // Get position of first element in this->mat
    const T& front() const override { return this->mat.front(); }
    T& front() override { return this->mat.front(); }
};

template <typename T, typename U> class IndexMatrix : public Matrix<T> {
private:
    std::vector<T> vals;
    U n_rows_vals;
    U n_cols_vals;
    std::vector<U> indices;

public:
    IndexMatrix() = default;
    ~IndexMatrix() = default;

    // Resize a matrix
    void resize(const uint32_t new_rows, const uint32_t new_cols, const T initial) override;

    // Access individual elements
    T& operator()(uint32_t row, uint32_t col) override;
    const T& operator()(uint32_t row, uint32_t col) const override;

    // Get position of first element in the values
    const T& front() const override { return this->vals.front(); }
    T& front() override { return this->vals.front(); }

    // Set the indices from some input
    void swap_indices(std::vector<T> &_indices) { this->indices.swap(_indices); };

    // Set the values from some input
    void swap_vals(std::vector<T> &_vals) { this->vals.swap(_vals); };
};
}

#include "../src/Matrix.cpp"
#include "../src/DenseMatrix.cpp"
#include "../src/IndexMatrix.cpp"

#endif
