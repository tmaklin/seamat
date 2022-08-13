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

#include "/home/temaklin/Projects/software/seamat/external/BitMagic-7.12.3/src/bm.h"
#include "/home/temaklin/Projects/software/seamat/external/BitMagic-7.12.3/src/bmsparsevec.h"

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
// Forward declare implementations of abstract class Matrix
template <typename T> class DenseMatrix;
template <typename T> class SparseMatrix;

template <typename T> class Matrix {
private:
    size_t rows;
    size_t cols;

protected:
    // Derived classes can use these to resize the base class
    void resize_rows(const size_t new_rows) { this->rows = new_rows; };
    void resize_cols(const size_t new_cols) { this->cols = new_cols; };

public:
    //////
    // Pure virtuals - must override in derived classes.
    ////
    // Access individual elements
    virtual T& operator()(size_t row, size_t col) =0;
    virtual const T& operator()(size_t row, size_t col) const =0;

    // Mathematical operators
    // Matrix-matrix in-place summation and subtraction
    virtual Matrix<T>& operator+=(const Matrix<T>& rhs) =0;
    virtual Matrix<T>& operator-=(const Matrix<T>& rhs) =0;

    // In-place right multiplication
    virtual Matrix<T>& operator*=(const Matrix<T>& rhs) =0;
    // In-place left multiplication
    virtual Matrix<T>& operator%=(const Matrix<T>& rhs) =0;

    // Matrix-scalar, in-place
    virtual Matrix<T>& operator+=(const T& rhs) =0;
    virtual Matrix<T>& operator-=(const T& rhs) =0;
    virtual Matrix<T>& operator*=(const T& rhs) =0;
    virtual Matrix<T>& operator/=(const T& rhs) =0;

    //////
    // Implemented in Matrix.cpp - these functions only use the pure
    // virtual functions and the private member variables, and are
    // generic in the sense that they work with any input class
    // derived from Matrix<T>.
    ////
    // Mathematical operators
    // Matrix-matrix summation and subtraction
    DenseMatrix<T>& operator+(const Matrix<T>& rhs) const;
    DenseMatrix<T>& operator-(const Matrix<T>& rhs) const;

    // Matrix-matrix right multiplication
    DenseMatrix<T> operator*(const Matrix<T>& rhs) const;
    // Matrix-matrix left multiplication
    DenseMatrix<T> operator%(const Matrix<T>& rhs) const;

    // Matrix-scalar
    DenseMatrix<T> operator+(const T& rhs);
    DenseMatrix<T> operator-(const T& rhs);
    DenseMatrix<T> operator*(const T& rhs);
    DenseMatrix<T> operator/(const T& rhs);

    // Matrix-matrix comparisons
    bool operator==(const Matrix<T>& rhs) const;

    // Matrix-vector multiplication
    template <typename U>
    std::vector<T> operator*(const std::vector<U>& rhs) const;
    template <typename U>
    std::vector<U> operator*(const std::vector<U>& rhs) const;

    // Matrix-vector right multiplication, store result in arg
    template <typename U, typename V>
    void right_multiply(const std::vector<U>& rhs, std::vector<V> &result) const;
    template <typename U, typename V>
    void logspace_right_multiply(const std::vector<U>& rhs, std::vector<V>& result) const;

    // LogSumExp a column
    template <typename V>
    T log_sum_exp_col(const size_t col_id) const;

    // Generic transpose
    DenseMatrix<T> transpose() const;

    // Get the number of rows in the matrix
    size_t get_rows() const { return this->rows; }
    // Get the number of columns of the matrix
    size_t get_cols() const { return this->cols; }

    // Turn an arbitrary matrix into a dense matrix
    DenseMatrix<T> densify() const { return DenseMatrix<T>(this); }
    // Turn an arbitrary matrix into a column sparse matrix
    SparseMatrix<T> sparsify(const T &zero_val) const { return SparseMatrix<T>(this, zero_val); }

};

template <typename T> class DenseMatrix : public Matrix<T> {
private:
    std::vector<T> mat;

public:
    DenseMatrix() = default;
    ~DenseMatrix() = default;
    // Parameter constructor
    DenseMatrix(size_t _rows, size_t _cols, const T& _initial);
    // Copy constructor from contiguous 2D vector
    DenseMatrix(const std::vector<T> &rhs, const size_t _rows, const size_t _cols);
    // Copy constructor from 2D vector
    DenseMatrix(const std::vector<std::vector<T>> &rhs);
    // Copy constructor from another matrix
    DenseMatrix(const Matrix<T> &rhs);

    // Assignment operator
    DenseMatrix<T>& operator=(const Matrix<T>& rhs);

   // Resize a matrix
    void resize(const size_t new_rows, const size_t new_cols, const T initial);

    // Access individual elements
    T& operator()(size_t row, size_t col) override;
    const T& operator()(size_t row, size_t col) const override;

    // Mathematical operators
    // Matrix-matrix in-place summation and subtraction
    DenseMatrix<T>& operator+=(const Matrix<T>& rhs) override;
    DenseMatrix<T>& operator-=(const Matrix<T>& rhs) override;

    // In-place right multiplication
    DenseMatrix<T>& operator*=(const Matrix<T>& rhs) override;
    // In-place left multiplication
    DenseMatrix<T>& operator%=(const Matrix<T>& rhs) override;

    // Matrix-scalar, in-place
    DenseMatrix<T>& operator+=(const T& rhs);
    DenseMatrix<T>& operator-=(const T& rhs);
    DenseMatrix<T>& operator*=(const T& rhs);
    DenseMatrix<T>& operator/=(const T& rhs);

    // Fill a matrix with the sum of two matrices in-place
    template <typename V, typename U>
    void sum_fill(const Matrix<V>& rhs1, const Matrix<U>& rhs2);

};

template <typename T, typename U> class IndexMatrix : public Matrix<T> {
private:
    size_t n_rows_vals;
    size_t n_cols_vals;
    std::unique_ptr<Matrix<T>> vals;
    std::unique_ptr<Matrix<U>> indices;

public:
    IndexMatrix() = default;
    ~IndexMatrix() = default;

    // Initialize from vals and indices
    IndexMatrix(const Matrix<T> &_vals, const Matrix<U> &_indices, const bool store_as_sparse);

    // Access individual elements
    T& operator()(size_t row, size_t col) override;
    const T& operator()(size_t row, size_t col) const override;

    // Mathematical operators
    // Matrix-matrix in-place summation and subtraction
    IndexMatrix<T, U>& operator+=(const Matrix<T>& rhs) override;
    IndexMatrix<T, U>& operator-=(const Matrix<T>& rhs) override;

    // In-place right multiplication
    IndexMatrix<T, U>& operator*=(const Matrix<T>& rhs) override;
    // In-place left multiplication
    IndexMatrix<T, U>& operator%=(const Matrix<T>& rhs) override;

    // Matrix-scalar, in-place
    IndexMatrix<T, U>& operator+=(const T& rhs) override;
    IndexMatrix<T, U>& operator-=(const T& rhs) override;
    IndexMatrix<T, U>& operator*=(const T& rhs) override;
    IndexMatrix<T, U>& operator/=(const T& rhs) override;

};

template <typename T> class SparseMatrix : public Matrix<T> {
private:
    // TODO generic sparse matrix (instead of zero)
    //
    // Sparse matrix implemented in the compressed row storage (CRS) format.
    // See link below for reference.
    // https://netlib.org/linalg/html_templates/node91.html#SECTION00931100000000000000
    //
    std::vector<T> vals;
    std::vector<size_t> row_ptr;
    std::vector<size_t> col_ind;
    T zero_val;

    T* get_address(size_t row, size_t col);
    const T* get_address(size_t row, size_t col) const;

public:
    SparseMatrix() = default;
    ~SparseMatrix() = default;
    // Parameter constructor
    SparseMatrix(size_t _rows, size_t _cols, const T& _initial);
    // Initialize from a DenseMatrix
    SparseMatrix(const Matrix<T> &_vals, const T& _zero_val);
    // Initialize from a 2D vector
    SparseMatrix(const std::vector<std::vector<T>> &rhs, const T& _zero_val);
    // Copy constructor from contiguous 2D vector
    SparseMatrix(const std::vector<T> &rhs, const size_t _rows, const size_t _cols, const T& _zero_val);

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

template <typename T> class SparseIntegerTypeMatrix : public Matrix<T> {
private:
    // Specialization of SparseIntegerTypeMatrix for integer types using BitMagic's sparse integer vectors
    //
    bm::sparse_vector<T, bm::bvector<>> vals;
    T zero_val;

public:
    SparseIntegerTypeMatrix() = default;
    ~SparseIntegerTypeMatrix() = default;
    // Parameter constructor
    SparseIntegerTypeMatrix(size_t _rows, size_t _cols, const T& _initial);
    // Initialize from a DenseMatrix
    SparseIntegerTypeMatrix(const Matrix<T> &_vals, const T& _zero_val);
    // Initialize from a 2D vector
    SparseIntegerTypeMatrix(const std::vector<std::vector<T>> &rhs, const T& _zero_val);
    // Copy constructor from contiguous 2D vector
    SparseIntegerTypeMatrix(const std::vector<T> &rhs, const size_t _rows, const size_t _cols, const T& _zero_val);

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

}

#include "../src/Matrix.cpp"
#include "../src/DenseMatrix.cpp"
#include "../src/IndexMatrix.cpp"
#include "../src/SparseMatrix.cpp"
#include "../src/SparseIntegerTypeMatrix.cpp"

#endif
