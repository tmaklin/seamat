// seamat: templatized matrix library
// https://github.com/tmaklin/seamat
//
// Copyright (C) 2021 Tommi Mäklin (tommi@maklin.fi)
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
#include "IndexMatrix_unittest.hpp"

////
// Move constructor tests
// Dense-dense move constructor
TEST_F(IndexMatrixTest, MoveConstructorTest_DenseDenseWorks) {
    std::vector<uint32_t> indices = { 1, 1, 1, 2, 0, 0, 0,
				      1, 1, 2, 2, 1, 0, 0,
				      2, 2, 0, 0, 0, 0, 0 };
    std::vector<double> vals = { 0.0, 1.0, 2.0,
				 0.0, 1.0, 2.0,
				 0.0, 1.0, 2.0 };

    seamat::IndexMatrix<double, uint32_t> got(seamat::DenseMatrix<double>(vals, 3, 3), seamat::DenseMatrix<uint32_t>(indices, 3, 7));
    EXPECT_EQ(this->n_rows, got.get_rows());
    EXPECT_EQ(this->n_cols, got.get_cols());
    for (uint32_t i = 0; i < this->n_rows; ++i) {
	for (uint32_t j = 0; j < this->n_cols; ++j) {
	    EXPECT_EQ(this->expected_mat[i*this->n_cols + j], got(i, j));
	}
    }
}
// Dense-sparse move constructor
TEST_F(IndexMatrixTest, MoveConstructorTest_DenseSparseWorks) {
    std::vector<uint32_t> indices = { 1, 1, 1, 2, 0, 0, 0,
				      1, 1, 2, 2, 1, 0, 0,
				      2, 2, 0, 0, 0, 0, 0 };
    std::vector<double> vals = { 0.0, 1.0, 2.0,
				 0.0, 1.0, 2.0,
				 0.0, 1.0, 2.0 };

    seamat::IndexMatrix<double, uint32_t> got(seamat::DenseMatrix<double>(vals, 3, 3), seamat::SparseMatrix<uint32_t>(indices, 3, 7, 0));
    EXPECT_EQ(this->n_rows, got.get_rows());
    EXPECT_EQ(this->n_cols, got.get_cols());
    for (uint32_t i = 0; i < this->n_rows; ++i) {
	for (uint32_t j = 0; j < this->n_cols; ++j) {
	    EXPECT_EQ(this->expected_mat[i*this->n_cols + j], got(i, j));
	}
    }
}
// Sparse-dense move constructor
TEST_F(IndexMatrixTest, MoveConstructorTest_SparseDenseWorks) {
    std::vector<uint32_t> indices = { 1, 1, 1, 2, 0, 0, 0,
				      1, 1, 2, 2, 1, 0, 0,
				      2, 2, 0, 0, 0, 0, 0 };
    std::vector<double> vals = { 0.0, 1.0, 2.0,
				 0.0, 1.0, 2.0,
				 0.0, 1.0, 2.0 };

    seamat::IndexMatrix<double, uint32_t> got(seamat::SparseMatrix<double>(vals, 3, 3, 0.0), seamat::DenseMatrix<uint32_t>(indices, 3, 7));
    EXPECT_EQ(this->n_rows, got.get_rows());
    EXPECT_EQ(this->n_cols, got.get_cols());
    for (uint32_t i = 0; i < this->n_rows; ++i) {
	for (uint32_t j = 0; j < this->n_cols; ++j) {
	    EXPECT_EQ(this->expected_mat[i*this->n_cols + j], got(i, j));
	}
    }
}
// Sparse-sparse move constructor
TEST_F(IndexMatrixTest, MoveConstructorTest_SparseSparseWorks) {
    std::vector<uint32_t> indices = { 1, 1, 1, 2, 0, 0, 0,
				      1, 1, 2, 2, 1, 0, 0,
				      2, 2, 0, 0, 0, 0, 0 };
    std::vector<double> vals = { 0.0, 1.0, 2.0,
				 0.0, 1.0, 2.0,
				 0.0, 1.0, 2.0 };

    seamat::IndexMatrix<double, uint32_t> got(seamat::SparseMatrix<double>(vals, 3, 3, 0.0), seamat::SparseMatrix<uint32_t>(indices, 3, 7, 0));
    EXPECT_EQ(this->n_rows, got.get_rows());
    EXPECT_EQ(this->n_cols, got.get_cols());
    for (uint32_t i = 0; i < this->n_rows; ++i) {
	for (uint32_t j = 0; j < this->n_cols; ++j) {
	    EXPECT_EQ(this->expected_mat[i*this->n_cols + j], got(i, j));
	}
    }
}

////
// Copy constructor tests
// Dense-dense copy constructor
TEST_F(IndexMatrixTest, CopyConstructorTest_DenseDenseWorks) {
    std::vector<uint32_t> indices = { 1, 1, 1, 2, 0, 0, 0,
				      1, 1, 2, 2, 1, 0, 0,
				      2, 2, 0, 0, 0, 0, 0 };
    std::vector<double> vals = { 0.0, 1.0, 2.0,
				 0.0, 1.0, 2.0,
				 0.0, 1.0, 2.0 };
    seamat::DenseMatrix<double> _vals(vals, 3, 3);
    seamat::DenseMatrix<uint32_t> _indices(indices, 3, 7);
    seamat::IndexMatrix<double, uint32_t> got(_vals, _indices);
    EXPECT_EQ(this->n_rows, got.get_rows());
    EXPECT_EQ(this->n_cols, got.get_cols());
    for (uint32_t i = 0; i < this->n_rows; ++i) {
	for (uint32_t j = 0; j < this->n_cols; ++j) {
	    EXPECT_EQ(this->expected_mat[i*this->n_cols + j], got(i, j));
	}
    }
}
// Dense-sparse copy constructor
TEST_F(IndexMatrixTest, CopyConstructorTest_DenseSparseWorks) {
    std::vector<uint32_t> indices = { 1, 1, 1, 2, 0, 0, 0,
				      1, 1, 2, 2, 1, 0, 0,
				      2, 2, 0, 0, 0, 0, 0 };
    std::vector<double> vals = { 0.0, 1.0, 2.0,
				 0.0, 1.0, 2.0,
				 0.0, 1.0, 2.0 };
    
    seamat::DenseMatrix<double> _vals(vals, 3, 3);
    seamat::SparseMatrix<uint32_t> _indices(indices, 3, 7, 0);
    seamat::IndexMatrix<double, uint32_t> got(_vals, _indices);
    EXPECT_EQ(this->n_rows, got.get_rows());
    EXPECT_EQ(this->n_cols, got.get_cols());
    for (uint32_t i = 0; i < this->n_rows; ++i) {
	for (uint32_t j = 0; j < this->n_cols; ++j) {
	    EXPECT_EQ(this->expected_mat[i*this->n_cols + j], got(i, j));
	}
    }
}
// Sparse-dense copy constructor
TEST_F(IndexMatrixTest, CopyConstructorTest_SparseDenseWorks) {
    std::vector<uint32_t> indices = { 1, 1, 1, 2, 0, 0, 0,
				      1, 1, 2, 2, 1, 0, 0,
				      2, 2, 0, 0, 0, 0, 0 };
    std::vector<double> vals = { 0.0, 1.0, 2.0,
				 0.0, 1.0, 2.0,
				 0.0, 1.0, 2.0 };
    
    seamat::SparseMatrix<double> _vals(vals, 3, 3, 0.0);
    seamat::DenseMatrix<uint32_t> _indices(indices, 3, 7);
    seamat::IndexMatrix<double, uint32_t> got(_vals, _indices);
    EXPECT_EQ(this->n_rows, got.get_rows());
    EXPECT_EQ(this->n_cols, got.get_cols());
    for (uint32_t i = 0; i < this->n_rows; ++i) {
	for (uint32_t j = 0; j < this->n_cols; ++j) {
	    EXPECT_EQ(this->expected_mat[i*this->n_cols + j], got(i, j));
	}
    }
}
// Sparse-sparse copy constructor
TEST_F(IndexMatrixTest, CopyConstructorTest_SparseSparseWorks) {
    std::vector<uint32_t> indices = { 1, 1, 1, 2, 0, 0, 0,
				      1, 1, 2, 2, 1, 0, 0,
				      2, 2, 0, 0, 0, 0, 0 };
    std::vector<double> vals = { 0.0, 1.0, 2.0,
				 0.0, 1.0, 2.0,
				 0.0, 1.0, 2.0 };
    seamat::SparseMatrix<double> _vals(vals, 3, 3, 0.0);
    seamat::SparseMatrix<uint32_t> _indices(indices, 3, 7, 0);
    seamat::IndexMatrix<double, uint32_t> got(_vals, _indices);
    EXPECT_EQ(this->n_rows, got.get_rows());
    EXPECT_EQ(this->n_cols, got.get_cols());
    for (uint32_t i = 0; i < this->n_rows; ++i) {
	for (uint32_t j = 0; j < this->n_cols; ++j) {
	    EXPECT_EQ(this->expected_mat[i*this->n_cols + j], got(i, j));
	}
    }
}
