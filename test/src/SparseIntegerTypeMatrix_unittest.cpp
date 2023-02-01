// seamat: templatized matrix library
// https://github.com/tmaklin/seamat
//
// Copyright (C) 2021 Tommi Mäklin (tommi@maklin.fi)
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
#include "SparseIntegerTypeMatrix_unittest.hpp"

TEST_F(SparseIntegerTypeMatrixTest, ParameterConstructor) {
    seamat::SparseIntegerTypeMatrix<unsigned> got(this->n_rows, this->n_cols, this->initial_val);
    EXPECT_EQ(this->n_rows, got.get_rows());
    EXPECT_EQ(this->n_cols, got.get_cols());
    for (uint32_t i = 0; i < this->n_rows; ++i) {
	for (uint32_t j = 0; j < this->n_cols; ++j) {
	    EXPECT_EQ(this->initial_val, got(i, j));
	}
    }
}

TEST_F(SparseIntegerTypeMatrixTest, CopyConstructorWorks) {
    seamat::SparseIntegerTypeMatrix<unsigned> input(this->n_rows, this->n_cols, this->initial_val);
    seamat::SparseIntegerTypeMatrix<unsigned> got(input, this->initial_val);
    EXPECT_EQ(this->n_rows, got.get_rows());
    EXPECT_EQ(this->n_cols, got.get_cols());
    for (uint32_t i = 0; i < this->n_rows; ++i) {
	for (uint32_t j = 0; j < this->n_cols; ++j) {
	    EXPECT_EQ(this->initial_val, got(i, j));
	}
    }
}

TEST_F(SparseIntegerTypeMatrixTest, CopyConstructorFromVectorWorks) {
    seamat::SparseIntegerTypeMatrix<unsigned> got(this->expected_mat, this->n_rows, this->n_cols, (double)0.0);
    EXPECT_EQ(this->n_rows, got.get_rows());
    EXPECT_EQ(this->n_cols, got.get_cols());
    for (uint32_t i = 0; i < got.get_rows(); ++i) {
	for (uint32_t j = 0; j < got.get_cols(); ++j) {
	    EXPECT_EQ(this->expected_mat[i*this->n_cols + j], got(i, j));
	}
    }
}

TEST_F(SparseIntegerTypeMatrixTest, CopyConstructorFrom2DVectorWorks) {
    std::vector<std::vector<unsigned>> input(3, std::vector<unsigned>());
    input[0] = { 0, 0, 0, 0, 0, 0, 0 };
    input[1] = { 0, 2, 0, 0, 0, 0, 0 };
    input[2] = { 2, 0, 0, 0, 0, 0, 2 };

    seamat::SparseIntegerTypeMatrix<unsigned> got(input, 0.0);
    EXPECT_EQ(this->n_rows, got.get_rows());
    EXPECT_EQ(this->n_cols, got.get_cols());
    for (uint32_t i = 0; i < this->n_rows; ++i) {
	for (uint32_t j = 0; j < this->n_cols; ++j) {
	    EXPECT_EQ(this->expected_mat[i*this->n_cols + j], got(i, j));
	}
    }
}

TEST_F(SparseIntegerTypeMatrixTest, ElementAssignmentWorks) {
    seamat::SparseIntegerTypeMatrix<unsigned> got(this->expected_mat, this->n_rows, this->n_cols, 0.0);
    got(2, 5) = this->new_val;
    EXPECT_EQ(this->new_val, got(2, 5));
}

TEST_F(SparseIntegerTypeMatrixTest, AssignmentOperatorWorks) {
    seamat::SparseIntegerTypeMatrix<unsigned> got(2, 5, 0.0);
    seamat::SparseIntegerTypeMatrix<unsigned> to_assign(this->expected_mat, this->n_rows, this->n_cols, 0.0);
    got = to_assign;
    EXPECT_EQ(this->n_rows, got.get_rows());
    EXPECT_EQ(this->n_cols, got.get_cols());
    for (uint32_t i = 0; i < this->n_rows; ++i) {
	for (uint32_t j = 0; j < this->n_cols; ++j) {
	    EXPECT_EQ(this->expected_mat[i*this->n_cols + j], got(i, j));
	}
    }    
}
