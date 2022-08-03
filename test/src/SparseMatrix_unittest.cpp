#include "SparseMatrix_unittest.hpp"

TEST_F(SparseMatrixTest, ParameterConstructorWorks) {
    seamat::SparseMatrix<double> got(this->n_rows, this->n_cols, this->initial_val);
    EXPECT_EQ(this->n_rows, got.get_rows());
    EXPECT_EQ(this->n_cols, got.get_cols());
    for (uint32_t i = 0; i < this->n_rows; ++i) {
	for (uint32_t j = 0; j < this->n_cols; ++j) {
	    EXPECT_EQ(this->initial_val, got(i, j));
	}
    }
}

TEST_F(SparseMatrixTest, CopyConstructorWorks) {
    seamat::SparseMatrix<double> input(this->n_rows, this->n_cols, this->initial_val);
    seamat::SparseMatrix<double> got(input);
    EXPECT_EQ(this->n_rows, got.get_rows());
    EXPECT_EQ(this->n_cols, got.get_cols());
    for (uint32_t i = 0; i < this->n_rows; ++i) {
	for (uint32_t j = 0; j < this->n_cols; ++j) {
	    EXPECT_EQ(this->initial_val, got(i, j));
	}
    }
}

TEST_F(SparseMatrixTest, CopyConstructorFromVectorWorks) {
    seamat::SparseMatrix<double> got(this->expected_mat, this->n_rows, this->n_cols, (double)0.0);
    EXPECT_EQ(this->n_rows, got.get_rows());
    EXPECT_EQ(this->n_cols, got.get_cols());
    for (uint32_t i = 0; i < got.get_rows(); ++i) {
	for (uint32_t j = 0; j < got.get_cols(); ++j) {
	    EXPECT_EQ(this->expected_mat[i*this->n_cols + j], got(i, j));
	}
    }
}

TEST_F(SparseMatrixTest, CopyConstructorFrom2DVectorWorks) {
    std::vector<std::vector<double>> input(3, std::vector<double>());
    input[0] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    input[1] = { 0.0, 2.7, 0.0, 0.0, 0.0, 0.0, 0.0 };
    input[2] = { 2.7, 0.0, 0.0, 0.0, 0.0, 0.0, 2.7 };

    seamat::SparseMatrix<double> got(input, 0.0);
    EXPECT_EQ(this->n_rows, got.get_rows());
    EXPECT_EQ(this->n_cols, got.get_cols());
    for (uint32_t i = 0; i < this->n_rows; ++i) {
	for (uint32_t j = 0; j < this->n_cols; ++j) {
	    EXPECT_EQ(this->expected_mat[i*this->n_cols + j], got(i, j));
	}
    }
}

TEST_F(SparseMatrixTest, ElementAssignmentWorks) {
    seamat::SparseMatrix<double> got(this->expected_mat, this->n_rows, this->n_cols, 0.0);
    got(2, 5) = this->new_val;
    EXPECT_EQ(this->new_val, got(2, 5));
}

TEST_F(SparseMatrixTest, AssignmentOperatorWorks) {
    seamat::SparseMatrix<double> got(2, 5, 0.0);
    seamat::SparseMatrix<double> to_assign(this->expected_mat, this->n_rows, this->n_cols, 0.0);
    got = to_assign;
    EXPECT_EQ(this->n_rows, got.get_rows());
    EXPECT_EQ(this->n_cols, got.get_cols());
    for (uint32_t i = 0; i < this->n_rows; ++i) {
	for (uint32_t j = 0; j < this->n_cols; ++j) {
	    EXPECT_EQ(this->expected_mat[i*this->n_cols + j], got(i, j));
	}
    }    
}

TEST_F(SparseMatrixTest, ResizeThrowsRuntimeError) {
    seamat::SparseMatrix<double> got(2, 5, 0.0);
    EXPECT_THROW(got.resize(this->n_rows, this->n_cols, this->initial_val), std::runtime_error);
}
