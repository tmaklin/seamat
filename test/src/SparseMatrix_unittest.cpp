#include "SparseMatrix_unittest.hpp"

TEST_F(SparseMatrixTest, ParameterConstructorWorks) {
    seamat::SparseMatrix<double> got(this->n_rows, this->n_cols, this->initial_val);
    EXPECT_EQ(this->n_rows, got.get_rows());
    EXPECT_EQ(this->n_cols, got.get_cols());
    for (uint32_t i = 0; i < this->n_rows; ++i) {
	for (uint32_t j = 0; j < this->n_cols; ++j) {
	    EXPECT_EQ(this->expected_mat[i*this->n_cols + j], got(i, j));
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
	    EXPECT_EQ(this->expected_mat[i*this->n_cols + j], got(i, j));
	}
    }
}

TEST_F(SparseMatrixTest, CopyConstructorFromVectorWorks) {
    seamat::SparseMatrix<double> got(this->expected_mat, this->n_rows, this->n_cols, (double)0.0);
    EXPECT_EQ(this->n_rows, got.get_rows());
    EXPECT_EQ(this->n_cols, got.get_cols());
    for (uint32_t i = 0; i < got.get_rows(); ++i) {
	for (uint32_t j = 0; j < got.get_cols(); ++j) {
	    std::cerr << "(" << i << "," << j << "): " << expected_mat[i*this->n_cols + j] << '\t' << got(i, j) << std::endl;
	    EXPECT_EQ(this->expected_mat[i*this->n_cols + j], got(i, j));
	}
    }
}

TEST_F(SparseMatrixTest, CopyConstructorFrom2DVectorWorks) {
    std::vector<std::vector<double>> input(3, std::vector<double>(7, 2.7));
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

TEST_F(SparseMatrixTest, ResizeWorks) {
    seamat::SparseMatrix<double> got(2, 5, 0.0);
    got.resize(this->n_rows, this->n_cols, this->initial_val);
    EXPECT_EQ(this->n_rows, got.get_rows());
    EXPECT_EQ(this->n_cols, got.get_cols());
    for (uint32_t i = 0; i < 1; ++i) {
	for (uint32_t j = 0; j < 4; ++j) {
	    EXPECT_EQ(0.0, got(i, j));
	}
    }    
    for (uint32_t i = 1; i < this->n_rows; ++i) {
	for (uint32_t j = 4; j < this->n_cols; ++j) {
	    EXPECT_EQ(2.7, got(i, j));
	}
    }    
}
