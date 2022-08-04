#include "Matrix_unittest.hpp"

TEST_F(MatrixOperatorsTest, DenseSummationWorks) {
    seamat::DenseMatrix<double> got(this->lhs, this->n_rows, this->n_cols);
    seamat::DenseMatrix<double> right(this->rhs, this->n_rows, this->n_cols);
    //seamat::DenseMatrix<double> got = left + right;
    got += right;
    EXPECT_EQ(this->n_rows, got.get_rows());
    EXPECT_EQ(this->n_cols, got.get_cols());
    for (uint32_t i = 0; i < this->n_rows; ++i) {
	for (uint32_t j = 0; j < this->n_cols; ++j) {
	    EXPECT_NEAR(this->expected[i*this->n_cols + j], got(i, j), 1e-5);
	}
    }
}
