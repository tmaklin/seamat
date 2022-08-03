#ifndef SEAMAT_SPARSEMATRIX_UNITTEST_HPP
#define SEAMAT_SPARSEMATRIX_UNITTEST_HPP

#include "gtest/gtest.h"

#include "Matrix.hpp"

// Test parameter constructor
class SparseMatrixTest : public ::testing::Test {
    protected:
    void SetUp() override {
	this->n_rows = 3;
	this->n_cols = 7;
	this->initial_val = 2.7;
	this->new_val = 3.14;
	this->expected_mat = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			       0.0, 2.7, 0.0, 0.0, 0.0, 0.0, 0.0,
			       2.7, 0.0, 0.0, 0.0, 0.0, 0.0, 2.7 };
    }
    void TearDown() override {
	this->expected_mat.clear();
	this->expected_mat.shrink_to_fit();
    }

    // Test inputs
    uint32_t n_rows;
    uint32_t n_cols;
    double initial_val;
    double new_val;

    // Expecteds
    std::vector<double> expected_mat;
};

#endif
