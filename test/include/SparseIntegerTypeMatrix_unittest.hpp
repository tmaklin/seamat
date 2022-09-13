// seamat: templatized matrix library
// https://github.com/tmaklin/seamat
//
// Copyright (C) 2021 Tommi Mäklin (tommi@maklin.fi)
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
#ifndef SEAMAT_SPARSEINTEGERTYPEMATRIX_UNITTEST_HPP
#define SEAMAT_SPARSEINTEGERTYPEMATRIX_UNITTEST_HPP

#include "gtest/gtest.h"

#include "Matrix.hpp"

// Test parameter constructor
class SparseIntegerTypeMatrixTest : public ::testing::Test {
    protected:
    void SetUp() override {
	this->n_rows = 3;
	this->n_cols = 7;
	this->initial_val = 2;
	this->new_val = 3;
	this->expected_mat = { 0, 0, 0, 0, 0, 0, 0,
			       0, 2, 0, 0, 0, 0, 0,
			       2, 0, 0, 0, 0, 0, 2 };
    }
    void TearDown() override {
	this->expected_mat.clear();
	this->expected_mat.shrink_to_fit();
    }

    // Test inputs
    uint32_t n_rows;
    uint32_t n_cols;
    int initial_val;
    int new_val;

    // Expecteds
    std::vector<unsigned> expected_mat;
};

#endif
