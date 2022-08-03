// seamat: templatized matrix library
// https://github.com/tmaklin/seamat
//
// Copyright (C) 2021 Tommi Mäklin (tommi@maklin.fi)
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
#ifndef SEAMAT_INDEX_MATRIX_CPP
#define SEAMAT_INDEX_MATRIX_CPP
#include "Matrix.hpp"

#include <cmath>

#include "openmp_config.hpp"

namespace seamat {
// Resize a matrix
template<typename T, typename U>
void IndexMatrix<T,U>::resize(const uint32_t new_rows, const uint32_t new_cols, const T initial) {
    if (new_rows != this->get_rows() || new_cols != this->get_cols()) {
	indices->resize(new_rows, new_cols, initial);
	this->resize_rows(new_rows);
	this->resize_cols(new_cols);
    }
}

// Access individual elements
template <typename T, typename U>
T& IndexMatrix<T,U>::operator()(uint32_t row, uint32_t col) {
    uint32_t out_col = (*this->indices)(row, col);
    return (*this->vals)(col, out_col);
}

// Access individual elements (const)
template <typename T, typename U>
const T& IndexMatrix<T,U>::operator()(uint32_t row, uint32_t col) const {
    uint32_t out_col = (*this->indices)(row, col);
    return (*this->vals)(col, out_col);
}
}

#endif
