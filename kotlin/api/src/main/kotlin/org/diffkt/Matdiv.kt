/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

import org.diffkt.external.SparseOps
import shapeTyping.annotations.SType

@SType("A: Dim, B: Dim")
fun @SType("[A,B]") DTensor.matdiv(that: @SType("[B,B]") DTensor): DTensor {
    // matrix-division is not supported by dense tensors operation yet
    require(this is SparseFloatTensor && that is SparseFloatTensor) {
        "Matrix-division is not yet supported in between dense tensors"}
    require(this.rank == 2 && that.rank == 2) {
        "Only 2 dimensional tensors are supported in Matrix division"}
    // inversion only supports square tensors
    require(that.shape[0] == that.shape[1]) {
      "Cannot divide by shape ${that.shape}: non square tensor is not supported" }
    require(this.shape[1] == that.shape[1]) {
      "Cannot divide shape ${this.shape} by ${that.shape}: inner dims do not match" }
    return SparseOps.matdiv(this, that)
}
