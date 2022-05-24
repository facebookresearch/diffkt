/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.external

object Predicate {
    fun ifThenElse(p: FloatArray, a: FloatArray, b: FloatArray, size: Int): FloatArray {
        val res = FloatArray(size)
        External.ifThenElse(p, a, b, res, size)
        return res
    }
}
