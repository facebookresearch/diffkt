/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.external

internal object Math {

    fun plus(a: FloatArray, b: FloatArray, size: Int): FloatArray {
        val res = FloatArray(size)
        External.plus(a, b, res, size)
        return res
    }

    fun minus(a: FloatArray, b: FloatArray, size: Int): FloatArray {
        val res = FloatArray(size)
        External.minus(a, b, res, size)
        return res
    }

    fun unaryMinus(a: FloatArray, size: Int): FloatArray {
        val res = FloatArray(size)
        External.unaryMinus(a, res, size)
        return res
    }

    fun exp(a: FloatArray, size: Int): FloatArray {
        val res = FloatArray(size)
        External.exp(a, res, size)
        return res
    }

    fun log(a: FloatArray, size: Int): FloatArray {
        val res = FloatArray(size)
        External.log(a, res, size)
        return res
    }

    fun times(a: FloatArray, b: FloatArray, size: Int): FloatArray {
        val res = FloatArray(size)
        External.times(a, b, res, size)
        return res
    }

    fun lgamma(a: FloatArray): FloatArray {
        val res = FloatArray(a.size)
        External.lgamma(a, res, a.size)
        return res
    }

    fun digamma(a: FloatArray): FloatArray {
        val res = FloatArray(a.size)
        External.digamma(a, res, a.size)
        return res
    }

    fun polygamma(n: Int, a: FloatArray): FloatArray {
        val res = FloatArray(a.size)
        External.polygamma(n, a, res, a.size)
        return res
    }

    // Scalars going to C++ for now.
    fun lgamma(f: Float): Float {
        return External.lgamma(f)
    }

    fun digamma(f: Float): Float {
        return External.digamma(f)
    }

    fun polygamma(n: Int, f: Float): Float {
        return External.polygamma(n, f)
    }

}
