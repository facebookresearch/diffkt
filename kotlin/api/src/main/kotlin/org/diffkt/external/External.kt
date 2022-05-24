/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.external

internal object External: ExternalLib {
    private const val DYLIB_NAME = "libops_jni"
    private var _isLoaded = false

    override val isLoaded get() = _isLoaded

    init {
        try {
            loadLib(DYLIB_NAME)
            _isLoaded = true
        } catch (e: Exception) { }
    }

    // Math

    external fun plus(
        a: FloatArray,
        b: FloatArray,
        res: FloatArray,
        size: Int
    )

    external fun minus(
        a: FloatArray,
        b: FloatArray,
        res: FloatArray,
        size: Int
    )

    external fun unaryMinus(
        a: FloatArray,
        res: FloatArray,
        size: Int
    )

    external fun times(
        a: FloatArray,
        b: FloatArray,
        res: FloatArray,
        size: Int
    )

    external fun exp(
        a: FloatArray,
        res: FloatArray,
        size: Int
    )

    external fun log(
        a: FloatArray,
        res: FloatArray,
        size: Int
    )

    @JvmStatic
    external fun lgamma(f: Float): Float

    external fun lgamma(
        a: FloatArray,
        res: FloatArray,
        size: Int
    )

    @JvmStatic
    external fun digamma(f: Float): Float

    external fun digamma(
        a: FloatArray,
        res: FloatArray,
        size: Int
    )

    @JvmStatic
    external fun polygamma(n: Int, f: Float): Float

    external fun polygamma(
        n: Int,
        a: FloatArray,
        res: FloatArray,
        size: Int
    )

    // Predicate

    external fun ifThenElse(
        p: FloatArray,
        a: FloatArray,
        b: FloatArray,
        res: FloatArray,
        size: Int
    )
}
