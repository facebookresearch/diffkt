/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.customReverse

import org.diffkt.*
import org.diffkt.reverse.ReverseScalar
import org.diffkt.reverse.ReverseTensor
import kotlin.math.cos
import kotlin.math.sin

/**
 * Demonstrate how to implement a custom reverse implementation for your operation.
 */
fun mySin(x: DTensor): DTensor {
    return when (x) {
        is FloatTensor -> x.map { v -> sin(v) }
        is ReverseTensor -> if (x is DScalar) {
            object : ReverseScalar(mySin(x.primal) as DScalar, x.derivativeID) {
                override fun backpropagate() {
                    x.pushback(myCos(x.primal).expandToTangent(upstream) * upstream)
                }
            }
        } else {
            object : ReverseTensor(mySin(x.primal) , x.derivativeID) {
                override fun backpropagate() {
                    x.pushback(myCos(x.primal).expandToTangent(upstream) * upstream)
                }
            }
        }
        // Ideally you'd have a general fallback using (possibly less efficient) primitives
        else -> sin(x)
    }
}

fun myCos(x: DTensor): DTensor {
    return when (x) {
        is FloatTensor -> x.map { v -> cos(v) }
        is ReverseTensor -> if (x is DScalar) {
            object : ReverseScalar(myCos(x.primal) as DScalar, x.derivativeID) {
                override fun backpropagate() {
                    x.pushback(-mySin(x.primal).expandToTangent(upstream) * upstream)
                }
            }
        } else {
            object : ReverseTensor(myCos(x.primal) , x.derivativeID) {
                override fun backpropagate() {
                    x.pushback(-mySin(x.primal).expandToTangent(upstream) * upstream)
                }
            }
        }
        else -> cos(x)
    }
}

fun main() {
    val x = tensorOf(1f, 1.1f, 2.3f, 2.7f)
    println(mySin(x))
    println(sin(x))
    println(reverseDerivative(x, ::mySin))
    println(reverseDerivative(x, ::sin))
}