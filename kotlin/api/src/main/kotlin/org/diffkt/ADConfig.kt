/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.adOptimize

annotation class StackImpl
annotation class BoxedPrimitive(val value:String)
annotation class ScalarRoot
annotation class ToUnboxedFunction(val functionName:String)
annotation class ToReverse(val fqClass:String)
annotation class DTensorRoot
annotation class ReverseScalarOperations
annotation class ForwardDifferentiable(val tangentProperty: String)

/**
 * a function is marked as a scalar noop if it accepts exactly one potentially
 * active operand and returns the same type. The return value MUST be the implicit receiver
 */
annotation class ScalarNoop

/**
 * The reverse differentiable scalar type should be annotated with this annotation
 * so the compiler plugin can build implementations of reverse nodes
 */
annotation class ReverseDifferentiable(val primalField:String, val upstreamField:String, val backpropogateMethod:String, val pushbackMethod:String, val derivativeID:String)

/**
 * This is used by the compiler when it cannot inline a function call.
 * The compiler expects the signature of the function annotated with this annotation
 * to be (DTensor, (DTensor) -> DTensor) -> Pair<DTensor, (DTensor)->DTensor>
 */
annotation class PrimalAndPullback

@StackImpl
class CodeGenStack<T> {
    val data = arrayListOf<T>()
    fun push(d:T){
        data.add(d)
    }
    fun pop():T {
        val x = data.last()
        data.removeLast()
        return x
    }
    fun top():T = data.last()
    fun notEmpty() = data.isNotEmpty()
}