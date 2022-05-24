/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

/**
 * A differentiable tensor of rank 0 containing a single float (a FloatTensor wrapper around a float).
 *
 *  @property value The floating point number to wrap with a FloatTensor.
 *  @constructor Creates a FloatScalar initialized to value.
 */
class FloatScalar(val value: Float) : FloatTensor(), DScalar {
    override val derivativeID: DerivativeID get() = NoDerivativeID
    override val operations: Operations
        get() = FloatScalarOperations
    override val shape: Shape get() = Shape()
    override val primal: DScalar get() = this

    override fun at(pos: Int): Float {
        require(pos == 0) { "indexing a scalar ..." }
        return value
    }

    override fun map(f: (Float)->Float): FloatScalar = FloatScalar(f(value))

    companion object {

        /**
         * A FloatScalar with the value of 0f.
         */
        val ZERO: FloatScalar = FloatScalar(0f)

        /**
         * A FloatScalar with the value of 1f.
         */
        val ONE: FloatScalar = FloatScalar(1f)

        /**
         * A FloatScalar with the value of PI.
         */
        val PI: FloatScalar = FloatScalar(Math.PI.toFloat())
    }

    override fun toString(): String = "${value}"
    override fun toCodeString(): String = "${value}f"
    override fun hashCode(): Int = FloatScalar::class.hashCode() * 101 + value.hashCode()
    override fun equals(other: Any?): Boolean = other is FloatScalar && this.value.equals(other.value)
}
