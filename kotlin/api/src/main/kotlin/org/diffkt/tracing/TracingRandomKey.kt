/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.tracing

import org.diffkt.*
import org.diffkt.random.RandomKey
import java.lang.IllegalArgumentException

sealed interface TracingRandomKey : RandomKey, Traceable {

    val traceId: TraceId

    sealed interface RandomBase: TracingRandomKey {

        override fun floats(shape: Shape): TracingTensor {
            return if (shape.isScalar) TracingScalar.FloatScalar(this) else TracingTensor.RandomFloats(this, shape)
        }

        override fun split(n: Int): List<RandomKey> {
            return Split(this, n).parts
        }

        override fun permitReuse(): RandomKey {
            TODO("Not yet implemented")
        }

        override fun gaussian(shape: Shape): DTensor {
            TODO("Not yet implemented")
        }

        override fun gamma(alpha: FloatTensor): DTensor {
            TODO("Not yet implemented")
        }
    }

    class RandomTensorWrapper(val key: RandomKey): DTensor {
        override val derivativeID: DerivativeID = NoDerivativeID
        override val primal: DTensor = this
        override val operations: Operations
            get() = throw IllegalAccessError("MockDTensor does not implement tensor operations")
    }

    class SplitRandom(val keys: List<RandomKey>): RandomKey {
        override fun split(n: Int) = throw IllegalArgumentException("Illegal operation on MockRandom")
        override fun floats(shape: Shape) = throw IllegalArgumentException("Illegal operation on MockRandom")
        override fun permitReuse() = throw IllegalArgumentException("Illegal operation on MockRandom")
        override fun gaussian(shape: Shape) = throw IllegalArgumentException("Illegal operation on MockRandom")
        override fun gamma(alpha: FloatTensor) = throw IllegalArgumentException("Illegal operation on MockRandom")
    }

    class Variable(val index: Int, override val traceId: TraceId, val name: String? = null): RandomBase {
        override fun <R> accept(v: TracingVisitor<R>): R = v.visitRandomVariable(this)
        private val precomputedHashCode = combineHash("Random.Variable", index, traceId)
        override fun toString(): String = name ?: "r$index"
        override fun hashCode(): Int = precomputedHashCode
        override fun equals(other: Any?): Boolean {
            return other is Variable && other.traceId == this.traceId && other.index == this.index
        }
    }

    class Split(val key: TracingRandomKey, val n: Int): RandomBase {
        override fun <R> accept(v: TracingVisitor<R>): R = v.visitRandomSplit(this)
        override val traceId: TraceId = key.traceId
        private val precomputedHashCode = combineHash("Random.split", n)
        override fun toString(): String = "split(key = $key, $n)"
        override fun hashCode(): Int = precomputedHashCode
        override fun equals(other: Any?): Boolean = other is Split && key == other.key && n == other.n
        val parts = LazyList<TracingRandomKey>(n) { SplitPart(this, it) }
    }

    class SplitPart(val split: TracingRandomKey, val splitIndex: Int): RandomBase {
        override fun <R> accept(v: TracingVisitor<R>): R = v.visitRandomSplitPart(this)
        override val traceId: TraceId = split.traceId
        private val precomputedHashCode = combineHash("Random.SplitPart", split, splitIndex)
        override fun toString(): String = "$split[$splitIndex]"
        override fun hashCode(): Int = precomputedHashCode
        override fun equals(other: Any?): Boolean = other is SplitPart && split == other.split
                && splitIndex == other.splitIndex
    }

}