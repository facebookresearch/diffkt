/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.tracing

import org.diffkt.*

interface TracingScalar : TracingTensor, DScalar {
    override val primal: DScalar get() = this
    override val operations: Operations get() = TracingTensorOperations

    class Constant(values: DScalar) : TracingTensor.Constant(values), TracingScalar {
        override val shape: Shape get() = Shape()
        override fun <R> accept(v: TracingVisitor<R>): R { return v.visitConstant(this) }
    }
    class Variable(varIndex: Int, name: String? = null, traceId: TraceId) : TracingTensor.Variable(varIndex, name, Shape(), traceId), TracingScalar {
        override val shape: Shape get() = Shape()
    }
    class Plus(left: TracingScalar, right: TracingScalar) : TracingTensor.Plus(left, right), TracingScalar {
        override val shape: Shape get() = Shape()
    }
    class Minus(left: TracingScalar, right: TracingScalar) : TracingTensor.Minus(left, right), TracingScalar {
        override val shape: Shape get() = Shape()
    }
    class TimesScalar(left: TracingScalar, right: TracingScalar) : TracingTensor.TimesScalar(left, right), TracingScalar {
        override val shape: Shape get() = Shape()
    }
    class Div(left: TracingScalar, right: TracingScalar) : TracingTensor.Div(left, right), TracingScalar {
        override val shape: Shape get() = Shape()
    }
    class Zero() : TracingTensor.Zero(Shape()), TracingScalar {
        override val shape: Shape get() = Shape()
    }
    class UnaryMinus(x: TracingScalar) : TracingTensor.UnaryMinus(x), TracingScalar {
        override val shape: Shape get() = Shape()
    }
    class Matmul(
        x: TracingTensor, y: TracingTensor, a: Shape, b: Shape, c: Shape, d: Shape
    ) : TracingTensor.Matmul(x, y, a, b, c, d), TracingScalar {
        override val shape: Shape get() = Shape()
    }
    class Sin(x: TracingScalar) : TracingTensor.Sin(x), TracingScalar {
        override val shape: Shape get() = Shape()
    }
    class Cos(x: TracingScalar) : TracingTensor.Cos(x), TracingScalar {
        override val shape: Shape get() = Shape()
    }
    class Tan(x: TracingScalar) : TracingTensor.Tan(x), TracingScalar {
        override val shape: Shape get() = Shape()
    }
    class Atan(x: TracingScalar) : TracingTensor.Atan(x), TracingScalar {
        override val shape: Shape get() = Shape()
    }
    class Exp(x: TracingScalar) : TracingTensor.Exp(x), TracingScalar {
        override val shape: Shape get() = Shape()
    }
    class Ln(x: TracingScalar) : TracingTensor.Ln(x), TracingScalar {
        override val shape: Shape get() = Shape()
    }
    class Lgamma(x: TracingScalar) : TracingTensor.Lgamma(x), TracingScalar {
        override val shape: Shape get() = Shape()
    }
    class Digamma(x: TracingScalar) : TracingTensor.Digamma(x), TracingScalar {
        override val shape: Shape get() = Shape()
    }
    class Polygamma(n: Int, x: TracingScalar) : TracingTensor.Polygamma(n, x), TracingScalar {
        override val shape: Shape get() = Shape()
    }
    class Sqrt(x: TracingScalar) : TracingTensor.Sqrt(x), TracingScalar {
        override val shape: Shape get() = Shape()
    }
    class Tanh(x: TracingScalar) : TracingTensor.Tanh(x), TracingScalar {
        override val shape: Shape get() = Shape()
    }
    class Split(from: TracingScalar, shapes: List<Shape>) : TracingTensor.Split(from, shapes), TracingScalar {
        override val shape: Shape get() = Shape()
    }
    class SplitPart(from: TracingTensor, index: Int) : TracingTensor.SplitPart(from, index, Shape()), TracingScalar {
        override val shape: Shape get() = Shape()
    }
    class Pow(base: TracingScalar, exponent: Float): TracingTensor.Pow(base, exponent), TracingScalar {
        override val shape: Shape get() = Shape()
    }
    class View1(x: TracingTensor, indexes: IntArray) : TracingTensor.View1(x, indexes, Shape()), TracingScalar {
        override val shape: Shape get() = Shape()
    }
    class View2(x: TracingTensor, index: Int, axis: Int) : TracingTensor.View2(x, index, axis, Shape()), TracingScalar {
        override val shape: Shape get() = Shape()
    }
    class ReshapeToScalar(val x: TracingTensor) : TracingScalar {
        val precomputedHashCode = combineHash(ReshapeToScalar::class, x)
        override val shape: Shape get() = Shape()
        override fun <R> accept(v: TracingVisitor<R>): R { return v.visitReshapeToScalar(this) }
        override fun hashCode(): Int = precomputedHashCode
        override fun equals(other: Any?): Boolean =
            other is ReshapeToScalar &&
                    this.precomputedHashCode == other.precomputedHashCode &&
                    this.x == other.x
    }
    class Squeeze(x: TracingTensor) : TracingTensor.Squeeze(x, 0), TracingScalar {
        override val shape: Shape get() = Shape()
    }
    class Relu(x: TracingTensor) : TracingTensor.Relu(x), TracingScalar {
        override val shape: Shape get() = Shape()
    }
    class ReluGrad(x: TracingTensor, upstream: TracingTensor) : TracingTensor.ReluGrad(x, upstream), TracingScalar {
        override val shape: Shape get() = Shape()
    }
    class Sigmoid(x: TracingTensor) : TracingTensor.Sigmoid(x), TracingScalar {
        override val shape: Shape get() = Shape()
    }
    open class Sum(x: TracingTensor) : TracingTensor.Sum(x, axes = IntArray(x.rank) { it }, keepDims = false), TracingScalar {
        override val shape: Shape get() = Shape()
    }
    class Compare(left: TracingTensor, right: TracingTensor, comparison: ComparisonKind) : TracingTensor.Compare(left, right, comparison), TracingScalar {
        override val shape: Shape get() = Shape()
    }
    class IfThenElse(
        cond: TracingTensor,
        whenTrue: TracingTensor,
        whenFalse: TracingTensor
    ) : TracingTensor.IfThenElse(cond, whenTrue, whenFalse), TracingScalar {
        override val shape: Shape get() = Shape()
    }

    class FloatScalar(override val key: TracingRandomKey): TracingTensor.RandomFloats(key, Shape()), TracingScalar {
        override val shape: Shape = Shape()
    }
}
