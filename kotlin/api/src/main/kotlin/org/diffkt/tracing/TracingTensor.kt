/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.tracing

import org.diffkt.*
import kotlin.math.ceil
import kotlin.math.pow

interface TracingTensor : DTensor, Traceable {
    override val derivativeID: DerivativeID get() = NoDerivativeID
    override val primal: DTensor get() = this
    override val operations: Operations get() = TracingTensorOperations
    fun floatEval(vars: FloatArray): Float = throw IllegalArgumentException(this::class.qualifiedName)

    companion object {
        fun wrap(x: DTensor): TracingTensor {
            return when (x) {
                is TracingTensor -> x
                is DScalar -> TracingScalar.Constant(x)
                else -> TracingTensor.Constant(x)
            }
        }

        abstract class TracingTensorBase(override val shape: Shape) : TracingTensor {
            init { require(shape.isScalar == (this is DScalar)) }
            override fun toString() = this.rawPrintedForm()
        }
    }

    open class Constant(val values: DTensor) : TracingTensorBase(values.shape) {
        init {
            require(values.derivativeID == NoDerivativeID)
        }
        val precomputedHashCode = combineHash(Constant::class, values)
        override fun <R> accept(v: TracingVisitor<R>): R = v.visitConstant(this)
        override fun floatEval(vars: FloatArray): Float = (values as FloatScalar).value
        override fun hashCode(): Int = precomputedHashCode
        override fun equals(other: Any?): Boolean =
            this === other ||
            other is Constant &&
            this.precomputedHashCode == other.precomputedHashCode &&
            this.values == other.values
        companion object {
            fun wrap(values: DTensor) = if (values is DScalar) TracingScalar.Constant(values) else TracingTensor.Constant(values)
        }
    }

    open class Variable(val varIndex: Int, var name: String? = null, shape: Shape, val traceId: TraceId) : TracingTensorBase(shape) {
        val precomputedHashCode = combineHash("Variable", varIndex, name, shape, traceId)
        override fun <R> accept(v: TracingVisitor<R>): R = v.visitVariable(this)
        override fun floatEval(vars: FloatArray): Float = vars[varIndex]
        override fun hashCode(): Int = precomputedHashCode
        override fun equals(other: Any?): Boolean =
            this === other ||
            other is Variable &&
                    this.precomputedHashCode == other.precomputedHashCode &&
                    this.varIndex == other.varIndex &&
                    this.name == other.name &&
                    this.shape == other.shape &&
                    this.traceId == other.traceId
    }

    open class Plus(val left: TracingTensor, val right: TracingTensor) : TracingTensorBase(left.shape) {
        val precomputedHashCode = combineHash("Plus", left, right)
        override fun <R> accept(v: TracingVisitor<R>): R = v.visitPlus(this)
        override fun floatEval(vars: FloatArray): Float {
            // This reduces the depth of recursion for deeply nested sums
            var sum = 0f
            var next = this
            while (true) {
                if (next.left is Plus) {
                    sum += next.right.floatEval(vars)
                    next = next.left as Plus
                } else if (next.right is Plus) {
                    sum += next.left.floatEval(vars)
                    next = next.right as Plus
                } else {
                    break;
                }
            }
            return sum + next.left.floatEval(vars) + next.right.floatEval(vars)
        }
        override fun hashCode(): Int = precomputedHashCode
        override fun equals(other: Any?): Boolean =
            this === other ||
            other is Plus &&
                    this.precomputedHashCode == other.precomputedHashCode &&
                    this.left == other.left &&
                    this.right == other.right
    }

    open class Minus(val left: TracingTensor, val right: TracingTensor) : TracingTensorBase(left.shape) {
        val precomputedHashCode = combineHash("Minus", left, right)
        override fun <R> accept(v: TracingVisitor<R>): R = v.visitMinus(this)
        override fun floatEval(vars: FloatArray): Float = left.floatEval(vars) - right.floatEval(vars)
        override fun hashCode(): Int = precomputedHashCode
        override fun equals(other: Any?): Boolean =
            this === other ||
            other is Minus &&
                    this.precomputedHashCode == other.precomputedHashCode &&
                    this.left == other.left &&
                    this.right == other.right
    }

    open class Times(val left: TracingTensor, val right: TracingTensor) : TracingTensorBase(left.shape) {
        val precomputedHashCode = combineHash("Times", left, right)
        override fun <R> accept(v: TracingVisitor<R>): R = v.visitTimes(this)
        override fun floatEval(vars: FloatArray): Float = left.floatEval(vars) * right.floatEval(vars)
        override fun hashCode(): Int = precomputedHashCode
        override fun equals(other: Any?): Boolean =
            this === other ||
            other is Times &&
                    this.precomputedHashCode == other.precomputedHashCode &&
                    this.left == other.left &&
                    this.right == other.right
    }

    open class TimesScalar(val left: TracingScalar, val right: TracingTensor) : TracingTensorBase(right.shape) {
        val precomputedHashCode = combineHash("TimesScalar", left, right)
        override fun <R> accept(v: TracingVisitor<R>): R = v.visitTimesScalar(this)
        override fun floatEval(vars: FloatArray): Float = left.floatEval(vars) * right.floatEval(vars)
        override fun hashCode(): Int = precomputedHashCode
        override fun equals(other: Any?): Boolean =
            this === other ||
            other is TimesScalar &&
                    this.precomputedHashCode == other.precomputedHashCode &&
                    this.left == other.left &&
                    this.right == other.right
    }

    open class Div(val left: TracingTensor, val right: TracingTensor) : TracingTensorBase(left.shape) {
        val precomputedHashCode = combineHash("Div", left, right)
        override fun <R> accept(v: TracingVisitor<R>): R = v.visitDiv(this)
        override fun floatEval(vars: FloatArray): Float = left.floatEval(vars) / right.floatEval(vars)
        override fun hashCode(): Int = precomputedHashCode
        override fun equals(other: Any?): Boolean =
            this === other ||
            other is Div &&
                    this.precomputedHashCode == other.precomputedHashCode &&
                    this.left == other.left &&
                    this.right == other.right
    }

    open class Zero(shape: Shape) : TracingTensorBase(shape) {
        init { require(shape.isScalar == (this is DScalar)) }
        val precomputedHashCode = combineHash("Zero", shape)
        override fun <R> accept(v: TracingVisitor<R>): R = v.visitZero(this)
        override fun floatEval(vars: FloatArray): Float = 0f
        override fun hashCode(): Int = precomputedHashCode
        override fun equals(other: Any?): Boolean =
            this === other ||
            other is Zero &&
                    this.precomputedHashCode == other.precomputedHashCode &&
                    this.shape == other.shape
    }

    class IdentityGradient(val halfShape: Shape) : TracingTensorBase(halfShape+halfShape) {
        val precomputedHashCode = combineHash("IdentityGradient", halfShape)
        override fun <R> accept(v: TracingVisitor<R>): R = v.visitIdentityGradient(this)
        override fun floatEval(vars: FloatArray): Float = 1f
        override fun hashCode(): Int = precomputedHashCode
        override fun equals(other: Any?): Boolean =
            this === other ||
            other is IdentityGradient &&
                    this.precomputedHashCode == other.precomputedHashCode &&
                    this.shape == other.shape
    }

    open class UnaryMinus(val x: TracingTensor) : TracingTensorBase(x.shape) {
        val precomputedHashCode = combineHash("UnaryMinus", x)
        override fun <R> accept(v: TracingVisitor<R>): R = v.visitUnaryMinus(this)
        override fun floatEval(vars: FloatArray): Float = - x.floatEval(vars)
        override fun hashCode(): Int = precomputedHashCode
        override fun equals(other: Any?): Boolean =
            this === other ||
            other is UnaryMinus &&
                    this.precomputedHashCode == other.precomputedHashCode &&
                    this.shape == other.shape
    }

    open class Matmul(
        val x: TracingTensor, val y: TracingTensor, val a: Shape, val b: Shape, val c: Shape, val d: Shape
    ) : TracingTensorBase(a + b + d) {
        init {
            assert(x.shape == a + b + c)
            assert(y.shape == a + c + d)
        }
        val precomputedHashCode = combineHash("Matmul", x, y, a, b, c, d)
        override fun <R> accept(v: TracingVisitor<R>): R = v.visitMatmul(this)
        override fun floatEval(vars: FloatArray): Float = x.floatEval(vars) * y.floatEval(vars)
        override fun hashCode(): Int = precomputedHashCode
        override fun equals(other: Any?): Boolean =
            this === other ||
            other is Matmul &&
                    this.precomputedHashCode == other.precomputedHashCode &&
                    this.x == other.x &&
                    this.y == other.y &&
                    this.a == other.a &&
                    this.b == other.b &&
                    this.c == other.c &&
                    this.d == other.d
    }

    open class OuterProduct(val x: TracingTensor, val y: TracingTensor) : TracingTensorBase(x.shape + y.shape) {
        val precomputedHashCode = combineHash("OuterProduct", x, y)
        override fun <R> accept(v: TracingVisitor<R>): R = v.visitOuterProduct(this)
        override fun floatEval(vars: FloatArray): Float = x.floatEval(vars) * y.floatEval(vars)
        override fun hashCode(): Int = precomputedHashCode
        override fun equals(other: Any?): Boolean =
            this === other ||
            other is OuterProduct &&
                    this.precomputedHashCode == other.precomputedHashCode &&
                    this.x == other.x &&
                    this.y == other.y
    }

    open class Sin(val x: TracingTensor) : TracingTensorBase(x.shape) {
        val precomputedHashCode = combineHash("Sin", x)
        override fun <R> accept(v: TracingVisitor<R>): R = v.visitSin(this)
        override fun floatEval(vars: FloatArray): Float = kotlin.math.sin(x.floatEval(vars))
        override fun hashCode(): Int = precomputedHashCode
        override fun equals(other: Any?): Boolean =
            this === other ||
            other is Sin &&
                    this.precomputedHashCode == other.precomputedHashCode &&
                    this.x == other.x
    }

    open class Cos(val x: TracingTensor) : TracingTensorBase(x.shape) {
        val precomputedHashCode = combineHash("Cos", x)
        override fun <R> accept(v: TracingVisitor<R>): R = v.visitCos(this)
        override fun floatEval(vars: FloatArray): Float = kotlin.math.cos(x.floatEval(vars))
        override fun hashCode(): Int = precomputedHashCode
        override fun equals(other: Any?): Boolean =
            this === other ||
            other is Cos &&
                    this.precomputedHashCode == other.precomputedHashCode &&
                    this.x == other.x
    }

    open class Tan(val x: TracingTensor) : TracingTensorBase(x.shape) {
        val precomputedHashCode = combineHash("Tan", x)
        override fun <R> accept(v: TracingVisitor<R>): R = v.visitTan(this)
        override fun floatEval(vars: FloatArray): Float = kotlin.math.tan(x.floatEval(vars))
        override fun hashCode(): Int = precomputedHashCode
        override fun equals(other: Any?): Boolean =
            this === other ||
            other is Tan &&
                    this.precomputedHashCode == other.precomputedHashCode &&
                    this.x == other.x
    }

    open class Atan(val x: TracingTensor) : TracingTensorBase(x.shape) {
        val precomputedHashCode = combineHash("Atan", x)
        override fun <R> accept(v: TracingVisitor<R>): R = v.visitAtan(this)
        override fun floatEval(vars: FloatArray): Float = kotlin.math.atan(x.floatEval(vars))
        override fun hashCode(): Int = precomputedHashCode
        override fun equals(other: Any?): Boolean =
            this === other ||
                    other is Atan &&
                    this.precomputedHashCode == other.precomputedHashCode &&
                    this.x == other.x
    }

    open class Exp(val x: TracingTensor) : TracingTensorBase(x.shape) {
        val precomputedHashCode = combineHash("Exp", x)
        override fun <R> accept(v: TracingVisitor<R>): R = v.visitExp(this)
        override fun floatEval(vars: FloatArray): Float = kotlin.math.exp(x.floatEval(vars))
        override fun hashCode(): Int = precomputedHashCode
        override fun equals(other: Any?): Boolean =
            this === other ||
            other is Exp &&
                    this.precomputedHashCode == other.precomputedHashCode &&
                    this.x == other.x
    }

    open class Ln(val x: TracingTensor) : TracingTensorBase(x.shape) {
        val precomputedHashCode = combineHash("Ln", x)
        override fun <R> accept(v: TracingVisitor<R>): R = v.visitLn(this)
        override fun floatEval(vars: FloatArray): Float = kotlin.math.ln(x.floatEval(vars))
        override fun hashCode(): Int = precomputedHashCode
        override fun equals(other: Any?): Boolean =
            this === other ||
            other is Ln &&
                    this.precomputedHashCode == other.precomputedHashCode &&
                    this.x == other.x
    }

    open class Lgamma(val x: TracingTensor) : TracingTensorBase(x.shape) {
        val precomputedHashCode = combineHash("Lgamma", x)
        override fun <R> accept(v: TracingVisitor<R>): R = v.visitLgamma(this)
        override fun floatEval(vars: FloatArray): Float = org.diffkt.external.Math.lgamma(x.floatEval(vars))
        override fun hashCode(): Int = precomputedHashCode
        override fun equals(other: Any?): Boolean =
            this === other ||
            other is Lgamma &&
                    this.precomputedHashCode == other.precomputedHashCode &&
                    this.x == other.x
    }

    open class Digamma(val x: TracingTensor) : TracingTensorBase(x.shape) {
        val precomputedHashCode = combineHash("Digamma", x)
        override fun <R> accept(v: TracingVisitor<R>): R = v.visitDigamma(this)
        override fun floatEval(vars: FloatArray): Float = org.diffkt.external.Math.digamma(x.floatEval(vars))
        override fun hashCode(): Int = precomputedHashCode
        override fun equals(other: Any?): Boolean =
            this === other ||
            other is Digamma &&
                    this.precomputedHashCode == other.precomputedHashCode &&
                    this.x == other.x
    }

    open class Polygamma(val n: Int, val x: TracingTensor) : TracingTensorBase(x.shape) {
        val precomputedHashCode = combineHash("Polygamma", n, x)
        override fun <R> accept(v: TracingVisitor<R>): R = v.visitPolygamma(this)
        override fun floatEval(vars: FloatArray): Float = org.diffkt.external.Math.polygamma(n, x.floatEval(vars))
        override fun hashCode(): Int = precomputedHashCode
        override fun equals(other: Any?): Boolean =
            this === other ||
            other is Polygamma &&
                    this.precomputedHashCode == other.precomputedHashCode &&
                    this.n == other.n &&
                    this.x == other.x
    }

    open class Sqrt(val x: TracingTensor) : TracingTensorBase(x.shape) {
        val precomputedHashCode = combineHash("Sqrt", x)
        override fun <R> accept(v: TracingVisitor<R>): R = v.visitSqrt(this)
        override fun floatEval(vars: FloatArray): Float = kotlin.math.sqrt(x.floatEval(vars))
        override fun hashCode(): Int = precomputedHashCode
        override fun equals(other: Any?): Boolean =
            this === other ||
            other is Sqrt &&
                    this.precomputedHashCode == other.precomputedHashCode &&
                    this.x == other.x
    }

    open class Tanh(val x: TracingTensor) : TracingTensorBase(x.shape) {
        val precomputedHashCode = combineHash("Tanh", x)
        override fun <R> accept(v: TracingVisitor<R>): R = v.visitTanh(this)
        override fun floatEval(vars: FloatArray): Float = kotlin.math.tanh(x.floatEval(vars))
        override fun hashCode(): Int = precomputedHashCode
        override fun equals(other: Any?): Boolean =
            this === other ||
            other is Tanh &&
                    this.precomputedHashCode == other.precomputedHashCode &&
                    this.x == other.x
    }

    class Meld(val values: List<TracingTensor>) : TracingTensorBase(Shape(values.map { it.size }.sum())) {
        val precomputedHashCode = combineHash("Meld", *values.toTypedArray())
        override fun <R> accept(v: TracingVisitor<R>): R = v.visitMeld(this)
        override fun hashCode(): Int = precomputedHashCode
        override fun equals(other: Any?): Boolean =
            this === other ||
            other is Meld &&
                    this.precomputedHashCode == other.precomputedHashCode &&
                    this.values == other.values
    }

    open class Split(val x: TracingTensor, val shapes: List<Shape>) : TracingTensorBase(x.shape) {
        init { require(shapes.map { it.product }.sum() == x.size) }
        val precomputedHashCode = combineHash("Split", x, *shapes.toTypedArray())
        val splitValues = shapes.indices.map {
            val partShape = shapes[it]
            if (partShape.isScalar) TracingScalar.SplitPart(this, it) else SplitPart(this, it, partShape)
        }

        override fun <R> accept(v: TracingVisitor<R>): R = v.visitSplit(this)
        override fun hashCode(): Int = precomputedHashCode
        override fun equals(other: Any?): Boolean =
            this === other ||
            other is Split &&
                    this.precomputedHashCode == other.precomputedHashCode &&
                    this.shapes == other.shapes
    }

    open class SplitPart(val from: TracingTensor, val index: Int, shape: Shape) : TracingTensorBase(shape) {
        val precomputedHashCode = combineHash("SplitPart", from, index)
        override fun <R> accept(v: TracingVisitor<R>): R = v.visitSplitPart(this)
        override fun hashCode(): Int = precomputedHashCode
        override fun equals(other: Any?): Boolean =
            this === other ||
            other is SplitPart &&
                    this.precomputedHashCode == other.precomputedHashCode &&
                    this.from == other.from &&
                    this.index == other.index
    }

    class Concat(val slices: List<TracingTensor>, val axis: Int) : TracingTensorBase(slices[0].shape.updated(axis, slices.map { it.shape[axis] }.sum())) {
        val precomputedHashCode = combineHash("Concat", slices, axis)
        override fun <R> accept(v: TracingVisitor<R>): R = v.visitConcat(this)
        override fun hashCode(): Int = precomputedHashCode
        override fun equals(other: Any?): Boolean =
            this === other ||
            other is Concat &&
                    this.precomputedHashCode == other.precomputedHashCode &&
                    this.slices == other.slices &&
                    this.axis == other.axis
    }

    class BroadcastTo(val x: TracingTensor, shape: Shape) : TracingTensorBase(shape) {
        val precomputedHashCode = combineHash("BroadcastTo", x, shape)
        override fun <R> accept(v: TracingVisitor<R>): R = v.visitBroadcastTo(this)
        override fun hashCode(): Int = precomputedHashCode
        override fun equals(other: Any?): Boolean =
            this === other ||
            other is BroadcastTo &&
                    this.precomputedHashCode == other.precomputedHashCode &&
                    this.x == other.x &&
                    this.shape == other.shape
    }

    class ConvImpl(
        val signal: TracingTensor,
        val filter: TracingTensor,
        val hStride: Int,
        val vStride: Int,
        val padding: Convolve.Padding2D) : TracingTensor {
        val precomputedHashCode = combineHash("ConvImpl", signal, filter, hStride, vStride, padding)
        override val shape: Shape = run {
            val signalShape = signal.shape
            val filterShape = filter.shape
            val numsignal = signalShape[Convolve.N_AXIS]
            val numfilter = filterShape[Convolve.N_AXIS]

            val endRow = signalShape[Convolve.H_AXIS] + padding.bottom - filterShape[Convolve.H_AXIS]
            val endCol = signalShape[Convolve.W_AXIS] + padding.right - filterShape[Convolve.W_AXIS]

            val outHeight = ceil((endRow + padding.top + 1).toFloat() / vStride).toInt()
            val outWidth = ceil((endCol + padding.left + 1).toFloat() / hStride).toInt()

            val outShape = Shape(numsignal, outHeight, outWidth, numfilter)
            outShape
        }

        override fun <R> accept(v: TracingVisitor<R>): R = v.visitConvImpl(this)
        override fun hashCode(): Int = precomputedHashCode
        override fun equals(other: Any?): Boolean =
            this === other ||
            other is ConvImpl &&
                    this.precomputedHashCode == other.precomputedHashCode &&
                    this.signal == other.signal &&
                    this.filter == other.filter &&
                    this.hStride == other.hStride &&
                    this.vStride == other.vStride &&
                    this.padding == other.padding
    }

    class Expand(val x: TracingTensor, shape: Shape) : TracingTensorBase(shape) {
        val precomputedHashCode = combineHash("Expand", x, shape)
        override fun <R> accept(v: TracingVisitor<R>): R = v.visitExpand(this)
        override fun hashCode(): Int = precomputedHashCode
        override fun equals(other: Any?): Boolean =
            this === other ||
            other is Expand &&
                    this.precomputedHashCode == other.precomputedHashCode &&
                    this.x == other.x &&
                    this.shape == other.shape
    }

    class Flip(val x: TracingTensor, val axes: IntArray) : TracingTensorBase(x.shape) {
        val precomputedHashCode = combineHash("Flip", x, axes.contentHashCode())
        override fun <R> accept(v: TracingVisitor<R>): R = v.visitFlip(this)
        override fun hashCode(): Int = precomputedHashCode
        override fun equals(other: Any?): Boolean =
            this === other ||
            other is Flip &&
                    this.precomputedHashCode == other.precomputedHashCode &&
                    this.x == other.x &&
                    this.axes contentEquals other.axes
    }

    class LogSoftmax(val x: TracingTensor, val axis: Int) : TracingTensorBase(x.shape) {
        val precomputedHashCode = combineHash("LogSoftmax", x, axis)
        override fun <R> accept(v: TracingVisitor<R>): R = v.visitLogSoftmax(this)
        override fun hashCode(): Int = precomputedHashCode
        override fun equals(other: Any?): Boolean =
            this === other ||
            other is LogSoftmax &&
                    this.precomputedHashCode == other.precomputedHashCode &&
                    this.x == other.x &&
                    this.axis == other.axis
    }

    class LogSoftmaxGrad(
        val x: TracingTensor,
        val axis: Int,
        val logSoftmax: TracingTensor,
        val upstream: TracingTensor
    ) : TracingTensorBase(x.shape) {
        val precomputedHashCode = combineHash("LogSoftmaxGrad", x, axis, logSoftmax, upstream)
        override fun <R> accept(v: TracingVisitor<R>): R = v.visitLogSoftmaxGrad(this)
        override fun hashCode(): Int = precomputedHashCode
        override fun equals(other: Any?): Boolean =
            this === other ||
            other is LogSoftmaxGrad &&
                    this.precomputedHashCode == other.precomputedHashCode &&
                    this.x == other.x &&
                    this.axis == other.axis &&
                    this.logSoftmax == other.logSoftmax &&
                    this.upstream == other.upstream
    }

    open class Pow(val base: TracingTensor, val exponent: Float) : TracingTensorBase(base.shape) {
        val precomputedHashCode = combineHash("Pow", base, exponent)
        override fun <R> accept(v: TracingVisitor<R>): R = v.visitPow(this)
        override fun floatEval(vars: FloatArray): Float = base.floatEval(vars).pow(exponent)
        override fun hashCode(): Int = precomputedHashCode
        override fun equals(other: Any?): Boolean =
            this === other ||
            other is Pow &&
                    this.precomputedHashCode == other.precomputedHashCode &&
                    this.base == other.base &&
                    this.exponent.equals(other.exponent)
    }

    open class View1(val x: TracingTensor, val indexes: IntArray, shape: Shape) : TracingTensorBase(shape) {
        val precomputedHashCode = combineHash("View1", x, indexes.contentHashCode(), shape)
        override fun <R> accept(v: TracingVisitor<R>): R = v.visitView1(this)
        override fun hashCode(): Int = precomputedHashCode
        override fun equals(other: Any?): Boolean =
            this === other ||
            other is View1 &&
                    this.precomputedHashCode == other.precomputedHashCode &&
                    this.x == other.x &&
                    this.indexes contentEquals other.indexes &&
                    this.shape == other.shape
    }

    open class View2(val x: TracingTensor, val index: Int, val axis: Int, shape: Shape) : TracingTensorBase(shape) {
        val precomputedHashCode = combineHash("View2", x, index, axis, shape)
        override fun <R> accept(v: TracingVisitor<R>): R = v.visitView2(this)
        override fun hashCode(): Int = precomputedHashCode
        override fun equals(other: Any?): Boolean =
            this === other ||
            other is View2 &&
                    this.precomputedHashCode == other.precomputedHashCode &&
                    this.x == other.x &&
                    this.index == other.index &&
                    this.axis == other.axis &&
                    this.shape == other.shape
    }

    class View3(val x: TracingTensor, val index: IntRange, val axis: Int
    ) : TracingTensorBase(x.shape.updated(axis, 1 + (index.endInclusive - index.start) / index.step)) {
        val precomputedHashCode = combineHash("View3", x, index, axis)
        override fun <R> accept(v: TracingVisitor<R>): R = v.visitView3(this)
        override fun hashCode(): Int = precomputedHashCode
        override fun equals(other: Any?): Boolean =
            this === other ||
            other is View3 &&
                    this.precomputedHashCode == other.precomputedHashCode &&
                    this.index == other.index &&
                    this.axis == other.axis
    }

    class Reshape(val x: TracingTensor, shape: Shape) : TracingTensorBase(shape) {
        val precomputedHashCode = combineHash("Reshape", x, shape)
        override fun <R> accept(v: TracingVisitor<R>): R = v.visitReshape(this)
        override fun hashCode(): Int = precomputedHashCode
        override fun equals(other: Any?): Boolean =
            this === other ||
            other is Reshape &&
                    this.precomputedHashCode == other.precomputedHashCode &&
                    this.x == other.x &&
                    this.shape == other.shape
    }

    open class Squeeze(val x: TracingTensor, val axis: Int) : TracingTensorBase(x.shape.remove(axis)) {
        val precomputedHashCode = combineHash("Squeeze", x, axis)
        override fun <R> accept(v: TracingVisitor<R>): R = v.visitSqueeze(this)
        override fun hashCode(): Int = precomputedHashCode
        override fun equals(other: Any?): Boolean =
            this === other ||
            other is Squeeze &&
                    this.precomputedHashCode == other.precomputedHashCode &&
                    this.x == other.x &&
                    this.axis == other.axis
    }

    class Unsqueeze(val x: TracingTensor, val axis: Int
    ) : TracingTensorBase(x.shape.take(axis) + 1 + x.shape.drop(axis)) {
        val precomputedHashCode = combineHash("Unsqueeze", x, axis)
        override fun <R> accept(v: TracingVisitor<R>): R = v.visitUnsqueeze(this)
        override fun hashCode(): Int = precomputedHashCode
        override fun equals(other: Any?): Boolean =
            this === other ||
            other is Unsqueeze &&
                    this.precomputedHashCode == other.precomputedHashCode &&
                    this.x == other.x &&
                    this.axis == other.axis
    }

    class Transpose(val x: TracingTensor, val axes: IntArray
    ) : TracingTensorBase(Shape(axes.map { i -> x.shape.dims[i] })) {
        val precomputedHashCode = combineHash("Transpose", x, axes.contentHashCode())
        override fun <R> accept(v: TracingVisitor<R>): R = v.visitTranspose(this)
        override fun hashCode(): Int = precomputedHashCode
        override fun equals(other: Any?): Boolean =
            this === other ||
            other is Transpose &&
                    this.precomputedHashCode == other.precomputedHashCode &&
                    this.x == other.x &&
                    this.axes contentEquals other.axes
    }

    open class Relu(val x: TracingTensor) : TracingTensorBase(x.shape) {
        val precomputedHashCode = combineHash("Relu", x)
        override fun <R> accept(v: TracingVisitor<R>): R = v.visitRelu(this)
        override fun floatEval(vars: FloatArray): Float {
            val xx = x.floatEval(vars)
            return if (xx > 0f) xx else 0f
        }
        override fun hashCode(): Int = precomputedHashCode
        override fun equals(other: Any?): Boolean =
            this === other ||
            other is Relu &&
                    this.precomputedHashCode == other.precomputedHashCode &&
                    this.x == other.x
    }

    open class ReluGrad(val x: TracingTensor, val upstream: TracingTensor) : TracingTensorBase(x.shape) {
        val precomputedHashCode = combineHash("ReluGrad", x, upstream)
        override fun <R> accept(v: TracingVisitor<R>): R = v.visitReluGrad(this)
        override fun floatEval(vars: FloatArray): Float {
            val xx = x.floatEval(vars)
            return if (xx > 0f) upstream.floatEval(vars) else 0f
        }
        override fun hashCode(): Int = precomputedHashCode
        override fun equals(other: Any?): Boolean =
            this === other ||
            other is ReluGrad &&
                    this.precomputedHashCode == other.precomputedHashCode &&
                    this.x == other.x &&
                    this.upstream == other.upstream
    }

    open class Sigmoid(val x: TracingTensor) : TracingTensorBase(x.shape) {
        val precomputedHashCode = combineHash("Sigmoid", x)
        override fun <R> accept(v: TracingVisitor<R>): R = v.visitSigmoid(this)
        override fun floatEval(vars: FloatArray): Float = sigmoidElem(x.floatEval(vars))
        override fun hashCode(): Int = precomputedHashCode
        override fun equals(other: Any?): Boolean =
            this === other ||
            other is Sigmoid &&
                    this.precomputedHashCode == other.precomputedHashCode &&
                    this.x == other.x
    }

    open class Sum(val x: TracingTensor, val axes: IntArray, val keepDims: Boolean
    ) : TracingTensorBase(if (keepDims)
                Shape(x.shape.dims.mapIndexed { ix, it -> if (ix in axes) 1 else it })
            else
                Shape(x.shape.dims.filterIndexed { ix, _ -> ix !in axes })) {
        val precomputedHashCode = combineHash("Sum", x, axes.contentHashCode(), keepDims)
        override fun <R> accept(v: TracingVisitor<R>): R = v.visitSum(this)
        override fun hashCode(): Int = precomputedHashCode
        override fun equals(other: Any?): Boolean =
            this === other ||
            other is Sum &&
                    this.precomputedHashCode == other.precomputedHashCode &&
                    this.x == other.x &&
                    this.axes contentEquals other.axes &&
                    this.keepDims == other.keepDims
    }

    open class AvgPool(val x: TracingTensor, val poolHeight: Int, val poolWidth: Int) : TracingTensor {
        val precomputedHashCode = combineHash("AvgPool", x, poolHeight, poolWidth)
        override val shape = run {
            val numItems = x.shape[Convolve.N_AXIS]
            val inHeight = x.shape[Convolve.H_AXIS]
            val inWidth = x.shape[Convolve.W_AXIS]
            val channels = x.shape.drop(Convolve.C_AXIS)

            val outHeight = inHeight / poolHeight
            val outWidth = inWidth / poolWidth

            val outShape = Shape(numItems, outHeight, outWidth) + channels
            outShape
        }

        override fun <R> accept(v: TracingVisitor<R>): R = v.visitAvgPool(this)
        override fun hashCode(): Int = precomputedHashCode
        override fun equals(other: Any?): Boolean =
            this === other ||
            other is AvgPool &&
                    this.precomputedHashCode == other.precomputedHashCode &&
                    this.x == other.x &&
                    this.poolHeight == other.poolHeight &&
                    this.poolWidth == other.poolWidth
    }

    open class AvgPoolGrad(val x: TracingTensor, val poolHeight: Int, val poolWidth: Int) : TracingTensor {
        val precomputedHashCode = combineHash("AvgPoolGrad", x, poolHeight, poolWidth)
        override val shape = run {
            val numItems = x.shape[Convolve.N_AXIS]
            val inHeight = x.shape[Convolve.H_AXIS]
            val inWidth = x.shape[Convolve.W_AXIS]
            val channels = x.shape.drop(Convolve.C_AXIS)
            val outHeight = inHeight * poolHeight
            val outWidth = inWidth * poolWidth

            val outShape = Shape(numItems, outHeight, outWidth) + channels
            outShape
        }

        override fun <R> accept(v: TracingVisitor<R>): R = v.visitAvgPoolGrad(this)
        override fun hashCode(): Int = precomputedHashCode
        override fun equals(other: Any?): Boolean =
            this === other ||
            other is AvgPoolGrad &&
                    this.precomputedHashCode == other.precomputedHashCode &&
                    this.x == other.x &&
                    this.poolHeight == other.poolHeight &&
                    this.poolWidth == other.poolWidth
    }

    class MaxPoolWithIndices(
        val x: TracingTensor,
        val poolHeight: Int,
        val poolWidth: Int,
        val withIndices: Boolean
    ) : TracingTensor {
        val precomputedHashCode = combineHash("MaxPoolWithIndices", x, poolHeight, poolWidth, withIndices)
        override val shape = run {
            val numItems = x.shape[Convolve.N_AXIS]
            val inHeight = x.shape[Convolve.H_AXIS]
            val inWidth = x.shape[Convolve.W_AXIS]
            val numChannels = x.shape[Convolve.C_AXIS]
            val outHeight = inHeight / poolHeight
            val outWidth = inWidth / poolWidth
            val outShape = Shape(numItems, outHeight, outWidth, numChannels)
            outShape
        }

        override fun <R> accept(v: TracingVisitor<R>): R = v.visitMaxPoolWithIndices(this)
        override fun hashCode(): Int = precomputedHashCode
        override fun equals(other: Any?): Boolean =
            this === other ||
            other is MaxPoolWithIndices &&
                    this.precomputedHashCode == other.precomputedHashCode &&
                    this.x == other.x &&
                    this.poolHeight == other.poolHeight &&
                    this.poolWidth == other.poolWidth &&
                    this.withIndices == other.withIndices
    }

    class Gather(
        val x: TracingTensor,
        val indexes: List<Int>,
        val axis: Int,
        val paddingIndex: Int) : TracingTensorBase(x.shape.updated(axis, indexes.size)) {
        val precomputedHashCode = combineHash("Gather", x, indexes, axis, paddingIndex)
        override fun <R> accept(v: TracingVisitor<R>): R = v.visitGather(this)
        override fun hashCode(): Int = precomputedHashCode
        override fun equals(other: Any?): Boolean =
            this === other ||
            other is Gather &&
                    this.precomputedHashCode == other.precomputedHashCode &&
                    this.x == other.x &&
                    this.indexes == other.indexes &&
                    this.axis == other.axis &&
                    this.paddingIndex == other.paddingIndex
    }

    class GatherAtIndices(val x: TracingTensor, val indexes: List<IntArray>) : TracingTensorBase(run {
            val shapePerIndex = x.shape.drop(indexes[0].size)
            val newShape = shapePerIndex.prepend(indexes.size)
            newShape
        }) {
        val precomputedHashCode = combineHash("GatherAtIndices", x, indexes)
        override fun <R> accept(v: TracingVisitor<R>): R = v.visitGatherAtIndices(this)
        override fun hashCode(): Int = precomputedHashCode
        override fun equals(other: Any?): Boolean =
            this === other ||
            other is GatherAtIndices &&
                    this.precomputedHashCode == other.precomputedHashCode &&
                    this.x == other.x &&
                    this.indexes == other.indexes
    }

    class Scatter(
        val x: TracingTensor,
        val indexes: List<Int>,
        val axis: Int,
        newShape: Shape,
        val paddingIndex: Int) : TracingTensorBase(newShape) {
        val precomputedHashCode = combineHash("Scatter", x, indexes, shape, paddingIndex)
        override fun <R> accept(v: TracingVisitor<R>): R = v.visitScatter(this)
        override fun hashCode(): Int = precomputedHashCode
        override fun equals(other: Any?): Boolean =
            this === other ||
            other is Scatter &&
                    this.precomputedHashCode == other.precomputedHashCode &&
                    this.x == other.x &&
                    this.indexes == other.indexes &&
                    this.axis == other.axis &&
                    this.shape == other.shape &&
                    this.paddingIndex == other.paddingIndex
    }

    class ScatterAtIndices(
        val x: TracingTensor,
        val indexes: List<IntArray>,
        shape: Shape
    ) : TracingTensorBase(shape) {
        val precomputedHashCode = combineHash("ScatterAtIndices", x, indexes, shape)
        override fun <R> accept(v: TracingVisitor<R>): R = v.visitScatterAtIndices(this)
        override fun hashCode(): Int = precomputedHashCode
        override fun equals(other: Any?): Boolean =
            this === other ||
            other is ScatterAtIndices &&
                    this.precomputedHashCode == other.precomputedHashCode &&
                    this.x == other.x &&
                    this.indexes == other.indexes
    }

    open class Compare(
        val left: TracingTensor,
        val right: TracingTensor,
        val comparison: ComparisonKind
    ) : TracingTensorBase(left.shape) {
        val precomputedHashCode = combineHash("Compare", left, right, comparison)
        override fun <R> accept(v: TracingVisitor<R>): R = v.visitCompare(this)
        override fun floatEval(vars: FloatArray): Float =
            if (compare(left.floatEval(vars), right.floatEval(vars), comparison)) 1f else 0f
        override fun hashCode(): Int = precomputedHashCode
        override fun equals(other: Any?): Boolean =
            this === other ||
                    other is Compare &&
                    this.left == other.left &&
                    this.right == other.right &&
                    this.comparison == other.comparison
    }

    open class IfThenElse(
        val cond: TracingTensor,
        val whenTrue: TracingTensor,
        val whenFalse: TracingTensor
    ) : TracingTensorBase(whenTrue.shape) {
        val precomputedHashCode = combineHash("IfThenElse", cond, whenTrue, whenFalse)
        override fun <R> accept(v: TracingVisitor<R>): R = v.visitIfThenElse(this)
        override fun floatEval(vars: FloatArray): Float =
            ifThenElse(cond.floatEval(vars), whenTrue.floatEval(vars), whenFalse.floatEval(vars))
        override fun hashCode(): Int = precomputedHashCode
        override fun equals(other: Any?): Boolean =
            this === other ||
                    other is IfThenElse &&
                    this.cond == other.cond &&
                    this.whenTrue == other.whenTrue &&
                    this.whenFalse == other.whenFalse
    }

    open class RandomFloats(open val key: TracingRandomKey, shape: Shape): TracingTensorBase(shape) {
        private val precomputedHashCode = combineHash("Random.Floats", key, shape)
        val traceId: TraceId get() = key.traceId
        override fun <R> accept(v: TracingVisitor<R>): R = v.visitRandomFloats(this)
        override fun toString(): String = "$key.floats($shape)"
        override fun hashCode(): Int = precomputedHashCode
        override fun equals(other: Any?): Boolean {
            return other is RandomFloats && other.key == this.key && shape == other.shape
        }
    }
}
