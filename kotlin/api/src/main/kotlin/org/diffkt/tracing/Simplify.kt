/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.tracing

import org.diffkt.*

fun <TData : Any> simplify(data: TData): TData {
    return simplifyingWrapper.wrap(data)
}

private object simplifyingWrapper : Wrapper() {
    override fun wrapDTensor(value: DTensor): DTensor {
        return when (value) {
            is TracingTensor ->
                simplifyingTracingRewriter.rewrite(value) as TracingTensor
            else -> value
        }
    }
}

private object simplifyingTracingRewriter : DeepTracingRewriter() {
    override fun rewriteOne(x: Traceable): Traceable {
        // produce a node with updated child nodes
        var withRewrittenChildren = super.rewriteOne(x)
        // then apply optimizations at the root
        return optimizer.rewrite(withRewrittenChildren)
    }
}

private object optimizer : ShallowTracingRewriter() {
    fun rewrite(x: Traceable): Traceable {
        // Apply local optimizations until a fixed point is reached
        var result = x
        while (true) {
            val optimized = visit(result)
            if (optimized === result) return result
            result = optimized
        }
    }

    fun rewrite(x: TracingTensor): TracingTensor = rewrite(x as Traceable) as TracingTensor

    override fun visitDiv(x: TracingTensor.Div): TracingTensor {
        if (x.left is TracingScalar.UnaryMinus) {
            return -(x.left.x / x.right) as TracingTensor
        }
        return x
    }
    fun removeBroadcast(left: TracingTensor, right: TracingTensor): Pair<TracingTensor, TracingTensor> {
        var l = left
        var r = right
        while (l is TracingTensor.Expand) l = l.x
        while (r is TracingTensor.Expand) r = r.x
        if (l is TracingTensor.Unsqueeze && r !is TracingTensor.Unsqueeze) {
            while (l is TracingTensor.Unsqueeze && l.let { ll -> (0 until ll.axis).all { ll.x.shape[it] == 1 }})
                l = l.x
        } else if (r is TracingTensor.Unsqueeze && l !is TracingTensor.Unsqueeze) {
            while (r is TracingTensor.Unsqueeze && r.let { rr -> (0 until rr.axis).all { rr.x.shape[it] == 1 }})
                r = r.x
        }
        return Pair(l, r)
    }
    override fun visitTimesScalar(x: TracingTensor.TimesScalar): TracingTensor {
        val left = x.left
        val right = x.right
        return when {
            right is TracingScalar.Constant && (right.values as FloatScalar).value == 1.0f -> left
            left is TracingScalar.Constant && (left.values as FloatScalar).value == 1.0f -> right
            right is TracingScalar.Zero -> right
            left is TracingScalar.Zero -> left
            right is TracingScalar.Constant && (right.values as FloatScalar).value == 0f -> right
            left is TracingScalar.Constant && (left.values as FloatScalar).value == 0f -> left
            left is TracingScalar.Constant && (left.values as FloatScalar).value == -1.0f ->
                right.operations.unaryMinus(right) as TracingTensor
            right is TracingScalar.Constant && (right.values as FloatScalar).value == -1.0f ->
                left.operations.unaryMinus(left) as TracingTensor
            right is TracingScalar.UnaryMinus ->
                -(x.left * (x.right as TracingScalar.UnaryMinus).x) as TracingTensor
            left is TracingScalar.UnaryMinus ->
                -((x.left as TracingScalar.UnaryMinus).x * x.right) as TracingTensor
            left is TracingTensor.UnaryMinus && right is TracingTensor.UnaryMinus ->
                (left.x * right.x) as TracingTensor
            left is TracingScalar.Pow && left.exponent == -1.0f ->
                (right / left.base) as TracingTensor // A.pow(-1)*B == B/A
            right is TracingScalar.Pow && right.exponent == -1.0f ->
                (left / right.base) as TracingTensor // B.pow(-1)*A == A/B
            left === x.left && right === x.right -> x
            else -> x.operations.timesScalar(left, right, NoDerivativeID) as TracingTensor
        }
    }

    override fun visitTimes(x: TracingTensor.Times): TracingTensor {
        val (left, right) = removeBroadcast(x.left, x.right)
        return when {
            right is TracingScalar -> visitTimesScalar(TracingTensor.TimesScalar(right, left))
            left is TracingScalar -> visitTimesScalar(TracingTensor.TimesScalar(left, right))
            right is TracingTensor.Constant && right.values is FloatTensor && right.values.all { it == 1F } -> left
            left is TracingTensor.Constant && left.values is FloatTensor && left.values.all { it == 1F } -> right
            left === x.left && right === x.right -> x
            else -> x.operations.times(left, right, NoDerivativeID) as TracingTensor
        }
    }

    override fun visitPlus(x: TracingTensor.Plus): TracingTensor {
        val (left, right) = removeBroadcast(x.left, x.right)
        return when {
            left is TracingTensor.Zero -> right
            right is TracingTensor.Zero -> left
            right is TracingTensor.Constant && right.values is FloatTensor && right.values.all { it == 0F } -> left
            left is TracingTensor.Constant && left.values is FloatTensor && left.values.all { it == 0F } -> right
            right is TracingTensor.UnaryMinus -> (left - right.x) as TracingTensor
            left == x.left && right == x.right -> x
            left is TracingScalar && right is TracingScalar -> TracingScalar.Plus(left, right)
            else -> x.operations.plus(left, right, NoDerivativeID) as TracingTensor
        }
    }

    override fun visitMinus(x: TracingTensor.Minus): TracingTensor {
        val (left, right) = removeBroadcast(x.left, x.right)
        return when {
            right is TracingTensor.Zero -> left
            left is TracingTensor.Zero -> right.operations.unaryMinus(right) as TracingTensor
            left is TracingTensor.Constant && left.values is FloatTensor && left.values.all { it == 0F } -> right.operations.unaryMinus(right) as TracingTensor
            right is TracingTensor.Constant && right.values is FloatTensor && right.values.all { it == 0F } -> left
            left is TracingTensor.UnaryMinus -> TracingTensorOperations.unaryMinus(TracingTensorOperations.plus(left.x, right, NoDerivativeID)) as TracingTensor
            left == x.left && right == x.right -> x
            else -> x.operations.minus(left, right, NoDerivativeID) as TracingTensor
        }
    }

    override fun visitUnaryMinus(x: TracingTensor.UnaryMinus): TracingTensor {
        val xx = x.x
        if (xx is TracingTensor.TimesScalar) {
            // replace -(A * -B) with (A * B)
            val l = xx.left
            val r = xx.right
            if (r is TracingTensor.UnaryMinus) {
                return (l * r.x) as TracingTensor
            }
        }
        if (xx is TracingTensor.UnaryMinus) {
            return xx.x
        }
        if (xx is TracingTensor.Constant) {
            val newValue = if (xx.values == FloatScalar.ZERO) FloatScalar.ZERO else -xx.values
            return TracingTensor.Constant.wrap(newValue)
        }
        if (xx is TracingTensor.Zero) {
            return xx
        }
        return x
    }

    override fun visitSin(x: TracingTensor.Sin): TracingTensor {
        val xx = x.x
        if (xx is TracingTensor.UnaryMinus) {
            // move the negation sign up over the sin
            val a = xx.x
            return -sin(a) as TracingTensor
        }
        return x
    }

    override fun visitPow(x: TracingTensor.Pow): TracingTensor {
        val xx = x.base
        if (xx is TracingTensor.Pow) {
            val pow1 = xx.exponent
            val pow2 = x.exponent
            val newExponent = pow1 * pow2
            return xx.base.pow(newExponent) as TracingTensor
        }
        return x
    }
    override fun visitView2(x: TracingTensor.View2): TracingTensor {
        fun makeView2(original: TracingTensor, shape: Shape): TracingTensor {
            // TODO: add more general handling for broadcasted arguments coming in here
            if (original is DScalar && x is DScalar) return original
            val o = original.broadcastTo(shape) as TracingTensor
            return if (x is DScalar) TracingScalar.View2(o, x.index, x.axis) else TracingTensor.View2(o, x.index, x.axis, shape)
        }
        // for many operations, view2(op(x)) == op(view2(x))
        // In this way we can push the view2 operation down until it can be eliminated.
        when (x.x) {
            is TracingTensor.Plus -> {
                val left = rewrite(makeView2(x.x.left, x.x.shape))
                val right = rewrite(makeView2(x.x.right, x.x.shape))
                if (left == right) {
                    val k2 = TracingScalar.Constant(FloatScalar(2f))
                    return if (x is DScalar) TracingScalar.TimesScalar(k2, right as TracingScalar) else TracingTensor.Plus(k2, right)
                }
                return if (x is DScalar) TracingScalar.Plus(left as TracingScalar, right as TracingScalar) else TracingTensor.Plus(left, right)
            }
            is TracingTensor.Minus -> {
                val left = rewrite(makeView2(x.x.left, x.x.shape))
                val right = rewrite(makeView2(x.x.right, x.x.shape))
                if (left == right) {
                    return TracingTensor.Constant.wrap(FloatTensor.zeros(x.shape))
                }
                return if (x is DScalar) TracingScalar.Minus(left as TracingScalar, right as TracingScalar) else TracingTensor.Plus(left, right)
            }
            is TracingTensor.Times -> {
                val left = rewrite(makeView2(x.x.left, x.x.shape))
                val right = rewrite(makeView2(x.x.right, x.x.shape))
                return if (x is DScalar) TracingScalar.TimesScalar(left as TracingScalar, right as TracingScalar) else TracingTensor.Times(left, right)
            }
            is TracingTensor.TimesScalar -> {
                val left = x.x.left
                val right = rewrite(makeView2(x.x.right, x.x.shape))
                return if (x is DScalar) TracingScalar.TimesScalar(left, right as TracingScalar) else TracingTensor.TimesScalar(left, right)
            }
            is TracingTensor.UnaryMinus -> {
                val arg = rewrite(makeView2(x.x.x, x.x.shape))
                return if (x is DScalar) TracingScalar.UnaryMinus(arg as TracingScalar) else TracingTensor.UnaryMinus(arg)
            }
            is TracingTensor.Sin -> {
                val arg = rewrite(makeView2(x.x.x, x.x.shape))
                return if (x is DScalar) TracingScalar.Sin(arg as TracingScalar) else TracingTensor.Sin(arg)
            }
            is TracingTensor.Cos -> {
                val arg = rewrite(makeView2(x.x.x, x.x.shape))
                return if (x is DScalar) TracingScalar.Cos(arg as TracingScalar) else TracingTensor.Cos(arg)
            }
            is TracingTensor.Constant -> {
                val result = x.x.values.view(x.index, x.axis)
                return TracingTensor.wrap(result)
            }
            is TracingTensor.Zero -> {
                return if (x is DScalar) TracingScalar.Zero() else TracingTensor.Zero(x.shape)
            }
            is TracingTensor.IdentityGradient -> {
                val result = StridedFloatTensor.identityGradient(x.x.halfShape).view(x.index, x.axis)
                return TracingTensor.wrap(result)
            }
            is TracingTensor.Reshape -> {
                // For a scalar s, s.reshape(1)[0] is the same as s.
                // TODO: generalize this to nonscalars where a dimension is added and then removed.
                if (x.x.rank == 1 && x.x.x is DScalar && x.index == 0)
                    return x.x.x
            }
            is TracingTensor.BroadcastTo -> {
                // If the broadcast is simply expanding a vector, we can eliminate the broadcast and the view.
                // TODO: we can generalize this
                if (x.x.rank == 1)
                    return TracingScalar.View2(x.x.x, 0, 0)
            }
        }
        return super.visitView2(x) as TracingTensor
    }

    override fun visitView3(x: TracingTensor.View3): TracingTensor {
        val xx = x.x
        if (xx is TracingTensor.Zero) {
            // A view of zero is zero.
            return TracingTensorOperations.zeroOfSameKind(x, x.shape) as TracingTensor
        }
        return x
    }

    override fun visitSplitPart(x: TracingTensor.SplitPart): TracingTensor {
        if (x.from !is TracingTensor.Split)
            return x
        // We optimize the pattern meld(x1, x2, ...).split(s1, s2, ...).splitPart(i)
        // Where s1 is the shape of x1, s2 is the shape of x2, etc.  In this case we
        // can replace the whole expression with xi.
        val xsplitfrom = x.from.x
        if (xsplitfrom is TracingTensor.Meld && xsplitfrom.values.map{it.shape} == x.from.shapes)
            return xsplitfrom.values[x.index]
        if (x.from.shapes.size == 1) {
            assert(x.index == 0)
            return xsplitfrom.reshape(x.from.shapes[0]) as TracingTensor
        }
        if (xsplitfrom is TracingTensor.IdentityGradient) {
            return makeConstant(x)
        }
        if (xsplitfrom is TracingTensor.Zero)
            return if (x is TracingScalar) TracingScalar.Zero() else TracingTensor.Zero(x.shape)
        if (xsplitfrom.rank == 1 && x.from.shapes.all { it.isScalar })
            return TracingScalar.View2(xsplitfrom, x.index, axis = 0)
        return x
    }

    val emptyVariables = Array<DTensor?>(0) { null }
    val constantEvaluator = TracingEvaluator(emptyVariables, TraceId())
    private fun makeConstant(x: TracingTensor): TracingTensor {
        val t = constantEvaluator.evaluate(x)
        return TracingTensor.wrap(t)
    }

    override fun visitIfThenElse(x: TracingTensor.IfThenElse): TracingTensor {
        return if (x.whenTrue == x.whenFalse) x.whenTrue else x
    }
}
