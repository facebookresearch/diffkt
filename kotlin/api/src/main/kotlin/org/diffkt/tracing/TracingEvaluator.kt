/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.tracing

import org.diffkt.*
import org.diffkt.model.*
import org.diffkt.random.RandomKey
import java.lang.IllegalStateException

fun <W : Wrappable<W>> W.eval(variables: Array<DTensor?>, traceId: TraceId): W {
    val te = TracingEvaluator(variables, traceId)
    return te.evaluate(this)
}

// TODO: collapse variable/randoms
internal class TracingEvaluator(
    val variables: Array<DTensor?>,
    val traceId: TraceId
) : TracingVisitor<DTensor> {
    val wrapper = object : Wrapper() {
        override fun wrapDTensor(value: DTensor): DTensor {
            return when (value) {
                is TracingTensor -> evaluate(value)
                else -> value
            }
        }

        override fun wrapRandomKey(value: RandomKey): RandomKey {
            return when (value) {
                is TracingRandomKey -> (evaluate(value) as TracingRandomKey.RandomTensorWrapper).key
                else -> value
            }
        }
    }

    fun <W : Wrappable<W>> evaluate(w: W): W = wrapper.wrap(w)

    fun evaluate(x: Traceable): DTensor {
        return x.accept(this)
    }

    override fun visitConstant(x: TracingTensor.Constant) = x.values
    override fun visitVariable(x: TracingTensor.Variable): DTensor {
        // If we see a value from another trace, treat it as a constant.
        // This handles the case when we capture a variable from some outer scope.
        // We don't want to evaluate it, just return the same thing.
        if (x.traceId != traceId) return x
        return variables[x.varIndex] ?: throw IllegalArgumentException("Variable $x not declared")
    }
    override fun visitPlus(x: TracingTensor.Plus): DTensor {
        val l = evaluate(x.left)
        val r = evaluate(x.right)
        val lops = l.operations
        val ldid = l.derivativeID
        return if (lops == r.operations && ldid == r.derivativeID) lops.plus(l, r, ldid) else l + r
    }
    override fun visitMinus(x: TracingTensor.Minus): DTensor {
        val l = evaluate(x.left)
        val r = evaluate(x.right)
        val lops = l.operations
        val ldid = l.derivativeID
        return if (lops == r.operations && ldid == r.derivativeID) lops.minus(l, r, ldid) else l - r
    }
    override fun visitTimes(x: TracingTensor.Times): DTensor {
        val l = evaluate(x.left)
        val r = evaluate(x.right)
        val lops = l.operations
        val ldid = l.derivativeID
        return if (lops == r.operations && ldid == r.derivativeID) lops.times(l, r, ldid) else l * r
    }
    override fun visitTimesScalar(x: TracingTensor.TimesScalar): DTensor {
        val l = evaluate(x.left)
        val r = evaluate(x.right)
        val lops = l.operations
        val ldid = l.derivativeID
        return if (lops == r.operations && ldid == r.derivativeID) lops.timesScalar(l as DScalar, r, ldid) else l * r
    }
    override fun visitDiv(x: TracingTensor.Div): DTensor {
        val l = evaluate(x.left)
        val r = evaluate(x.right)
        val lops = l.operations
        val ldid = l.derivativeID
        return if (lops == r.operations && ldid == r.derivativeID) lops.div(l, r, ldid) else l / r
    }
    override fun visitZero(x: TracingTensor.Zero): DTensor {
        // TODO: Can we abstract this to use any tensor implementation (even tracing)?
        return FloatTensor.zeros(x.shape)
    }
    override fun visitIdentityGradient(x: TracingTensor.IdentityGradient): DTensor {
        // TODO: Can we abstract this to use any tensor implementation (even tracing)?
        return StridedFloatTensor.identityGradient(x.halfShape)
    }
    override fun visitUnaryMinus(x: TracingTensor.UnaryMinus): DTensor {
        val arg = evaluate(x.x)
        return arg.operations.unaryMinus(arg)
    }
    override fun visitMatmul(x: TracingTensor.Matmul) =
        evaluate(x.x).matmul(evaluate(x.y), x.a, x.b, x.c, x.d)
    override fun visitOuterProduct(x: TracingTensor.OuterProduct) =
        evaluate(x.x).outerProduct(x.y)
    override fun visitSin(x: TracingTensor.Sin) = sin(evaluate(x.x))
    override fun visitCos(x: TracingTensor.Cos) = cos(evaluate(x.x))
    override fun visitTan(x: TracingTensor.Tan) = tan(evaluate(x.x))
    override fun visitAtan(x: TracingTensor.Atan) = atan(evaluate(x.x))
    override fun visitExp(x: TracingTensor.Exp) = exp(evaluate(x.x))
    override fun visitLn(x: TracingTensor.Ln) = ln(evaluate(x.x))
    override fun visitLgamma(x: TracingTensor.Lgamma) = lgamma(evaluate(x.x))
    override fun visitDigamma(x: TracingTensor.Digamma) = digamma(evaluate(x.x))
    override fun visitPolygamma(x: TracingTensor.Polygamma) = polygamma(x.n, evaluate(x.x))
    override fun visitSqrt(x: TracingTensor.Sqrt) = sqrt(evaluate(x.x))
    override fun visitTanh(x: TracingTensor.Tanh) = tanh(evaluate(x.x))
    override fun visitMeld(x: TracingTensor.Meld) = meld(x.values.map { evaluate(it) })
    override fun visitSplit(x: TracingTensor.Split): DTensor = TranslatedSplit(evaluate(x.x).split(x.shapes))
    private class TranslatedSplit(val parts: List<DTensor>): DTensor {
        override val derivativeID = NoDerivativeID
        override val primal = this
        override val operations get() = throw NotImplementedError()
        override val shape: Shape = Shape()
    }
    override fun visitSplitPart(x: TracingTensor.SplitPart) = (evaluate(x.from) as TranslatedSplit).parts[x.index]
    override fun visitConcat(x: TracingTensor.Concat) = concat(x.slices.map { evaluate(it) }, x.axis)
    override fun visitBroadcastTo(x: TracingTensor.BroadcastTo) = evaluate(x.x).broadcastTo(x.shape)
    override fun visitConvImpl(x: TracingTensor.ConvImpl) =
        conv2d(
            evaluate(x.signal),
            evaluate(x.filter), x.hStride, x.vStride, x.padding)
    override fun visitExpand(x: TracingTensor.Expand) = evaluate(x.x).expand(x.shape)
    override fun visitFlip(x: TracingTensor.Flip) = evaluate(x.x).flip(x.axes)
    override fun visitLogSoftmax(x: TracingTensor.LogSoftmax) = evaluate(x.x).logSoftmax(x.axis)
    override fun visitLogSoftmaxGrad(x: TracingTensor.LogSoftmaxGrad): DTensor {
        val xx = evaluate(x.x)
        val l = evaluate(x.logSoftmax)
        val u = evaluate(x.upstream)
        val sample = highestDerivativeID(xx, l, u)
        return sample.operations.logSoftmaxGrad(xx, x.axis, l, u)
    }
    override fun visitPow(x: TracingTensor.Pow) = evaluate(x.base).pow(x.exponent)
    override fun visitView1(x: TracingTensor.View1) = evaluate(x.x).view(x.indexes)
    override fun visitView2(x: TracingTensor.View2) = evaluate(x.x).view(x.index, x.axis)
    override fun visitView3(x: TracingTensor.View3) = evaluate(x.x).view(x.index, x.axis)
    override fun visitReshape(x: TracingTensor.Reshape) = evaluate(x.x).reshape(x.shape)
    override fun visitReshapeToScalar(x: TracingScalar.ReshapeToScalar) = evaluate(x.x).reshape(Shape())
    override fun visitSqueeze(x: TracingTensor.Squeeze): DTensor = evaluate(x.x).squeeze(x.axis)
    override fun visitUnsqueeze(x: TracingTensor.Unsqueeze) = evaluate(x.x).unsqueeze(x.axis)
    override fun visitTranspose(x: TracingTensor.Transpose) = evaluate(x.x).transpose(x.axes)
    override fun visitRelu(x: TracingTensor.Relu) = evaluate(x.x).relu()
    override fun visitReluGrad(x: TracingTensor.ReluGrad) = reluGrad(
        evaluate(x.x),
        evaluate(x.upstream)
    )
    override fun visitSigmoid(x: TracingTensor.Sigmoid) = sigmoid(evaluate(x.x))
    override fun visitSum(x: TracingTensor.Sum) = evaluate(x.x).sum(x.axes, x.keepDims)
    override fun visitAvgPool(x: TracingTensor.AvgPool) = avgPool(evaluate(x.x), x.poolHeight, x.poolWidth)
    override fun visitAvgPoolGrad(x: TracingTensor.AvgPoolGrad) = avgPoolGrad(evaluate(x.x), x.poolHeight, x.poolWidth)
    override fun visitMaxPoolWithIndices(x: TracingTensor.MaxPoolWithIndices) =
        maxPool(evaluate(x.x), x.poolHeight, x.poolWidth)
    override fun visitGather(x: TracingTensor.Gather) = evaluate(x.x).gather(x.indexes, x.axis, x.paddingIndex)
    override fun visitGatherAtIndices(x: TracingTensor.GatherAtIndices): DTensor {
        TODO("Not yet implemented")
    }
    override fun visitScatter(x: TracingTensor.Scatter) = evaluate(x.x)
        .scatter(x.indexes, x.axis, x.shape, x.paddingIndex)
    override fun visitScatterAtIndices(x: TracingTensor.ScatterAtIndices): DTensor {
        TODO("Not yet implemented")
    }

    override fun visitCompare(x: TracingTensor.Compare): DTensor {
        val left = evaluate(x.left)
        val right = evaluate(x.right)
        return compare(left, right, x.comparison)
    }

    override fun visitIfThenElse(x: TracingTensor.IfThenElse): DTensor {
        val cond = evaluate(x.cond)
        val whenTrue = evaluate(x.whenTrue)
        val whenFalse = evaluate(x.whenFalse)
        return ifThenElse(cond, whenTrue, whenFalse)
    }

    override fun visitRandomFloats(x: TracingTensor.RandomFloats): DTensor {
        if (x.traceId != traceId) return x
        return evaluateRandomKey(x.key).floats(x.shape)
    }

    override fun visitRandomVariable(x: TracingRandomKey.Variable): DTensor {
        return (variables[x.index] as? TracingRandomKey.RandomTensorWrapper)
            ?: throw IllegalArgumentException("Variable $x not declared")
    }

    override fun visitRandomSplit(x: TracingRandomKey.Split): DTensor {
        return TracingRandomKey.RandomTensorWrapper(evaluateRandomKey(x))
    }

    override fun visitRandomSplitPart(x: TracingRandomKey.SplitPart): DTensor {
        return TracingRandomKey.RandomTensorWrapper(evaluateRandomKey(x))
    }

    // Avoids boxing/unboxing non-variable MockDTensors, since RandomKeys only have RandomKeys as child nodes
    private fun evaluateRandomKey(x: TracingRandomKey): RandomKey {
        if (x.traceId != traceId) return x
        return when (x) {
            is TracingRandomKey.Variable -> (variables[x.index] as? TracingRandomKey.RandomTensorWrapper)?.key
                ?: throw IllegalArgumentException("Variable $x not declared")
            is TracingRandomKey.SplitPart -> {
                val split = evaluateRandomKey(x.split)
                (split as TracingRandomKey.SplitRandom).keys[x.splitIndex]
            }
            is TracingRandomKey.Split -> {
                val key = evaluateRandomKey(x.key)
                TracingRandomKey.SplitRandom(key.split(x.n))
            }
            is PrintedRandomKey -> throw IllegalStateException("Invalid operation on printed key")
        }
    }
}
