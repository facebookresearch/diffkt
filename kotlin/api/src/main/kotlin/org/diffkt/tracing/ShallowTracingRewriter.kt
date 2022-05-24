/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.tracing

internal open class ShallowTracingRewriter : TracingVisitor<Traceable> {
    open fun defaultShallowRewrite(x: Traceable): Traceable {
        return x
    }

    override fun visitConstant(x: TracingTensor.Constant): Traceable {
        return defaultShallowRewrite(x)
    }

    override fun visitVariable(x: TracingTensor.Variable): Traceable {
        return defaultShallowRewrite(x)
    }

    override fun visitPlus(x: TracingTensor.Plus): Traceable {
        return defaultShallowRewrite(x)
    }

    override fun visitMinus(x: TracingTensor.Minus): Traceable {
        return defaultShallowRewrite(x)
    }

    override fun visitTimes(x: TracingTensor.Times): Traceable {
        return defaultShallowRewrite(x)
    }

    override fun visitTimesScalar(x: TracingTensor.TimesScalar): Traceable {
        return defaultShallowRewrite(x)
    }

    override fun visitDiv(x: TracingTensor.Div): Traceable {
        return defaultShallowRewrite(x)
    }

    override fun visitZero(x: TracingTensor.Zero): Traceable {
        return defaultShallowRewrite(x)
    }

    override fun visitIdentityGradient(x: TracingTensor.IdentityGradient): Traceable {
        return defaultShallowRewrite(x)
    }

    override fun visitUnaryMinus(x: TracingTensor.UnaryMinus): Traceable {
        return defaultShallowRewrite(x)
    }

    override fun visitMatmul(x: TracingTensor.Matmul): Traceable {
        return defaultShallowRewrite(x)
    }

    override fun visitOuterProduct(x: TracingTensor.OuterProduct): Traceable {
        return defaultShallowRewrite(x)
    }
    
    override fun visitSin(x: TracingTensor.Sin): Traceable {
        return defaultShallowRewrite(x)
    }

    override fun visitCos(x: TracingTensor.Cos): Traceable {
        return defaultShallowRewrite(x)
    }

    override fun visitTan(x: TracingTensor.Tan): Traceable {
        return defaultShallowRewrite(x)
    }

    override fun visitAtan(x: TracingTensor.Atan): Traceable {
        return defaultShallowRewrite(x)
    }

    override fun visitExp(x: TracingTensor.Exp): Traceable {
        return defaultShallowRewrite(x)
    }

    override fun visitLn(x: TracingTensor.Ln): Traceable {
        return defaultShallowRewrite(x)
    }

    override fun visitLgamma(x: TracingTensor.Lgamma): Traceable {
        return defaultShallowRewrite(x)
    }

    override fun visitDigamma(x: TracingTensor.Digamma): Traceable {
        return defaultShallowRewrite(x)
    }

    override fun visitPolygamma(x: TracingTensor.Polygamma): Traceable {
        return defaultShallowRewrite(x)
    }

    override fun visitSqrt(x: TracingTensor.Sqrt): Traceable {
        return defaultShallowRewrite(x)
    }

    override fun visitTanh(x: TracingTensor.Tanh): Traceable {
        return defaultShallowRewrite(x)
    }

    override fun visitMeld(x: TracingTensor.Meld): Traceable {
        return defaultShallowRewrite(x)
    }

    override fun visitSplit(x: TracingTensor.Split): Traceable {
        return defaultShallowRewrite(x)
    }

    override fun visitSplitPart(x: TracingTensor.SplitPart): Traceable {
        return defaultShallowRewrite(x)
    }

    override fun visitConcat(x: TracingTensor.Concat): Traceable {
        return defaultShallowRewrite(x)
    }

    override fun visitBroadcastTo(x: TracingTensor.BroadcastTo): Traceable {
        return defaultShallowRewrite(x)
    }

    override fun visitConvImpl(x: TracingTensor.ConvImpl): Traceable {
        return defaultShallowRewrite(x)
    }

    override fun visitExpand(x: TracingTensor.Expand): Traceable {
        return defaultShallowRewrite(x)
    }

    override fun visitFlip(x: TracingTensor.Flip): Traceable {
        return defaultShallowRewrite(x)
    }

    override fun visitLogSoftmax(x: TracingTensor.LogSoftmax): Traceable {
        return defaultShallowRewrite(x)
    }

    override fun visitLogSoftmaxGrad(x: TracingTensor.LogSoftmaxGrad): Traceable {
        return defaultShallowRewrite(x)
    }

    override fun visitPow(x: TracingTensor.Pow): Traceable {
        return defaultShallowRewrite(x)
    }

    override fun visitView1(x: TracingTensor.View1): Traceable {
        return defaultShallowRewrite(x)
    }

    override fun visitView2(x: TracingTensor.View2): Traceable {
        return defaultShallowRewrite(x)
    }

    override fun visitView3(x: TracingTensor.View3): Traceable {
        return defaultShallowRewrite(x)
    }

    override fun visitReshape(x: TracingTensor.Reshape): Traceable {
        return defaultShallowRewrite(x)
    }

    override fun visitReshapeToScalar(x: TracingScalar.ReshapeToScalar): Traceable {
        return defaultShallowRewrite(x)
    }

    override fun visitSqueeze(x: TracingTensor.Squeeze): Traceable {
        return defaultShallowRewrite(x)
    }

    override fun visitUnsqueeze(x: TracingTensor.Unsqueeze): Traceable {
        return defaultShallowRewrite(x)
    }

    override fun visitTranspose(x: TracingTensor.Transpose): Traceable {
        return defaultShallowRewrite(x)
    }

    override fun visitRelu(x: TracingTensor.Relu): Traceable {
        return defaultShallowRewrite(x)
    }

    override fun visitReluGrad(x: TracingTensor.ReluGrad): Traceable {
        return defaultShallowRewrite(x)
    }

    override fun visitSigmoid(x: TracingTensor.Sigmoid): Traceable {
        return defaultShallowRewrite(x)
    }

    override fun visitSum(x: TracingTensor.Sum): Traceable {
        return defaultShallowRewrite(x)
    }

    override fun visitAvgPool(x: TracingTensor.AvgPool): Traceable {
        return defaultShallowRewrite(x)
    }

    override fun visitAvgPoolGrad(x: TracingTensor.AvgPoolGrad): Traceable {
        return defaultShallowRewrite(x)
    }

    override fun visitMaxPoolWithIndices(x: TracingTensor.MaxPoolWithIndices): Traceable {
        return defaultShallowRewrite(x)
    }

    override fun visitGather(x: TracingTensor.Gather): Traceable {
        return defaultShallowRewrite(x)
    }

    override fun visitGatherAtIndices(x: TracingTensor.GatherAtIndices): Traceable {
        return defaultShallowRewrite(x)
    }

    override fun visitScatter(x: TracingTensor.Scatter): Traceable {
        return defaultShallowRewrite(x)
    }

    override fun visitScatterAtIndices(x: TracingTensor.ScatterAtIndices): Traceable {
        return defaultShallowRewrite(x)
    }

    override fun visitRandomFloats(x: TracingTensor.RandomFloats):  Traceable {
        return defaultShallowRewrite(x)
    }

    override fun visitRandomVariable(x: TracingRandomKey.Variable): Traceable {
        return defaultShallowRewrite(x)
    }

    override fun visitRandomSplit(x: TracingRandomKey.Split): Traceable {
        return defaultShallowRewrite(x)
    }

    override fun visitRandomSplitPart(x: TracingRandomKey.SplitPart): Traceable {
        return defaultShallowRewrite(x)
    }

    override fun visitCompare(x: TracingTensor.Compare): Traceable {
        return defaultShallowRewrite(x)
    }

    override fun visitIfThenElse(x: TracingTensor.IfThenElse): Traceable {
        return defaultShallowRewrite(x)
    }
}
