/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.tracing

interface TracingVisitor<R> {
    fun visit(x: Traceable): R = x.accept(this)
    fun visitConstant(x: TracingTensor.Constant): R
    fun visitVariable(x: TracingTensor.Variable): R
    fun visitPlus(x: TracingTensor.Plus): R
    fun visitMinus(x: TracingTensor.Minus): R
    fun visitTimes(x: TracingTensor.Times): R
    fun visitTimesScalar(x: TracingTensor.TimesScalar): R
    fun visitDiv(x: TracingTensor.Div): R
    fun visitZero(x: TracingTensor.Zero): R
    fun visitIdentityGradient(x: TracingTensor.IdentityGradient): R
    fun visitUnaryMinus(x: TracingTensor.UnaryMinus): R
    fun visitMatmul(x: TracingTensor.Matmul): R
    fun visitOuterProduct(x: TracingTensor.OuterProduct): R
    fun visitSin(x: TracingTensor.Sin): R
    fun visitCos(x: TracingTensor.Cos): R
    fun visitTan(x: TracingTensor.Tan): R
    fun visitAtan(x: TracingTensor.Atan): R
    fun visitExp(x: TracingTensor.Exp): R
    fun visitLn(x: TracingTensor.Ln): R
    fun visitLgamma(x: TracingTensor.Lgamma): R
    fun visitDigamma(x: TracingTensor.Digamma): R
    fun visitPolygamma(x: TracingTensor.Polygamma): R
    fun visitSqrt(x: TracingTensor.Sqrt): R
    fun visitTanh(x: TracingTensor.Tanh): R
    fun visitMeld(x: TracingTensor.Meld): R
    fun visitSplit(x: TracingTensor.Split): R
    fun visitSplitPart(x: TracingTensor.SplitPart): R
    fun visitConcat(x: TracingTensor.Concat): R
    fun visitBroadcastTo(x: TracingTensor.BroadcastTo): R
    fun visitConvImpl(x: TracingTensor.ConvImpl): R
    fun visitExpand(x: TracingTensor.Expand): R
    fun visitFlip(x: TracingTensor.Flip): R
    fun visitLogSoftmax(x: TracingTensor.LogSoftmax): R
    fun visitLogSoftmaxGrad(x: TracingTensor.LogSoftmaxGrad): R
    fun visitPow(x: TracingTensor.Pow): R
    fun visitView1(x: TracingTensor.View1): R
    fun visitView2(x: TracingTensor.View2): R
    fun visitView3(x: TracingTensor.View3): R
    fun visitReshape(x: TracingTensor.Reshape): R
    fun visitReshapeToScalar(x: TracingScalar.ReshapeToScalar): R
    fun visitSqueeze(x: TracingTensor.Squeeze): R
    fun visitUnsqueeze(x: TracingTensor.Unsqueeze): R
    fun visitTranspose(x: TracingTensor.Transpose): R
    fun visitRelu(x: TracingTensor.Relu): R
    fun visitReluGrad(x: TracingTensor.ReluGrad): R
    fun visitSigmoid(x: TracingTensor.Sigmoid): R
    fun visitSum(x: TracingTensor.Sum): R
    fun visitAvgPool(x: TracingTensor.AvgPool): R
    fun visitAvgPoolGrad(x: TracingTensor.AvgPoolGrad): R
    fun visitMaxPoolWithIndices(x: TracingTensor.MaxPoolWithIndices): R
    fun visitGather(x: TracingTensor.Gather): R
    fun visitGatherAtIndices(x: TracingTensor.GatherAtIndices): R
    fun visitScatter(x: TracingTensor.Scatter): R
    fun visitScatterAtIndices(x: TracingTensor.ScatterAtIndices): R
    fun visitCompare(x: TracingTensor.Compare): R
    fun visitIfThenElse(x: TracingTensor.IfThenElse): R
    fun visitRandomFloats(x: TracingTensor.RandomFloats): R
    fun visitRandomVariable(x: TracingRandomKey.Variable): R
    fun visitRandomSplit(x: TracingRandomKey.Split): R
    fun visitRandomSplitPart(x: TracingRandomKey.SplitPart): R
}
