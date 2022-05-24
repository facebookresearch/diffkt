/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.model

import org.diffkt.*

/**
 * The batchNorm op used for training
 *
 * @param input a tensor the last axis of which is the features dimension of size C
 * @param scaleShift the combined scale and shift tensor, with shape (2, C)
 * @return BatchNormResult
 */
fun batchNorm(
    input: DTensor,
    scaleShift: DTensor
): BatchNormResult {
    require(input.rank >= 2)
    require(scaleShift.shape == Shape(2, input.shape.last))
    val (operations, derivativeId) = commonKind(input, scaleShift)
    return operations.batchNorm(input, scaleShift, derivativeId)
}

internal fun baseBatchNorm(input: DTensor, scaleShift: DTensor): BatchNormResult  {
    val c = input.shape.last
    val n = input.shape.product / c.toFloat()
    val axes = IntArray(input.rank-1) { it }
    val sum = input.sum(axes)
    val sumOfSquares = input.pow(2).sum(axes)
    val mean = sum / n
    val variance = sumOfSquares / n - mean.pow(2)

    // In computing the standard deviation, we add an epsilon "to improve stability".
    // See "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"
    // https://arxiv.org/abs/1502.03167 Algorithm 1 (page 3).
    // This prevents us from dividing by zero when computing normalizedInput below.
    val stddev = sqrt(variance + BATCHNORM_EPSILON)

    // Normalize the input
    val normalizedInput = (input - mean) / stddev
    val scale = scaleShift[0]
    val shift = scaleShift[1]
    val result = scale * normalizedInput + shift
    return BatchNormResult(result, n, sum, sumOfSquares, mean, variance)
}

const val BATCHNORM_EPSILON = 1.0e-5f

class BatchNormResult(
    /**
     * The result of batch normalization
     */
    val result: DTensor,
    /**
     * The size of the batch
     */
    val n: Float,
    /**
     * The simple sum of the data in the batch.
     */
    val sum: DTensor,
    /**
     * The sum of the squares of the data items in the batch.
     */
    val sumOfSquares: DTensor,
    /**
     * The mean of the data, which is sum / n
     */
    val mean: DTensor,
    /**
     * The variance of the data, which is E[x^2] - E[x]^2
     */
    val variance: DTensor)
{
    companion object {
        fun fromMeanAndVariance(result: DTensor, mean: DTensor, variance: DTensor): BatchNormResult {
            val n = result.shape.product / result.shape.last.toFloat()
            val sum = mean * n // since E[x] is sum/N, sum is N * E[x]
            // (1) variance = E[x^2] - E[x]^2           // from definition
            // (2) N*variance = N * E[x^2] - N * E[x]^2 // multiply (1) by N
            // (3) N*variance = Sum(x^2) - N * E[x]^2   // subst definition of E[x^2] = Sum[x^2]/N
            // (4) N*variance = Sum(x^2) - sum^2 / N    // subst sum is N * E[x]
            // (5) Sum(x^2) = N*variance + sum^2 / N    // rearrange terms
            val sumOfSquares = n * variance + sum.pow(2) / n
            return BatchNormResult(result, n, sum, sumOfSquares, mean, variance)
        }
    }
}
