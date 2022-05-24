/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.random

import org.diffkt.*

/**
 * Samples from a gamma distribution with a shape parameter [alpha] (also known as k) and a scale/ rate parameter of 1
 */
fun RandomKey.gamma(shape: Shape, alpha: DTensor): DTensor {
    val broadcastAlpha =  alpha.broadcastTo(shape)
    return broadcastAlpha.operations.gamma(broadcastAlpha, this)
}

/**
 * Gamma distribution with a shape parameter [k], which is equal to alpha, and a scale parameter [θ].
 */
fun RandomKey.gammaWithScale(shape: Shape, k: DTensor, theta: DTensor): DTensor {
    return gamma(shape, k) * theta.broadcastTo(shape)
}

/**
 * Gamma distribution with a shape parameter [alpha], which is equal to k, and a rate parameter [β] = 1/θ
 */
fun RandomKey.gammaWithRate(shape: Shape, alpha: DTensor, beta: DTensor): DTensor {
    return gamma(shape, alpha) / beta.broadcastTo(shape)
}
