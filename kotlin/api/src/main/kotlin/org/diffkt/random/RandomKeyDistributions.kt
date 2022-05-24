/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.random

import org.diffkt.*

/**
 * Samples from a gaussian distribution with [mean] and [std] (standard deviation)
 */
fun RandomKey.gaussian(shape: Shape, mean: DTensor, std: DTensor): DTensor {
    return gaussian(shape) * std + mean
}

/**
 * Samples from a cauchy distribution with a loc of 1 and a scale of 0
 * The cumulative distribution function (CDF) is F = (1 / 𝜋) arctan((x - [loc]) / [scale]) + (1 / 2)
 * The random variable Y = F(x) has a uniform distribution.
 * Therefore, we can invert F and use a uniform distribution to simulate random variable X
 * https://en.wikipedia.org/wiki/Cauchy_distribution
 */
fun RandomKey.cauchy(shape: Shape): DTensor {
    return tan(FloatScalar.PI * (this.floats(shape) - 0.5f))
}

/**
 * Samples from a cauchy distribution with [loc] and [scale]
 */
fun RandomKey.cauchy(shape: Shape, loc: DTensor, scale: DTensor): DTensor {
    return cauchy(shape) * scale + loc
}

/**
 * Samples from a chi square distribution
 * The chi square distribution X is a special case of the gamma distribution
 * X ~ Gamma([dof] / 2, 1 / 2)
 * https://en.wikipedia.org/wiki/Chi-squared_distribution
 */
fun RandomKey.chiSquare(shape: Shape, dof: DTensor): DTensor {
    return gammaWithRate(shape, dof / 2f, FloatScalar(0.5f))
}
