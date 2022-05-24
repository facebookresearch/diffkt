/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

import kotlin.math.max

object Convolve {
    const val N_AXIS = 0
    const val H_AXIS = 1
    const val W_AXIS = 2
    const val C_AXIS = 3

    data class Padding2D(val top: Int, val bottom: Int, val left: Int, val right: Int) {
        constructor(height: Int, width: Int) : this(height, height, width, width)
        constructor(n: Int) : this(n, n, n, n)
    }

    sealed class PaddingStyle {
        abstract fun getPadding(inputShape: Shape, filterhape: Shape, verticalStride: Int, horizontalStride: Int): Padding2D

        object Valid : PaddingStyle() {
            override fun getPadding(inputShape: Shape, filterhape: Shape, verticalStride: Int, horizontalStride: Int): Padding2D {
                return Padding2D(top = 0, bottom = 0, left = 0, right = 0)
            }
        }

        object Same : PaddingStyle() {
            override fun getPadding(inputShape: Shape, filterhape: Shape, verticalStride: Int, horizontalStride: Int): Padding2D {
                val (inHeight, inWidth) = Pair(inputShape[1], inputShape[2])
                val (filterHeight, filterWidth) = Pair(filterhape[1], filterhape[2])

                val verticalPad = if (inHeight % verticalStride == 0)
                    max(filterHeight - verticalStride, 0)
                else
                    max(filterHeight - (inHeight % verticalStride), 0)
                val horizontalPad = if (inWidth % horizontalStride == 0)
                    max(filterWidth - horizontalStride, 0)
                else
                    max(filterWidth - (inWidth % horizontalStride), 0)

                val padTop = verticalPad / 2
                val padLeft = horizontalPad / 2

                return Padding2D(
                        top = padTop,
                        bottom = verticalPad - padTop,
                        left = padLeft,
                        right = horizontalPad - padLeft
                )
            }
        }

        object Full : PaddingStyle() {
            override fun getPadding(inputShape: Shape, filterhape: Shape, verticalStride: Int, horizontalStride: Int): Padding2D {
                val (verticalPad, horizontalPad) = Pair(filterhape[1] - 1, filterhape[2] - 1)
                return Padding2D(
                        top = verticalPad,
                        bottom = verticalPad,
                        left = horizontalPad,
                        right = horizontalPad
                )
            }
        }

        data class Explicit(val top: Int, val bottom: Int, val left: Int, val right: Int) : PaddingStyle() {
            constructor(height: Int, width: Int) : this(height, height, width, width)
            constructor(n: Int) : this(n, n, n, n)

            override fun getPadding(inputShape: Shape, filterhape: Shape, verticalStride: Int, horizontalStride: Int): Padding2D {
                return Padding2D(top, bottom, left, right)
            }
        }
    }
}

fun conv2d(
        signal: DTensor,
        filter: DTensor,
        hStride: Int,
        vStride: Int,
        paddingStyle: Convolve.PaddingStyle = Convolve.PaddingStyle.Same): DTensor {
    require(signal.rank == 4) { "conv2D signal must be rank 4, was ${signal.rank}" }
    require(filter.rank == 4) { "conv2D filter must be rank 4, was ${filter.rank}" }
    paddingStyle.getPadding(signal.shape, filter.shape, vStride, hStride)
    return convImpl(signal, filter, hStride, vStride, paddingStyle.getPadding(signal.shape, filter.shape, hStride, vStride))
}

fun conv2d(
        signal: DTensor,
        filter: DTensor,
        hStride: Int,
        vStride: Int,
        padding: Convolve.Padding2D): DTensor {
    require(signal.rank == 4) { "conv2D signal must be rank 4, was ${signal.rank}" }
    require(filter.rank == 4) { "conv2D filter must be rank 4, was ${filter.rank}" }
    return convImpl(signal, filter, hStride, vStride, padding)
}

internal fun convImpl(signal: DTensor, filter: DTensor, hStride: Int, vStride: Int, padding: Convolve.Padding2D): DTensor {
    val (operations, derivativeId) = commonKind(signal, filter)
    return operations.convImpl(signal, filter, hStride, vStride, padding, derivativeId)
}
