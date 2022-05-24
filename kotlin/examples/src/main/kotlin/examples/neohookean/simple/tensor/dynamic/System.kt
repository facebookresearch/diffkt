/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.neohookean.simple.tensor.dynamic

import org.diffkt.*

class System(val triangles: NeohookeanTriangles) {
    fun makeBackwardEulerLoss(s0: Vertices, h: Float): (DTensor) -> DScalar {
        fun loss(x: DTensor): DScalar {
            val x0 = s0.x
            val v0 = s0.v
            val y = x0 + v0 * h
            val m = s0.m
            val inertia = 0.5f * (m * (x - y).pow(2).sum(1)).sum()
            val h2 = h * h
            return h2 * triangles.energy(x) + inertia
        }
        return ::loss
    }
}