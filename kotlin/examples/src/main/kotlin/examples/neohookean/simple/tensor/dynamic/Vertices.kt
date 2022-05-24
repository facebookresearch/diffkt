/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.neohookean.simple.tensor.dynamic

import org.diffkt.*

class Vertices(val x: DTensor, val v: DTensor, val m: DTensor) {
    fun preBackwardEulerOptim(h: Float): Vertices = withX(this.x + h * this.v)
    fun postBackwardEulerOptim(x: DTensor, h: Float): Vertices = withXV(x, (x - this.x) / h)

    fun withX(x: DTensor): Vertices = Vertices(x, v, m)
    fun withV(v: DTensor): Vertices = Vertices(x, v, m)
    fun withXV(x: DTensor, v: DTensor): Vertices = Vertices(x, v, m)
}