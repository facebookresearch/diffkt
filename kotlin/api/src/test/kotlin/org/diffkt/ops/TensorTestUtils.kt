/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.ops

import org.diffkt.DScalar
import org.diffkt.DTensor
import org.diffkt.Shape
import org.diffkt.split

val DTensor.elements: List<DScalar>
    get() {
        val scalarShape = Shape()
        val shapes = List(this.size) { scalarShape }
        return this.split(shapes).map { it as DScalar }
    }
