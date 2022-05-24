/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.model

import org.diffkt.*
import java.nio.ByteBuffer

interface Trainable<T : Trainable<T>> : Differentiable<T>, OnDevice {
    fun extractTangent(output: DTensor, extractor: (input: DTensor, output: DTensor)->DTensor): Tangent
    fun trainingStep(optim: Optimizer<*>, tangent: Tangent): T

    fun store(into: ByteBuffer): ByteBuffer
    fun load(from: ByteBuffer): T

    override fun cpu(): T
    override fun gpu(): T

    interface Tangent
}
