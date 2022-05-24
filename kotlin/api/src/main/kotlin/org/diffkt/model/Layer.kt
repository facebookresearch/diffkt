/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.model

import org.diffkt.DTensor
import org.diffkt.OnDevice

interface Layer<T: Layer<T>> : OnDevice {
    operator fun invoke(vararg inputs: DTensor): DTensor

    /**
     * Helper to check that the layer was called with a single input.
     * Returns that input if successful, else errors.
     */
    fun getSingleInput(inputs: Array<out DTensor>): DTensor {
        require(inputs.size == 1) { "${this.javaClass} layer called with ${inputs.size}, takes 1." }
        return inputs[0]
    }

    override fun cpu(): Layer<T> = this
    override fun gpu(): Layer<T> = this
}

infix fun <T: Layer<T>> DTensor.into(layer: Layer<T>): DTensor = layer.invoke(this)

interface LayerSingleInput<T: LayerSingleInput<T>> : Layer<T> {
    operator fun invoke(input: DTensor): DTensor

    override fun invoke(vararg inputs: DTensor) = invoke(getSingleInput(inputs))
}
