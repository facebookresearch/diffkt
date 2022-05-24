/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.model

import org.diffkt.combineHash

class DummyModel : Model<DummyModel>() {
    override val layers: List<Layer<*>> = listOf()
    override fun withLayers(newLayers: List<Layer<*>>) = this
    override fun hashCode(): Int = combineHash("DummyModel")
    override fun equals(other: Any?): Boolean = other is DummyModel
}