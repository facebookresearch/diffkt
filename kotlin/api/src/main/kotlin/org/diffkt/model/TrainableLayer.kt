/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.model

interface TrainableLayer<T : TrainableLayer<T>> : TrainableComponent<T>, Layer<T> {
    override fun cpu(): T = withTrainables(trainables.map { it.cpu() })
    override fun gpu(): T = withTrainables(trainables.map { it.gpu() })
}

interface TrainableLayerSingleInput<T : TrainableLayerSingleInput<T>> : TrainableLayer<T>, TrainableComponent<T>, LayerSingleInput<T>

