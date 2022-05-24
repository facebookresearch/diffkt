/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.model

import org.diffkt.*

object ReluLayer: LayerSingleInput<ReluLayer> {
    override fun invoke(input: DTensor): DTensor {
        return relu(input)
    }
}
