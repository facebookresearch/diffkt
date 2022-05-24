/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

import shapeTyping.annotations.SType

@JvmName("DScalarRelu")
fun DScalar.relu(): DScalar = relu(this)

fun relu(x: DScalar): DScalar = x.operations.relu(x) as DScalar

@SType("S: Shape")
@JvmName("DTensorRelu")
fun @SType("S") DTensor.relu(): @SType("S") DTensor = relu(this)

@SType("S: Shape")
fun relu(x: @SType("S") DTensor): @SType("S") DTensor = x.operations.relu(x)


fun reluGrad(x: DScalar, upstream: DScalar): DScalar  {
    val (op, did) = commonKind(x, upstream)
    return op.reluGrad(x, upstream, did) as DScalar
}

fun reluGrad(x: DTensor, upstream: DTensor): DTensor {
    val (op, did) = commonKind(x, upstream)
    return op.reluGrad(x.expandAndBroadcastToTangent(upstream), upstream, did)
}
