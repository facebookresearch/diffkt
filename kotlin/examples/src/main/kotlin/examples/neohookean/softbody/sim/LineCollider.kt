/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.neohookean.softbody.sim

import org.diffkt.*

class LineCollider(val pos: Vector2, val normal: Vector2, val collisionK: Float, val frictionEps: Float = 0.01f, val frictionK: Float = 1000f) {
    val tangent: Vector2 by lazy {
        Vector2(-normal.y, normal.x)
    }

    // contact and friction model from https://arxiv.org/abs/2102.05791
    // extended to support arbitrary line orientations
    fun energy(state0: SystemState, state: SystemState, h: Float): DScalar {
        return state0.vertices.zip(state.vertices) { vertex0, vertex ->
            val dNormal = (vertex.pos - this.pos).dot(normal)
            val collisionEnergy = ifThenElse(dNormal lt 0f, collisionK * dNormal * dNormal, FloatScalar.ZERO)

            val dNormalForFriction = (vertex0.pos - this.pos).dot(normal) - frictionEps
            val frictionEnergy = if (dNormalForFriction.basePrimal().value < 0f) {
                val vel = (vertex.pos - vertex0.pos) / h
                val velTangent = vel.dot(tangent)
                frictionK * velTangent * velTangent * -dNormalForFriction
            } else {
                FloatScalar(0f)
            }

            collisionEnergy + frictionEnergy
        }.sum()
    }
}