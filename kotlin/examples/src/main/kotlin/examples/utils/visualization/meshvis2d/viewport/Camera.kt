/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.utils.visualization.meshvis2d.viewport

import examples.utils.visualization.meshvis2d.math.*

class Camera(val worldToViewportTransform: Transform2 = Transform2.identity()) {
    fun lookAtWithWorldWidth(
        worldCenter: Vector2, worldWidth: Float,
        viewportWidth: Float, viewportHeight: Float
    ): Camera {
        val worldToViewportScale = viewportWidth / worldWidth
        return lookAtWithWorldToViewportScale(worldCenter, worldToViewportScale, viewportWidth, viewportHeight)
    }

    fun lookAtWithWorldToViewportScale(
        worldCenter: Vector2, worldToViewportScale: Float,
        viewportWidth: Int, viewportHeight: Int
    ): Camera {
        return lookAtWithWorldToViewportScale(worldCenter, worldToViewportScale, viewportWidth.toFloat(), viewportHeight.toFloat())
    }

    fun lookAtWithWorldToViewportScale(
        worldCenter: Vector2, worldToViewportScale: Float,
        viewportWidth: Float, viewportHeight: Float
    ): Camera {
        val linear = Matrix2x2(
            worldToViewportScale, 0f,
            0f, -worldToViewportScale
        )
        val translation = Vector2(
            viewportWidth  * 0.5f - worldCenter.x * worldToViewportScale,
            viewportHeight * 0.5f + worldCenter.y * worldToViewportScale
        )
        return Camera(Transform2(translation, linear))
    }

    fun lookAtWithWorldWidth(worldCenter: Vector2, worldWidth: Float, viewportWidth: Int, viewportHeight: Int): Camera {
        return lookAtWithWorldWidth(worldCenter, worldWidth, viewportWidth.toFloat(), viewportHeight.toFloat())
    }

    fun lookAtViewportPos(pv: Vector2, worldToViewportScale: Float): Camera {
        val pw = this.viewportToWorld(pv)
        val tx = pv.x - worldToViewportScale * pw.x
        val ty = pv.y + worldToViewportScale * pw.y
        val transform = Transform2(
            Vector2(tx, ty),
            Matrix2x2(
                worldToViewportScale, 0f,
                0f, -worldToViewportScale
            )
        )
        return Camera(transform)
    }

    fun worldToViewport(b: Vector2): Vector2 = worldToViewportTransform.apply(b)

    fun viewportToWorld(b: Vector2): Vector2 {
        val viewportToWorldTransform = worldToViewportTransform.inv
        return viewportToWorldTransform.apply(b)
    }

    fun inferWorldToViewportScale(): Float = worldToViewportTransform.linear.m00
}