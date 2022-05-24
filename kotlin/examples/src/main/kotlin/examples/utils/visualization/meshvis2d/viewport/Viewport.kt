/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.utils.visualization.meshvis2d.viewport

import examples.utils.visualization.meshvis2d.math.Vector2

abstract class Viewport(
    val width: Int,
    val height: Int,
    var camera: Camera = Camera(),
    private var _worldCenter: Vector2 = Vector2(0f, 0f),
) {
    val worldCenter get() = _worldCenter

    fun lookAt(worldCenter: Vector2) {
        camera = camera.lookAtWithWorldToViewportScale(
            worldCenter, camera.inferWorldToViewportScale(),
            width, height
        )
        _worldCenter = worldCenter
    }

    fun lookAt(worldCenter: Vector2, worldWidth: Float) {
        camera = camera.lookAtWithWorldWidth(worldCenter, worldWidth, width, height)
        this._worldCenter = worldCenter
    }

    fun zoom(viewportCenter: Vector2, dScale: Float) {
        val worldToViewportScale = camera.inferWorldToViewportScale()
        val newWorldToViewportScale = worldToViewportScale + dScale
        camera = camera.lookAtViewportPos(viewportCenter, newWorldToViewportScale)
        _worldCenter = camera.viewportToWorld(
            Vector2(width * 0.5f, height * 0.5f)
        )
    }

    abstract fun show(
        initialScene: Scene,
        update: (Scene, List<DragEvent>) -> Scene
    )
}