/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.utils.visualization.meshvis2d.openrndr

import examples.utils.visualization.meshvis2d.math.Vector2
import examples.utils.visualization.meshvis2d.mesh.Line
import examples.utils.visualization.meshvis2d.mesh.Mesh
import examples.utils.visualization.meshvis2d.mesh.Triangle
import examples.utils.visualization.meshvis2d.mesh.Vertex
import examples.utils.visualization.meshvis2d.viewport.Camera
import org.openrndr.MouseEvent
import org.openrndr.color.ColorRGBa
import org.openrndr.draw.Drawer
import org.openrndr.draw.LineJoin
import org.openrndr.shape.contour

fun MouseEvent.toVector2(): Vector2 = Vector2(this.position.x.toFloat(), this.position.y.toFloat())

fun Vector2.openrndr(): org.openrndr.math.Vector2 {
    return org.openrndr.math.Vector2(x.toDouble(), y.toDouble())
}

fun Vertex.render(camera: Camera, drawer: Drawer) {
    val worldPos = pos
    val viewportPos = camera.worldToViewport(worldPos).openrndr()
    val worldToViewportScale = camera.inferWorldToViewportScale()
    val viewportRadius = renderRadius.toDouble() * worldToViewportScale
    drawer.fill = ColorRGBa.PINK
    drawer.circle(viewportPos, viewportRadius)
}

fun List<Vertex>.renderLine(line: Line, camera: Camera, drawer: Drawer) {
    val viewportA = camera.worldToViewport(this[line.a].pos)
    val viewportB = camera.worldToViewport(this[line.b].pos)

    val ctr = contour {
        moveTo(viewportA.openrndr())
        lineTo(viewportB.openrndr())
    }
    drawer.contour(ctr)
}

fun List<Vertex>.renderTriangle(triangle: Triangle, camera: Camera, drawer: Drawer) {
    val viewportA = camera.worldToViewport(this[triangle.a].pos)
    val viewportB = camera.worldToViewport(this[triangle.b].pos)
    val viewportC = camera.worldToViewport(this[triangle.c].pos)

    drawer.fill = ColorRGBa.TRANSPARENT
    drawer.lineJoin = LineJoin.ROUND

    val ctr = contour {
        moveTo(viewportA.openrndr())
        lineTo(viewportB.openrndr())
        lineTo(viewportC.openrndr())
        close()
    }
    drawer.contour(ctr)
}

fun Mesh.render(camera: Camera, drawer: Drawer) {
    lines.forEach { vertices.renderLine(it, camera, drawer) }
    triangles.forEach { vertices.renderTriangle(it, camera, drawer) }
    vertices.forEach { it.render(camera, drawer) }
}