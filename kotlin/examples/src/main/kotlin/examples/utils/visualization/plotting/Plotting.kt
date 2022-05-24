/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.utils.visualization.plotting

import jetbrains.datalore.base.geometry.DoubleVector
import jetbrains.datalore.plot.MonolithicAwt
import jetbrains.datalore.vis.svg.SvgSvgElement
import jetbrains.datalore.vis.swing.BatikMapperComponent
import jetbrains.datalore.vis.swing.BatikMessageCallback
import jetbrains.letsPlot.intern.Plot
import jetbrains.letsPlot.intern.toSpec
import javax.swing.JFrame
import javax.swing.SwingUtilities
import javax.swing.WindowConstants

object Plotting {
    // Setup
    private val SVG_COMPONENT_FACTORY_BATIK =
        { svg: SvgSvgElement -> BatikMapperComponent(svg, BATIK_MESSAGE_CALLBACK) }

    private val BATIK_MESSAGE_CALLBACK = object : BatikMessageCallback {
        override fun handleMessage(message: String) {
            println(message)
        }

        override fun handleException(e: Exception) {
            if (e is RuntimeException) {
                throw e
            }
            throw RuntimeException(e)
        }
    }

    private val AWT_EDT_EXECUTOR = { runnable: () -> Unit ->
        // Just invoke in the current thread.
        runnable.invoke()
    }

    fun displayPlot(p: Plot, plotName: String) {
        SwingUtilities.invokeLater {
            // Create Swing Panel showing the plot.
            val plotSpec = p.toSpec()
            val plotSize = DoubleVector(600.0, 300.0)

            val component = MonolithicAwt.buildPlotFromRawSpecs(
                plotSpec,
                plotSize,
                SVG_COMPONENT_FACTORY_BATIK, AWT_EDT_EXECUTOR
            ) {
                for (message in it) {
                    println("PLOT MESSAGE: $message")
                }
            }

            // Show plot in Swing frame.
            val frame = JFrame(plotName)
            frame.contentPane.add(component)
            frame.defaultCloseOperation = WindowConstants.DISPOSE_ON_CLOSE
            frame.pack()
            frame.isVisible = true
        }
    }
}

fun Plot.display(name: String) {
    Plotting.displayPlot(this, name)
}
