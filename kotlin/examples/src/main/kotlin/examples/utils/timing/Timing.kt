/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.utils.timing

import kotlin.system.measureNanoTime

inline fun warmupAndTime(f: () -> Unit, iters: Int = 10): Long {
    // warm up
    for (i in 0 until iters) f()
    // average time per run
    var totalTime = 0L
    for (i in 0 until iters) {
        totalTime += measureNanoTime(f)
    }
    return totalTime / iters
}
