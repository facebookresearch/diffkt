/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

enum class Device {
    CPU,
    GPU
}

interface OnDevice {
    fun cpu(): OnDevice

    fun gpu(): OnDevice

    fun to(device: Device): OnDevice = when (device) {
        Device.CPU -> cpu()
        Device.GPU -> gpu()
    }
}
