/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import org.gradle.internal.os.OperatingSystem
import org.gradle.nativeplatform.platform.internal.DefaultNativePlatform
import java.io.ByteArrayOutputStream

/* openRNDR options */
val openrndrVersion = "0.3.58"
val orxVersion = "0.3.58"

// all feature options https://github.com/openrndr/openrndr-template/blob/master/build.gradle.kts#L13
// if we need olive or kintetic, be certain to add the extra dependencies from the template
val orxFeatures = setOf("orx-compositor", "orx-image-fit", "orx-panel")

val supportedPlatforms = setOf("windows", "macos", "linux-x64", "linux-arm64")

val openrndrOs = if (project.hasProperty("targetPlatform")) {
    val platform : String = project.property("targetPlatform") as String
    if (platform !in supportedPlatforms) {
        throw IllegalArgumentException("target platform not supported: $platform")
    } else {
        platform
    }
} else when (OperatingSystem.current()) {
    OperatingSystem.WINDOWS -> "windows"
    OperatingSystem.MAC_OS -> "macos"
    OperatingSystem.LINUX -> when(val h = DefaultNativePlatform("current").architecture.name) {
        "x86-64" -> "linux-x64"
        "aarch64" -> "linux-arm64"
        else ->throw IllegalArgumentException("architecture not supported: $h")
    }
    else -> throw IllegalArgumentException("os not supported")
}

fun DependencyHandler.orx(module: String): Any {
    return "org.openrndr.extra:$module:$orxVersion"
}

fun DependencyHandler.openrndr(module: String): Any {
    return "org.openrndr:openrndr-$module:$openrndrVersion"
}

fun DependencyHandler.openrndrNatives(module: String): Any {
    return "org.openrndr:openrndr-$module-natives-$openrndrOs:$openrndrVersion"
}

fun DependencyHandler.orxNatives(module: String): Any {
    return "org.openrndr.extra:$module-natives-$openrndrOs:$orxVersion"
}

dependencies {
    implementation(project(":api"))
    implementation(project(":data"))
    implementation(project(":distributions"))
    testImplementation(project(":testutils"))

    // for openRNDR
    runtimeOnly(openrndr("gl3"))
    runtimeOnly(openrndrNatives("gl3"))
    implementation(openrndr("openal"))
    runtimeOnly(openrndrNatives("openal"))
    implementation(openrndr("core"))
    for (feature in orxFeatures) {
        implementation(orx(feature))
    }

    // lets-plot
    implementation("org.jetbrains.lets-plot:lets-plot-batik:1.5.6")
    api("org.jetbrains.lets-plot:lets-plot-common:1.5.6")
    api("org.jetbrains.lets-plot-kotlin:lets-plot-kotlin-api:1.1.0")
}

plugins {
    application
}

// Task to run examples.
// Example usage: ./gradlew :examples:run -Ppackage=brachistochrone
tasks.run {
    val packageName = (project.properties["package"] as? String)
    if (packageName != null) {
        application {
            mainClass.set("examples.$packageName.MainKt")
            applicationDefaultJvmArgs = listOf("-XstartOnFirstThread")
        }
    }
}

fun createBenchmarkRun(packageName: String, iteration: Int, runArgs: String, outputFile: File? = null): JavaExec {
    return tasks.create("${packageName}_${iteration}_$runArgs", JavaExec::class) {
        classpath(sourceSets.main.get().runtimeClasspath)
        mainClass.set("examples.$packageName.MainKt")
        setArgsString("$runArgs benchmark")
        jvmArgs = listOf("-XstartOnFirstThread")
        standardOutput = ByteArrayOutputStream()
        doLast {
            outputFile?.appendText(standardOutput.toString())
        }
    }
}

// Task to run benchmarks.
// Example usage:
// ./gradlew :examples:benchmark -Ppackage=poissonBlending.gn -Piters=3 -Pargs="1 1000; 5 500" -Pfile=results.txt
tasks.create("benchmark") {
    val packageName = (project.properties["package"] as? String)
    val iters = (project.properties["iters"] as? String)?.toInt() ?: 5
    val args = ((project.properties["args"] as? String) ?: "").split(";").map { it.trim() }
    val outputFile = (project.properties["file"] as? String)?.let { File(it) }
    if (packageName != null) {
        for (arg in args) {
            for (i in 0 until iters) {
                dependsOn(createBenchmarkRun(packageName, i, arg, outputFile))
            }
        }
    }
}

// Send stdin to the our application during `gradle run` tasks
// Source: https://stackoverflow.com/a/46662535
val run: JavaExec by tasks
run.standardInput = System.`in`
