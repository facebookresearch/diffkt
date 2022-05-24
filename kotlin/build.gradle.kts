/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
    // Kotlin
    val ktVersion: String by System.getProperties()
    kotlin("jvm") version ktVersion
}

repositories {
    jcenter() // or maven(url="https://dl.bintray.com/kotlin/dokka")
    mavenCentral()
}

// Allow testResults to be accessed in sub-projects if needed.
val testResults: MutableList<String> by extra { mutableListOf() }

allprojects {
    group = "diffkt"
    apply(plugin = "java")

    java {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }

    tasks.withType<KotlinCompile>() {
        kotlinOptions.jvmTarget = "1.8"
        kotlinOptions.freeCompilerArgs += "-XXLanguage:+ProperCheckAnnotationsTargetInTypeUsePositions"
    }

    repositories {
        jcenter()
        maven(url = "https://maven.openrndr.org")
        maven(url = "https://maven.pkg.jetbrains.space/kotlin/p/kotlin/bootstrap")
    }

    tasks.withType<Test> {
        maxHeapSize = "8g"
        // Filter out GPU tests by default
        if (!project.hasProperty("gpu"))
            systemProperties["kotest.tags.exclude"] = "Gpu"

        testLogging {
            events = setOf(
                org.gradle.api.tasks.testing.logging.TestLogEvent.FAILED,
                org.gradle.api.tasks.testing.logging.TestLogEvent.SKIPPED,
                org.gradle.api.tasks.testing.logging.TestLogEvent.STANDARD_OUT,
                org.gradle.api.tasks.testing.logging.TestLogEvent.STANDARD_ERROR
            )
            exceptionFormat = org.gradle.api.tasks.testing.logging.TestExceptionFormat.FULL
            showExceptions = true
            showCauses = true
            showStackTraces = true

            // set options for log level DEBUG and INFO
            debug {
                events = this@testLogging.events + org.gradle.api.tasks.testing.logging.TestLogEvent.PASSED
                exceptionFormat = org.gradle.api.tasks.testing.logging.TestExceptionFormat.FULL
            }
            info {
                events = debug.events
                exceptionFormat = debug.exceptionFormat
            }
        }
        addTestListener(object : TestListener {
            override fun beforeTest(descriptor: TestDescriptor?) = Unit
            override fun afterTest(descriptor: TestDescriptor?, result: TestResult?) = Unit
            override fun beforeSuite(descriptor: TestDescriptor?) = Unit
            override fun afterSuite(descriptor: TestDescriptor?, result: TestResult?) {
                result?.let {
                    if (descriptor?.parent == null) { // Only summarize for a whole (sub) project
                        val summary = """
                            ${this@withType.project.name}:${this@withType.name} results: ${it.resultType} (${it.testCount} tests, ${it.successfulTestCount} passed, ${it.failedTestCount} failed, ${it.skippedTestCount} skipped)
                            Report file: ${this@withType.reports.html.entryPoint}
                            """.trimIndent()
                        if (it.resultType == TestResult.ResultType.SUCCESS) {
                            testResults.add(0, summary)
                        } else {
                            testResults += summary
                        }
                    }
                }
            }
        })
        useJUnitPlatform()
    }
}

subprojects {
    apply(plugin = "kotlin")

    val kotestVersion: String by System.getProperties()
    val junitVersion: String by System.getProperties()
    dependencies {
        implementation(kotlin("stdlib-jdk8"))
        implementation(kotlin("reflect"))
        implementation("org.apache.commons:commons-compress:1.2")
        testImplementation("io.kotest:kotest-runner-junit5-jvm:$kotestVersion")
            ?.because("kotest framework")
        testImplementation(kotlin("test"))
        testImplementation(kotlin("test-junit", junitVersion))
        testImplementation("io.kotest:kotest-assertions-core-jvm:$kotestVersion")
            ?.because("kotest core jvm assertions")
    }
}

// Define a pretty printer for test results
fun printResults(results: List<String>) {
    val maxLength = results.flatMap { it.lines().map { it.length } }.max() ?: 0
    println("┌${"─".repeat(maxLength + 2)}┐")
    println(
        results.joinToString("\n├${"─".repeat(maxLength + 2)}┤\n") {
            it.lines().joinToString("\n") { "│ $it${" ".repeat(maxLength - it.length)} │" }
        }
    )
    println("└${"─".repeat(maxLength + 2)}┘")
}

// Print results when build is finished
gradle.buildFinished {
    if (testResults.isNotEmpty()) {
        printResults(testResults)
    }
}
