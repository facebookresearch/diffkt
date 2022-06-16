/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import org.jetbrains.dokka.gradle.DokkaTask


plugins {
    `maven-publish`
    id("shapeKt") version "1.0"
    id("org.jetbrains.dokka") version "1.6.0"
    signing
}

repositories {
    jcenter() // or maven(url="https://dl.bintray.com/kotlin/dokka")
    mavenCentral()
    maven {
        name = "GitHubPackages"
        url = uri("https://maven.pkg.github.com/facebookresearch/diffkt")
        credentials {
            username = System.getenv("GITHUB_ACTOR")
            password = System.getenv("GITHUB_TOKEN")
        }
    }
    mavenLocal()
    mavenCentral()
}

dependencies {
    implementation(group = "net.bytebuddy", name = "byte-buddy", version="1.12.7")
    compileOnly("shapeKt:annotations:1.0")
    compileOnly("shapeKt:shape-functions:1.0")
    testCompileOnly("shapeKt:annotations:1.0")
    testCompileOnly("shapeKt:shape-functions:1.0")
    testImplementation(project(":testutils"))
}

//sources
val sourcesJar by tasks.register<Jar>("sourcesJar") {
    group = "build"
    description = "Assembles a jar archive containing the main sources."
    archiveClassifier.set("sources")

    from(sourceSets["main"].allSource)
    from("LICENSE")
}

//documentation
val javadocJar by tasks.creating(Jar::class) {
    archiveClassifier.set("javadoc")
    from("$buildDir/javadoc")
}

tasks.register<Exec>("buildExternalJni") {
    commandLine("../../cpp/ops/scripts/build.sh")
}

tasks.register<Exec>("buildGpuJni") {
    commandLine("../../cpp/gpuops/build.sh")
}

tasks.named("compileKotlin") {
    dependsOn("buildExternalJni")
    if (project.hasProperty("gpu")) {
        dependsOn("buildGpuJni")
    }
}

tasks.withType<org.jetbrains.kotlin.gradle.tasks.KotlinCompile>() {
    kotlinOptions.freeCompilerArgs += "-Xserialize-ir=inline"
}

// To publish locally, run the following commands:
//          export GITHUB_USERNAME=${{YOUR USRNAME}}
//          export GITHUB_ACCESS_TOKEN=${{ACCESS TOKEN WITH PACKAGE WRITE ACCESS}}
//          ./gradlew publish -Pversion=0.1.0-$(git rev-parse --short HEAD)


val repositoryUsername: String by project
val repositoryPassword: String by project
val signingKey: String? by project
val signingPassword: String? by project


fun isRelease() = true // System.getenv("VERSION_NAME") != null

publishing {
    repositories {
        maven {
            val releasesRepoUrl = uri("https://oss.sonatype.org/service/local/staging/deploy/maven2/")
            val snapshotsRepoUrl = uri("https://oss.sonatype.org/content/repositories/snapshots/")
            name = "deploy"
            url = if (isRelease()) releasesRepoUrl else snapshotsRepoUrl
            credentials {
                username = System.getenv("repositoryUsername") ?: repositoryUsername
                password = System.getenv("repositoryPassword") ?: repositoryPassword
            }
        }
    }

    publications {
        create<MavenPublication>("Diffkt") {
            //from(components["javaPlatform"])
            from(components["java"])

            artifact(sourcesJar)
            artifact(javadocJar)

            pom {
                name.set("DiffKt")
                description.set("Automatic differentiation in Kotlin")
                url.set("https://github.com/facebookresearch/diffkt")

                groupId = "com.facebook.diffkt"
                artifactId = "diffkt"
                version = "0.0.1-DEV2"

                scm {
                    connection.set("scm:git:https://github.com/facebookresearch/diffkt")
                    developerConnection.set("scm:git:https://github.com/thomasnield/")
                    url.set("https://github.com/facebookresearch/diffkt")
                }

                licenses {
                    license {
                        name.set("MIT-1.0")
                        url.set("https://opensource.org/licenses/MIT")
                    }
                }

                developers {
                    developer {
                        id.set("thomasnield")
                        name.set("Thomas Nield")
                        email.set("thomasnield@live.com")
                    }
                }
            }
        }
    }
}

signing {
    val publications: PublicationContainer = (extensions.getByName("publishing") as PublishingExtension).publications

    if (isRelease()) {
        sign(publications)
    }
}

apply(plugin="org.jetbrains.dokka")
tasks.withType<DokkaTask>().configureEach{
    outputDirectory.set(rootDir.resolve("../docs/api"))
    dokkaSourceSets {
        configureEach {
            samples.from(rootDir.resolve("samples/src/main/kotlin"))
        }
    }
}
