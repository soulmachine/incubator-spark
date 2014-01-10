/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.examples

import scala.Array

import org.apache.spark._
import org.apache.spark.mllib.classification.rf.RandomForest
import org.apache.spark.mllib.regression.LabeledPoint

/**
 * Random forest based classification.
 */
object SparkRF {
  private val TRAIN_DATA = Array(
    LabeledPoint(1, Array(1, 85.0, 85, 0)),
    LabeledPoint(1, Array(1, 80.0, 90, 1)),
    LabeledPoint(0, Array(2, 83.0, 86, 0)),
    LabeledPoint(0, Array(0, 70.0, 96, 0)),
    LabeledPoint(0, Array(0, 68.0, 80, 0)),
    LabeledPoint(1, Array(0, 65.0, 70, 1)),
    LabeledPoint(0, Array(2, 64.0, 65, 1)),
    LabeledPoint(1, Array(1, 72.0, 95, 0)),
    LabeledPoint(0, Array(1, 69.0, 70, 0)),
    LabeledPoint(0, Array(0, 75.0, 80, 0)),
    LabeledPoint(0, Array(1, 75.0, 70, 1)),
    LabeledPoint(0, Array(2, 72.0, 90, 1)),
    LabeledPoint(0, Array(2, 81.0, 75, 0)),
    LabeledPoint(1, Array(0, 71.0, 91, 1)))

  private val TEST_DATA = Array(
    Array(0, 70.0, 96, 1),
    Array(2, 64.0, 65, 1),
    Array(1, 75.0, 90, 1))

  def main(args: Array[String]) {
    if (args.length == 0) {
      System.err.println("Usage: SparkRF <master> [<slices>]")
      System.exit(1)
    }
    val sc = new SparkContext(args(0), "SparkRF",
      System.getenv("SPARK_HOME"), SparkContext.jarOfClass(this.getClass))
    val numSlices = if (args.length > 1) args(1).toInt else 2
    val points = sc.parallelize(TRAIN_DATA, numSlices).cache()

    println("Training data (the first column is the label):")
    TRAIN_DATA.foreach { point =>
      print(point.label + " ")
      println(point.features.mkString(" "))
    }

    println("Begin training.")
    val forest = RandomForest.train(points, true, Array(true, false, false, true),
      17, 20, false, TRAIN_DATA(0).features.length, 0)
    println("Training completed.")

    println("Use the random forest for prediction.")
    TEST_DATA.foreach { point =>
      print(point.mkString(" "))
      println(" => " + forest.predict(point))
    }
    System.exit(0)
  }
}
