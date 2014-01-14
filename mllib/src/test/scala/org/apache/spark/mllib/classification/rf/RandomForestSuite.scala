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

package org.apache.spark.mllib.classification.rf

import scala.util.Random
import scala.collection.mutable.ArrayBuffer

import org.scalatest.BeforeAndAfterAll
import org.scalatest.FunSuite
import org.junit.Assert._

import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LabeledPoint


class RandomForestSuite extends FunSuite with BeforeAndAfterAll {
  /** "Close enough" value for floating-point comparisons. */
  @transient private val EPSILON = 0.000001
  @transient private var sc: SparkContext = _
  @transient private val seed = 17
  @transient private val rnd = new Random(seed)

  @transient private val metaInfo = DataMetaInfo(classification = true,
    Array(true, false, false, true), 2, Array(3, -1,-1,2))

  @transient private val TRAIN_DATA = Array(
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

  @transient private val TEST_DATA = Array(
    Array(0, 70.0, 96, 1),
    Array(2, 64.0, 65, 1),
    Array(1, 75.0, 90, 1))

  override def beforeAll() {
    sc = new SparkContext("local", "test")
  }

  override def afterAll() {
    sc.stop()
    System.clearProperty("spark.driver.port")
  }

  private def generateTrainingDataA(): Array[Data] = {
    val data = Data(metaInfo, TRAIN_DATA)
    val points = Array.fill(3)(new ArrayBuffer[LabeledPoint])

    for (point <- data.points) {
      if (point.features(0) == 0.0d) {
        points(0) += point
      } else {
        points(1) += point
      }
    }

    points.map(p => new Data(metaInfo, p.toArray))
  }

  private def generateTrainingDataB(): Array[Data] = {
    val dataset = new Array[Data](3)
    val metaInfo = DataMetaInfo(classification = false, Array(true, false), -1, Array(3, -1))

    dataset(0) = Data(metaInfo, Array.tabulate[LabeledPoint](20) { i =>
      if (i % 3 == 0) {
        LabeledPoint(i + 20, Array(0.0, 40 - i))
      } else if (i % 3 == 1) {
        LabeledPoint(40 - i, Array(1.0, i + 20))
      } else {
        LabeledPoint(i + 20, Array(2.0, i + 20))
      }
    })

    dataset(1) = Data(metaInfo, Array.tabulate[LabeledPoint](20) { i =>
      if (i % 2 == 0) {
        LabeledPoint(i + 10, Array(0.0, 50 - i))
      } else {
        LabeledPoint(50 - i, Array(1.0, i + 10))
      }
    })

    dataset(2) = Data(metaInfo, Array.tabulate[LabeledPoint](10) { i =>
      LabeledPoint(i + 20, Array(0.0, 40 - i))
    })

    dataset
  }

  private def buildForest(dataset: Array[Data]): RandomForestModel = {
    val trees = Array.tabulate[Node](dataset.length) { i =>
      val data = dataset(i)
      val builder = new DecisionTreeBuilder(data.metainfo)
        .setM(data.metainfo.categorical.length)
        .setMinSplitNum(0)
      builder.build(rnd, data)
    }

    new RandomForestModel(trees, dataset(0).metainfo)
  }

  test ("RegressionComputeSplit") {
    val dataset = generateTrainingDataB()
    val computeSplit = new RegressionComputeSplit()

    var split = computeSplit(dataset(0), 1)
    assertEquals(180.0, split.ig, EPSILON)
    assertEquals(38.0, split.split, EPSILON)

    val lesserThan = (point: LabeledPoint) => { point.features(1) < 38.0 }
    split = computeSplit(dataset(0).subset(lesserThan), 1)
    assertEquals(76.5, split.ig, EPSILON)
    assertEquals(21.5, split.split, EPSILON)

    split = computeSplit(dataset(1), 0)
    assertEquals(2205.0, split.ig, EPSILON)
    assertEquals(Double.NaN, split.split, EPSILON)

    val equal = (point: LabeledPoint) => { point.features(0) == 0.0 }
    split = computeSplit(dataset(1).subset(equal), 1)
    assertEquals(250.0, split.ig, EPSILON)
    assertEquals(41.0, split.split, EPSILON)
  }

  test("local Random forest for classification") {
    val dataset = generateTrainingDataA()
    val forest = buildForest(dataset)
    val predictions = forest.predict(TEST_DATA)

    assertArrayEquals(Array(1.0, Double.NaN, Double.NaN), predictions(0), 0.0)
    assertArrayEquals(Array(1.0, 0.0, Double.NaN), predictions(1), 0.0)
    assertArrayEquals(Array(1.0, 1.0, Double.NaN), predictions(2), 0.0)

    assert(1.0 == forest.predict(TEST_DATA(0)))
    // This one is tie-broken, either 1 or 0 is OK
    val label = forest.predict(TEST_DATA(1))
    assert(1.0 == label || 0.0 == label)
    assert(1.0 == forest.predict(TEST_DATA(2)))
  }

  test("local Random forest for regression") {
    val dataset = generateTrainingDataB()
    val forests = new Array[RandomForestModel](dataset.length)

    for (i <- 0 until dataset.length) {
      val subDataset = new Array[Data](dataset.length - 1)
      var k = 0
      for (j <- 0 until dataset.length) {
        if (j != i) {
          subDataset(k) = dataset(j)
          k += 1
        }
      }
      forests(i) = buildForest(subDataset)
    }

    var predictions = forests(0).predict(dataset(0).points.map(_.features))
    assertArrayEquals(Array(20.0, 20.0), predictions(0), EPSILON)
    assertArrayEquals(Array(39.0, 29.0), predictions(1), EPSILON)
    assertArrayEquals(Array(37.0, 29.0), predictions(2), EPSILON)
    assertArrayEquals(Array(22.0, 23.0), predictions(17), EPSILON)

    predictions = forests(1).predict(dataset(1).points.map(_.features))
    assertArrayEquals(Array(30.0, 29.0), predictions(19), EPSILON)

    predictions = forests(2).predict(dataset(2).points.map(_.features))
    assertArrayEquals(Array(29.0, 28.0), predictions(9), EPSILON)

    assertEquals(20.0, forests(0).predict(dataset(0).points(0).features), EPSILON)
    assertEquals(34.0, forests(0).predict(dataset(0).points(1).features), EPSILON)
    assertEquals(33.0, forests(0).predict(dataset(0).points(2).features), EPSILON)
    assertEquals(22.5, forests(0).predict(dataset(0).points(17).features), EPSILON)

    assertEquals(29.5, forests(1).predict(dataset(1).points(19).features), EPSILON)
    assertEquals(28.5, forests(2).predict(dataset(2).points(9).features), EPSILON)
  }

  test("Spark Random forest for classification using full data") {
    val dataset = generateTrainingDataA()
    val data = dataset(0).points ++ dataset(1).points
    // since training data is insufficient, duplicate itself
    val dataRDD  = sc.parallelize(data ++ data, 2)

    val iteration = 100
    var error = 0
    for (i <- 0 until iteration) {
      val forest = RandomForest.train( dataRDD, metaInfo.categorical,
        rnd.nextInt(), 20, false, metaInfo.categorical.length, 0)

      if (1.0 != forest.predict(TEST_DATA(0))) error += 1
      if (1.0 != forest.predict(TEST_DATA(2))) error += 1
    }
    assert (error < 2 * iteration * 0.1)    // error rate must be less than 10%
  }

  test("Spark Random forest for classification using partial data") {
    val dataset = generateTrainingDataA()
    val data = dataset(0).points ++ dataset(1).points
    // since training data is insufficient, duplicate itself
    val dataRDD  = sc.parallelize(data ++ data ++ data, 2)

    val iteration = 100
    var error = 0
    for (i <- 0 until iteration) {
      val forest = RandomForest.train( dataRDD, metaInfo.categorical,
        rnd.nextInt(), 20, true, metaInfo.categorical.length, 0)

      if (1.0 != forest.predict(TEST_DATA(0))) error += 1
      if (1.0 != forest.predict(TEST_DATA(2))) error += 1
    }
    assert (error < 2 * iteration * 0.2)    // error rate must be less than 20%
  }
}
