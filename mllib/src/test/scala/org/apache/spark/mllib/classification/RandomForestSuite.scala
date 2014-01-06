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

package org.apache.spark.mllib.classification

import scala.util.Random
import scala.collection.mutable.ListBuffer
import org.scalatest.BeforeAndAfterAll
import org.scalatest.FunSuite
import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.rf.{Node, Data, DataMetainfo}
import org.apache.spark.mllib.classification.rf.{DecisionTreeBuilder, RandomForestModel}
import org.apache.spark.mllib.classification.rf.DefaultComputeSplit
import org.apache.spark.mllib.classification.rf.ClassificationComputeSplit
import org.apache.spark.mllib.classification.rf.RegressionComputeSplit


class RandomForestSuite extends FunSuite with BeforeAndAfterAll {
  /** "Close enough" value for floating-point comparisons. */
  @transient private val EPSILON = 0.000001
  @transient private var sc: SparkContext = _
  @transient private val rnd = new Random()
  @transient private val TRAIN_DATA = List(LabeledPoint(1,Array(1,85.0,85,0)),
    LabeledPoint(1,Array(1,80.0,90,1)), LabeledPoint(0,Array(2,83.0,86,0)),
    LabeledPoint(0,Array(0,70.0,96,0)), LabeledPoint(0,Array(0,68.0,80,0)), LabeledPoint(1,Array(0,65.0,70,1)),
    LabeledPoint(0,Array(2,64.0,65,1)), LabeledPoint(1,Array(1,72.0,95,0)),
    LabeledPoint(0,Array(1,69.0,70,0)), LabeledPoint(0,Array(0,75.0,80,0)), LabeledPoint(0,Array(1,75.0,70,1)),
    LabeledPoint(0,Array(2,72.0,90,1)), LabeledPoint(0,Array(2,81.0,75,0)),
    LabeledPoint(1,Array(0,71.0,91,1)))

  @transient private val TEST_DATA = Array(Array(1.0,70,96,1), Array(2.0,64,65,1), Array(0.0,75,90,1))

  override def beforeAll() {
    sc = new SparkContext("local", "test")
  }

  override def afterAll() {
    sc.stop()
    System.clearProperty("spark.driver.port")
  }
  
  private def generateTrainingDataA(): Array[Data] = {
    val metainfo = DataMetainfo(true, Array(true, false, false, true),
      2, Array(3, -1,-1,2))
    val data = Data(metainfo, TRAIN_DATA)
    val points = Array.fill(3)(new ListBuffer[LabeledPoint])
    
    for (point <- data.points) {
      if (point.features(0) == 0.0d) {
        points(0) += point
      } else {
        points(1) += point
      }
    }
    points.map(p => new Data(metainfo, p.toList))
  }

  private def generateTrainingDataB(): Array[Data] = {
    val datas = new Array[Data](3)
    var metainfo = DataMetainfo(false, Array(true, false), -1, Array(3, -1))
    // Training data
    var trainData = new Array[LabeledPoint](20)
    for (i <- 0 until trainData.length) {
      trainData(i) = if (i % 3 == 0)      LabeledPoint(i+20, Array(0.0, 40-i))
                     else if (i % 3 == 1) LabeledPoint(40-i, Array(1.0, i+20))
                     else                 LabeledPoint(i+20, Array(2.0, i+20))
    }
    datas(0) = Data(metainfo, trainData.toList)

    // Training data
    trainData = new Array[LabeledPoint](20)
    for (i <- 0 until trainData.length) {
      trainData(i) = if (i % 2 == 0) LabeledPoint(i+10, Array(0.0, 50-i))
                     else LabeledPoint(50-i, Array(1.0, i+10))
    }
    datas(1) = Data(metainfo, trainData.toList)

    // Training data
    trainData = new Array[LabeledPoint](10)
    for (i <- 0 until trainData.length) {
      trainData(i) = LabeledPoint(i+20, Array(0.0, 40-i))
    }
    datas(2) = Data(metainfo, trainData.toList)

    datas
  }
  
  private def buildForest(datas: Array[Data]): RandomForestModel = {
    val trees = new Array[Node](datas.length)
    for (i <- 0 until datas.length) {
      val data = datas(i)
      // build tree
      val builder = new DecisionTreeBuilder()
      builder.setM(data.metainfo.categorical.length - 1)
      builder.setMinSplitNum(0)
      trees(i) = builder.build(rnd, data)
    }
    new RandomForestModel(trees, datas(0).metainfo)
  }
  
  test ("ClassificationComputeSplit") {
    val ref = new DefaultComputeSplit()
    val opt = new ClassificationComputeSplit()
    val metainfo = DataMetainfo(true, Array(true, false, false, true),
      2, Array(3, -1,-1,2))
    val data = Data(metainfo, TRAIN_DATA)
    
    for (f <- 0 until data.metainfo.categorical.length) {
      val expected = ref(data, f)
      val actual = opt(data, f)
      
      if (!metainfo.categorical(f)) {
          assert(math.abs(expected.split - actual.split) < EPSILON)
      }
    }
  }
  
  test ("RegressionComputeSplit") {
    val datas = generateTrainingDataB()

    val computeSplit = new RegressionComputeSplit()
    var split = computeSplit(datas(0), 1)
    assert(math.abs(180.0 - split.ig) < EPSILON)
    assert(math.abs(38.0 - split.split) < EPSILON)
    
    val lesserThan = (point: LabeledPoint) => { point.features(1) < 38.0 }
    split = computeSplit(datas(0).subset(lesserThan), 1)
    assert(math.abs(76.5 - split.ig) < EPSILON)
    assert(math.abs(21.5 - split.split) < EPSILON)

    split = computeSplit(datas(1), 0)
    assert(math.abs(2205.0 - split.ig) < EPSILON)
    assert(split.split.isNaN)
    
    val equal = (point: LabeledPoint) => { point.features(0) == 0.0 }
    split = computeSplit(datas(1).subset(equal), 1)
    assert(math.abs(250.0 - split.ig) < EPSILON)
    assert(math.abs(41.0 - split.split) < EPSILON)
  }
  
  test("DecisionTreeBuilder") {
    // classification
    val points = List (
      LabeledPoint(1, Array( 0.25, 0.0, 0.0, 5.143998668220409E-4, 0.019847102289905324, 3.5216524641879855E-4, 0.0, 0.6225857142857143)),
      LabeledPoint(0, Array( 0.25, 0.0, 0.0, 0.0010504411519893459, 0.005462138323171171, 0.0026130744829756746, 0.0, 0.4964857142857143)),
      LabeledPoint(1, Array( 0.25, 0.0, 0.0, 0.0010504411519893459, 0.005462138323171171, 0.0026130744829756746, 0.0, 0.4964857142857143)),
      LabeledPoint(0, Array( 0.25, 0.0, 0.0, 5.143998668220409E-4, 0.019847102289905324, 3.5216524641879855E-4, 0.0, 0.6225857142857143))
    )
    val metainfo = DataMetainfo(true, Array(false,false,false,false,false,false,false,false), 2, null)
    val data = Data(metainfo, points)
    val builder = new DecisionTreeBuilder()
    builder.build(rnd, data)
    
    // regression
    val points2 = List (
      LabeledPoint(4, Array( 0.25, 0.0, 0.0, 5.143998668220409E-4, 0.019847102289905324, 3.5216524641879855E-4, 0.0, 0.6225857142857143)),
      LabeledPoint(3, Array( 0.25, 0.0, 0.0, 0.0010504411519893459, 0.005462138323171171, 0.0026130744829756746, 0.0, 0.4964857142857143)),
      LabeledPoint(4, Array( 0.25, 0.0, 0.0, 0.0010504411519893459, 0.005462138323171171, 0.0026130744829756746, 0.0, 0.4964857142857143)),
      LabeledPoint(3, Array( 0.25, 0.0, 0.0, 5.143998668220409E-4, 0.019847102289905324, 3.5216524641879855E-4, 0.0, 0.6225857142857143))
    )
    val metainfo2 = DataMetainfo(false, Array(false,false,false,false,false,false,false,false), -1, null)
    val data2 = Data(metainfo2, points2)
    val builder2 = new DecisionTreeBuilder()
    builder2.build(rnd, data2)
  }
  
  test("Random forest for classification") {
    // Training data
    val datas = generateTrainingDataA()
    // Build Forest
    val forest = buildForest(datas)
    
    println(forest.predict(TEST_DATA(0)))
    println(forest.predict(TEST_DATA(1)))
    println(forest.predict(TEST_DATA(2)))
    assert(math.abs(1.0 - forest.predict(TEST_DATA(0))) < EPSILON)
    // This one is tie-broken -- 1 is OK too
    // assert(math.abs(0.0 - forest.predict(TEST_DATA(1))) < EPSILON)
    assert(math.abs(1.0 - forest.predict(TEST_DATA(2))) < EPSILON)
  }
  
  test("Random forest for regression") {
    
  }
}
