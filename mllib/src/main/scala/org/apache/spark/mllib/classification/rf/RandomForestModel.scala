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

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.classification.ClassificationModel

/**
 * Classification model trained using Logistic Regression.
 *
 * @param trees trees that are built on training data.
 * @param metaInfo the metainfo of data.
 */
class RandomForestModel(val trees: Array[Node], val metaInfo: DataMetaInfo) extends ClassificationModel {

  /*@transient*/ private val rng = new Random()

  def predict(testData: RDD[Array[Double]]): RDD[Double] = testData.map(predict)

  def predict(testData: Array[Double]): Double = {
    if (!metaInfo.classification) {
      var sum = 0.0
      var cnt = 0
      trees.foreach { tree =>
        val prediction = tree.classify(testData)
        if (!prediction.isNaN) {
          sum += prediction
          cnt += 1
        }
      }
      if (cnt > 0) sum / cnt else Double.NaN
    } else {
      val predictions = new Array[Int](metaInfo.nbLabels)
      trees.foreach { tree =>
        val prediction = tree.classify(testData)
        if (!prediction.isNaN) {
          predictions(prediction.toInt) += 1
        }
      }

      if (predictions.sum == 0) Double.NaN  // no prediction available
      else DataUtils.maxIndex(rng, predictions)
    }
  }
}

/**
 * Represents an abstract node of a decision tree
 */
abstract class Node extends Serializable {

  /**
   * predicts the label for the instance
   *
   * @return -1 if the label cannot be predicted
   */
  def classify(x : Array[Double]): Double

  /**
   * @return the total number of nodes of the tree
   */
  def numNodes(): Long

  /**
   * @return the maximum depth of the tree
   */
  def maxDepth(): Long
}

/**
 * Represents a Leaf node
 */
class Leaf(val label: Double) extends Node {

  override def classify(x: Array[Double]): Double = label

  override def maxDepth(): Long = 1

  override def numNodes(): Long = 1
}

class CategoricalNode(private val feature: Int,
    private val values: Array[Double],
    private val children: Array[Node]) extends Node {

  def this() = this(0, null, null)

  override def classify(x: Array[Double]): Double = {
    val index = values.indexOf(x(feature))

    if (index == -1) Double.NaN else children(index).classify(x)
  }

  override def maxDepth(): Long =  children.map(child => child.maxDepth()).max + 1

  override def numNodes(): Long = children.map(child => child.numNodes()).sum + 1
}

/**
 * Represents a node that splits using a numerical attribute
 *
 * @param feature numerical attribute to split for
 * @param split split value
 * @param loChild child node when attribute's value < split value
 * @param hiChild child node when attribute's value >= split value
 */
class NumericalNode(private val feature: Int,
    private val split: Double,
    private val loChild: Node,
    private val hiChild: Node) extends Node {

  override def classify(x: Array[Double]): Double = {
    if (x(feature) < split) loChild.classify(x) else hiChild.classify(x)
  }

  override def maxDepth(): Long = 1 + math.max(loChild.maxDepth(), hiChild.maxDepth())

  override def numNodes(): Long = 1 + loChild.numNodes() + hiChild.numNodes()
}
