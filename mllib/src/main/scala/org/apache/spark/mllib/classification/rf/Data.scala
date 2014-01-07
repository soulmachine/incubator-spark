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

import scala.collection.mutable
import scala.util.Random

import org.apache.spark.mllib.regression.LabeledPoint

/**
 * Holds a list of vectors and their corresponding `DataMetaInfo`. contains various operations that
 * deals with the vectors (subset, count,...)
 */
private [rf] case class Data(metainfo: DataMetaInfo, points: Array[LabeledPoint]) {

  def size: Int = points.size
  def isEmpty: Boolean = points.isEmpty

  /**
   * finds all distinct values of a given feature
   */
  def values(feature: Int): Array[Double] = {
    val result = new mutable.HashSet[Double]
    points.foreach(result += _.features(feature))
    result.toArray
  }

  /**
   * @return the subset from this data that matches the given condition
   */
  def subset(condition: LabeledPoint => Boolean): Data = {
    new Data(metainfo, points.filter(condition))
  }

  /**
   * If data has N cases, sample N cases at random -but with replacement.
   */
  def bagging(rnd: Random): Data = {
    val dataSize = size
    val bag = Array.fill[LabeledPoint](dataSize) {
      points(rnd.nextInt(dataSize))
    }

    new Data(metainfo, bag)
  }

  /**
   * checks if all the vectors have identical label values
   */
  def identicalLabel(): Boolean = {
    isEmpty || {
      val first = points.head.label
      points.tail.find(_.label != first).isEmpty
    }
  }

  /**
   * finds the majority label, breaking ties randomly.
   *
   * This method can be used when the criterion variable is the categorical attribute.
   *
   * @return the majority label value
   */
  def majorityLabel(rnd: Random): Int = {
    val counts = new Array[Int](metainfo.nbLabels)
    points.foreach(point => counts(point.label.toInt) += 1)
    DataUtils.maxIndex(rnd, counts)
  }

  def entropy(): Double = {
    val counts = new Array[Int](metainfo.nbLabels)
    points.foreach(point => counts(point.label.toInt) += 1)
    ClassificationComputeSplit.entropy(counts, size)
  }
}

/**
 * The data meta info of training data.
 *
 * @param classification Whether the label is categorical or numerical.
 * @param categorical Whether the label is categorical or numerical, true if categorical,
 *                    false if numerical.when one feature is categorical, the corresponding
 *                    value is true; when numerical, false.
 * @param nbLabels number of label values
 * @param nbValues nbValues[i] means number of feature i's values, if feature i is numerical,
 *                 nbValues[i] equals to -1
 */
private [rf] case class DataMetaInfo(classification: Boolean, categorical: Array[Boolean],
    nbLabels: Int, nbValues: Array[Int]) extends Serializable

