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

import scala.util.Sorting
import scala.math.Ordering
import org.apache.spark.mllib.regression.LabeledPoint


/**
 * Contains enough information to identify each split.
 *
 * @param feature the feature to split for
 * @param ig Information Gain of the split
 * @param split split value for NUMERICAL features
 */
private [rf] case class Split(feature: Int, ig: Double, split: Double = Double.NaN) {
  override def toString: String = "feature: " + feature + ", ig: " + ig + ", split: " + split
}

/**
 * Computes the best split using the Information Gain measure.
 */
private [rf] abstract class ComputeSplit extends ((Data, Int) => Split)

object ComputeSplit {
  val LOG2 = math.log(2.0)
}

private [rf] class DefaultComputeSplit extends ComputeSplit {
  override def apply(data: Data, feature: Int): Split = {
    if (!data.metainfo.categorical(feature)) {
      val values = data.values(feature)
      var bestIg = -1.0
      var bestSplit = 0.0

      for (value <- values) {
        val ig = DefaultComputeSplit.numericalIg(data, feature, value)
        if (ig > bestIg) {
          bestIg = ig
          bestSplit = value
        }
      }

      new Split(feature, bestIg, bestSplit)
    } else {
      val ig = DefaultComputeSplit.categoricalIg(data, feature)

      new Split(feature, ig)
    }
  }
}

private [rf] object DefaultComputeSplit {
  /**
   * Computes the split for a CATEGORICAL attribute.
   */
  private def categoricalIg(data: Data, feature: Int): Double = {
    val values = data.values(feature)
    val hy = data.entropy() // H(Y)
    var hyx = 0.0 // H(Y|X)
    val invDataSize = 1.0 / data.size

    for (value <- values) {
      val equal = (point: LabeledPoint) => { point.features(feature) == value }
      val subset = data.subset(equal)
      hyx += subset.size * invDataSize * subset.entropy()
    }

    hy - hyx
  }

  /**
   * Computes the best split for a NUMERICAL attribute.
   */
  private def numericalIg(data: Data, feature: Int, split: Double): Double = {
    var hy = data.entropy()
    val invDataSize = 1.0 / data.size

    val lesserThan = (point: LabeledPoint) => { point.features(feature) < split }
    val greaterThan = (point: LabeledPoint) => !lesserThan(point)
    // LO subset
    var subset = data.subset(lesserThan)
    hy -= subset.size * invDataSize * subset.entropy()

    // HI subset
    subset = data.subset(greaterThan)
    hy -= subset.size * invDataSize * subset.entropy()

    hy
  }
}

/**
 * Classification problem implementation of ComputeSplit.
 *
 * This class can be used when the criterion variable is the categorical attribute.
 */
private [rf] class ClassificationComputeSplit extends ComputeSplit {
  override def apply(data: Data, feature: Int): Split = {
    if (data.metainfo.categorical(feature)) ClassificationComputeSplit.categoricalSplit(data, feature)
    else ClassificationComputeSplit.numericalSplit(data, feature)
  }
}

private [rf] object ClassificationComputeSplit {
  /**
   * Computes the split for a CATEGORICAL attribute.
   */
  private def categoricalSplit(data: Data, feature: Int): Split = {
    val values = data.values(feature)
    val counts = Array.ofDim[Int](values.length, data.metainfo.nbLabels)
    val countAll = new Array[Int](data.metainfo.nbLabels)

    computeFrequencies(data, feature, values, counts, countAll)

    val dataSize = data.points.size
    val invDataSize = 1.0 / dataSize
    val hd = entropy(countAll, dataSize)  // H(D)
    var hdf = 0.0                         // H(D|F)

    for (index <- 0 until values.length) {
      val size = counts(index).sum
      hdf += size * invDataSize * entropy(counts(index), size)
    }

    val ig = hd - hdf
    new Split(feature, ig)
  }

  /**
   * Computes the best split for a NUMERICAL attribute.
   */
  private def numericalSplit(data: Data, feature: Int): Split = {
    val values = sortedValues(data, feature)

    val counts = Array.ofDim[Int](values.length, data.metainfo.nbLabels)
    val countGreater = new Array[Int](data.metainfo.nbLabels)
    val countLess = new Array[Int](data.metainfo.nbLabels)

    computeFrequencies(data, feature, values, counts, countGreater)

    val dataSize = data.points.size
    val invDataSize = 1.0 / dataSize
    val hd = entropy(countGreater, dataSize)  // H(D)

    var best = -1
    var bestIg = -1.0

    // try each possible split value
    for (index <- 0 until values.length) {
      var ig = hd

      // instance with attribute value < values[index]
      var size = countLess.sum
      ig -= size * invDataSize * entropy(countLess, size)

      // instance with attribute value >= values[index]
      size = countGreater.sum
      ig -= size * invDataSize * entropy(countGreater, size)

      if (ig > bestIg) {
        bestIg = ig
        best = index
      }

      DataUtils.add(countLess, counts(index))
      DataUtils.dec(countGreater, counts(index))
    }

    if (best == -1) {
      throw new IllegalStateException("no best split found !")
    } else {
      new Split(feature, bestIg, values(best))
    }
  }

  /**
   * Computes the Entropy
   *
   * @param counts   counts[i] = numPoints with label i
   * @param dataSize total number of data points
   */
  def entropy(counts: Array[Int], dataSize: Int): Double = {
    if (dataSize == 0) return 0.0

    var h = 0.0
    val invDataSize = 1.0 / dataSize

    for (count <- counts) {
      if (count > 0) {
        val p = count * invDataSize
        h += -p * Math.log(p) / ComputeSplit.LOG2
      }
    }

    h
  }

  /**
   * Return the sorted list of distinct values for the given attribute
   */
  private def sortedValues(data: Data, feature: Int): Array[Double] = {
    val values = data.values(feature)
    Sorting.quickSort(values)
    values
  }

  /** compute frequencies. */
  private def computeFrequencies(data: Data, feature: Int, values: Array[Double],
      counts: Array[Array[Int]], countAll: Array[Int]) {
    for (point <- data.points) {
      counts(values.indexOf(point.features(feature)))(point.label.toInt) += 1
      countAll(point.label.toInt) += 1
    }
  }
}

/**
 * Regression problem implementation of ComputeSplit.
 *
 * This class can be used when the criterion variable is the numerical attribute.
 */
private [rf] class RegressionComputeSplit extends ComputeSplit {
  override def apply(data: Data, feature: Int): Split = {
    if (data.metainfo.categorical(feature)) {
      RegressionComputeSplit.categoricalSplit(data, feature)
    } else {
      RegressionComputeSplit.numericalSplit(data, feature)
    }
  }
}

private [rf] object RegressionComputeSplit {
  /**
   * Comparator for Instance sort
   */
  private class PointOrdering(private val feature: Int) extends Ordering[LabeledPoint] {
    def compare(p1: LabeledPoint, p2: LabeledPoint): Int =
      p1.features(feature).compare(p2.features(feature))
  }

  /**
   * Computes the split for a CATEGORICAL attribute.
   */
  private def categoricalSplit(data: Data, feature: Int): Split = {
    val ra = Array.fill(data.metainfo.nbValues(feature))(new FullRunningAverage())
    val sk = new Array[Double](data.metainfo.nbValues(feature))
    val totalRa = new FullRunningAverage()
    var totalSk = 0.0

    for (point <- data.points) {
      // computes the variance
      val x = point.features(feature).toInt
      val y = point.label
      if (ra(x).count == 0) {
        ra(x).addItem(y)
        sk(x) = 0.0
      } else {
        val mean = ra(x).average
        ra(x).addItem(y)
        sk(x) += (y - mean) * (y - ra(x).average)
      }

      // total variance
      if (totalRa.count == 0) {
        totalRa.addItem(y)
        totalSk = 0.0
      } else {
        val mean = totalRa.average
        totalRa.addItem(y)
        totalSk += (y - mean) * (y - totalRa.average)
      }
    }

    // computes the variance gain
    var ig = totalSk
    for (aSk <- sk) {
      ig -= aSk
    }

    new Split(feature, ig)
  }

  /**
   * Computes the best split for a NUMERICAL attribute.
   */
  private def numericalSplit(data: Data, feature: Int): Split = {
    val ra = Array.fill(2)(new FullRunningAverage())  // less, greater
    val sk = new Array[Double](2)   // less, greater

    // points sort
    val sortedPoints = data.points.toArray
    implicit val ordering = new PointOrdering(feature)
    Sorting.quickSort(sortedPoints)

    for (point <- sortedPoints) {
      val y = point.label
      if (ra(1).count == 0) {
        ra(1).addItem(y)
        sk(1) = 0.0
      } else {
        val mean = ra(1).average
        ra(1).addItem(y)
        sk(1) += (y - mean) * (y - ra(1).average)
      }
    }

    val totalSk = sk(1)

    // find the best split point
    var split = Double.NaN
    var preSplit = Double.NaN
    var bestVal = Double.MaxValue
    var bestSk = 0.0

    // computes total variance
    for (point <- sortedPoints) {
      val y = point.label

      if (point.features(feature) > preSplit) {
        val curVal = sk(0) / ra(0).count + sk(1) / ra(1).count
        if (curVal < bestVal) {
          bestVal = curVal
          bestSk = sk(0) + sk(1)
          split = (point.features(feature) + preSplit) / 2.0
        }
      }

      // computes the variance
      if (ra(0).count == 0) {
        ra(0).addItem(y)
        sk(0) = 0.0
      } else {
        val mean = ra(0).average
        ra(0).addItem(y)
        sk(0) += (y - mean) * (y - ra(0).average)
      }

      val mean = ra(1).average
      ra(1).removeItem(y)
      sk(1) -= (y - mean) * (y - ra(1).average)

      preSplit = point.features(feature)
    }

    // computes the variance gain
    val ig = totalSk - bestSk

    new Split(feature, ig, split)
  }
}
