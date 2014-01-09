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

import org.apache.spark.Logging
import org.apache.spark.mllib.regression.LabeledPoint


/**
 * Abstract base class for TreeBuilders
 */
abstract class TreeBuilder extends Logging {

  /**
   * Builds a Decision tree using the training data
   *
   * @param rnd random-numbers generator
   * @param data training data
   * @return root Node
   */
  def build(rnd: Random, data: Data): Node
}

/**
 * Builds a classification tree or regression tree.
 *
 * A classification tree is built when the criterion variable is the categorical attribute.
 * A regression tree is built when the criterion variable is the numerical attribute.
 *
 * @param m number of attributes to select randomly at each node
 * @param minSplitNum minimum number for split
 * @param minVarianceProportion minimum proportion of the total variance for split
 */
class DecisionTreeBuilder(private var m: Int = 0, private var minSplitNum: Double = 2.0,
    private var minVarianceProportion: Double = 1.0e-3) extends TreeBuilder {
  /**
   * indicates which CATEGORICAL attributes have already been selected in the parent nodes
   */
  private var selected: Array[Boolean] = _

  /**
   * IgSplit implementation
   */
  private var computeSplit: ComputeSplit = _

  /**
   * minimum variance for split
   */
  private var minVariance: Double = Double.NaN

  /**
   * Builds a Decision tree using the training data
   *
   * @param rnd random-numbers generator
   * @param data training data
   * @return root Node
   */
  def build(rnd: Random, data: Data): Node = {
    if (data.isEmpty) {
      return new Leaf(Double.NaN)
    }

    if (selected == null) {
      selected = new Array[Boolean](data.metainfo.categorical.length)
    }
    if (m == 0) {
      // set default m
      val e = data.metainfo.categorical.length
      if (data.metainfo.classification) {
        // classification
        m = math.ceil(math.sqrt(e)).toInt
      } else {
        // regression
        m = math.ceil(e / 3.0).toInt
      }
    }

    if (computeSplit == null) {
      if (data.metainfo.classification) {
        // classification
        computeSplit = new ClassificationComputeSplit()
      } else {
        // regression
        computeSplit = new RegressionComputeSplit()
      }
    }

    var sum = 0.0
    if (data.metainfo.classification) {
      // classification
      if (isIdentical(data)) {
        return new Leaf(data.majorityLabel(rnd))
      }
      if (data.identicalLabel()) {
        return new Leaf(data.points(0).label)
      }
    } else {
      // regression
      // sum and sum squared of a label is computed
      var sumSquared = 0.0
      for (point <- data.points) {
        val label = point.label
        sum += label
        sumSquared += label * label
      }

      // computes the variance
      val variance = sumSquared - (sum * sum) / data.size

      // computes the minimum variance
      if (minVariance.compare(Double.NaN) == 0) {
        minVariance = variance / data.size * minVarianceProportion
        logDebug("minVariance:" + minVariance)
      }

      // variance is compared with minimum variance
      if ((variance / data.size) < minVariance) {
        logDebug("variance(" + variance / data.size + ") < minVariance(" +
            minVariance + ") Leaf(" + sum / data.size + ")")
        return new Leaf(sum / data.size)
      }
    }

    val features = DecisionTreeBuilder.randomFeatures(rnd, selected, m)
    if (features == null || features.length == 0) {
      // we tried all the features and could not split the data anymore
      val label = if (data.metainfo.classification) data.majorityLabel(rnd) else sum / data.size
      logWarning("feature which can be selected is not found Leaf(" + label + ")")
      return new Leaf(label)
    }

    // find the best split
    var best: Split = null
    for (f <- features) {
      val split = computeSplit(data, f)
      if (best == null || best.ig < split.ig) {
        best = split
      }
    }

    // information gain is near to zero.
    if (best.ig < DecisionTreeBuilder.EPSILON) {
      val label = if (data.metainfo.classification) data.majorityLabel(rnd) else sum / data.size
      logDebug("ig is near to zero Leaf(" + label + ")")
      return new Leaf(label)
    }

    logDebug("best split " + best)

    val alreadySelected: Boolean = selected(best.feature)
    if (alreadySelected) {
      // attribute already selected
      logWarning("attribute " + best.feature + " already selected in a parent node")
    }

    var root: Node = null
    if (!data.metainfo.categorical(best.feature)) {
      var temp: Array[Boolean] = null

      val less = (point: LabeledPoint) => { point.features(best.feature) < best.split }
      val greater = (point: LabeledPoint) => !less(point)
      val loSubset = data.subset(less)
      val hiSubset = data.subset(greater)

      if (loSubset.isEmpty || hiSubset.isEmpty) {
        // the selected attribute did not change the data, avoid using it in the child notes
        selected(best.feature) = true
      } else {
        // the data changed, so we can unselect all previousely selected NUMERICAL attributes
        temp = selected
        selected = DecisionTreeBuilder.cloneCategoricalAttributes(data.metainfo, selected)
      }

      // size of the subset is less than the minSpitNum
      if (loSubset.size < minSplitNum || hiSubset.size < minSplitNum) {
        // branch is not split
        val label = if (data.metainfo.classification) data.majorityLabel(rnd) else sum / data.size
        logDebug("branch is not split Leaf(" + label + ")")
        return new Leaf(label)
      }

      val loChild = build(rnd, loSubset)
      val hiChild = build(rnd, hiSubset)

      // restore the selection state of the attributes
      if (temp != null) {
        selected = temp
      } else {
        selected(best.feature) = alreadySelected
      }

      root = new NumericalNode(best.feature, best.split, loChild, hiChild)
    } else { // CATEGORICAL attribute
      val values = data.values(best.feature)

      var cnt = 0
      val subsets = new Array[Data](values.length)
      for (index <- 0 until values.length) {
        val equal = (point: LabeledPoint) => { point.features(best.feature) == values(index) }
        subsets(index) = data.subset(equal)
        if (subsets(index).size >= minSplitNum) {
          cnt += 1
        }
      }

      // size of the subset is less than 2
      if (cnt < 2) {
        // branch is not split
        val label = if (data.metainfo.classification) data.majorityLabel(rnd) else sum / data.size
        logDebug("branch is not split Leaf(" + label + ")")
        return new Leaf(label)
      }

      selected(best.feature) = true

      val children = new Array[Node](values.length)
      for (index <- 0 until values.length) {
        children(index) = build(rnd, subsets(index))
      }

      selected(best.feature) = alreadySelected

      root = new CategoricalNode(best.feature, values, children)
    }

    root
  }

  /**
   * checks if all the vectors have identical attribute values. Ignore selected attributes.
   *
   * @return `true` is all the vectors are identical or the data is empty, false otherwise
   */
  private def isIdentical(data: Data): Boolean = {
    if (data.isEmpty) return true

    val first = data.points(0)
    for (feature <- 0 until selected.length) {
      if (!selected(feature)) {
        for (index <- 1 until data.size) {
          if (data.points(index).features(feature) != first.features(feature)) {
            return false
          }
        }
      }
    }

    true
  }
}

object DecisionTreeBuilder extends Logging {
  private val NO_FEATURES = new Array[Int](0)
  private val EPSILON = 1.0e-6

  /**
   * Make a copy of the selection state of the attributes, unselect all numerical attributes
   *
   * @param selected selection state to clone
   * @return cloned selection state
   */
  private def cloneCategoricalAttributes(metainfo: DataMetaInfo,
      selected: Array[Boolean]): Array[Boolean] = {
    val cloned = new Array[Boolean](selected.length)

    for (i <- 0 until selected.length) {
      cloned(i) = metainfo.categorical(i) && selected(i)
    }

    cloned
  }

  /**
   * Randomly selects m features to consider for split
   *
   * @param rnd      random-numbers generator
   * @param selected features' state (selected or not)
   * @param m        number of features to choose
   * @return list of selected features' indices, or null if all features have already been selected
   */
  def randomFeatures(rnd: Random, selected: Array[Boolean], m: Int): Array[Int] = {
    val nonSelected = selected.indices.filterNot(selected)

    if (nonSelected.isEmpty) {
      logWarning("All features are selected !")
      NO_FEATURES
    } else if (nonSelected.size <= m) {
      nonSelected.toArray
    } else {
      rnd.shuffle(nonSelected).take(m).toArray
    }
  }
}
