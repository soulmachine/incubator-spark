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
import org.apache.spark.mllib.regression._

/**
 * Builds a random forest using partial data. Each worker uses only the data given by its partition.
 *
 * @param metaInfo MetaInfo of training data.
 * @param nbTrees Number of trees to build.
 * @param seed Random seed.
 */
class RandomForest(private val metaInfo: DataMetaInfo, private val nbTrees: Int, seed: Int)
    extends Serializable{
  /**
   * Run the algorithm with the configured parameters on an input RDD of LabeledPoint .
   */
  def run(input: RDD[LabeledPoint]) : RandomForestModel = {
    val numPartitions = input.partitions.length

    val trees = input.mapPartitionsWithIndex { (index, iterator) =>
      val rnd = new Random(seed)
      val data = Data(metaInfo, iterator.toArray)
      val builder = new DecisionTreeBuilder()
      val numTrees = nbTreesOfPartition(numPartitions, index)

      val trees = new Array[Node](numTrees)
      for (i <- 0 until numTrees) {
        trees(i) = builder.build(rnd, data.bagging(rnd))
      }

      trees.toIterator
    }

    new RandomForestModel(trees.collect(), metaInfo, seed)
  }

  /**
   * Compute the number of trees for a given partition. The first partition (0) may be longer than
   * the rest of partition because of the remainder.
   *
   * @param nbPartitions total number of partitions
   * @param partition partition to compute the number of trees for
   */
  def nbTreesOfPartition(nbPartitions: Int, partition: Int): Int = {
    var result = nbTrees / nbPartitions
    // if nbTrees is less than nbPartitions, each partition at least builds one tree
    if (result <= 0) result = 1
    else {
      if (partition == 0) {
        result += nbTrees - result * nbPartitions
      }
    }
    result
  }
}

object RandomForest {
  /**
   * Train a Random Forest model given an RDD of (label, features) pairs.
   *
   * @param input RDD of (label, array of features) pairs, for categorical features,
   *        they should be converted to Integers, starting from 0.
   * @param seed seed Random seed
   * @param classification Whether the label is categorical or numerical.
   * @param categorical Whether the label is categorical or numerical, true if categorical,
   *                    false if numerical.when one feature is categorical, the corresponding
   *                    value is true; when numerical, false.
   * @param nbLabels Number of label values, just for classification problem
   * @param nbValues nbValues[i] means number of feature i's values, if feature i is numerical,
   *                 nbValues[i] is -1
   * @param nbTrees Number of trees to build, should be greater than number of partitions.
   */
  def train(
      input: RDD[LabeledPoint],
      seed: Int,
      classification: Boolean,
      categorical: Array[Boolean],
      nbLabels: Int = -1,
      nbValues: Array[Int] = null,
      nbTrees: Int = -1) : RandomForestModel = {
    train(input, seed, new DataMetaInfo(classification, categorical, nbLabels, nbValues), nbTrees)
  }

  /**
   * Train a Random Forest model given an RDD of (label, features) pairs.
   *
   * @param input RDD of (label, array of features) pairs, for categorical features,
   *        they should be converted to Integers, starting from 0.
   * @param seed seed Random seed
   * @param metaInfo Meta info of training data
   * @param nbTrees Number of trees to build, should be greater than number of partitions.
   */
  private[classification] def train(
       input: RDD[LabeledPoint],
       seed: Int,
       metaInfo: DataMetaInfo,
       nbTrees: Int): RandomForestModel = {
    new RandomForest(metaInfo, nbTrees, seed).run(input)
  }
}
