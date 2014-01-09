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

import scala.concurrent._
import ExecutionContext.Implicits.global
import scala.concurrent.duration.Duration
import scala.util.Random

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression._

/**
 * Builds a random forest.
 * @param metaInfo MetaInfo of training data.
 * @param seed Random seed.
 */
class RandomForest(private val metaInfo: DataMetaInfo, seed: Int)
    extends Serializable{

  /**
   * Builds a random forest using full data. Each tree uses the data sampled from the full data.
   *
   * Each sampled RDD has been coalesced to one partition, thus we can avoid data network IO.
   */
  def runFull(input: RDD[LabeledPoint], nbTrees: Int, m: Int,
      minSplitNum: Int): RandomForestModel = {
    val seeds = {
      val rnd = new Random(seed)
      Array.fill(nbTrees)(rnd.nextInt())
    }

    val trees = Array.tabulate[RDD[Node]](nbTrees) { i =>
      input.sample(withReplacement = true, 1.0, seeds(i)).coalesce(1).mapPartitions { iterator =>
        val rnd = new Random(seeds(i))
        val data = Data(metaInfo, iterator.toArray)
        val builder = new DecisionTreeBuilder(m, minSplitNum)
        val tree = builder.build(rnd, data)

        List(tree).iterator
      }
    }.reduce(_ ++ _)

    new RandomForestModel(trees.collect(), metaInfo)
  }

  /**
   * Builds a random forest using partial data. Each tree uses only the data given by its partition.
   *
   * Each sampled RDD has been coalesced to one partition, thus we can avoid data network IO.
   */
  def runPartial(input: RDD[LabeledPoint], nbTrees: Int, m: Int,
       minSplitNum: Int): RandomForestModel = {
    val numPartitions = input.partitions.length
    val partitionToRDDs =  Array.tabulate(numPartitions) { i =>
      input.mapPartitionsWithIndex{ (index, iterator) =>
        if (index == i) iterator else List.empty[LabeledPoint].iterator
      }.coalesce(1)
    }

    val trees = partitionToRDDs.map { rdd =>
      rdd.mapPartitionsWithIndex { (index, iterator) =>
        val data = Data(metaInfo, iterator.toArray)
        val numTrees = RandomForest.nbTreesOfPartition(nbTrees, numPartitions, index)
        val seeds = {
          val rnd = new Random(seed)
          Array.fill(numTrees)(rnd.nextInt())
        }

        val futures = Array.tabulate(numTrees){ i=>
          val rnd = new Random(seeds(i))
          val builder = new DecisionTreeBuilder(m, minSplitNum)
          future {builder.build(rnd, data.bagging(rnd)) }
        }
        val trees = Array.tabulate(numTrees)(i=> Await.result(futures(i), Duration.Inf))

        trees.toIterator
      }
    }

    new RandomForestModel(trees.map(_.collect()).flatten, metaInfo)
  }
}

object RandomForest {
  /**
   * Train a Random Forest model given an RDD of (label, features) pairs.
   *
   * @param input RDD of (label, array of features) pairs, for categorical features,
   *        they should be converted to Integers, starting from 0.
   * @param partial If the training data is too big to be loaded into one machine, set this to
   *                tree, so that each tree will be trained using partial data.
   * @param seed seed Random seed
   * @param classification Whether the label is categorical or numerical.
   * @param categorical Whether the label is categorical or numerical, true if categorical,
   *                    false if numerical.when one feature is categorical, the corresponding
   *                    value is true; when numerical, false.
   * @param nbLabels Number of label values, just for classification problem
   * @param nbValues nbValues[i] means number of feature i's values, if feature i is numerical,
   *                 nbValues[i] is -1
   * @param nbTrees Number of trees to build, should be greater than number of partitions.
   * @param m number of attributes to select randomly at each node, default is sqrt(dimension)
   * @param minSplitNum minimum number for split, default is 2
   */
  def train(
      input: RDD[LabeledPoint],
      partial: Boolean,
      seed: Int,
      classification: Boolean,
      categorical: Array[Boolean],
      nbLabels: Int,
      nbValues: Array[Int],
      nbTrees: Int,
      m: Int = 0,
      minSplitNum: Int = 0) : RandomForestModel = {
    train(input, partial, seed, new DataMetaInfo(classification, categorical, nbLabels, nbValues),
      nbTrees, m, minSplitNum)
  }

  private[classification] def train(
       input: RDD[LabeledPoint],
       partial: Boolean,
       seed: Int,
       metaInfo: DataMetaInfo,
       nbTrees: Int,
       m: Int,
       minSplitNum: Int): RandomForestModel = {
    val rm = new RandomForest(metaInfo, seed)
    if  (partial) {
      rm.runPartial(input, nbTrees, m, minSplitNum)
    } else {
      rm.runFull(input, nbTrees, m, minSplitNum)
    }
  }

  /**
   * Compute the number of trees for a given partition. The first partition (0) may be longer than
   * the rest of partition because of the remainder.
   *
   * @param nbTrees total number of trees
   * @param nbPartitions total number of partitions
   * @param partition partition to compute the number of trees for
   */
  private[rf] def nbTreesOfPartition(nbTrees: Int, nbPartitions: Int, partition: Int): Int = {
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
