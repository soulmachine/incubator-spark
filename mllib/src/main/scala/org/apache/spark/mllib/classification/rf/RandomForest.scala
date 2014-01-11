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

import scala.concurrent.{Await, Future}
import scala.concurrent.duration.Duration
import scala.concurrent.ExecutionContext.Implicits.global
import scala.util.Random

import org.apache.spark.mllib.regression._
import org.apache.spark.rdd.RDD

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
  def runFull(
      input: RDD[LabeledPoint],
      nbTrees: Int,
      m: Int,
      minSplitNum: Int,
      minVarianceProportion: Double)
    : RandomForestModel =
  {
    val seeds = {
      val rnd = new Random(seed)
      Array.fill(nbTrees)(rnd.nextInt())
    }

    val trees = Array.tabulate[RDD[Node]](nbTrees) { i =>
      input.sample(withReplacement = true, 1.0, seeds(i)).coalesce(1).mapPartitions { iterator =>
        val rnd = new Random(seeds(i))
        val data = Data(metaInfo, iterator.toArray)
        val builder = new DecisionTreeBuilder(m, minSplitNum, minVarianceProportion)
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
  def runPartial(
      input: RDD[LabeledPoint],
      nbTrees: Int,
      m: Int,
      minSplitNum: Int,
      minVarianceProportion: Double)
    : RandomForestModel =
  {
    val numPartitions = input.partitions.length
    val partitionToRDDs =  Array.tabulate(numPartitions) { i =>
      input.mapPartitionsWithIndex{ (index, iterator) =>
        if (index == i) iterator else List.empty[LabeledPoint].iterator
      }.coalesce(1)
    }

    val trees = partitionToRDDs.zipWithIndex.map { case (rdd, index) =>
      rdd.mapPartitions { iterator =>
        val data = Data(metaInfo, iterator.toArray)
        val numTrees = RandomForest.nbTreesOfPartition(nbTrees, numPartitions, index)
        val rnd = new Random(seed)
        val futures = Seq.tabulate(numTrees) { i =>
          val r = new Random(rnd.nextInt())
          val builder = new DecisionTreeBuilder(m, minSplitNum, minVarianceProportion)
          Future {
            builder.build(r, data.bagging(r))
          }
        }

        val trees = Await.result(Future.sequence(futures), Duration.Inf)
        trees.iterator
      }
    }.reduce(_ ++ _)

    new RandomForestModel(trees.collect(), metaInfo)
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
   * @param seed Random seed
   * @param classification Whether the label is categorical or numerical.
   * @param categorical If the j'th feature is categorical, then categorical(j)=true;
   *                    if the j'th feature is numerical, then categorical(j)=false.
   * @param nbTrees Number of trees to build, should be greater than or equal to the number
   *                of partitions.
   * @param m number of attributes to select randomly at each node, default is sqrt(dimension)
   * @param minSplitNum minimum number for split, default is 2
   * @param minVarianceProportion minimum proportion of the total variance for split, default
   *                              is 1.0e-3
   */
  def train(
      input: RDD[LabeledPoint],
      classification: Boolean,
      categorical: Array[Boolean],
      seed: Int,
      nbTrees: Int,
      partial: Boolean = false,
      m: Int = 0,
      minSplitNum: Int = 2,
      minVarianceProportion: Double = 1.0e-3)
    : RandomForestModel =
  {
    require(nbTrees > input.partitions.size)
    val metaInfo = generateMetaInfo(input, classification, categorical)
    train(input, metaInfo, seed, nbTrees, partial, m, minSplitNum, minVarianceProportion)
  }

  private[rf] def train(
      input: RDD[LabeledPoint],
      metaInfo: DataMetaInfo,
      seed: Int,
      nbTrees: Int,
      partial: Boolean,
      m: Int,
      minSplitNum: Int,
      minVarianceProportion: Double)
    : RandomForestModel =
  {
    val rm = new RandomForest(metaInfo, seed)
    if  (partial) {
      rm.runPartial(input, nbTrees, m, minSplitNum, minVarianceProportion)
    } else {
      rm.runFull(input, nbTrees, m, minSplitNum, minVarianceProportion)
    }
  }

  /**
   *  Generate meta info.
   */
  private[rf] def generateMetaInfo(
      data: RDD[LabeledPoint],
      classification: Boolean,
      categorical: Array[Boolean])
    : DataMetaInfo =
  {
    val D = categorical.length
    val maxPoint = data.fold(LabeledPoint(0.0, new Array[Double](D))) {(result, point) =>
      for (i <- point.features.indices) {
        result.features(i) = result.features(i).max(point.features(i))
      }
      LabeledPoint(result.label.max(point.label), result.features)
    }

    val nbLabels = if (classification) maxPoint.label.toInt + 1 else -1
    val nbValues = maxPoint.features.zip(categorical).map { case (feature, isCategorical) =>
      if (isCategorical) feature.toInt + 1 else -1
    }

    DataMetaInfo(classification, categorical, nbLabels, nbValues)
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
    val n = nbTrees / nbPartitions
    if (partition == 0) nbTrees - n * (nbPartitions - 1) else n
  }
}
