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

private [rf] object DataUtils {

  /**
   * foreach i : array1[i] += array2[i]
   */
  def add(array1: Array[Int], array2: Array[Int]) {
    require (array1.length == array2.length, "array1.length != array2.length")
    for (index <- 0 until array1.length) {
      array1(index) += array2(index)
    }
  }

  /**
   * foreach i : array1[i] -= array2[i]
   */
  def dec(array1: Array[Int], array2: Array[Int]) {
    require (array1.length == array2.length, "array1.length != array2.length")
    for (index <- 0 until array1.length) {
      array1(index) -= array2(index)
    }
  }

  /**
   * return the index of the maximum of the array, breaking ties randomly
   *
   * @param rnd used to break ties
   * @return index of the maximum
   */
  def maxIndex(rnd: Random, values: Array[Int]): Int = {
    val maxValue = values.max
    val maxIndices = values.zipWithIndex.filter(_._1 == maxValue).map(_._2)
    if (maxIndices.length > 1) maxIndices(rnd.nextInt(maxIndices.length)) else maxIndices(0)
  }
}