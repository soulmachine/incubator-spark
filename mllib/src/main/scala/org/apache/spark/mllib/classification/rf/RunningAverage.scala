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

/**
 * Interface for classes that can keep track of a running average of a series of numbers. One can
 * add to or remove from the series, as well as update a datum in the series. The class does not
 * actually keep track of the series of values, just its running average, so it doesn't even matter
 * if you remove/change a value that wasn't added.
 */
private [rf] abstract class RunningAverage {

  /**
   * @param item new item to add to the running average
   * @throws IllegalArgumentException if datum is [[scala.Double.NaN]]
   */
  def addItem(item: Double): Unit

  /**
   * @param item item to remove to the running average
   * @throws IllegalArgumentException if datum is [[scala.Double.NaN]]
   * @throws IllegalStateException if count is 0
   */
  def removeItem(item: Double): Unit

  /**
   * @param delta amount by which to change a datum in the running average
   * @throws IllegalArgumentException if delta is [[scala.Double.NaN]]
   * @throws IllegalStateException if count is 0
   */
  def changeItem(delta: Double): Unit

  def count: Int

  def average: Double
}

/**
 * A simple class that can keep track of a running average of a series of numbers. One can add to or
 * remove from the series, as well as update a datum in the series. The class does not actually keep
 * track of the series of values, just its running average, so it doesn't even matter if you
 * remove/change a value that wasn't added.
 */
private [rf] class FullRunningAverage(private var count_ : Int, private var average_ : Double)
    extends RunningAverage with Serializable {

  def this() = this(0, Double.NaN)

  def addItem(item: Double) {
    this.synchronized {
      count_ += 1
      if (count_ == 1) {
        average_ = item
      } else {
        average_ = (average_ * (count_ - 1) + item) / count_
      }
    }
  }

  def removeItem(item: Double) {
    this.synchronized {
      if (count_ == 0) throw new IllegalStateException()
      count_ -= 1
      if (count_ == 0) {
        average_ = Double.NaN
      } else {
        average_ = (average_ * (count_ + 1) - item) / count_
      }
    }
  }

  def changeItem(delta: Double) {
    this.synchronized {
      if (count_ == 0) throw new IllegalStateException()
      average_ += delta / count_
    }
  }

  def count = {
    this.synchronized {
      count_
    }
  }

  def average = {
    this.synchronized {
      average_
    }
  }
}
