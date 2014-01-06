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
 * <p>
 * Interface for classes that can keep track of a running average of a series of numbers. One can add to or
 * remove from the series, as well as update a datum in the series. The class does not actually keep track of
 * the series of values, just its running average, so it doesn't even matter if you remove/change a value that
 * wasn't added.
 * </p>
 */
abstract class RunningAverage {
  
  /**
   * @param item new item to add to the running average
   * @throws IllegalArgumentException if datum is {@link Double#NaN}
   */
  def addItem(item: Double): Unit
  
  /**
   * @param item item to remove to the running average
   * @throws IllegalArgumentException if datum is {@link Double#NaN}
   * @throws IllegalStateException if count is 0
   */
  def removeItem(item: Double): Unit
  
  /**
   * @param delta amount by which to change a datum in the running average
   * @throws IllegalArgumentException if delta is {@link Double#NaN}
   * @throws IllegalStateException if count is 0
   */
  def changeItem(delta: Double): Unit
  
  def getCount(): Int
  
  def getAverage(): Double
}

/**
 * <p>
 * A simple class that can keep track of a running average of a series of numbers. One can add to or
 * remove from the series, as well as update a datum in the series. The class does not actually keep
 * track of the series of values, just its running average, so it doesn't even matter if you 
 * remove/change a value that wasn't added.
 * </p>
 */
class FullRunningAverage(private var count: Int, private var average: Double) 
    extends RunningAverage with Serializable {
  
  def this() = this(0, Double.NaN)

  def addItem(item: Double) {
    this.synchronized {
      count += 1
      if (count == 1) {
        average = item
      } else {
        average = (average * (count - 1) + item) / count
      }
    }
  }
  
  def removeItem(item: Double) {
    this.synchronized {
      if (count == 0) throw new IllegalStateException()
      count -= 1
      if (count == 0) {
        average = Double.NaN
      } else {
        average = (average * (count + 1) - item) / count
      }
    }
  }
  
  def changeItem(delta: Double) {
    this.synchronized {
      if (count == 0) throw new IllegalStateException()
      average += delta / count
    }
  }
  
  def getCount(): Int = {
    this.synchronized {
      count
    }
  }
  
  def getAverage(): Double = {
    this.synchronized {
      average
    }
  }
}
