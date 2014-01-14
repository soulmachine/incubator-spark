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

package org.apache.spark.mllib.examples;

import java.util.Arrays;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.rf.RandomForest;
import org.apache.spark.mllib.classification.rf.RandomForestModel;
import org.apache.spark.mllib.regression.LabeledPoint;

/**
 * Random forest based classification using ML Lib.
 */
public class JavaRF {
  private static final LabeledPoint[] TRAIN_DATA = {
      new LabeledPoint(1, new double[]{1, 85.0, 85, 0}),
      new LabeledPoint(1, new double[]{1, 80.0, 90, 1}),
      new LabeledPoint(0, new double[]{2, 83.0, 86, 0}),
      new LabeledPoint(0, new double[]{0, 70.0, 96, 0}),
      new LabeledPoint(0, new double[]{0, 68.0, 80, 0}),
      new LabeledPoint(1, new double[]{0, 65.0, 70, 1}),
      new LabeledPoint(0, new double[]{2, 64.0, 65, 1}),
      new LabeledPoint(1, new double[]{1, 72.0, 95, 0}),
      new LabeledPoint(0, new double[]{1, 69.0, 70, 0}),
      new LabeledPoint(0, new double[]{0, 75.0, 80, 0}),
      new LabeledPoint(0, new double[]{1, 75.0, 70, 1}),
      new LabeledPoint(0, new double[]{2, 72.0, 90, 1}),
      new LabeledPoint(0, new double[]{2, 81.0, 75, 0}),
      new LabeledPoint(1, new double[]{0, 71.0, 91, 1})
  };

  private static final double[][] TEST_DATA = {
      {0, 70.0, 96, 1},
      {2, 64.0, 65, 1},
      {1, 75.0, 90, 1}
  };

  public static void main(String[] args) {
    if (args.length == 0) {
      System.err.println("Usage: JavaRF <master> [<slices>]");
      System.exit(1);
    }
    JavaSparkContext sc = new JavaSparkContext(args[0], "JavaRF",
        System.getenv("SPARK_HOME"), JavaSparkContext.jarOfClass(JavaRF.class));
    int numSlices = 2;
    if (args.length > 1) numSlices = Integer.parseInt(args[1]);
    JavaRDD<LabeledPoint> points = sc.parallelize(Arrays.asList(TRAIN_DATA), numSlices).cache();

    System.out.println("Training data(the first column is the label):");
    for (LabeledPoint point : TRAIN_DATA) {
      System.out.print(point.label() + " ");
      for (int i = 0; i < point.features().length; ++i) {
        System.out.print(point.features()[i] + " ");
      }
      System.out.println();
    }

    System.out.println("Begin training.");
    RandomForestModel forest = RandomForest.train( points.rdd(),
        new boolean[]{true, false, false, true}, 17, 20, false,
        TRAIN_DATA[0].features().length, 0);
    System.out.println("Training completed.");

    System.out.println("Use the random forest for prediction.");
    for (double[] point : TEST_DATA) {
      for (double x : point) {
        System.out.print(x + " ");
      }
      System.out.println(" => " + forest.predict(point));
    }
    System.exit(0);
  }
}
