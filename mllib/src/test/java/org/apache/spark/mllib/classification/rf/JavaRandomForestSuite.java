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

package org.apache.spark.mllib.classification.rf;

import scala.util.Random;

import com.google.common.collect.Lists;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.regression.LabeledPoint;

public class JavaRandomForestSuite implements Serializable {
  private transient JavaSparkContext sc;
  private transient final int seed = 17;
  private transient final Random rnd = new Random(seed);

  private transient final DataMetaInfo metaInfo = new DataMetaInfo(true,
      new boolean[]{true, false, false, true}, 2, new int[]{3, -1, -1, 2});

  private transient final LabeledPoint[] TRAIN_DATA = {
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

  private transient final double[][] TEST_DATA = {
      {0, 70.0, 96, 1},
      {2, 64.0, 65, 1},
      {1, 75.0, 90, 1}
  };

  @Before
  public void setUp() {
    sc = new JavaSparkContext("local", "JavaRandomForestSuite");
  }

  @After
  public void tearDown() {
    sc.stop();
    sc = null;
    System.clearProperty("spark.driver.port");
  }

  private Data[] generateTrainingDataA() {
    final Data data = new Data(metaInfo, TRAIN_DATA);
    @SuppressWarnings("unchecked")
    final List<LabeledPoint>[] points = new List[3];

    for (int i = 0; i < points.length; i++) {
      points[i] = Lists.newArrayList();
    }

    for (LabeledPoint point : data.points()) {
      if (point.features()[0] == 0.0) {
        points[0].add(point);
      } else {
        points[1].add(point);
      }
    }

    Data[] dataset = new Data[points.length];
    for (int i = 0; i < dataset.length; i++) {
      LabeledPoint[] tmp = new LabeledPoint[points[i].size()];
      points[i].toArray(tmp);
      dataset[i] = new Data(metaInfo, tmp);
    }

    return dataset;
  }

  @Test
  public void runRandomForestWithFullData() {
    final Data[] dataset = generateTrainingDataA();
    final List<LabeledPoint> points = new ArrayList<>();

    for (int i = 0; i < 2; ++i) { // since training data is insufficient, duplicate itself
      for (int j = 0; j < dataset[0].points().length; ++j) {
        points.add(dataset[0].points()[j]);
      }

      for (int j = 0; j < dataset[1].points().length; ++j) {
        points.add(dataset[1].points()[j]);
      }
    }

    final JavaRDD<LabeledPoint> dataRDD = sc.parallelize(points, 2);
    final int iteration = 100;
    int error = 0;

    for (int i = 0; i < iteration; ++i) {
      final RandomForestModel forest = RandomForest.train(dataRDD.rdd(),
          metaInfo.classification(), metaInfo.categorical(),  rnd.nextInt(),  20, false,
          metaInfo.categorical().length, 0, 1.0e-3);

      if (1.0 != forest.predict(TEST_DATA[0])) error += 1;
      if (1.0 != forest.predict(TEST_DATA[2])) error += 1;
    }

    assert (error < 2 * iteration * 0.1);    // error rate must be less than 10%
  }

  @Test
  public void runRandomForestWithPartialData() {
    final Data[] dataset = generateTrainingDataA();
    final List<LabeledPoint> points = new ArrayList<>();

    for (int i = 0; i < 3; ++i) { // since training data is insufficient, duplicate itself
      for (int j = 0; j < dataset[0].points().length; ++j) {
        points.add(dataset[0].points()[j]);
      }

      for (int j = 0; j < dataset[1].points().length; ++j) {
        points.add(dataset[1].points()[j]);
      }
    }

    final JavaRDD<LabeledPoint> dataRDD  = sc.parallelize(points, 2);
    final int iteration = 100;
    int error = 0;

    for (int i = 0; i < iteration; ++i) {
      final RandomForestModel forest = RandomForest.train(dataRDD.rdd(),
          metaInfo.classification(), metaInfo.categorical(),  rnd.nextInt(),  20, true,
          metaInfo.categorical().length, 0, 1.0e-3);

      if (1.0 != forest.predict(TEST_DATA[0])) error += 1;
      if (1.0 != forest.predict(TEST_DATA[2])) error += 1;
    }

    assert (error < 2 * iteration * 0.2);    // error rate must be less than 20%
  }
}
