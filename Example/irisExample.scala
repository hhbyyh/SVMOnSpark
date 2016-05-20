/**
 * Created by Tony on 17/5/2016.
 */
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{LogisticRegression, SVM}
import org.apache.spark.ml.feature.{MinMaxScaler, StandardScaler}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.SparkSession

object irisExample {
  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)
    val spark = SparkSession.builder.appName("svm").master("local").getOrCreate()

    val allData =  spark.sparkContext.textFile("./data/iris.data")
      .filter(!_.trim.isEmpty)
      .map(line => line.split(","))
      .map( x=> (x(0).toDouble, x(1).toDouble, x(2).toDouble, x(3).toDouble, x(4).trim))
      .map(arr =>  new LabeledPoint(if (arr._5 == "Iris-setosa") 1 else 0, Vectors.dense(arr._1, arr._2, arr._3, arr._4)))

    val trainDF = spark.createDataFrame(allData).cache()
    val testDF = spark.createDataFrame(allData).cache()

    val scaler = new MinMaxScaler().setInputCol("features").setOutputCol("out_features")
    val svm = new SVM().setKernelType("linear").setMaxIter(100).setFeaturesCol("out_features")
    val model = new Pipeline().setStages(Array(scaler, svm)).fit(trainDF)

    val result = model.transform(testDF).cache()
    println("total: " + testDF.count())
    println(result.filter("label = prediction").count())
  }

}
