import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.SVM
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.SparkSession

object Example {


  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)
    val spark = SparkSession.builder.appName("svm").master("local[8]").getOrCreate()

    val trainRDD = spark.sparkContext.textFile("data/mnist/mnist_train.csv", 8)
      .map(line => line.split(",")).map(arr => arr.map(_.toDouble))
      .map(arr => new LabeledPoint(if (arr(0) == 1) 1 else -1, Vectors.dense(arr.slice(1, 785))))
    val trainDF = spark.createDataFrame(trainRDD).cache()

    val model = new SVM()
      .setKernelType("linear").fit(trainDF)

    val testRDD = spark.sparkContext.textFile("data/mnist/mnist_test.csv", 8)
      .map(line => line.split(",")).map(arr => arr.map(_.toDouble))
      .map(arr => new LabeledPoint(if (arr(0) == 1) 1 else -1, Vectors.dense(arr.slice(1, 785))))
    val testDF = spark.createDataFrame(testRDD).cache()

    val result = model.transform(testDF).cache()
    result.show()
    println("total: " + testDF.count())
    println(result.filter("label = prediction").count())
  }

}
