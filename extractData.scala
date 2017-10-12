
import org.apache.spark.sql.functions._
import java.sql.Timestamp
import java.text.SimpleDateFormat
import java.io.PrintWriter
import java.io.File
import org.apache.spark.sql.types._
import org.apache.commons.lang3.time.DateUtils




import org.apache.spark.{SparkConf, SparkContext}

import scala.reflect.io.Directory
object extractData {
  def main(args: Array[String]):Unit = {
    /* args(0): startDay,"2017/07/07"
    *
     */
    val  fmt:SimpleDateFormat = new java.text.SimpleDateFormat("yyyy/MM/dd")

    val startDay = args(0)
    val endDay = args(1)
    //init the spark
    val conf = new SparkConf()
      .setMaster("yarn-cluster")
      .setAppName("Extract data")
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    val sc = new SparkContext(conf)
    sc.setLogLevel("WARN")
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._
    /**  提取数据
      * extracte data from elt2
      * write to the new file
      * @return true: success, false: failed
      */
    def deleteFile(path: String) : Unit = {
      import org.apache.hadoop.conf.Configuration
      import org.apache.hadoop.fs.{FileSystem, Path}
      val hdfs : FileSystem = FileSystem.get(new Configuration)
      val isExist = hdfs.exists(new Path(path))
      if (isExist){
//        println("welll")
        hdfs.delete(new Path(path), true)//true: delete files recursively
    }
    }
    def dayDiff(start:String,end:String):Int = {
      val dayms = 1000 * 3600 * 24.0
      val st = fmt.parse(start).getTime
      val endDate = fmt.parse(end).getTime
      val days = (endDate - st) / dayms
      days.toInt
    }
    //org.apache.spark.sql.DataFrame
    def extractedData(inputDir : String,outputDir : String) : Unit= {
      val df = sqlContext.read.parquet(inputDir);
      val SelectDf = df.select("tdid","app.appKey","receiveTime","platform","app.installTime","device.brand","device.standardModel","os.osStandardVersion")
        .filter("appKey = '323BE90BCA2213A07D18FE935A6BA9E5' or appKey = '323BE90BCA2213A07D18FE935A6BA9E5' " +
          "or appKey = '52F459D41AAB85846D4CA031B40FFECE' or appKey = '86C480859A2547CB85FB0BC5A6EC3943'" )
        .withColumn("receiveTime",(to_date(from_unixtime((df("receiveTime"))/1000))))

      val resDF = SelectDf.rdd.map(row => (row,1)).reduceByKey(_ + _).map{row =>
          val tdid =row._1.getAs[String](0);
          val appKey =row._1.getAs[String](1);
          val receiveTime =row._1.getAs[java.sql.Date](2);
          val platform =row._1.getAs[Int](3);
          val installTime =row._1.getAs[Long](4);
          val purchaseTime= row._1.getAs[String](5);
          val brand =row._1.getAs[String](6);
          val osStandardVersion =row._1.getAs[String](7);
          val freq =row._2;
          (tdid,appKey,receiveTime,platform,installTime,purchaseTime,brand,osStandardVersion,freq)}
        .toDF("tdid","appKey","receiveTime","platform","installTime","brand","standardModel","osStandardVersion","freq")

      val resDf = resDF.withColumn("installTime",to_date(from_unixtime(resDF("installTime")/1000)))
      resDf.repartition(10).write.format("parquet").save(outputDir)

    }
    val rootpath="/datascience/etl2/standard/ta/"

    val range = dayDiff(startDay,endDay)
    (0 to range).foreach{i =>
      val baseday = fmt.parse(startDay)
      val daytmp =fmt.format(DateUtils.addDays(baseday,i))
      val input = rootpath+daytmp +"/*/"
      val output="/datalab/user/frank.zhang/data/ta/" + daytmp
      deleteFile(output)
      extractedData(input,output)
//      println(output+" write done")

    }
  }
}
