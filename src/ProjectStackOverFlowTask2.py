#Import libraries
from pyspark.ml.feature import PCA
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import isnan, when, count, col

#Create Spark session
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
spark = SparkSession.builder.appName('project').getOrCreate()
sc = spark.sparkContext
sqlContext = SQLContext(sc)
sc.setLogLevel("ERROR")

#Read data 
post_data = sqlContext.read.format("csv") \
  .option("delimiter",",") \
  .option("header", "true") \
  .option("multiline","true") \
  .option("inferSchema", "true") \
  .option("quote", '"') \
  .option("escape", "\\") \
  .option("escape", '"') \
  .load('/Data/*.csv')
  
#Select features that are 
post_data_score = post_data.select(['Id0','Tags','ViewCount','OwnerUserId','AnswerCount','CommentCount','FavoriteCount','Reputation','Views','UpVotes','DownVotes','Score'])
#Filter out records with null tags
post_data_score = post_data_score.filter(post_data_score['Tags']!= 'null')
#Find columns which have null values
post_data_score.select([count(when(col(c).isNull(), c)).alias(c) for c in post_data_score.columns]).show()
#Replace null values by 0
post_data_score = post_data_score.na.fill(0)
#In this task, score will be our labels
post_data_score = post_data_score.withColumnRenamed('Score','label')

#Convert the multi-class labels into indices uniquely assigned to each tag. 
#Labels are convert from type string to corresponding float-type.
indexer = StringIndexer(inputCol="Tags", outputCol="indexedTags")
model_score = indexer.fit(post_data_score)
model_score.setHandleInvalid("error")
indexed_df_score = model_score.transform(post_data_score)

#Assemble columns into a single column of feature vectors using VectorAssembler (Feature transformation)
assembler = VectorAssembler().setInputCols(["indexedTags","ViewCount","OwnerUserId","AnswerCount","FavoriteCount","Reputation"]).setOutputCol("features")
assembled_df_score = assembler.transform(indexed_df_score).select("label", "features")

#Feature reduction
pca = PCA(k=3, inputCol="features", outputCol="pcaFeatures")
model = pca.fit(assembled_df_score)
result = model.transform(assembled_df_score).select("pcaFeatures","label")
result = result.withColumnRenamed('pcaFeatures','features')

#Split data into train and test
train_score, test_score = result.randomSplit([0.8, 0.2], seed = 2020)

#-------------------------------------------------------------------------------- Comparing performance of different classification models -----------------------------------------------------------------

#Declare the evaluation metric. (RSME and MAE)
evaluator_rmse = RegressionEvaluator( labelCol="label", predictionCol="prediction", metricName="rmse")
evaluator_mae = RegressionEvaluator( labelCol="label", predictionCol="prediction", metricName="mae")

#Linear Regressor
from pyspark.ml.regression import LinearRegression
lr_model = LinearRegression(maxIter=3, regParam=0.1, elasticNetParam=0.8)
lrModelfit = lr_model.fit(train_score)
predictions_score_lr = lrModelfit.transform(test_score)
predictions_score_lr.show()

rmse = evaluator_rmse.evaluate(predictions_score_lr)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

mae = evaluator_mae.evaluate(predictions_score_lr)
print("Mean Absolute Error (MAE) on test data = %g" % mae)

#Random Forest Regressor

from pyspark.ml.regression import RandomForestRegressor
rf_reg = RandomForestRegressor(featuresCol="features")
rf_regfit = rf_reg.fit(train_score)
predictions_score_rf = rf_regfit.transform(test_score)
predictions_score_rf.show()

rmse_rf = evaluator_rmse.evaluate(predictions_score_rf)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse_rf)

mae_rf = evaluator_mae.evaluate(predictions_score_rf)
print("Mean Absolute Error (MAE) on test data = %g" % mae_rf)

#Gradient-boosted trees Regressor

from pyspark.ml.regression import GBTRegressor
gbt_model = GBTRegressor(featuresCol="features", maxIter=5)
gbtModelfit = gbt_model.fit(train_score)
predictions_score_gbt = gbtModelfit.transform(test_score)
predictions_score_gbt.show()

rmse_gbt = evaluator_rmse.evaluate(predictions_score_gbt)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse_gbt)

mae_gbt = evaluator_mae.evaluate(predictions_score_gbt)
print("Mean Absolute Error (MAE) on test data = %g" % mae_gbt)

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------