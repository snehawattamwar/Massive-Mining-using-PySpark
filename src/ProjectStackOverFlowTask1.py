#Import libraries
import re
import string
import pyLDAvis
import numpy as np
import matplotlib.pyplot as plt
from pyspark.ml.clustering import LDA
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import CountVectorizer,IDF
from pyspark.sql.functions import explode, size
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

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
  
     
#We tried implementing the cleaning of data using filter. However, for the data format we posses, 
#we decided to call the clean_text over a lambda function in parallel for each row     
def clean_text(raw_html):
    #Remove HTML tags using re
    cleanr = re.compile('<.*?>')
    text = re.sub(cleanr, '', raw_html)
    cleantext = re.sub(r'<!DOCTYPE[^>[]*(\[[^]]*\])?>', '', text)
    #Remove digits
    cleannum = ''.join([i for i in cleantext if not i.isdigit()])
    #Convert text to lower
    input_str = cleannum.lower()
    #Clean punctuations
    exclude = set(string.punctuation)
    exclude.remove('#')
    input = ''.join(ch for ch in input_str if ch not in exclude)
    #Remove stopwords
    input_text = input.split()
    result = [i for i in input_text if not i in ENGLISH_STOP_WORDS]
    result_arr = [i for i in result if not len(i)<2]
    #Return cleaned text as array of words
    return result_arr 


#Select the required columns which contribute to the prediction of tags, i.e. mainly the title and body content of the post
#Filter data frame for only questions as the data contains both answer posts (0) and question posts (1)
post_tags = post_data.select(['Id0','PostTypeId','Title','Body','Tags']).filter(post_data['PostTypeId']==1)
#Drop PostTypeId
post_tags = post_tags.drop(post_tags['PostTypeId'])
#Filter out records with null tags
post_tags = post_tags.filter(post_tags['Tags']!= 'null')
post_tags.show()

#Explore data to estimate the distribution of tags over the data
tag_count = post_tags.groupby('Tags').count().withColumnRenamed('Tags','Tags_count')
tag_count.show()

#Plot a bar graph for tags vs count
plt.rcParams['figure.figsize'] = [20, 10]
tag_count.toPandas().plot(x = "Tags_count", y = "count", kind = "bar")
plt.show()

#Assign dataframe records with the count of respective tags tags 
posts_join_count = post_tags.join(tag_count,post_tags['Tags']==tag_count['Tags_count'])
#Set a threshold for the minimum number of count required by a tag in order to maintain uniformity and remove outliers
posts_join_count = posts_join_count.filter(posts_join_count['count']>10000)

#Drop extra columns
posts_join_count = posts_join_count.drop('Tags_count')
posts_join_count = posts_join_count.drop('count')

#Convert to rdd
post_tags_rdd = posts_join_count.rdd.map(tuple)
#The rdd has tuples of Id,Title,Body,Tags
post_tags_rdd_clean = post_tags_rdd.map(lambda x : (x[0],clean_text(str(x[1])),clean_text(str(x[2])),x[3]))
#Separate tags for obtaining the primary tag of the question. here, primary tag refers to the technology or programming language
post_tags_rdd_cleanTags = post_tags_rdd_clean.map(lambda x : (x[0],x[1],x[2],str(x[3]).replace('-', '><').strip("<").strip(">").split("><")[0]))

#Concatenate the Title and body content
post_tags_rdd_final = post_tags_rdd_cleanTags.map(lambda x : (x[0],x[1]+x[2],x[3]))
#Convert into datframe
final_dataframe = sqlContext.createDataFrame(post_tags_rdd_final,['Id','Text','Tags']) 
final_dataframe.show()

#Convert the multi-class labels into indices uniquely assigned to each tag. 
#Labels are convert from type string to corresponding float-type.
indexer = StringIndexer(inputCol="Tags", outputCol="label")
modelIndexer = indexer.fit(final_dataframe)
modelIndexer.setHandleInvalid("error")
final_indexed_dataframe = modelIndexer.transform(final_dataframe)
final_indexed_dataframe.show()

#Vectorize the text column using count vectorizer to extract relevant features
cv = CountVectorizer(inputCol="Text", outputCol="features", minDF=5.0)
count_vectorizer = cv.fit(final_indexed_dataframe)
df_cv = count_vectorizer.transform(final_indexed_dataframe)

#Calculate the TF-IDF of the terms for each row
idf = IDF(inputCol="features", outputCol="features_tfidf")
idf_model = idf.fit(df_cv)
df_tfidf = idf_model.transform(df_cv)
#Drop extra columns
df_tfidf_final = df_tfidf.drop('Id','Text','features')
df_tfidf_final = df_tfidf_final.withColumnRenamed('features_tfidf','features')

#Split data into train and test 
train, test = df_tfidf_final.randomSplit([0.8, 0.2], seed = 2020)

#-------------------------------------------------------------------------------- Comparing performance of different classification models -----------------------------------------------------------------

#Declare the evaluation metric. The default is F1 score.
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")

#Naive Bayes Classifier
from pyspark.ml.classification import NaiveBayes
nb = NaiveBayes(smoothing=1)
#Fit and transform
model = nb.fit(train)
predictions_nb = model.transform(test)
#Evaluate
evaluator.evaluate(predictions_nb)


#Random Forest Classifier
from pyspark.ml.classification import RandomForestClassifier
rf = RandomForestClassifier(numTrees = 10, maxDepth = 3, maxBins = 32)
#Fit and transform
rfModel = rf.fit(train)
predictions_rf = rfModel.transform(test)
#Evaluate
evaluator.evaluate(predictions_rf)

#LogisticRegression Classifier
lr = LogisticRegression(maxIter=10, tol=1E-6, fitIntercept=True)
# instantiate the One Vs Rest Classifier.
ovr = OneVsRest(classifier=lr)
# train the multiclass model.
ovrModel = ovr.fit(train)
# score the model on test data.
predictions_ovr = ovrModel.transform(test)
evaluator.evaluate(predictions_ovr)

#------------------------------------------------------------------------- Suggest/predict top 3 tags -------------------------------------------------------------------------------

#Vocabulary of the text data where the words are mapped to indices using the count vectorizer
vocabArray = count_vectorizer.vocabulary

#Get the top 3 indices with maximum TF-IDF values
def getIndices(arr):
  terms_indices = np.argsort(arr)[::-1][:3]
  return terms_indices

#Drop extra columns
suggest_tags = df_tfidf.drop('Text','features')
suggest_tags_rdd = suggest_tags.rdd.map(tuple)
#Conert sparse vector of tfIdf into array
suggest_tags_vector = suggest_tags_rdd.map(lambda x: (x[0],x[3].toArray()))
#Call getindices() in parallel
suggest_tags_indices = suggest_tags_vector.map(lambda x: (x[0],getIndices(x[1])))
#Get the words mapped to the respective indices
suggest_tags_words = suggest_tags_indices.map(lambda x: (x[0],[vocabArray[i] for i in x[1]]))
#Convert to dataframe such that the columns a re the Postid and the tag suggestions
suggest_tags_words_df = spark.createDataFrame(suggest_tags_words,['PostId','Suggestions'])
#Join the suggestions to the respective questions
suggestions_join = post_tags.join(suggest_tags_words_df, suggest_tags_words_df.PostId==post_tags.Id0)
suggestions_join.show(truncate=200)

#---------------------------------------------------------------------- Visualization -----------------------------------------------------------------------------------------

#Declare the LDA model
lda = LDA(k=5, maxIter=20, featuresCol='features')
lda_model = lda.fit(df_tfidf_final)
transformed = lda_model.transform(df_tfidf_final)
transformed.show()

#Get the tokenized text data from above
tokenized_df = final_dataframe.select('Text').selectExpr('Text as documents')

def pyldavis_data_format(tokenized_df, count_vectorizer, transformed, lda_model):
    word_counts = tokenized_df.select((explode(tokenized_df.documents)).alias("words")).groupby("words").count()
    word_counts_list = {r['words']:r['count'] for r in word_counts.collect()}
    word_counts_list = [word_counts_list[w] for w in count_vectorizer.vocabulary]

    #Create data with key-value pairs as expected by the pyLDAvis tool 
    data = {'topic_term_dists': np.array(lda_model.topicsMatrix().toArray()).T, 
            'doc_topic_dists': np.array([x.toArray() for x in transformed.select(["topicDistribution"]).toPandas()['topicDistribution']]),
            'doc_lengths': [r[0] for r in tokenized_df.select(size(tokenized_df.documents)).collect()],
            'vocab': count_vectorizer.vocabulary,
            'term_frequency': word_counts_list}
    return data

#Call format_data_to_pyldavis() with necessary parameters
data = pyldavis_data_format(tokenized_df, count_vectorizer, transformed, lda_model)

pylda_data = pyLDAvis.prepare(**data)
pyLDAvis.display(pylda_data)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------