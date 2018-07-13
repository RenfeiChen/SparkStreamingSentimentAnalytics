from pyspark.sql import SparkSession
import os


# Just for Windows User
os.environ['JAVA_HOME'] = 'F:\Java\jdk1.8.0_131'

# Create SparkSession
spark = SparkSession.builder.appName('nlp').getOrCreate()

# import the raw data from the dataset
data = spark.read.csv('training.1600000.processed.noemoticon.csv', inferSchema = True)
data = data.select(['_c0', '_c5'])
data = data.withColumnRenamed('_c0', 'class').withColumnRenamed('_c5', 'text')

from pyspark.ml.feature import (Tokenizer, StopWordsRemover, CountVectorizer, IDF, StringIndexer)

# handle the data through tokenizer, removing the unimportant word, word count and tf_idf
tokenizer = Tokenizer(inputCol='text', outputCol='token_text')
stop_remove = StopWordsRemover(inputCol='token_text', outputCol='stop_token')
count_vec = CountVectorizer(inputCol='stop_token',outputCol='c_vec')
idf = IDF(inputCol='c_vec', outputCol='tf_idf')
from_four_to_one = StringIndexer(inputCol='class', outputCol='label')


# turn the features into a vector
from pyspark.ml.feature import VectorAssembler
clean_up = VectorAssembler(inputCols=['tf_idf'], outputCol='features')

# create a pipeline to clean the data and the store other model in order to handle the test data
# which comes from tweets to get the same dimension as the training model
from pyspark.ml import Pipeline
data_prep_pipe = Pipeline(stages = [from_four_to_one, clean_up])
tokenized = tokenizer.transform(data)
removed = stop_remove.transform(tokenized)
cv_model = count_vec.fit(removed)
cv_result = cv_model.transform(removed)
idf_model = idf.fit(cv_result)
idf_result = idf_model.transform(cv_result)

cleaner = data_prep_pipe.fit(idf_result)
clean_data = cleaner.transform(idf_result)

# get the final data with the label and features
final_data = clean_data.select(['label', 'features'])
training, test = final_data.randomSplit([0.7, 0.3])

# the evaluation tool
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
acc_eval = MulticlassClassificationEvaluator()


# the NaiveBayes (the best model so we choose this one as the final model)

# from pyspark.ml.classification import NaiveBayes
# nb = NaiveBayes()
# nb_model = nb.fit(training)
# test_results = nb_model.transform(test)
# nb_acc = acc_eval.evaluate(test_results)
# print(nb_acc)

# the Random Forest

# from pyspark.ml.classification import RandomForestClassifier
# rf = RandomForestClassifier(labelCol='label', featuresCol='features')
# rf_model = rf.fit(training)
# test_results = rf_model.transform(test)
# rf_acc = acc_eval.evaluate(test_results)
# print(rf_acc)

# the Decision Tree

# from pyspark.ml.classification import DecisionTreeClassifier
# dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")
# dt_model = dt.fit(training)
# test_results = dt_model.transform(test)
# dt_acc = acc_eval.evaluate(test_results)
# print(dt_acc)

# the Logistic Regression

# from pyspark.ml.classification import LogisticRegression
# lr = LogisticRegression(labelCol="label", featuresCol="features")
# lr_model = lr.fit(training)
# test_results = lr_model.transform(test)
# lr_acc = acc_eval.evaluate(test_results)
# print(lr_acc)

# the final NaiveBayes Model

from pyspark.ml.classification import NaiveBayes
nb = NaiveBayes()
nb_model = nb.fit(final_data)

# handle the input tweets data while generating the csv files

index = 0
while (True):
    # get the input tweets data file name
    file = 'Data' + str(index) + '.csv'
    index = index + 1

    # clean the input file in order to get the same format as the training data
    test_input_data = spark.read.csv(file).withColumnRenamed('_c0', 'text')
    test_prep_pipe = Pipeline(stages=[clean_up])
    tokenized = tokenizer.transform(test_input_data)
    removed = stop_remove.transform(tokenized)
    cv_result_test = cv_model.transform(removed)
    idf_result_test = idf_model.transform(cv_result_test)
    test_cleaner = test_prep_pipe.fit(idf_result_test)
    test_clean_data = test_cleaner.transform(idf_result_test).select("features")

    # get the total number of the input tweets
    total = test_clean_data.count()

    # get the prediction from the clean tweets data
    test_result = nb_model.transform(test_clean_data).select('prediction')

    # get the number of the positive prediction
    positive_num = test_result.filter(test_result['prediction'] == 1).count()

    # draw the pie plot
    from matplotlib import pyplot as plt

    plt.figure(figsize=(6, 9))
    labels = ['Positive', 'Negative']
    sizes = [positive_num, total - positive_num]
    colors = ['red', 'yellowgreen']
    explode = (0.05, 0)
    patches, l_text, p_text = plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                                      labeldistance=1.1, autopct='%3.1f%%', shadow=False,
                                      startangle=90, pctdistance=0.6)
    plt.axis('equal')
    plt.legend()
    plt.show()
