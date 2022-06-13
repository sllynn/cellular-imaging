# Databricks notebook source
# MAGIC %md
# MAGIC # Model tuning and training
# MAGIC In this notebook we load the previously pre-processed patches from delta and train a binary classifier to classify tumor/normal patches. We use `mlFlow` to track and store the trained model. Note that you can use any other method for training the classifier. 

# COMMAND ----------

WSI_PATH='/databricks-datasets/med-images/camelyon16/'
BASE_PATH='/tmp/digital-pathology'
ANNOTATION_PATH = BASE_PATH+"/annotations"
PATCH_PATH = BASE_PATH+"/patches"

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 1. Load pre-processed features from Delta

# COMMAND ----------

dataset_df = spark.read.load(f'{BASE_PATH}/delta')
pos_ratio=dataset_df.selectExpr('round(avg(label),1) AS ratio').collect()[0]['ratio']
print(f"Number of patches: {dataset_df.count()} with ~%{100*pos_ratio} positive labels")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Model selection and hyperparameter tuning
# MAGIC ## <div style="text-align: left; line-height: 0; padding-top: 0px;"> <img src="https://secure.meetupstatic.com/photos/event/c/1/7/6/600_472189526.jpeg" width=100> + <img src="https://i.postimg.cc/TPmffWrp/hyperopt-new.png" width=40></div>
# MAGIC 
# MAGIC Before training the model, we would like to find the best set of parameters (judged based on the highest score). To this end we use HyperOpt to search over a grid of parameters.

# COMMAND ----------

import mlflow
import mlflow.spark

from pyspark.sql.functions import *
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from math import exp
import numpy as np
from hyperopt import fmin, hp, tpe, SparkTrials, STATUS_OK

# COMMAND ----------

(train_df,test_df)=dataset_df.randomSplit([0.7, 0.3], seed=42)

def params_to_lr(params):
  return {
    'max_iter':          params['max_iter'], 
    'reg_param':         exp(params['reg_param']), # exp() here because hyperparams are in log space
    'elastic_net_param': exp(params['elastic_net_param'])
  }

def tune_model(params):
  with mlflow.start_run(run_name='tunning-logistic-regression') as run:
    mlflow.spark.autolog()
    params=params_to_lr(params)    
    max_iter=params['max_iter']
    reg_param=params['reg_param']
    elastic_net_param=params['elastic_net_param']
    lr = LogisticRegression(featuresCol='features_vec',maxIter=max_iter, regParam=reg_param, elasticNetParam=elastic_net_param)
    model = lr.fit(train_df)
    _predictions = model.transform(test_df).select("prediction", "label")
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction")
    area_under_roc=evaluator.evaluate(_predictions)
  
  return {'status': STATUS_OK, 'loss': - area_under_roc}


# COMMAND ----------

# spark_trials = SparkTrials(parallelism=1)

search_space = {
  # use uniform over loguniform here simply to make metrics show up better in mlflow comparison, in logspace
  'max_iter': 10,
  'reg_param':      hp.uniform('reg_param', -2, 0),
  'elastic_net_param':   hp.uniform('elastic_net_param', -3, -1),
}

best_params = fmin(fn=tune_model, space=search_space, algo=tpe.suggest, max_evals=8, rstate=np.random.RandomState(43))
print(best_params)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Train a binary classifeir and log the model
# MAGIC Now we train a binary classifier using the best parameters during hyper parameter tuning step and log the model with mlflow.

# COMMAND ----------

model_name='wsi-spark-lr'
def train(params):
  mlflow.spark.autolog()
  params=params_to_lr(params)    
  max_iter=params['max_iter']
  reg_param=params['reg_param']
  elastic_net_param=params['elastic_net_param']
  lr = LogisticRegression(featuresCol='features_vec',maxIter=max_iter, regParam=reg_param, elasticNetParam=elastic_net_param)
  model = lr.fit(dataset_df)
  mlflow.spark.log_model(model,model_name)
  run_info = mlflow.active_run().info
  return(run_info)

# COMMAND ----------

# DBTITLE 1,train and  log the classifier
best_params['max_iter']=10;
run_info=train(best_params)

# COMMAND ----------

run_info.artifact_uri

# COMMAND ----------

print(f"use {run_info.artifact_uri}/{model_name} in the next step to load the model in the next step.")

# COMMAND ----------

