import os
from sklearn.metrics import accuracy_score
from sklearn import metrics
from autogluon.tabular import TabularPredictor
import json
import numpy as np
import pandas as pd 
from sklearn.model_selection import cross_val_score
import jsonpickle
import time

# folder structure:
# task_dir {
#     test.csv
#     train.csv
#     Autugluon {
#        pipelines_ranked {}
#        pipelines_scored {}
#        predictions {}
#        training_predictions {}
#     }
# }

def run_autogluon(task_dir, timeout_in_sec, n_jobs):
    
    print("Training AutoGluon...")
    
    #initialize AutoML
    os.chdir(task_dir)

    #training data
    train = pd.read_csv("train.csv")
    #X_train = train[:, :-1] 
    #y_train = train[:, -1]
    
    #testing data
    test = pd.read_csv("test.csv")
    #X_test = test[:, :-1] 
    #y_test = test[:, -1]
    
    os.mkdir("ag")
    os.chdir("ag")

    #train AutoML
    start_time = time.time() #start
    label = train.columns[-1]
    metric = 'roc_auc'
    autogluon =TabularPredictor(label, eval_metric=metric).fit(train, time_limit=timeout_in_sec, ag_args_fit={'num_cpus': n_jobs}) 
    stop_time = time.time() #stop
    
    #--------------EXPORT RESULTS----------------#
    lb = autogluon.leaderboard()
    top_model = lb.model[0]
    test_lb = autogluon.leaderboard(test, extra_metrics=['accuracy'])
    top_score = test_lb.loc[test_lb['model']==top_model]
    top_score = top_score.iloc[:, 0:4]
    columns_titles = ["model", "score_val","score_test", "accuracy"]
    top_score = top_score.reindex(columns=columns_titles)
    top_score.insert(0, "time", [timeout_in_sec])
    top_score.rename(columns={'accuracy':'test_accuracy'}, inplace=True)
    top_score.to_csv("score.csv")
    #display success message 
    with open(os.getcwd() + '/traintime.txt', 'w') as f:
        f.write("Training time limit: %s sec\n" % (timeout_in_sec))
        f.write("Actual time taken to train: %s sec\n" % (stop_time - start_time))
    print("...Autogluon export success! Execution time: %s seconds " % (stop_time - start_time))
