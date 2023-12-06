import os
from tpot import TPOTClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
import json
import numpy as np
import pandas as pd 
from deap import creator
from sklearn.model_selection import cross_val_score
from tpot.export_utils import generate_pipeline_code, expr_to_tree, set_param_recursive
import jsonpickle
import time

# folder structure:
# task_dir {
#     test.csv
#     train.csv
#     TPOT {
#        pipelines_ranked {}
#        pipelines_scored {}
#        predictions {}
#        training_predictions {}
#     }
# }

def run_tpot(task_dir, timeout_in_sec, n_jobs):
    
    print("Training TPOT...")
    
    #initialize AutoML
    os.chdir(task_dir)

    #training data
    tpot_train = np.genfromtxt("train.csv", delimiter=',', skip_header=1)
    tpot_X_train = tpot_train[:, :-1] 
    tpot_y_train = tpot_train[:, -1]
    
    #testing data
    tpot_test = np.genfromtxt("test.csv", delimiter=',', skip_header=1)
    tpot_X_test = tpot_test[:, :-1] 
    tpot_y_test = tpot_test[:, -1]
    
    #train AutoML
    start_time = time.time() #start
    timeout = timeout_in_sec/60
    tpot = TPOTClassifier(verbosity=0, n_jobs=n_jobs, scoring='roc_auc', random_state=1, max_time_mins=timeout, max_eval_time_mins=0.04, population_size=15)
    tpot.fit(tpot_X_train, tpot_y_train)
    stop_time = time.time() #stop
    
    #--------------EXPORT RESULTS----------------#
    os.chdir(task_dir)
    os.mkdir("TPOT")
    os.chdir("TPOT")
    os.mkdir("pipelines_ranked")
    os.mkdir("predictions")
    os.mkdir("training_predictions")
    os.mkdir("pipelines_scored")

    #export leaderboard
    pipelines = tpot.evaluated_individuals_
    pipelines_ranked = dict(sorted(pipelines.items(), key=lambda x: x[1]['internal_cv_score'], reverse = True)[:10])
    df = pd.DataFrame(pipelines_ranked.values(), index=pipelines_ranked.keys())
    df.to_csv('pipelines_ranked/leaderboard.csv')

    rank = 1;
    for pipeline_string in pipelines_ranked.keys():
    
        deap_pipeline = creator.Individual.from_string(pipeline_string, tpot._pset)
        sklearn_pipeline = tpot._toolbox.compile(expr=deap_pipeline)
        sklearn_pipeline_str = generate_pipeline_code(expr_to_tree(deap_pipeline, tpot._pset), tpot.operators)
    
        #save model:
        js = export_pipeline(sklearn_pipeline)
        js = jsonpickle.encode(js)
        json_txt = json.dumps(js, default = vars)
        path = os.getcwd() + '/pipelines_ranked/[' + str(rank) + ']' + pipeline_string[:50] + '(...).json'
        with open(path, 'w') as file:
            file.write(json_txt)
    
        #make predictions on test data:
        sklearn_pipeline.fit(tpot_X_train, tpot_y_train)
        results = sklearn_pipeline.predict_proba(tpot_X_test)
        tpot_test_preds = pd.DataFrame()
        tpot_test_preds['predict'] = sklearn_pipeline.predict(tpot_X_test)
        tpot_test_preds['p0']=results[:,0]
        tpot_test_preds['p1']=results[:,1]
        path = os.getcwd() + '/predictions/[' + str(rank) + ']' + pipeline_string[:50] + '(...).json.predictions.csv'; open(path, 'w')
        tpot_test_preds.to_csv(path, index=False)
    
        #make predictions on train data:
        results = sklearn_pipeline.predict_proba(tpot_X_train)
        tpot_train_preds = pd.DataFrame()
        tpot_train_preds['predict'] = sklearn_pipeline.predict(tpot_X_train)
        tpot_train_preds['p0']=results[:,0]
        tpot_train_preds['p1']=results[:,1]
        path = os.getcwd() + '/training_predictions/[' + str(rank) + ']' + pipeline_string[:50] + '(...)_train_predictions.csv'; open(path, 'w')
        tpot_train_preds.to_csv(path, index=False)
    
        #print scores on test & train data
        path = os.getcwd() + '/pipelines_scored/[' + str(rank) + ']' + pipeline_string[:50] + '(...).scores.csv';
        file = open(path,'w')
        #test:
        test_preds_pd = pd.DataFrame(tpot_test_preds)
        test_pd = pd.DataFrame(tpot_y_test)
        test_acc_score = accuracy_score(test_pd, test_preds_pd['predict'])
        test_auc_score =  metrics.roc_auc_score(test_pd, test_preds_pd['p1'])
        #train:
        train_preds_pd = pd.DataFrame(tpot_train_preds)
        train_pd = pd.DataFrame(tpot_y_train)
        train_acc_score = accuracy_score(train_pd, train_preds_pd['predict'])
        train_auc_score = metrics.roc_auc_score(train_pd, train_preds_pd['p1'])
        #write:
        file.write("Accuracy score on test data: " + str(test_acc_score) + "\n")
        file.write("AUC score on test data: " + str(test_auc_score) + "\n")
        file.write("Accuracy score on train data: " + str(train_acc_score) + "\n")
        file.write("AUC score on train data: " + str(train_auc_score) + "\n")
        file.close()
    
        rank+=1
    
    #display success message 
    with open(os.getcwd() + '/traintime.txt', 'w') as f:
        f.write("Training time limit: %s sec\n" % (timeout_in_sec))
        f.write("Actual time taken to train: %s sec\n" % (stop_time - start_time))
    print("...TPOT export success! Execution time: %s seconds " % (stop_time - start_time))
        
def fullname(o):
  return o.__module__ + "." + o.__class__.__name__

def export_pipeline(scikit_pipeline):
  steps_obj = {'steps':[]}
  for name, md in scikit_pipeline.steps:
      steps_obj['steps'].append({
          'name': name,
          'class_name': fullname(md),
          'params': md.get_params()
      })
  return steps_obj

        
