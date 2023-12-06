import os, logging, json
import pandas as pd 
import subprocess
from sklearn.metrics import accuracy_score
from sklearn import metrics
#from autonml import AutonML, createD3mDataset
from autonml import AutonML, create_d3m_dataset
import csv
import time
logging.basicConfig(level=logging.INFO)
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

def run_autonml(task_dir, timeout_in_sec, n_jobs):
    
    print("Training AUTO^nML...")
        
    #training/testing data
    os.chdir(task_dir)
    path_to_train_data = os.path.abspath("train.csv")
    path_to_test_data = os.path.abspath("test.csv")
    df = pd.read_csv('train.csv')
    target = list(df.columns)[-1]

    #create d3m TRAIN/TEST data files
    os.mkdir("AUTO^nML")
    os.chdir("AUTO^nML")
    os.mkdir("data")
    os.mkdir("output")
    path_to_data = os.path.abspath("data")
    path_to_output = os.path.abspath("output")
    create_d3m_dataset.run(path_to_train_data, path_to_test_data, path_to_data, target, "rocAuc", ["classification"])
    
    #train AutoML
    start_time = time.time() #start
    timeout = timeout_in_sec//60
    aml = AutonML(input_dir=path_to_data,
             output_dir=path_to_output,
             timeout=timeout, numcpus=n_jobs)
    aml.run()
    stop_time = time.time() #stop
          
    #--------------EXPORT RESULTS----------------#
    AutonML_dir = os.getcwd()
    os.chdir("output")
    curr_dir = os.getcwd()
    list_result = get_ls(curr_dir)
    output_dir = curr_dir + "/" + list_result[0]
    os.chdir(AutonML_dir)
    os.mkdir("pipelines_ranked")
    os.mkdir("predictions")
    os.mkdir("training_predictions")
    os.mkdir("pipelines_scored")
    os.mkdir("executables") 

    #copy (i.e. executables, pipelines_ranked, pred, training_pred):
    cmd1 = ('cp -r ' +  output_dir + '/executables/. ' + os.getcwd() + '/executables').split(maxsplit=3) 
    cmd2 = ('cp -r ' +  output_dir + '/pipelines_ranked/. ' + os.getcwd() + '/pipelines_ranked').split(maxsplit=3) 
    cmd3 = ('cp -r ' +  output_dir + '/predictions/. ' + os.getcwd() + '/predictions').split(maxsplit=3) 
    cmd4 = ('cp -r ' +  output_dir + '/training_predictions/. ' + os.getcwd() + '/training_predictions').split(maxsplit=3)

    subprocess.run(cmd1)
    subprocess.run(cmd2)
    subprocess.run(cmd3)
    subprocess.run(cmd4)

    ######### export leaderboard #############

    list_result = get_ls(os.getcwd() + '/pipelines_ranked')
    pipelines_ranked = {}
    for i in list_result:
        model_name = i.split('.'); model_name = model_name[0]
        path_to_json = os.getcwd() + '/pipelines_ranked/' + i
        f = open(path_to_json, "r")
        model_dict = json.loads(f.read())
        rank = model_dict['pipeline_rank']
        print(model_name + ":     " + str(rank))
        pipelines_ranked[model_name] = int(rank)
 
    pipelines_ranked = sorted(pipelines_ranked.items(), key=lambda x: x[1], reverse=False)

    with open(os.getcwd() + '/pipelines_ranked/leaderboard.csv','w') as f:
        w = csv.writer(f)
        w.writerow(['Pipeline', 'Rank'])
        for i in range(0, len(list_result)): 
            w.writerow(pipelines_ranked[i])
    
    ######### for test predictions ########
    list_result = get_ls(os.getcwd() + '/predictions')

    #transform test predictions and calculate scores
    for i in list_result:
        model_name = i.split('.'); model_name = model_name[0]

        #transform test predictions:
        raw_test_preds = pd.read_csv(os.getcwd() + '/predictions/' + i) 
        test_preds = pd.DataFrame(columns=['predict','p0','p1']).astype('int64')
        for index, row in raw_test_preds.iterrows():
            d3mIndex = str(int(row['d3mIndex']))
            test_preds.at[d3mIndex, 'p' + str(int(row[str(target)]))] = row['confidence']
            if (test_preds.at[d3mIndex,'p0'] >= test_preds.at[d3mIndex,'p1']):
                test_preds.at[d3mIndex, 'predict'] = 0
            elif (test_preds.at[d3mIndex,'p0'] < test_preds.at[d3mIndex,'p1']):
                test_preds.at[d3mIndex, 'predict'] = 1

        #rename old predictions, save transformed predictions with original name
        cmd1 = 'mv %s/predictions/%s %s/predictions/RAW-%s' % (os.getcwd(), i, os.getcwd(), i) #rename
        os.system(cmd1) 
        path = '%s/predictions/%s' %(os.getcwd(), i); open(path, 'w')
        test_preds.to_csv(path, index=False) #save 
    
        #import true labels:
        test_pd = pd.read_csv(path_to_test_data).iloc[:,-1]
        test_preds_pd = test_preds
    
        #calculate accuracy:
        test_acc_score = accuracy_score(test_pd, test_preds_pd['predict'])
        
        #calculate auc:
        test_auc_score = metrics.roc_auc_score(test_pd, test_preds_pd['p1'])
           
        path = os.getcwd() + '/pipelines_scored/' + model_name + '.scores.csv'
        file = open(path, 'a+')
        file.write("Accuracy score on test data: " + str(test_acc_score) + "\n")
        file.write("AUC score on test data: " + str(test_auc_score) + "\n")
        file.close()
    os.mkdir(os.getcwd() + '/predictions/RAW') #make directory
    cmd2 = 'mv %s/predictions/RAW-* %s/predictions/RAW/' % (os.getcwd(), os.getcwd()) #move
    os.system(cmd2)

    ######## for train predictions ########
    list_result = get_ls(os.getcwd() + '/training_predictions')

    #transform train predictions and calculate scores
    for i in list_result:
        model_name = i.split('_'); model_name = model_name[0]

        #transform train predictions:
        raw_train_preds = pd.read_csv(os.getcwd() + '/training_predictions/' + i) 
        train_preds = pd.DataFrame(columns=['predict','p0','p1']).astype('int64')
    
        row_iterator = raw_train_preds.iterrows()
        for index, row in row_iterator:
            d3mIndex = str(int(row[raw_train_preds.columns[0]]))
            train_preds.at[d3mIndex, 'p0'] = row['Prediction']
            next_row = next(row_iterator)[1] #skip to next row
            train_preds.at[d3mIndex, 'p1'] = next_row['Prediction']
            if (train_preds.at[d3mIndex,'p0'] >= train_preds.at[d3mIndex,'p1']):
                train_preds.at[d3mIndex, 'predict'] = 0
            elif (train_preds.at[d3mIndex,'p0'] < train_preds.at[d3mIndex,'p1']):
                train_preds.at[d3mIndex, 'predict'] = 1
    
        #rename old predictions, save transformed predictions with original name
        cmd1 = 'mv %s/training_predictions/%s %s/training_predictions/RAW-%s' % (os.getcwd(), i, os.getcwd(), i) #rename
        os.system(cmd1) 
        path = '%s/training_predictions/%s' %(os.getcwd(), i); open(path, 'w')
        train_preds.to_csv(path, index=False) #save 
    
        #import true labels:
        train_pd = pd.read_csv(path_to_train_data).iloc[:,-1]
        train_preds_pd = train_preds
    
        #calculate accuracy:
        train_acc_score = accuracy_score(train_pd, train_preds_pd['predict'])
    
        #calculate auc:
        train_auc_score = metrics.roc_auc_score(train_pd, train_preds_pd['p1'])
           
        path = os.getcwd() + '/pipelines_scored/' + model_name + '.scores.csv'
        file = open(path, 'a+')
        file.write("Accuracy score on train data: " + str(train_acc_score) + "\n")
        file.write("AUC score on train data: " + str(train_auc_score) + "\n")
        file.close()
    os.mkdir(os.getcwd() + '/training_predictions/RAW') #make directory
    cmd2 = 'mv %s/training_predictions/RAW-* %s/training_predictions/RAW/' % (os.getcwd(), os.getcwd()) #move
    os.system(cmd2)

    ######### display success message #########
    with open(os.getcwd() + '/traintime.txt', 'w') as f:
        f.write("Training time limit: %s sec\n" % (timeout_in_sec))
        f.write("Actual time taken to train: %s sec\n" % (stop_time - start_time))
    print("...AUTO^nML export success! Execution time: %s seconds " % (stop_time - start_time))

def get_ls(filepath):
    proc = subprocess.Popen(['ls', filepath], stdout=subprocess.PIPE, universal_newlines=True)
    list_result = proc.stdout.readlines()
    list_result = [x.rstrip('\n') for x in iter(list_result)]
    return list_result         
          
          
         
    
