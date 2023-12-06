import os
import h2o
import time
from h2o.automl import H2OAutoML
from sklearn.metrics import accuracy_score
from sklearn import metrics
import time

# folder structure:
# task_dir {
#     test.csv
#     train.csv
#     H2O {
#        pipelines_ranked {}
#        pipelines_scored {}
#        predictions {}
#        training_predictions {}
#     }
# }

def run_h2o(task_dir, timeout_in_sec):

    print("Training H2O...")

    #initialize AutoML
    os.chdir(task_dir)
    h2o.init()
    #training data
    h2o_train = h2o.import_file(path = "train.csv")
    target = h2o_train.names[-1]
    h2o_train[target] = h2o_train[target].asfactor()

    #testing data
    h2o_test = h2o.import_file(path = "test.csv")
    h2o_test[target] = h2o_test[target].asfactor()

    #train AutoML
    start_time = time.time() #start
    aml = H2OAutoML(max_runtime_secs=timeout_in_sec, nfolds=0, sort_metric='AUC')
    aml.train(y=target, training_frame=h2o_train)
    stop_time = time.time() #stop

    # #--------------EXPORT RESULTS----------------#
    # os.chdir(task_dir)
    # os.mkdir("H2O")
    # os.chdir("H2O")
    # os.mkdir("pipelines_ranked")
    # os.mkdir("predictions")
    # os.mkdir("training_predictions")
    # os.mkdir("pipelines_scored")

    # #export leaderboard
    # lb = h2o.automl.get_leaderboard(aml, extra_columns = 'ALL')
    # open('pipelines_ranked/leaderboard.csv', 'w')
    # h2o.export_file(lb, path = os.getcwd() + '/pipelines_ranked/leaderboard.csv', force = True)

    # for model in aml.leaderboard.as_data_frame()['model_id'][:10]:
    #     m = h2o.get_model(model)
    #     m_name =  m.params['model_id']['actual']['name']

    #     #save model
    #     path = os.getcwd() + '/pipelines_ranked/'
    #     m.save_model_details(path=path, force = True)

    #     #make predictions on test data
    #     h2o_test_preds = m.predict(h2o_test);
    #     path = os.getcwd() + '/predictions/' + str(m_name) + '.json.predictions.csv'; open(path, 'w')
    #     h2o.export_file(h2o_test_preds, path = path, force = True)

    #     #make predictions on train data
    #     h2o_train_preds = m.predict(h2o_train);
    #     path = os.getcwd() + '/training_predictions/' + str(m_name) + '_train_predictions.csv'; open(path, 'w')
    #     h2o.export_file(h2o_train_preds, path = path, force = True)

    #     #print scores on test & train data
    #     path = os.getcwd() + '/pipelines_scored/' + str(m_name) + '.scores.csv'
    #     file = open(path,'w')
    #     #test:
    #     test_preds_pd = h2o_test_preds.as_data_frame(use_pandas=True)
    #     test_pd = h2o_test.as_data_frame(use_pandas=True)
    #     test_acc_score = accuracy_score(test_pd[target], test_preds_pd['predict'])
    #     test_auc_score =  metrics.roc_auc_score(test_pd[target], test_preds_pd['p1'])
    #     #train:
    #     train_preds_pd = h2o_train_preds.as_data_frame(use_pandas=True)
    #     train_pd = h2o_train.as_data_frame(use_pandas=True)
    #     train_acc_score = accuracy_score(train_pd[target], train_preds_pd['predict'])
    #     train_auc_score = metrics.roc_auc_score(train_pd[target], train_preds_pd['p1'])
    #     #write:
    #     file.write("Accuracy score on test data: " + str(test_acc_score) + "\n")
    #     file.write("AUC score on test data: " + str(test_auc_score) + "\n")
    #     file.write("Accuracy score on train data: " + str(train_acc_score) + "\n")
    #     file.write("AUC score on train data: " + str(train_auc_score) + "\n")
    #     file.close()

    # #shutdown AutoML
    # h2o.cluster().shutdown()
    # time.sleep(2)

    # #display success message
    # with open(os.getcwd() + '/traintime.txt', 'w') as f:
    #     f.write("Training time limit: %s sec\n" % (timeout_in_sec))
    #     f.write("Actual time taken to train: %s sec\n" % (stop_time - start_time))
    # print("...H2O export success! Execution time: %s seconds " % (stop_time - start_time))
