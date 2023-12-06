import os, shutil
parent_dir = os.path.join(os.getcwd(), r'eval_results')
if os.path.exists(parent_dir):
    shutil.rmtree(parent_dir)
os.makedirs(parent_dir)
# os.chdir(parent_dir)

#import necessary functions/packages:
# import openml_dataset
# from autonml_script import run_autonml
# from h2o_script import run_h2o
# from tpot_script import run_tpot
# from ag_script import run_autogluon
from picard_script import run_picard
import pandas as pd
import numpy as np
from numpy import savetxt
import sys
import logging
from analysis.auc_rank import train_test_auc_picard
logging.basicConfig(filename=os.path.join(parent_dir, "errors.log"))

# ids= [823, 737, 740, 757, 792, 799, 803]     #OpenML IDs
# ids = [*range(166860, 166890)]
# timelimits=[60, 600, 1200]
picard_tr_auc_scores = []
picard_te_auc_scores = []
tasks = os.path.join(os.getcwd(), "Desktop/")
#for each OpenML dataset:
for task in os.listdir(tasks):
    print("*** For OpenML ID #" + str(task)+ " ***")
    try:
        if os.path.isfile(task):
            continue
        task_dir = os.path.join(tasks, task)
        os.chdir(parent_dir)
    #    #save train-test split
    #    (X_train, X_test, y_train, y_test, name, dummy, rows, classes, columns) = openml_dataset.get_openml_classification_dataset(id)
    #    train = pd.concat([pd.DataFrame(X_train), pd.DataFrame(y_train)], axis = 1)
    #    train.columns = columns
    #    train.to_csv("train.csv", index=False)
    #    test = pd.concat([pd.DataFrame(X_test), pd.DataFrame(y_test)], axis = 1)
    #    test.columns = columns
    #    test.to_csv("test.csv", index=False)
    except KeyboardInterrupt:
        sys.exit()
    except Exception as e:
        print("Preprocessing exception:", task)
        logging.exception(e, exc_info=True)
        continue

    # run picard
    try:
        train_aucs, test_aucs, train_auc_best, test_auc_best = run_picard(task_dir, task)
        picard_tr_auc_scores.append(train_auc_best)
        picard_te_auc_scores.append(test_auc_best)
    except KeyboardInterrupt:
        sys.exit()
    except Exception as e:
        print(task, "picard")
        logging.exception(e, exc_info=True)
    print("*** Finished OpenML ID #"  + str(task) + " ***")

train_test_auc_picard(picard_tr_auc_scores, picard_te_auc_scores)