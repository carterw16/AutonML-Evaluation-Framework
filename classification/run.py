import os, shutil, sys, logging
from pathlib import Path
from picard_script import run_picard
import pandas as pd
import numpy as np
from numpy import savetxt
sys.path.append('../')
from analysis.auc_rank import train_test_auc_picard

parent_dir = os.path.join(os.getcwd(), r'eval_results')
if os.path.exists(parent_dir):
    shutil.rmtree(parent_dir)
os.makedirs(parent_dir)
# print(Path(os.getcwd()).parent.absolute())

#import necessary functions/packages:
# import openml_dataset
# from autonml_script import run_autonml
# from h2o_script import run_h2o
# from tpot_script import run_tpot
# from ag_script import run_autogluon

logging.basicConfig(filename=os.path.join(parent_dir, "errors.log"))

# ids= [823, 737, 740, 757, 792, 799, 803]     #OpenML IDs
# ids = [*range(166860, 166890)]
# timelimits=[60, 600, 1200]
tasks = os.path.join(os.getcwd(), "Desktop/")
picard_tr_auc_scores = []
picard_te_auc_scores = []
#for each OpenML dataset:
for task in os.listdir(tasks):
    print("*** For OpenML ID #" + str(task)+ " ***")
    try:
        os.chdir(parent_dir)
        task_dir = os.path.join(tasks, task)
        if not os.path.isdir(task_dir):
            print("Ignoring file")
            continue
        task_id = task.split("_")[1]
        eval_task_dir = os.path.join(os.getcwd(), task)
        if os.path.exists(eval_task_dir):
            shutil.rmtree(eval_task_dir)
        os.makedirs(eval_task_dir)
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
        train_auc_best, test_auc_best, auton_auc_train, auton_auc_test = run_picard(task_dir, eval_task_dir, task_id)
        picard_tr_auc_scores.append(train_auc_best)
        picard_te_auc_scores.append(test_auc_best)
    except KeyboardInterrupt:
        sys.exit()
    except Exception as e:
        print(task, "picard failed")
        logging.exception(e, exc_info=True)
    print("*** Finished OpenML ID #"  + str(task) + " ***")
print("*** Finished All Tasks #")

# os.chdir(parent_dir)
train_test_auc_picard(picard_tr_auc_scores, picard_te_auc_scores, auton_auc_train, auton_auc_test)