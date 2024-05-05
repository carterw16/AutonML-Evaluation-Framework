import os, shutil, sys, logging
from pathlib import Path
from picard_script import run_picard
import pandas as pd
import numpy as np
from numpy import savetxt
import openml_dataset
import openml
from timeout_decorator import timeout
# sys.path.append('../')

parent_dir = os.path.join(os.getcwd(), r'eval_results')
if os.path.exists(parent_dir):
    shutil.rmtree(parent_dir)
os.makedirs(parent_dir)

logging.basicConfig(filename=parent_dir+"/errors.log")
tasks = os.path.join(os.getcwd(), "cleaned/")
picard_tr_r2_scores = []
picard_te_r2_scores = []
#for each OpenML dataset sorted by task ID:

for task in sorted(os.listdir(tasks)):
    print("*** For OpenML ID #" + str(task) + " ***")
    if task == "41065" or task == "23513" :
        continue
    # get task directory
    try:
        os.chdir(parent_dir)
        task_dir = os.path.join(tasks, task)
        if not os.path.isdir(task_dir):
            print("Ignoring file")
            continue
        task_id = task
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
        train_r2_best, test_r2_best = run_picard(
            task_dir, eval_task_dir, task_id)
        picard_tr_r2_scores.append(train_r2_best)
        picard_te_r2_scores.append(test_r2_best)
    except KeyboardInterrupt:
        sys.exit()
    except Exception as e:
        print(task, "picard failed")
        logging.exception(e, exc_info=True)
    except TimeoutError:
        print("Script execution timed out. Skipping...")
        continue
    print("*** Finished OpenML ID #" + str(task) + " ***")

# train_test_auc_picard(picard_tr_auc_scores,
                    #   picard_te_auc_scores, auton_auc_train, auton_auc_test)
