import os
parent_dir = "/home/xyang4/automl/Maya/d3m/re_exp_autonml/"
os.chdir(parent_dir)

#import necessary functions/packages:
import openml_dataset
from get_summary_new import get_summary
#from autosk import run_autosklearn
from auto_tpot import run_tpot
from h2o_automl import run_h2o
from autonml_script import run_autonml
#from randomforest import run_RF
import pandas as pd 
import numpy as np
from numpy import savetxt
import sys
import logging
logging.basicConfig(filename=parent_dir+"errors.log")
ids = [823]
#timelimits=[60]
#ids= [823, 737, 740, 757, 792, 799, 803, 842]     #OpenML IDs
#ids= [1004, 1006, 1011, 1013, 1038, 1116, 1486, 1220]     #OpenML IDs
#ids = [1037, 1566, 1597, 1558, 1024, 293, 354, 1369]     #OpenML IDs
#ids = [1558, 1024, 293]
#ids = [354, 1369]
#ids = [40978, 978, 1022, 718, 350, 1042]     #OpenML IDs
#ids = [23499, 1167, 1511, 1556, 1524, 346, 446, 890, 376, 1455, 1473, 1463, 1464, 1495, 41998, 40714, 41538, 41521, 41496, 42638, 42665, 40669, 40681, 40683]
#ids = [40690, 43098, 479, 476, 472, 724, 731, 729, 730, 726, 767, 764, 765, 787, 790, 795, 792, 791, 827, 829, 865, 859, 864, 860, 867, 899, 902, 905, 898, 900, 942, 944, 945, 946, 967, 961, 968, 960, 996, 997, 1026, 1025, 335, 333, 470, 43255, 13, 739, 733, 736, 784, 777, 780]
#ids = [246, 269, 120, 267, 258, 72, 153, 1211, 1219, 1235, 152, 264, 146, 260, 135, 256, 40514, 40518, 40515, 1178, 131, 124, 122, 73, 140, 121, 142, 77, 257, 126, 262, 128, 1205, 132, 1212, 1180, 1182, 1181, 1372, 293, 41150, 42206, 1217, 41228, 41836, 42397, 41147, 42252, 42256, 42769, 42742, 42750, 43489, 43439, 44036, 44038, 44077, 44092, 44121, 43903, 43975, 44081, 44159, 44161, 43948, 41159, 42758, 42812, 1046, 151, 4135, 139]
#ids = [334, 50, 333, 37, 335, 15, 451, 470, 43, 13, 782, 172, 885, 867, 875, 736, 916, 895, 1013, 974, 754, 969, 829, 448, 726, 464, 921, 346, 890, 784, 811, 747, 714, 902, 461, 955, 444, 748, 719, 860, 1075, 814, 450, 733, 730, 776, 911, 1026, 925, 744, 886, 900, 1011, 931, 949, 792, 795, 796, 996, 774, 909, 893, 906, 884, 804, 894, 770, 870, 908, 951, 997, 749, 791, 1014, 947, 841, 1005, 724, 950]
#ids = [764, 945, 941, 790 ,907, 946, 874, 481, 750, 1025, 818, 898, 848, 857, 817, 765, 1073, 1048, 1015, 944, 446, 780, 767, 827, 472, 777, 899, 891, 815, 887, 836, 961, 783, 864, 957, 852, 967, 960, 835, 968, 844, 755, 731, 831, 729, 1167, 1055, 459, 839, 854, 739]
timelimits=[60, 600, 1200]  #Time limits in seconds [60, 600, 1200]

#for each OpenML dataset: 
for id in ids: 
    print("*** For OpenML ID #" + str(id)+ " ***")
    try:
       task_dir = parent_dir + "Task_" + str(id) + "/" 
       os.mkdir(task_dir)
       os.chdir(task_dir)
       #save train-test split
       (X_train, X_test, y_train, y_test, name, dummy, rows, classes, columns) = openml_dataset.get_openml_classification_dataset(id)
       train = pd.concat([pd.DataFrame(X_train), pd.DataFrame(y_train)], axis = 1)
       train.columns = columns
       train.to_csv("train.csv", index=False)
       test = pd.concat([pd.DataFrame(X_test), pd.DataFrame(y_test)], axis = 1)
       test.columns = columns
       test.to_csv("test.csv", index=False)
    except KeyboardInterrupt:
        sys.exit()
    except Exception as e:
        print("Preprocessing:", id)
        logging.exception(e, exc_info=True)
        continue
    
    for timeout_in_sec in timelimits:
        try:
            timelimit_dir = task_dir + "timelimit:" + str(timeout_in_sec) + "sec/"
            os.mkdir(timelimit_dir)
        except KeyboardInterrupt:
            sys.exit()
        except Exception as e:
            print("Preprocessing: ", id, timeout_in_sec)
            logging.exception(e, exc_info=True)
        
        #define params
        n_jobs = 8
    
        #run AUTO^nML
        try:
            run_autonml(task_dir, timeout_in_sec, n_jobs)
            os.system("mv " + task_dir + "AUTO^nML/ " + timelimit_dir)
        except KeyboardInterrupt:
            sys.exit()
        except Exception as e:
            print(id, timeout_in_sec, "AUTO^nML")
            logging.exception(e, exc_info=True)
        
        #run Auto-sklearn
        #run_autosklearn(task_dir, timeout_in_sec, n_jobs) 
        #os.system("mv " + task_dir + "AUTO-SKLEARN/ " + timelimit_dir)
        
        #run H2O
        try:
            run_h2o(task_dir, timeout_in_sec)    
            os.system("mv " + task_dir + "H2O/ " + timelimit_dir)
        except KeyboardInterrupt:
            sys.exit()
        except Exception as e:
            print(id, timeout_in_sec, "H2O")
            logging.exception(e, exc_info=True)
        
        #run TPOT
        try:
            run_tpot(task_dir, timeout_in_sec, n_jobs)
            os.system("mv " + task_dir + "TPOT/ " + timelimit_dir)
        except KeyboardInterrupt:
            sys.exit()
        except Exception as e:
            print(id, timeout_in_sec, "TPOT")
            logging.exception(e, exc_info=True)
        
        #run separate program to collect accuacy/AUC scores
        try:
            get_summary(id, timeout_in_sec, parent_dir)
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print("get summary")
            logging.exception(e, exc_info=True)
        
        print("** Finished OpenML ID #" + str(id) + " For Time Limit: " + str(timeout_in_sec) + " seconds **")
        
    print("*** Finished OpenML ID #"  + str(id) + " ***")
