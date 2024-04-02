import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata
import glob
import copy
from datetime import datetime

dim_interval = 5
dim_split = list(range(0, 50, dim_interval))
inst_interval = 500
inst_split = list(range(0, 4000, inst_interval))

def plot_graph(picard_avg, picard_median, picard_ci, autonml_avg, autonml_median, autonml_ci,
               inputx, mode, csv_file, timestamp):

        plt.figure(figsize=(6.5, 6.5))

        ax = plt.gca()

        plt.errorbar(inputx, picard_avg, yerr=picard_ci, color='r', label='Picard mean', alpha=.5, fmt='o', markersize=8, capsize=10)
        plt.errorbar(inputx, autonml_avg, yerr=autonml_ci, color='b', label='Auto^nML mean', alpha=.5, fmt='o', markersize=8, capsize=10)
        # plt.errorbar(inputx + 0.05 * (inputx[-1] - inputx[-2]), h2o_avg, yerr=h2o_ci, color='b', label='H2O AutoML mean', alpha=.5, fmt='o', markersize=8, capsize=10)
        # plt.errorbar(inputx + 0.1 * (inputx[-1] - inputx[-2]), tpot_avg, yerr=tpot_ci, color='g', label='TPOT mean', alpha=.5, fmt='o', markersize=8, capsize=10)
        # plt.errorbar(inputx + 0.15 * (inputx[-1] - inputx[-2]), ag_avg, yerr=ag_ci, color='m', label='AutoGluon mean', alpha=.5, fmt='o', markersize=8, capsize=10)

        plt.xlim([0, inputx[-1] + 1 * (inputx[-1] - inputx[-2])])
        plt.ylim([2.5, 0.8])
        plt.xlabel("Training Time Limit (seconds)")
        if mode == "dimensionality":
            plt.xlabel("Dimensionality of datasets", fontsize=14)
        elif mode == "number of intances":
            plt.xlabel("Number of Instances of datasets", fontsize=14)
        plt.ylabel("Rank from Test Predictions accross datasets", fontsize=14)
        #plt.title("Average and Median Rank per AutoML on Test Predictions (" + str(timelimit) + "s)")
        plt.title("Average rank per AutoML on Test Predictions", fontsize=16)
        plt.legend(['Picard','Auto^nML'])
        ax.legend(loc='center left', bbox_to_anchor=(0.7, 0.5))

        plt.savefig(mode + "-" + "max" +str(inputx[-1]) + "-" + "interval" +str(inputx[-1] - inputx[-2]) + "-" + csv_file + timestamp + ".svg")

def bin_index(num, break_points):
    index = 0
    for break_point in break_points:
        if num >= break_point:
            index += 1
        else:
            break
    return index

def plot():
    now = datetime.now()
    timestamp = now.strftime("%m-%d-%Y %H:%M:%S")

    csv_files = glob.glob('*.csv')
    print(csv_files)
    for csv_file in csv_files:
        if csv_file == "merged_final_data.csv":

            data = pd.read_csv(csv_file)

            dim_bin, inst_bin = [], []
            for _ in range(len(dim_split)+1):
                dim_bin.append([])
            for _ in range(len(inst_split)+1):
                inst_bin.append([])

            for index, row in data.iterrows():
                number_of_features = row['NumberOfFeatures']
                dim_index = bin_index(number_of_features, dim_split)
                dim_bin[dim_index].append(index)

                number_of_instances = row['NumberOfInstances']
                inst_index = bin_index(number_of_instances, inst_split)
                inst_bin[inst_index].append(index)

            picard, autonml = [], []
            competition = data[['Picard Test AUC', 'AutonML Test AUC']]
            for _, row in competition.iterrows():
                ranking = rankdata(row)
                picard.append(ranking[0])
                autonml.append(ranking[1])

            # picard_avg_dim, picard_median_dim, picard_ci_dim = [], [], []
            # autonml_avg_dim, autonml_median_dim, autonml_ci_dim = [], [], []

            # z = 1.96 # 95% confidence

            # for b in dim_bin:
            #     if b != []:
            #         picard_bin = np.array(picard)[np.array(b)]
            #         autonml_bin = np.array(autonml)[np.array(b)]
            #         length = len(autonml)
            #         picard_avg_dim.append(np.mean(picard_bin))
            #         picard_median_dim.append(np.median(picard_bin))
            #         picard_ci_dim.append(z * np.std(picard_bin) / np.sqrt(length))
            #         autonml_avg_dim.append(np.mean(autonml_bin))
            #         autonml_median_dim.append(np.median(autonml_bin))
            #         autonml_ci_dim.append(z * np.std(autonml_bin) / np.sqrt(length))

            #     else:
            #         picard_avg_dim.append(-1)
            #         picard_median_dim.append(-1)
            #         picard_ci_dim.append(0)
            #         autonml_avg_dim.append(-1)
            #         autonml_median_dim.append(-1)
            #         autonml_ci_dim.append(0)

            # dimensionality = copy.deepcopy(dim_split)
            # dimensionality.insert(0, 0)
            # dimensionality = np.array(dimensionality) + int(dim_interval/2)

            # plot_graph(picard_avg_dim, picard_median_dim, picard_ci_dim, autonml_avg_dim, autonml_median_dim, autonml_ci_dim, dimensionality, "dimensionality", csv_file, timestamp)

            picard_avg_inst, picard_median_inst, picard_ci_inst = [], [], []
            autonml_avg_inst, autonml_median_inst, autonml_ci_inst = [], [], []
            # h2o_avg_inst, h2o_median_inst, h2o_ci_inst = [], [], []
            # tpot_avg_inst, tpot_median_inst, tpot_ci_inst = [], [], []
            # ag_avg_inst, ag_median_inst, ag_ci_inst = [], [], []

            z = 1.96 # 95% confidence

            for b in inst_bin:
                if b != []:
                    picard_bin = np.array(picard)[np.array(b)]
                    autonml_bin = np.array(autonml)[np.array(b)]
                    length = len(autonml)
                    picard_avg_inst.append(np.mean(picard_bin))
                    picard_median_inst.append(np.median(picard_bin))
                    picard_ci_inst.append(z * np.std(picard_bin) / np.sqrt(length))
                    autonml_avg_inst.append(np.mean(autonml_bin))
                    autonml_median_inst.append(np.median(autonml_bin))
                    autonml_ci_inst.append(z * np.std(autonml_bin) / np.sqrt(length))
                else:
                    picard_avg_inst.append(-1)
                    picard_median_inst.append(-1)
                    picard_ci_inst.append(0)
                    autonml_avg_inst.append(-1)
                    autonml_median_inst.append(-1)
                    autonml_ci_inst.append(0)

            numberOfIntances = copy.deepcopy(inst_split)
            numberOfIntances.insert(0, 0)
            numberOfIntances = np.array(numberOfIntances) + int(inst_interval/2)

            plot_graph(picard_avg_inst, picard_median_inst, picard_ci_inst, autonml_avg_inst, autonml_median_inst, autonml_ci_inst, numberOfIntances, "number of intances", csv_file, timestamp)

if __name__ == "__main__":
    plot()