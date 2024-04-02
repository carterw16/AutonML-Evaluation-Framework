import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import rankdata, sem
import glob

def train_test_auc_picard(train_auc_scores, test_auc_scores, auton_auc_train, auton_auc_test):
    # r2_picard = r2_score(train_auc_scores, test_auc_scores)

    reg = LinearRegression()

    x_picard = np.array(train_auc_scores).reshape(-1,1)
    y_picard = np.array(test_auc_scores)
    x_autonml = np.array(auton_auc_train).reshape(-1,1)
    y_autonml = np.array(auton_auc_test)
    reg.fit(x_picard, y_picard)
    r2_picard = reg.score(x_picard, y_picard)
    reg.fit(x_autonml, y_autonml)
    r2_autonml = reg.score(x_autonml, y_autonml)
    # Scatterplot of BEST AUC scores (train vs test) for each dataset
    plt.figure(figsize=(6, 6))
    plt.scatter(train_auc_scores, test_auc_scores, color='r', label='Picard (r2_score = %0.2f)' % r2_picard, alpha=.5)
    plt.scatter(auton_auc_train, auton_auc_test, color='b', label='AutonML (r2_score = %0.2f)' % r2_autonml, alpha=.5)
    plt.plot([0.3, 1.1], [0.3, 1.1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.3, 1.1])
    plt.ylim([0.3, 1.1])
    ax = plt.gca()
    ax.set_aspect(1)
    plt.xlabel("AUC from training predictions per Task", fontsize=14)
    plt.ylabel("AUC from testing predictions per Task", fontsize=14)
    plt.title("Best AUC for Picard/AutonML - Train Predictions vs Test Predictions", fontsize=11)
    plt.legend(loc="upper left", prop={'size': 9})
    plt.savefig("picard_auton_best_auc.svg")

def train_auc_picard_auton(train_auc_picard, train_auc_auton):
    # Scatterplot of BEST TRAIN AUC scores (picard vs auton) for each dataset
    plt.figure(figsize=(6, 6))
    plt.scatter(train_auc_picard, train_auc_auton, color='r', alpha=.5)
    plt.plot([0.3, 1.1], [0.3, 1.1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.3, 1.1])
    plt.ylim([0.3, 1.1])
    ax = plt.gca()
    ax.set_aspect(1)
    plt.xlabel("Picard AUC per Task", fontsize=14)
    plt.ylabel("AutonML AUC per Task", fontsize=14)
    plt.title("Best AUC for Picard vs AutonML - Train Predictions", fontsize=11)
    # plt.legend(loc="upper left", prop={'size': 9})
    plt.savefig("picard_auton_best_auc_train.svg")

def test_auc_picard_auton(test_auc_picard, test_auc_auton):
    # Scatterplot of BEST TEST AUC scores (picard vs auton) for each dataset
    plt.figure(figsize=(6, 6))
    plt.scatter(test_auc_picard, test_auc_auton, color='r', alpha=.5)
    plt.plot([0.3, 1.1], [0.3, 1.1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.3, 1.1])
    plt.ylim([0.3, 1.1])
    ax = plt.gca()
    ax.set_aspect(1)
    plt.xlabel("Picard AUC per Task", fontsize=14)
    plt.ylabel("AutonML AUC per Task", fontsize=14)
    plt.title("Best AUC for Picard vs AutonML - Test Predictions", fontsize=11)
    # plt.legend(loc="upper left", prop={'size': 9})
    plt.savefig("picard_auton_best_auc_test.svg")

def plot_best_scores():
	# Read csv file
	csv_file = "picard_scores.csv"
	df = pd.read_csv(csv_file)
	train_test_auc_picard(df['Picard Train AUC'], df['Picard Test AUC'], df['AutonML Train AUC'], df['AutonML Test AUC'])

def merge_data():
    old_data = pd.read_csv("old_data.csv")
    old_data = old_data[['Index', 'NumberOfFeatures', 'NumberOfInstances']]
    new_data = pd.read_csv("picard_scores.csv")
    # swap column headers for AutonML Train AUC to AutonML Test AUC and vice versa
    new_data = new_data.rename(columns={'AutonML Train AUC': 'AutonML Test AUC', 'AutonML Test AUC': 'AutonML Train AUC'})
    new_data = new_data[['Task ID', 'Picard Train AUC', 'Picard Test AUC', 'AutonML Train AUC', 'AutonML Test AUC']]
    # round all values in the dataframe to 3 decimal places
    new_data = new_data.round(3)
    merged_df = pd.merge(old_data, new_data, how='outer', left_on='Index', right_on='Task ID')
    merged_df.dropna(inplace=True)
    merged_df.reset_index(drop=True, inplace=True)

    merged_df.to_csv("merged_final_data.csv", index=False)
    return merged_df
    # print(merged_df[merged_df['Index'].isna()])

if __name__ == "__main__":

    df = pd.read_csv("merged_final_data.csv")
    picard_train_auc = df['Picard Train AUC']
    picard_test_auc = df['Picard Test AUC']
    autonml_train_auc = df['AutonML Train AUC']
    autonml_test_auc = df['AutonML Test AUC']
    # run plotting functions
    train_auc_picard_auton(picard_train_auc, autonml_train_auc)
    test_auc_picard_auton(picard_test_auc, autonml_test_auc)

    # plot_best_scores()