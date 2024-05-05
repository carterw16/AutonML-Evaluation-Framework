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

# Same as above, but for R^2 scores instead of AUC scores
def train_test_r2_picard(train_r2_scores, test_r2_scores, auton_r2_train, auton_r2_test):
    # r2_picard = r2_score(train_auc_scores, test_auc_scores)
    reg = LinearRegression()
    x_picard = np.array(train_r2_scores).reshape(-1,1)
    y_picard = np.array(test_r2_scores)
    x_autonml = np.array(auton_r2_train).reshape(-1,1)
    y_autonml = np.array(auton_r2_test)
    reg.fit(x_picard, y_picard)
    r2_picard = reg.score(x_picard, y_picard)
    reg.fit(x_autonml, y_autonml)
    r2_autonml = reg.score(x_autonml, y_autonml)
    # Scatterplot of BEST R^2 scores (train vs test) for each dataset
    plt.figure(figsize=(6, 6))
    plt.scatter(train_r2_scores, test_r2_scores, color='r', label='Picard (r2_score = %0.2f)' % r2_picard, alpha=.5)
    plt.scatter(auton_r2_train, auton_r2_test, color='b', label='AutonML (r2_score = %0.2f)' % r2_autonml, alpha=.5)
    plt.plot([0, 1.1], [0, 1.1], color='navy', lw=2, linestyle='--')
    plt.xlim([0, 1.1])
    plt.ylim([0, 1.1])
    ax = plt.gca()
    ax.set_aspect(1)
    plt.xlabel("R^2 from training predictions per Task", fontsize=14)
    plt.ylabel("R^2 from testing predictions per Task", fontsize=14)
    plt.title("Best R^2 for Picard/AutonML - Train Predictions vs Test Predictions", fontsize=11)
    plt.legend(loc="upper left", prop={'size': 9})
    plt.savefig("picard_auton_best_r2.svg")

def train_auc_picard_auton(train_auc_picard, train_auc_auton):
    # Scatterplot of BEST TRAIN AUC scores (picard vs auton) for each dataset
    plt.figure(figsize=(6, 6))
    plt.scatter(train_auc_auton, train_auc_picard, color='r', alpha=.5)
    plt.plot([0.3, 1.1], [0.3, 1.1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.3, 1.1])
    plt.ylim([0.3, 1.1])
    ax = plt.gca()
    ax.set_aspect(1)
    plt.xlabel("AutonML AUC per Task", fontsize=14)
    plt.ylabel("Picard AUC per Task", fontsize=14)
    plt.title("Best AUC for AutonML vs Picard - Train Predictions", fontsize=11)
    # plt.legend(loc="upper left", prop={'size': 9})
    plt.savefig("picard_auton_best_auc_train.svg")

def test_auc_picard_auton(test_auc_picard, test_auc_auton):
    # Scatterplot of BEST TEST AUC scores (picard vs auton) for each dataset
    plt.figure(figsize=(6, 6))
    plt.scatter(test_auc_auton, test_auc_picard, color='r', alpha=.5)
    plt.plot([0.3, 1.1], [0.3, 1.1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.3, 1.1])
    plt.ylim([0.3, 1.1])
    ax = plt.gca()
    ax.set_aspect(1)
    plt.xlabel("AutonML AUC per Task", fontsize=14)
    plt.ylabel("Picard AUC per Task", fontsize=14)
    plt.title("Best AUC for AutonML vs Picard - Test Predictions", fontsize=11)
    # plt.legend(loc="upper left", prop={'size': 9})
    plt.savefig("picard_auton_best_auc_test.svg")

# Same as two functions above but for r^2 scores instead of AUC scores
def train_r2_picard_auton(train_r2_picard, train_r2_auton):
    # Scatterplot of BEST TRAIN R^2 scores (picard vs auton) for each dataset
    plt.figure(figsize=(6, 6))
    plt.scatter(train_r2_auton, train_r2_picard, color='r', alpha=.5)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    ax = plt.gca()
    ax.set_aspect(1)
    plt.xlabel("AutonML R^2 per Task", fontsize=14)
    plt.ylabel("Picard R^2 per Task", fontsize=14)
    plt.title("Best R^2 for AutonML vs Picard - Train Predictions", fontsize=11)
    # plt.legend(loc="upper left", prop={'size': 9})
    plt.savefig("picard_auton_best_r2_train.svg")

def test_r2_picard_auton(test_r2_picard, test_r2_auton):
    # Scatterplot of BEST TEST R^2 scores (picard vs auton) for each dataset
    plt.figure(figsize=(6, 6))
    plt.scatter(test_r2_auton, test_r2_picard, color='r', alpha=.5)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    ax = plt.gca()
    ax.set_aspect(1)
    plt.xlabel("AutonML R^2 per Task", fontsize=14)
    plt.ylabel("Picard R^2 per Task", fontsize=14)
    plt.title("Best R^2 for AutonML vs Picard - Test Predictions", fontsize=11)
    # plt.legend(loc="upper left", prop={'size': 9})
    plt.savefig("picard_auton_best_r2_test.svg")

def merge_data():
    old_data = pd.read_csv("old_regression_data.csv")
    old_data = old_data[['Index', 'numberOfFeatures', 'numberOfInstances','aml1200algo', 'aml1200train', 'aml1200test']]
    new_data = pd.read_csv("picard_reg_scores.csv")
    new_data = new_data.round(3)
    merged_df = pd.merge(old_data, new_data, how='outer', left_on='Index', right_on='Task ID')
    merged_df.dropna(inplace=True)
    merged_df.reset_index(drop=True, inplace=True)

    merged_df.to_csv("merged_final_reg_data.csv", index=False)
    return merged_df
