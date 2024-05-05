import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata
import glob
import copy
from datetime import datetime
import seaborn as sns

def algo_bar_plot(df, metric="AUC"):
  fig, ax = plt.subplots(figsize=(8, 8))
  x = range(len(df))
  ax.bar(x, df['AutonML_Frequency'], width=0.4, label='AutonML', color='blue')
  ax.bar(x, df['Picard_Frequency'], width=0.4, label='Picard', color='red',align='edge')
  ax.set_xticks(x)
  ax.set_xticklabels(frequency_df['Algorithm'], rotation=45)
  ax.set_ylabel('Frequency')
  ax.set_title('Frequency of Classification Algorithms - AutonML and Picard')
  ax.legend()

  plt.tight_layout()
  plt.savefig(metric + '_algo_bar_plot_.svg')

def cross_map_plot(df, metric="AUC"):
  cross_tab = pd.crosstab(df['Picard Algorithm'], df['aml1200algo'])

  # Plotting the heatmap
  plt.figure(figsize=(12, 8))
  sns.heatmap(cross_tab, annot=True, cmap='Blues', fmt='d')
  plt.title('Cross-Mapping of Regression Algorithms by AutonML and Picard')
  plt.xlabel('AutonML Algorithms')
  plt.ylabel('Picard Algorithms')
  plt.xticks(rotation=45)
  plt.yticks(rotation=0)
  plt.tight_layout()
  plt.savefig(metric + '_cross_map_plot.svg')

def algo_scatterplot(df, metric="AUC"):
  plt.figure(figsize=(10, 6))
  # specify the colors for each algorithm
  algo_colors = {
                  # 'extra_trees': 'blue',
                 'randomforestregressor': 'purple',
                 'gradientboostingregressor': 'green',
                #  'xgboost_gbtree': 'red',
                 'baggingregressor': 'orange',
                 'lassocv': 'gold',
                 'linearregression': 'grey',
                 'adaboostregressor': 'violet',
                 'ridge': 'cyan',
                 'lasso': 'black',
                 'elasticnet': 'brown',
                 'svr': 'magenta',
                }


  sns.scatterplot(data=df, x='Train R2', y='Test R2', hue='Picard Algorithm', palette=algo_colors, s=100)
  # sns.scatterplot(data=df, x='AutonML Train AUC', y='AutonML Test AUC', hue='aml1200algo', palette='bright', s=100)

  # Enhancing the plot
  plt.title(f'Picard Training vs. Testing {metric} Scores  by Algorithm')
  plt.xlabel('Picard Train R^2')
  plt.ylabel('Picard Test R^2')
  plt.plot([0, 1.1], [0,1.1], color='navy', lw=2, linestyle='--')
  plt.xlim([0, 1.1])
  plt.ylim([0, 1.1])
  plt.grid(True)  # Add gridlines for better readability
  plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')  # Place legend outside of the plot
  plt.tight_layout()
  # keep size of figure consistent as square
  plt.gca().set_aspect('equal', adjustable='box')
  plt.savefig(metric + '_algo_train_test_picard.svg')
