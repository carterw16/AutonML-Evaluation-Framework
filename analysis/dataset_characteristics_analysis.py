import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata
import glob
import copy
from datetime import datetime
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score

dim_interval = 10
dim_split = list(range(0, 50, dim_interval))
inst_interval = 500
inst_split = list(range(0, 4000, inst_interval))

def plot_graph(picard_avg, picard_median, picard_ci, autonml_avg, autonml_median, autonml_ci,
               inputx, mode, csv_file, timestamp):

        plt.figure(figsize=(8, 6.5))
        ax = plt.gca()

        plt.errorbar(inputx, picard_avg, yerr=picard_ci, color='r', label='Picard mean', alpha=.5, fmt='o', markersize=8, capsize=10)
        plt.errorbar(inputx, autonml_avg, yerr=autonml_ci, color='b', label='Auto^nML mean', alpha=.5, fmt='o', markersize=8, capsize=10)
        plt.xlim([0, inputx[-1] + 1 * (inputx[-1] - inputx[-2])])
        plt.ylim([2.5, 0.8])
        plt.xlabel("Training Time Limit (seconds)")
        if mode == "dimensionality":
            plt.xlabel("Dimensionality of datasets", fontsize=14)
        elif mode == "number of intances":
            plt.xlabel("Number of instances per dataset", fontsize=14)
        plt.ylabel("Rank from Test Predictions - Picard vs AutonML", fontsize=14)
        #plt.title("Average and Median Rank per AutoML on Test Predictions (" + str(timelimit) + "s)")
        plt.title("Datast Dimensionality vs Average Test Ranking by R^2", fontsize=16)
        plt.legend(['Picard','Auto^nML'], loc='lower left', prop={'size': 14})
        # ax.legend(loc='center left', bbox_to_anchor=(0.7, 0.5))

        plt.savefig(mode + "-" + "regression" + "-" + csv_file + timestamp + ".svg")

def plot_train_test_drop(picard_train, picard_test, autonml_train, autonml_test, sizes, metric="AUC", mode="features"):

    picard_train = np.array(picard_train)
    picard_test = np.array(picard_test)
    autonml_train = np.array(autonml_train)
    autonml_test = np.array(autonml_test)
    picard_diff = (picard_test - picard_train) / picard_train
    autonml_diff = (autonml_test - autonml_train) / autonml_train
    if mode == "features":
        # Normalize sizes for visibility in the plot, adjust the scale factor as needed
        scaled_sizes = 500 * (sizes / np.max(sizes))**.75
    elif mode == "instances":
        scaled_sizes = 500 * (sizes / np.max(sizes))**.75

    # make the difference between large and small sizes less extreme
    fig = plt.figure(figsize=(7, 7))
    # make the dots only outlines
    plt.scatter(autonml_diff, picard_diff, color='b', alpha=.5)

    plt.plot([-0.3, .3], [-0.3, .3], color='navy', lw=2, linestyle='--')

    plt.xlim([-.3, .3])
    plt.ylim([-.3, .3])
    # print(np.sum(autonml_diff < picard_diff) / len(autonml_diff))
    # calculate outliers for picard and autonml
    picard_outliers = np.abs(picard_diff) > 1.5*np.std(picard_diff)
    autonml_outliers = np.abs(autonml_diff) > 1.5*np.std(autonml_diff)
    print(f"{np.sum(picard_outliers)} picard outliers and {np.sum(autonml_outliers)} autonml outliers")
    plt.xlabel(f"AutonML {metric} percent change from train to test", fontsize=14)

    plt.ylabel(f"Picard {metric} percent change from train to test", fontsize=14)

    plt.title(f"Percent Change in {metric}: Train to Test Results - AutonML vs. Picard", fontsize=14)
    plt.savefig('percent_drop' + "-" + metric + ".svg")

def algo_boxplot(data, x='aml1200algo', y='aml1200test', metric='AUC'):
    plt.figure(figsize=(12, 6))
    # Creating a boxplot to compare the training and testing scores for each algorithm
    sns.boxplot(data=data, x=x, y=y)
    plt.xticks(rotation=45)
    plt.title('Distribution of Test ' + metric + ' Scores by Algorithm')
    plt.ylabel('Test ' + metric)
    plt.xlabel('Algorithm')
    plt.ylim(-.4, 1.05)
    # keep the y axis size consistent
    # plt.gca().set_ylim(bottom=-.75, top=1)
    plt.tight_layout()
    # keep the y axis scale consistent

    # keep size of plot within figure consistent but taller
    # plt.gca().set_size_inches((12, 6))
    # plt.gca().set_aspect(aspect='auto',adjustable='box', anchor='C')
    # make the plot within the figure taller
    # adjust y axis height
    # plt.gca().set_position([.0, .05, 1, .5]) # left, bottom, width, height

    plt.savefig('algo_boxplot' + "-" + x + "-" + metric + ".svg")


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
        if csv_file == "merged_final_reg_data.csv":

            data = pd.read_csv(csv_file)

            dim_bin, inst_bin = [], []
            for _ in range(len(dim_split)+1):
                dim_bin.append([])
            for _ in range(len(inst_split)+1):
                inst_bin.append([])

            for index, row in data.iterrows():
                number_of_features = row['numberOfFeatures']
                dim_index = bin_index(number_of_features, dim_split)
                dim_bin[dim_index].append(index)

                number_of_instances = row['numberOfInstances']
                inst_index = bin_index(number_of_instances, inst_split)
                inst_bin[inst_index].append(index)

            picard, autonml = [], []
            competition = data[['Test R2', 'aml1200test']]
            for _, row in competition.iterrows():
                # reverse the rankings so high values are better
                ranking = rankdata(-row)
                # ranking = rankdata(row)
                picard.append(ranking[0])
                autonml.append(ranking[1])

            picard_avg_dim, picard_median_dim, picard_ci_dim = [], [], []
            autonml_avg_dim, autonml_median_dim, autonml_ci_dim = [], [], []

            z = 1.96 # 95% confidence

            for b in dim_bin:
                if b != []:
                    picard_bin = np.array(picard)[np.array(b)]
                    autonml_bin = np.array(autonml)[np.array(b)]
                    length = len(autonml)
                    picard_avg_dim.append(np.mean(picard_bin))
                    picard_median_dim.append(np.median(picard_bin))
                    picard_ci_dim.append(z * np.std(picard_bin) / np.sqrt(length))
                    autonml_avg_dim.append(np.mean(autonml_bin))
                    autonml_median_dim.append(np.median(autonml_bin))
                    autonml_ci_dim.append(z * np.std(autonml_bin) / np.sqrt(length))

                else:
                    picard_avg_dim.append(-1)
                    picard_median_dim.append(-1)
                    picard_ci_dim.append(0)
                    autonml_avg_dim.append(-1)
                    autonml_median_dim.append(-1)
                    autonml_ci_dim.append(0)

            dimensionality = copy.deepcopy(dim_split)
            dimensionality.insert(0, 0)
            dimensionality = np.array(dimensionality) + int(dim_interval/2)

            plot_graph(picard_avg_dim, picard_median_dim, picard_ci_dim, autonml_avg_dim, autonml_median_dim, autonml_ci_dim, dimensionality, "dimensionality", csv_file, timestamp)

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

            # plot_graph(picard_avg_inst, picard_median_inst, picard_ci_inst, autonml_avg_inst, autonml_median_inst, autonml_ci_inst, numberOfIntances, "number of intances", csv_file, timestamp)

def decision_tree_analysis(picard_score, auton_score, task):
    # Load and prepare data
    data = pd.read_csv('merged_final_reg_data.csv')

    # Calculate the relative difference and determine the 'similar' threshold using the mean
    data['Relative_Difference'] = np.abs(data[picard_score] - data[auton_score]) / data[picard_score]
    threshold_similar = data['Relative_Difference'].quantile(0.25)
    print(threshold_similar)
    # Create a new column for the three-class target
    data['Performance_Comparison'] = np.where(
        data['Relative_Difference'] < threshold_similar, 'Similar',
        np.where(data[picard_score] > data[auton_score], 'Picard Better', 'AutonML Better')
    )

    print(data['Performance_Comparison'].value_counts())
    encoder = LabelEncoder()
    data['Performance_Comparison_encoded'] = encoder.fit_transform(data['Performance_Comparison'])

    # One-Hot Encoding for algorithm names
    aml_algo_dummies = pd.get_dummies(data['aml1200algo'], prefix='aml_algo')
    picard_algo_dummies = pd.get_dummies(data['Picard Algorithm'], prefix='picard_algo')

    # Concatenate the one-hot encoded columns with the original DataFrame
    data_encoded = pd.concat([data, aml_algo_dummies, picard_algo_dummies], axis=1)

    # Now, 'X' will include these one-hot encoded features instead of the encoded algorithm names
    feature_columns = ['numberOfFeatures', 'numberOfInstances'] + list(aml_algo_dummies.columns) + list(picard_algo_dummies.columns)
    X = data_encoded[feature_columns]

    y = data['Performance_Comparison_encoded']
    print(data_encoded.head())

    # Setup cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Determine the best tree depth using cross-validation
    best_depth = None
    best_accuracy = 0
    for depth in range(1, 10):
        tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
        # Cross-validation for each depth
        cv_accuracies = cross_val_score(tree, X, y, cv=kf, scoring='accuracy')
        cv_accuracy = np.mean(cv_accuracies)
        if cv_accuracy > best_accuracy:
            best_accuracy = cv_accuracy
            best_depth = depth
        tree.fit(X, y)
        y_pred = tree.predict(X)
        accuracy = accuracy_score(y, y_pred)
        print(f'Accuracy on the entire dataset: {accuracy:.2f}')

        plot_dec_tree(tree, feature_columns, encoder.classes_.tolist(), cv_accuracy, task)
        # Predict probabilities
        y_score = tree.predict_proba(X)

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        n_classes = y_score.shape[1]

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y == i, y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # plot_dec_tree(final_tree, feature_columns, encoder.classes_.tolist(), accuracy, task)
        # Plot all ROC curves
        plt.figure()
        class_names = encoder.inverse_transform([0, 1, 2])
        colors = ['orange', 'green', 'purple']
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, label=f'{class_names[i]} vs the rest (AUC = {roc_auc[i]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curves for Decision Tree Classifier - Regression Test Results')
        plt.legend(loc="lower right")
        plt.show()

    print("Best Tree Depth:", best_depth)

    # Train the final model
    final_tree = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
    final_tree.fit(X, y)

    y_pred = final_tree.predict(X)
    accuracy = accuracy_score(y, y_pred)
    print(f'Accuracy on the entire dataset: {accuracy:.2f}')

    # Predict probabilities
    y_score = final_tree.predict_proba(X)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = y_score.shape[1]

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y == i, y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plot_dec_tree(final_tree, feature_columns, encoder.classes_.tolist(), accuracy, task)
    # Plot all ROC curves
    plt.figure()
    class_names = encoder.inverse_transform([0, 1, 2])
    colors = ['orange', 'green', 'purple']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, label=f'{class_names[i]} vs the rest (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves for Decision Tree Classifier - Regression Test Results')
    plt.legend(loc="lower right")
    plt.show()

def analyze_performance_drop(picard_train, picard_test, autonml_train, autonml_test, task):
    # Load the dataset
    data = pd.read_csv("merged_final_reg_data.csv")
    # Calculate percent drop for Picard and AutonML
    data['Picard_Percent_Drop'] = (data[picard_train] - data[picard_test]) / data[picard_train]
    data['AutonML_Percent_Drop'] = (data[autonml_train] - data[autonml_test]) / data[autonml_train]

    # Calculate the absolute difference in percent drop and determine the threshold for 'similar'
    data['Percent_Drop_Difference'] = np.abs(data['Picard_Percent_Drop'] - data['AutonML_Percent_Drop'])
    threshold_similar = data['Percent_Drop_Difference'].quantile(0.25)

    # Label the comparison
    data['Drop_Comparison'] = np.where(
        data['Percent_Drop_Difference'] <= threshold_similar, 'Similar',
        np.where(data['Picard_Percent_Drop'] < data['AutonML_Percent_Drop'], 'Picard Better', 'AutonML Better')
    )

    # Encode the target variable
    encoder = LabelEncoder()
    data['Drop_Comparison_encoded'] = encoder.fit_transform(data['Drop_Comparison'])

    # One-Hot Encoding for algorithm names
    aml_algo_dummies = pd.get_dummies(data['aml1200algo'], prefix='aml_algo')
    picard_algo_dummies = pd.get_dummies(data['Picard Algorithm'], prefix='picard_algo')

    # Concatenate the one-hot encoded columns with the original DataFrame
    data_encoded = pd.concat([data, aml_algo_dummies, picard_algo_dummies], axis=1)

    # Now, 'X' will include these one-hot encoded features instead of the encoded algorithm names
    feature_columns = ['numberOfFeatures', 'numberOfInstances'] + list(aml_algo_dummies.columns) + list(picard_algo_dummies.columns)
    X = data_encoded[feature_columns]
    y = data['Drop_Comparison_encoded']

    # Perform k-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    depths = range(1, 6)
    mean_accuracies = []

    for depth in depths:
        tree = DecisionTreeClassifier(max_depth=depth, random_state=42, min_samples_split=20)
        accuracies = cross_val_score(tree, X, y, cv=kf, scoring='accuracy')
        mean_accuracies.append(np.mean(accuracies))

        tree.fit(X, y)
        plot_dec_tree(tree, feature_columns, encoder.classes_.tolist(), depth, task)
        # Predict probabilities
        y_score = tree.predict_proba(X)

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        n_classes = y_score.shape[1]

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y == i, y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot all ROC curves
        plt.figure()
        class_names = encoder.inverse_transform([0, 1, 2])
        colors = ['orange', 'green', 'purple']
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, label=f'{class_names[i]} vs the rest (AUC = {roc_auc[i]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curves for Decision Tree Classifier - Regression Overfitting')
        plt.legend(loc="lower right")
        plt.show()
    print(mean_accuracies)
    # Train the final model using the best depth
    best_depth = depths[np.argmax(mean_accuracies)]
    final_tree = DecisionTreeClassifier(max_depth=best_depth, random_state=42, min_samples_split=20)
    final_tree.fit(X, y)

    y_pred = final_tree.predict(X)
    accuracy = accuracy_score(y, y_pred)
    print(f'Accuracy on the entire dataset: {accuracy:.2f}')

    plot_dec_tree(final_tree, feature_columns, encoder.classes_.tolist(), accuracy, task)

def plot_dec_tree(tree, feature_names, class_names, accuracy, task):
    plt.figure(figsize=(16,8))
    pt = plot_tree(tree, feature_names=feature_names, class_names=class_names, filled=True, rounded=True, fontsize=16, label='all', impurity=False)
    plt.title('Decision Tree Showing Factors Affecting Overfitting - ' + task, fontsize=18)
    plt.savefig('pdrop_decision_tree_regression.svg')
