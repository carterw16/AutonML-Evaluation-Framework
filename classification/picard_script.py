# Copyright (c) 2023 Carnegie Mellon University
# This code is subject to the license terms contained in the LICENSE file.
import os, sys, glob
from pathlib import Path
import pandas as pd
import csv
import warnings
import code
import matplotlib.pyplot as plt
sys.path.append('../')
from analysis.auc_rank import train_test_auc_picard

PATH_TO_PICARD="/Users/carterweaver/Desktop/Summer2023/Auton/ngautonml"
sys.path.append(PATH_TO_PICARD)

from ngautonml.wrangler.wrangler import Wrangler
from ngautonml.wrangler.dataset import DatasetKeys
from ngautonml.problem_def.problem_def  import ProblemDefinition
from ngautonml.algorithms.impl.algorithm_auto import FakeAlgorithmCatalogAuto

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def write_best_to_csv(task_id, train_auc, test_auc, auton_auc_train, auton_auc_test):
	csv_file = "picard_scores.csv"  # CSV file to store results
	row = [task_id, train_auc, test_auc, auton_auc_train, auton_auc_test]
	if not os.path.isfile(csv_file):
		# Create the CSV file with headers if it doesn't exist
		with open(csv_file, 'w', newline='') as file:
			writer = csv.writer(file)
			writer.writerow(["Task ID", "Picard Train AUC", "Picard Test AUC", "AutonML Train AUC", "AutonML Test AUC"])

	with open(csv_file, 'a', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(row)


def write_task_to_csv(task_id, train_auc_scores, test_auc_scores):
	csv_file = f"picard_scores_{task_id}.csv"

	with open(csv_file, 'w', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(["pipeline_des", "train_auc", "test_auc"])

		for pipeline_des, train_auc in train_auc_scores.items():
			test_auc = test_auc_scores[pipeline_des]
			writer.writerow([pipeline_des, train_auc, test_auc])


def plot_task_scores_picard(task_id):
	csv_files = glob.glob(f"picard_scores_{task_id}.csv")

	if not csv_files:
			print(f"No CSV file found for Task {task_id}")
			return

	csv_file = csv_files[0]
	df = pd.read_csv(csv_file)
	fig = plt.figure(figsize=(8, 8))

	for index, row in df.iterrows():
		pipeline_des = row['pipeline_des']
		train_auc = row['train_auc']
		test_auc = row['test_auc']

		plt.scatter(train_auc, test_auc, label=pipeline_des)

	plt.plot([0, 1], [0, 1], 'r--')
	plt.xlabel("Train AUC")
	plt.ylabel("Test AUC")
	plt.title(f"Train AUC vs. Test AUC - Task {task_id}")
	plt.grid(True)
	plt.legend(
		loc='lower center',
		bbox_to_anchor=(0.5, 1.1),
    fontsize='small',
	)
	plt.tight_layout()
	plt.savefig(f"picard_auc_task_{task_id}.svg")


def run_picard(task_dir, eval_task_dir, task_id):
	eval_dir = os.getcwd()
	os.chdir(eval_task_dir)

	#Read AutonML AUC scores
	csv_file = os.path.join(
		task_dir, f'timelimit_1200sec', f'timelimit-1200sec-summary.csv')
	# Check if the CSV file exists
	if not os.path.exists(csv_file):
		print(f"CSV file '{csv_file}' not found.")

	# Read the CSV file and extract AutonML scores
	summary = pd.read_csv(csv_file)
	# print(summary.columns)
	for index, row in summary.iterrows():
		if row[0] == 'Auto^nML(Best)':
			auton_auc_train = float(row['AUC-Train'])
			auton_auc_test = float(row['AUC-Test'])
			print(auton_auc_train, auton_auc_test)


	# Read training/testing data
	train_path = os.path.join(task_dir, "train.csv")
	train_pd = pd.read_csv(train_path)
	test_path = os.path.join(task_dir, "test.csv")
	test_pd = pd.read_csv(test_path)

	# Set target column to last column
	target = train_pd.columns[-1]

  	# Set problem definition and fit pipelines to task
	pdef_dict = {
		"dataset": {
			"config": "local",
			"test_path": test_path,
			"train_path": train_path,
			"column_roles": {
				"target" : {
					"name": target
				}
			}
		},
		"problem_type": {
			"task": "binary_classification"
		},
		"output": {},
		"metrics": {
			"roc_auc_score": {}
		}
	}
	problem_def = ProblemDefinition(pdef_dict)
	wrangler = Wrangler(problem_definition=problem_def)
	wrangler_result = wrangler.fit_predict_rank()

	train_data = wrangler.dataset(pd.read_csv(train_path))
	tr_ground_truth = wrangler.dataset(data=train_data.dataframe[[target]], key=DatasetKeys.GROUND_TRUTH)
	test_data = wrangler.dataset(pd.read_csv(test_path))
	te_ground_truth = wrangler.dataset(data=test_data.dataframe[[target]], key=DatasetKeys.GROUND_TRUTH)

	# predict on train/test data and get train/test rankings
	train_result = wrangler_result.train_results
	train_rankings = wrangler_result.rankings

	test_result = wrangler_result.test_results
	test_rankings = wrangler.rank(results=test_result, ground_truth=te_ground_truth)
	# Initialize dict mapping (pipeline_des, auc_score)
	train_auc_scores = {}
	test_auc_scores = {}

	# Populate scores dicts with train/test scores for top pipelines (on train data)
	train_auc_pipelines = train_rankings["roc_auc_score"].best(10)
	for pipeline in train_auc_pipelines:
		des = pipeline.pipeline_des
		train_auc_scores[des] = pipeline.score
		test_auc_scores[des] = test_rankings["roc_auc_score"].scores_as_dict[des].score

	# Get scores for pipeline with best train score
		best_pipeline = train_auc_pipelines[0]
		train_auc_best = best_pipeline.score
		test_auc_best = test_rankings["roc_auc_score"].scores_as_dict[best_pipeline.pipeline_des].score

	# Write scores for each pipeline to new csv file in Picard folder
	picard_dir = os.path.join(os.getcwd(), "Picard")
	if os.path.exists(picard_dir):
		shutil.rmtree(picard_dir)
	os.makedirs(picard_dir)
	os.chdir(picard_dir)

	write_task_to_csv(task_id, train_auc_scores, test_auc_scores)
	# Save scatterplot of auc scores in Picard folder
	plot_task_scores_picard(task_id)

	#write:
	os.chdir(eval_dir)
	write_best_to_csv(task_id, train_auc_best, test_auc_best,
	                  auton_auc_train, auton_auc_test)
	return train_auc_best, test_auc_best, auton_auc_train, auton_auc_test

# Write a function that reads each column of picard_scores.csv into a list and passes it into train_test_auc_picard
def plot_best_scores():
	# Read csv file
	csv_file = "eval_results/picard_scores.csv"
	df = pd.read_csv(csv_file)
	train_test_auc_picard(df['Picard Train AUC'], df['Picard Test AUC'], df['AutonML Train AUC'], df['AutonML Test AUC'])

# run_picard("../ngautonml/examples/classification",0)
# plot_best_scores()