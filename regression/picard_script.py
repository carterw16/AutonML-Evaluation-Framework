# Copyright (c) 2023 Carnegie Mellon University
# This code is subject to the license terms contained in the LICENSE file.
import os, sys, glob
from pathlib import Path
import pandas as pd
import csv
import warnings
import code
import matplotlib.pyplot as plt
from timeout_decorator import timeout

PATH_TO_PICARD="/Users/carterweaver/Desktop/Summer2023/Auton/ngautonml"
sys.path.append(PATH_TO_PICARD)

from ngautonml.wrangler.wrangler import Wrangler
from ngautonml.wrangler.dataset import DatasetKeys
from ngautonml.problem_def.problem_def  import ProblemDefinition
from ngautonml.algorithms.impl.algorithm_auto import FakeAlgorithmCatalogAuto

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def write_best_to_csv(task_id, train_r2, test_r2):
	csv_file = "picard_reg_scores.csv"  # CSV file to store results
	row = [task_id, train_r2, test_r2]
	if not os.path.isfile(csv_file):
		# Create the CSV file with headers if it doesn't exist
		with open(csv_file, 'w', newline='') as file:
			writer = csv.writer(file)
			writer.writerow(["Task ID", "Train R2", "Test R2"])

	with open(csv_file, 'a', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(row)


def write_task_to_csv(task_id, train_rmse_scores, test_rmse_scores):
	csv_file = f"picard_scores_{task_id}.csv"

	with open(csv_file, 'w', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(["pipeline_des", "train_rmse", "test_rmse"])

		for pipeline_des, train_rmse in train_rmse_scores.items():
			test_rmse = test_rmse_scores[pipeline_des]
			writer.writerow([pipeline_des[:10], train_rmse, test_rmse])

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
		train_rmse = row['train_rmse']
		test_rmse = row['test_rmse']

		plt.scatter(train_rmse, test_rmse, label=pipeline_des)

	# plt.plot([0, max(train_rmse_scores.values())], [0, max(test_rmse_scores.values())], 'r--')
	# draw a line from (0,0) to the top right of the figure along the diagonal

	plt.plot([0, 1], [0, 1], 'r--')
	plt.xlabel("Train R^2 Score")
	plt.ylabel("Test R^2 Score")
	plt.title(f"Train R^2 vs. Test R^2 - Task {task_id}")
	plt.grid(True)
	plt.legend(
		loc='lower center',
		bbox_to_anchor=(0.5, 1.1),
    fontsize='small',
	)
	plt.tight_layout()
	plt.savefig(f"picard_r2_task_{task_id}.svg")

# @timeout(10)
def run_picard(task_dir, eval_task_dir, task_id):
	eval_dir = os.getcwd()
	os.chdir(eval_task_dir)

	# Read training/testing data
	train_path = os.path.join(task_dir, "train.csv")
	train_pd = pd.read_csv(train_path)
	test_path = os.path.join(task_dir, "test.csv")
	test_pd = pd.read_csv(test_path)

	# skip this run if train_pd has over 20000 rows or over 300 columns
	if len(train_pd) > 50000 or len(train_pd.columns) > 300:
		print(f"Skipping task {task_id}")
		return

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
			"task": "regression"
		},
		"cross_validation": {
			"k": 5
		},
		"output": {},
		"metrics": {
			"r2_score" : {},
		},
		"hyperparams": ["disable_grid_search"]
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
	train_r2_scores = {}
	test_r2_scores = {}
	print("ran picard script")
	# Populate scores dicts with train/test scores for top pipelines (on train data)

	train_r2_pipelines = train_rankings["r2_score"].best(10)
	for pipeline in train_r2_pipelines:
		des = pipeline.pipeline_des
		train_r2_scores[des] = pipeline.score
		test_r2_scores[des] = test_rankings["r2_score"].scores_as_dict[des].score

	# Get scores for pipeline with best train score
		best_pipeline = train_r2_pipelines[0]
		train_r2_best = best_pipeline.score
		test_r2_best = test_rankings["r2_score"].scores_as_dict[best_pipeline.pipeline_des].score

	# Write scores for each pipeline to new csv file in Picard folder
	picard_dir = os.path.join(os.getcwd(), "Picard")
	if os.path.exists(picard_dir):
		shutil.rmtree(picard_dir)
	os.makedirs(picard_dir)
	os.chdir(picard_dir)

	write_task_to_csv(task_id, train_r2_scores, test_r2_scores)
	# Save scatterplot of auc scores in Picard folder
	plot_task_scores_picard(task_id)

	#write:
	os.chdir(eval_dir)
	write_best_to_csv(task_id, train_r2_best, test_r2_best)
	return train_r2_best, test_r2_best


# run_picard("../ngautonml/examples/classification",0)
