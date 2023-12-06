# Copyright (c) 2023 Carnegie Mellon University
# This code is subject to the license terms contained in the LICENSE file.
import os, sys
from pathlib import Path
import pandas as pd
import csv
import warnings
import code

PATH_TO_PICARD="/Users/carterweaver/Desktop/Summer2023/Auton/ngautonml"
sys.path.append(PATH_TO_PICARD)

from automl.wrangler.wrangler import Wrangler
from automl.wrangler.dataset import DatasetKeys
from automl.problem_def.problem_def  import ProblemDefinition
from automl.algorithms.impl.algorithm_auto import FakeAlgorithmCatalogAuto

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

PROBLEM_DEF = '''{{
    "dataset": {{
        "config": "local",
        "test_path": "{test_path}",
        "train_path": "{train_path}",
        "column_roles": {{
            "target" : {{
                "name": "{class_name}"
            }}
        }}
    }},
    "problem_type": {{
        "task": "binary_classification"
    }},
    "output": {{}},
    "metrics": {{
        "roc_auc_score": {{}}
    }}
}}'''

def write_to_csv(task_id, train_auc, test_auc):
	csv_file = "picard_results.csv"  # CSV file to store results
	row = [task_id, train_auc, test_auc]
	if not os.path.isfile(csv_file):
		# Create the CSV file with headers if it doesn't exist
		with open(csv_file, 'w', newline='') as file:
			writer = csv.writer(file)
			writer.writerow(["Task ID", "Train AUC", "Test AUC"])

	with open(csv_file, 'a', newline='') as file:
		writer = csv.writer(file)
		writer.writerow(row)


def run_picard(task_dir, task_id):

	# Read training/testing data
	train_path = os.path.join(task_dir, "train.csv")
	train_pd = pd.read_csv(train_path)

	test_path = os.path.join(task_dir, "test.csv")
	test_pd = pd.read_csv(test_path)

  # Set problem definition and fit pipelines to task
	pdef_str = PROBLEM_DEF.format(train_path=train_path, test_path=test_path, class_name=train_pd.columns[-1])
	problem_def = ProblemDefinition(pdef_str)
	wrangler = Wrangler(
			problem_definition=problem_def,
			)
	wrangler_result = wrangler.fit_predict_rank()

	train_data = wrangler.dataset(pd.read_csv(train_path))
	tr_ground_truth = wrangler.dataset(data=train_data.dataframe[[train_pd.columns[-1]]], key=DatasetKeys.GROUND_TRUTH)

	test_data = wrangler.dataset(pd.read_csv(test_path))
	te_ground_truth = wrangler.dataset(data=test_data.dataframe[[train_pd.columns[-1]]], key=DatasetKeys.GROUND_TRUTH)

	# predict on train/test data and get train/test rankings
	train_result = wrangler.predict(train_data)
	train_rankings = wrangler.ranker.rank(results=train_result, metrics=wrangler.lookup_metrics(), ground_truth=tr_ground_truth)

	test_result = wrangler.predict(test_data)
	test_rankings = wrangler.ranker.rank(results=test_result, metrics=wrangler.lookup_metrics(), ground_truth=te_ground_truth)

	# Initialize dict mapping (pipeline_des, auc_score)
	train_auc_scores = {}
	test_auc_scores = {}

	train_auc_pipelines = train_rankings["roc_auc_score"].best(5)
	for pipeline in train_auc_pipelines:
		des = pipeline.pipeline_des
		train_auc_scores[des] = pipeline.score
		test_auc_scores[des] = test_rankings["roc_auc_score"].scores_as_dict[des].score

	best_pipeline = train_auc_pipelines[0]
	train_auc_best = best_pipeline.score
	test_auc_best = test_rankings["roc_auc_score"].scores_as_dict[best_pipeline.pipeline_des].score

	#write:
	write_to_csv(task_id, train_auc_best, test_auc_best)
	return train_auc_scores, test_auc_scores, train_auc_best, test_auc_best


# run_picard("../ngautonml/examples/classification",0)
