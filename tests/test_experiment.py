import numpy as np
from code import run
import os

datasets = [
    os.path.join('data', x)
    for x in os.listdir('data')
    if '.csv' in x
]

def test_experiment_run_decision_tree():
    accuracies = {}
    for data_path in datasets:
        learner_type = 'decision_tree'
        confusion_matrix, accuracy, precision, recall, f1_measure = (
            run(data_path, learner_type, 1.0)
        )
        accuracies[data_path] = accuracy
    dataset = [dataset for dataset in datasets if 'ivy-league.csv' in dataset][0]
    assert (accuracies[dataset] > .7)


def test_experiment_run_prior_probability():
    accuracies = {}
    for data_path in datasets:
        learner_type = 'prior_probability'
        confusion_matrix, accuracy, precision, recall, f1_measure = (
            run(data_path, learner_type, 1.0)
        )
        accuracies[data_path] = accuracy
    dataset = [dataset for dataset in datasets if 'ivy-league.csv' in dataset][0]
    assert (accuracies[dataset] > .2)

def test_experiment_run_and_compare():
    for data_path in datasets:
        accuracies = {}
        learner_types = ['prior_probability', 'decision_tree']
        for learner_type in learner_types:
            accuracies[learner_type] = run(data_path, learner_type, 1.0)[1]
        if 'candy' in data_path or 'ivy' in data_path:
            assert (
                accuracies['decision_tree'] > accuracies['prior_probability']
            )
