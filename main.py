import argparse
import copy
import json
import os
import pandas as pd
import shutil
import sys
import tarfile
import time

from sklearn.model_selection import StratifiedKFold
from tqdm.contrib.concurrent import process_map

from algorithm.qknn import run_qknn
from postprocessing.process_results import process_results


def preprocess_experiment_config(config):
    if not os.path.isabs(config['dataset']):
        config['dataset'] = os.path.abspath(config['dataset'])

    if config['class_mapping'] is not None and not os.path.isabs(config['class_mapping']):
        config['class_mapping'] = os.path.abspath(config['class_mapping'])

    if not os.path.isabs(config['res_dir']):
        config['res_dir'] = os.path.abspath(config['res_dir'])

    # If it is a classical execution, overwrite the encoding field and the dist_estimates field
    if config['knn']['exec_type'] == 'classical':
        config['knn']['encoding'] = 'classical'
        config['knn']['dist_estimates'] = ['exact']


def print_config_dict(d, level=0):
    for k, v in sorted(d.items()):
        if isinstance(v, dict):
            print('\t'*level + k+':')
            print_config_dict(v, level+1)
        else:
            print('\t'*level + '{}: {}'.format(k, v))


def run_test(test_config):
    # Mute the subprocess
    sys.stdout = open(os.devnull, 'w')

    # Run the algorithm
    (predicted_knn_indices_filepath, _, _, predicted_label_filepath), expected_knn_indices_filepath, _, \
    (algorithm_exec_time, classical_exp_exec_time) = \
        run_qknn(test_config['training_data'], test_config['test_instance'], test_config['k'],
                 test_config['exec_type'], test_config['encoding'], test_config['backend_name'],
                 test_config['job_name'], test_config['shots'], test_config['pseudocounts'],
                 test_config['seed_simulator'], test_config['seed_transpiler'], test_config['dist_estimates'],
                 test_config['res_dir'], test_config['classical_expectation'], test_config['verbose'],
                 test_config['store_results'], test_config['save_circuit_plot'])

    # Retrieve the results from the output files
    predicted_knn_indices, predicted_label, expected_knn_indices = None, None, None

    with open(predicted_knn_indices_filepath) as predicted_knn_indices_file:
        predicted_knn_indices = json.load(predicted_knn_indices_file)

    with open(predicted_label_filepath) as predicted_label_file:
        predicted_label = json.load(predicted_label_file)

    if test_config['classical_expectation']:
        with open(expected_knn_indices_filepath) as expected_knn_indices_file:
            expected_knn_indices = json.load(expected_knn_indices_file)

    return predicted_knn_indices, predicted_label, expected_knn_indices, (algorithm_exec_time, classical_exp_exec_time)


def list_to_csv_field(values_list):
    return '[' + ' '.join([str(value) for value in values_list]) + ']'


def run_fold(config, dataset, train, test, i, fold_res_dir, res_file):
    print('Fold {}'.format(i))
    fold_start_time = time.time()

    # Save the training set and the test set for the current fold
    training_data_file = os.path.join(fold_res_dir, 'training_data.csv')
    dataset.iloc[train].to_csv(training_data_file, index=False)

    test_instances_file = os.path.join(fold_res_dir, 'test_instances.csv')
    dataset.iloc[test].to_csv(test_instances_file, index=False)

    # Create a directory for the test instances without label
    test_instances_no_label_dir = os.path.join(fold_res_dir, 'test_instances_no_label')
    os.makedirs(test_instances_no_label_dir)

    # Prepare test directories, files and configs
    eval_nearest_neighbors = config['eval_nearest_neighbors']
    expected_labels = []
    test_configs = []
    for j in range(0, len(test)):
        test_j_res_dir = os.path.join(fold_res_dir, 'test_{}'.format(j))
        os.makedirs(test_j_res_dir)

        test_instance_no_label_file = os.path.join(test_instances_no_label_dir, f'test_instance_{j}.csv')
        dataset.iloc[[test[j]], :-1].to_csv(test_instance_no_label_file, index=False)

        test_instance_label = dataset.iloc[test[j], -1]
        test_instance_label_filepath = os.path.join(test_j_res_dir, 'correct_label.txt')
        with open(test_instance_label_filepath, 'w') as test_instance_label_file:
            test_instance_label_file.write(str(test_instance_label))

        job_name = None if config['knn']['job_name_prefix'] is None \
            else '{}_f{}_t{}'.format(config['knn']['job_name_prefix'], i, j)
        test_config = {
            'training_data': training_data_file,
            'test_instance': test_instance_no_label_file,
            'job_name': job_name,
            'res_dir': test_j_res_dir,
            'classical_expectation': eval_nearest_neighbors,
            'verbose': False,
            'store_results': True
        }
        test_config = {**test_config, **copy.deepcopy(config['knn'])}

        expected_labels.append(test_instance_label)
        test_configs.append(test_config)

    # Parallel execution
    results = process_map(run_test, test_configs, max_workers=config['num_processes'])

    # Save the fold results
    dist_estimates = config['knn']['dist_estimates']
    for j, (exp_label, (pred_knn_indices, pred_label, exp_knn_indices, (alg_exec_time, cl_exp_exec_time))) \
            in enumerate(zip(expected_labels, results)):
        res_file.write('{},{},{},{},'.format(
            i, j, exp_label, ','.join([str(pred_label[dist_estimate]) for dist_estimate in dist_estimates]))
        )

        if eval_nearest_neighbors:
            res_file.write('{},{},'.format(
                list_to_csv_field(exp_knn_indices['exact']),
                ','.join([list_to_csv_field(pred_knn_indices[dist_estimate]) for dist_estimate in dist_estimates])
            ))

        res_file.write('{:.5f}'.format(alg_exec_time))

        if eval_nearest_neighbors:
            res_file.write(',{:.5f}'.format(cl_exp_exec_time))

        res_file.write('\n')

    # Save the execution time of the current fold
    fold_exec_time = time.time() - fold_start_time
    with open(os.path.join(fold_res_dir, 'fold_execution_time.txt'), 'w') as fold_exec_time_file:
        fold_exec_time_file.write('{:.5f} s\n'.format(fold_exec_time))

    # Compress fold results and delete the original directory
    with tarfile.open('{}.tar.gz'.format(fold_res_dir), 'w:gz') as tar_file:
        tar_file.add(fold_res_dir, arcname=os.path.basename(fold_res_dir))
    shutil.rmtree(fold_res_dir)

    print()


def run(config):
    start_time = time.time()
    
    # Show the experiment configuration
    print('Experiment Configuration\n')
    print_config_dict(config)
    print('\n')

    # Create the results directory
    res_dir = config['res_dir']
    os.makedirs(res_dir)

    # Save the experiment configuration
    with open(os.path.join(res_dir, 'exp_config.json'), 'w') as exp_config_file:
        json.dump(config, exp_config_file, ensure_ascii=False, indent=4)

    # Load the dataset
    dataset = pd.read_csv(config['dataset'], sep=',')
    shutil.copy2(config['dataset'], res_dir)

    # Copy the class mapping file (if present)
    if config['class_mapping'] is not None:
        shutil.copy2(config['class_mapping'], res_dir)

    # Instantiate some useful variables
    dist_estimates, eval_nearest_neighbors = config['knn']['dist_estimates'], config['eval_nearest_neighbors']

    # Create the results file
    res_filepath = os.path.join(res_dir, 'results.csv')
    with open(res_filepath, 'w') as res_file:
        # Prepare and write the header
        predicted_label_columns = 'predicted_label' if len(dist_estimates) == 1 \
            else ','.join([f'predicted_label_{dist_estimate}' for dist_estimate in dist_estimates])

        knn_indices_columns, cl_exp_exec_time_column = '', ''
        if eval_nearest_neighbors:
            knn_indices_columns += ',expected_knn_indices,'
            knn_indices_columns += 'predicted_knn_indices' if len(dist_estimates) == 1 \
                else ','.join([f'predicted_knn_indices_{dist_estimate}' for dist_estimate in dist_estimates])

            cl_exp_exec_time_column = ',classical_exp_exec_time'

        res_file.write('fold,test,expected_label,{}{},algorithm_exec_time{}\n'.format(
            predicted_label_columns, knn_indices_columns, cl_exp_exec_time_column
        ))

        # K-fold cross-validation
        kf = StratifiedKFold(n_splits=config['folds_number'], shuffle=True, random_state=config['k_fold_random_seed'])
        columns = len(dataset.columns)
        training_test_splits = [
            (train.tolist(), test.tolist())
            for (train, test) in kf.split(dataset.iloc[:, :-1], dataset.iloc[:, columns-1:columns])
        ]

        # Save the splits for a potential resume of the execution (not yet implemented)
        with open(os.path.join(res_dir, 'training_test_splits.json'), 'w') as tts_file:
            json.dump(training_test_splits, tts_file, ensure_ascii=False)

        # Iterate over folds
        for i, (train, test) in enumerate(training_test_splits):
            fold_i_res_dir = os.path.join(res_dir, 'fold_{}'.format(i))
            os.makedirs(fold_i_res_dir)

            run_fold(config, dataset, train, test, i, fold_i_res_dir, res_file)

    # Process the results
    print('\nResults')
    process_results(res_filepath, dist_estimates, eval_nearest_neighbors)

    # Show and save the execution time
    exec_time = time.time() - start_time
    print('\nExecution time: {:.5f} s'.format(exec_time))
    with open(os.path.join(res_dir, 'execution_time.txt'), 'w') as exec_time_file:
        exec_time_file.write('{:.5f} s\n'.format(exec_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for running experiments on the classical/quantum k-NN based '
                                                 'on the Euclidean distance.')
    parser.add_argument('config_filepath', metavar='config_filepath', type=str, nargs='?', default=None,
                        help='path of the (.json) file containing the configuration for the experiment.')
    args = parser.parse_args()

    config_filepath = args.config_filepath
    if config_filepath is not None:
        with open(config_filepath) as config_file:
            config = json.load(config_file)
            preprocess_experiment_config(config)
            run(config)
