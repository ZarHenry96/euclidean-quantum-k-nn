import numpy as np
import os
import sys

from scipy.stats import mode
from sklearn.neighbors import NearestNeighbors

from algorithm.utils import save_data_to_json_file


def print_cknn_results(verb_to_print, k, nearest_neighbors, training_instances, training_labels, target_label,
                       file=sys.stdout):
    if file == sys.stdout:
        print()
    print(f'Classically {verb_to_print} {k} nearest neighbors for the target instance provided:', file=file)
    index_max_chars = str(len(str(len(training_instances))))
    for index, distance in zip(nearest_neighbors[1][0], nearest_neighbors[0][0]):
        element = np.array2string(training_instances[index], separator=', ')
        print(('    index: {:'+index_max_chars+'d}, distance: {:.10f}, element: {}, label: {}')
              .format(index, distance, element, training_labels[index]), file=file)

    print(f'\nClassically {verb_to_print} target instance label: {target_label}', file=file)


def save_cknn_log(res_dir, filename, verb_to_print, k, nearest_neighbors, training_instances, training_labels,
                  target_label):
    filepath = os.path.join(res_dir, f'{filename}.txt')
    with open(filepath, 'w') as log_file:
        print_cknn_results(verb_to_print, k, nearest_neighbors, training_instances, training_labels, target_label,
                           file=log_file)

    return filepath


def classical_knn(training_df, target_df, k, original_training_df, save_results_to_file, res_dir,
                  expectation=False, verbose=True):
    cl_knn_output_dir = os.path.join(res_dir, 'classical_expectation' if expectation else 'output')
    if save_results_to_file:
        os.makedirs(cl_knn_output_dir, exist_ok=True)

    features_number = len(training_df.columns) - 1

    # Prepare the data for the k-NN
    training_instances = np.array(training_df.iloc[:, :features_number])
    training_labels = np.array(training_df.iloc[:, features_number])
    target_instance = np.array([target_df.iloc[0, :features_number]])

    # Fit the model to the training data
    knn_model = NearestNeighbors(n_neighbors=k, metric='euclidean', algorithm='brute')
    knn_model.fit(training_instances)

    # Determine the k nearest neighbors, and predict the target label accordingly
    knn = knn_model.kneighbors(target_instance)
    knn_df = original_training_df.iloc[knn[1][0], :].reset_index(drop=True)
    normalized_knn_df = training_df.iloc[knn[1][0], :].reset_index(drop=True)
    target_label = mode(normalized_knn_df.iloc[:, features_number]).mode[0]

    # Show the results
    verb_to_print = 'expected' if expectation else 'predicted'
    if verbose:
        print_cknn_results(verb_to_print, k, knn, training_instances, training_labels, target_label)

    # Save the results (if needed)
    knn_indices_out_file, knn_out_file, normalized_knn_out_file, target_label_out_file = None, None, None, None
    if save_results_to_file:
        save_cknn_log(cl_knn_output_dir, 'knn_log', verb_to_print, k, knn, training_instances, training_labels,
                      target_label)

        knn_indices_out_file = save_data_to_json_file(cl_knn_output_dir, 'k_nearest_neighbors_indices',
                                                      {'exact': knn[1][0].tolist()})

        knn_out_file = os.path.join(cl_knn_output_dir, 'k_nearest_neighbors.csv')
        knn_df.to_csv(knn_out_file, index=False)

        normalized_knn_out_file = os.path.join(cl_knn_output_dir, 'normalized_k_nearest_neighbors.csv')
        normalized_knn_df.to_csv(normalized_knn_out_file, index=False)

        target_label_out_file = save_data_to_json_file(cl_knn_output_dir, 'target_label', {'exact': target_label})

    return knn_indices_out_file, knn_out_file, normalized_knn_out_file, target_label_out_file
