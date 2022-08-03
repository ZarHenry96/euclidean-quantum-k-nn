import json
import numpy as np
import os
import sys

from sklearn.neighbors import NearestNeighbors


def print_cknn_results(verb_to_print, k, nearest_neighbors, training_instances, file=sys.stdout):
    if file == sys.stdout:
        print()
    print(f'Classically {verb_to_print} {k} nearest neighbors for the target instance provided:', file=file)
    index_max_chars = str(len(str(len(training_instances))))
    for index, distance in zip(nearest_neighbors[1][0], nearest_neighbors[0][0]):
        element = np.array2string(training_instances[index], separator=', ')
        print(('    distance: {:.10f}, index: {:'+index_max_chars+'d}, element: {}')
              .format(distance, index, element), file=file)


def save_cknn_log(res_dir, filename, verb_to_print, k, nearest_neighbors, training_instances):
    with open(os.path.join(res_dir, f'{filename}.txt'), 'w') as log_file:
        print_cknn_results(verb_to_print, k, nearest_neighbors, training_instances, file=log_file)


def classical_knn(training_df, target_df, k, original_training_df, save_results_to_file, res_dir,
                  expectation=False, verbose=True):
    cl_knn_output_dir = os.path.join(res_dir, 'classical_expectation' if expectation else 'output')
    if save_results_to_file:
        os.makedirs(cl_knn_output_dir, exist_ok=True)

    features_number = len(training_df.columns) - 1

    # Prepare the data for the k-NN
    training_instances = np.array(training_df.iloc[:, :features_number])
    target_instance = np.array([target_df.iloc[0, :features_number]])

    # Fit the model to the training data
    knn_model = NearestNeighbors(n_neighbors=k, metric='euclidean', algorithm='brute')
    knn_model.fit(training_instances)

    # Determine the nearest neighbors
    knn = knn_model.kneighbors(target_instance)
    knn_df = original_training_df.iloc[knn[1][0], :].reset_index(drop=True)
    normalized_knn_df = training_df.iloc[knn[1][0], :].reset_index(drop=True)

    # Show the results
    verb_to_print = 'expected' if expectation else 'predicted'
    if verbose:
        print_cknn_results(verb_to_print, k, knn, training_instances)

    # Save the results (if needed)
    knn_indices_out_file, knn_out_file, normalized_knn_out_file = None, None, None
    if save_results_to_file:
        save_cknn_log(cl_knn_output_dir, 'knn_log', verb_to_print, k, knn, training_instances)

        knn_indices_out_file = os.path.join(cl_knn_output_dir, 'k_nearest_neighbors_indices.json')
        with open(knn_indices_out_file, 'w') as json_file:
            json.dump({'exact': knn[1][0].tolist()}, json_file, ensure_ascii=False, indent=4)

        knn_out_file = os.path.join(cl_knn_output_dir, 'k_nearest_neighbors.csv')
        knn_df.to_csv(knn_out_file, index=False)

        normalized_knn_out_file = os.path.join(cl_knn_output_dir, 'normalized_k_nearest_neighbors.csv')
        normalized_knn_df.to_csv(normalized_knn_out_file, index=False)

    return knn_indices_out_file, knn_out_file, normalized_knn_out_file
