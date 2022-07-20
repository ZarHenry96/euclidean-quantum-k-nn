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
    nearest_neighbors = knn_model.kneighbors(target_instance)
    nearest_neighbors_df = original_training_df.iloc[nearest_neighbors[1][0], :]
    normalized_nearest_neighbors_df = training_df.iloc[nearest_neighbors[1][0], :]

    # Show the results
    verb_to_print = 'expected' if expectation else 'predicted'
    if verbose:
        print_cknn_results(verb_to_print, k, nearest_neighbors, training_instances)

    # Save the results (if needed)
    normalized_knn_filename = None
    if save_results_to_file:
        save_cknn_log(cl_knn_output_dir, 'knn_log', verb_to_print, k, nearest_neighbors, training_instances)

        knn_filename = os.path.join(cl_knn_output_dir, 'nearest_neighbors.csv')
        nearest_neighbors_df.to_csv(knn_filename, index=False)

        normalized_knn_filename = os.path.join(cl_knn_output_dir, 'normalized_nearest_neighbors.csv')
        normalized_nearest_neighbors_df.to_csv(normalized_knn_filename, index=False)

    normalized_nearest_neighbors_df.reset_index(drop=True, inplace=True)
    return normalized_nearest_neighbors_df, normalized_knn_filename
