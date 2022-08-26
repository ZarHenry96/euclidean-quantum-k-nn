import json
import numpy as np
import os

from scipy.stats import mode
from sklearn.neighbors import NearestNeighbors

from algorithm.utils import print_cknn_results, save_cknn_log, print_cknn_expectation_results, \
    save_cknn_expectation_log, save_cknn_distances, save_data_to_json_file


def run_cknn(training_df, target_df, k, N, d, original_training_df, res_dir, expectation=False, verbose=True,
             store_results=True):
    # Prepare the data for the k-NN
    training_instances = np.array(training_df.iloc[:, :d])
    target_instance = np.array([target_df.iloc[0, :d]])

    # Fit the model to the training data
    knn_model = NearestNeighbors(n_neighbors=N, metric='euclidean', algorithm='brute')
    knn_model.fit(training_instances)

    # Run the model to obtain data indices (and distances) sorted according to the Euclidean distance metric
    knn_model_output = knn_model.kneighbors(target_instance)
    sorted_indices, sorted_distances = knn_model_output[1][0], knn_model_output[0][0]
    knn_indices, knn_distances = sorted_indices[0: k], sorted_distances[0: k]

    # Extract the k nearest neighbors and predict the target label accordingly (if it is not an expectation execution)
    knn_df, normalized_knn_df, target_label = None, None, None
    if not expectation:
        knn_df = original_training_df.iloc[knn_indices, :].reset_index(drop=True)
        normalized_knn_df = training_df.iloc[knn_indices, :].reset_index(drop=True)
        target_label = mode(normalized_knn_df.iloc[:, d], keepdims=False).mode

    # Show the results (if needed)
    if verbose:
        if not expectation:
            print_cknn_results(N, sorted_indices, sorted_distances, k, normalized_knn_df, target_label)
        else:
            print_cknn_expectation_results(knn_indices, knn_distances, k)

    # Save the results (if needed)
    knn_indices_out_file, knn_out_file, normalized_knn_out_file, target_label_out_file = None, None, None, None
    if store_results:
        if not expectation:
            save_cknn_log(res_dir, 'knn_log', N, sorted_indices, sorted_distances, k, normalized_knn_df, target_label)
            distances_filename = 'knn_distances'
        else:
            save_cknn_expectation_log(res_dir, 'expectation_log', knn_indices, knn_distances, k)
            distances_filename = 'distances'

        distances_dict = {index: distance for index, distance in zip(sorted_indices, sorted_distances)}
        save_cknn_distances(res_dir, distances_filename, distances_dict)

        knn_indices_out_file = save_data_to_json_file(res_dir, 'k_nearest_neighbors_indices',
                                                      {'exact': knn_indices.tolist()})

        if not expectation:
            knn_out_file = os.path.join(res_dir, 'k_nearest_neighbors.csv')
            knn_df.to_csv(knn_out_file, index=False)

            normalized_knn_out_file = os.path.join(res_dir, 'normalized_k_nearest_neighbors.csv')
            normalized_knn_df.to_csv(normalized_knn_out_file, index=False)

            target_label_out_file = save_data_to_json_file(
                res_dir, 'target_label', {'exact': getattr(target_label, "tolist", lambda: target_label)()}
            )

    return knn_indices_out_file, knn_out_file, normalized_knn_out_file, target_label_out_file
