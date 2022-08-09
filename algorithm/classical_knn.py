import json
import numpy as np
import os
import sys

from scipy.stats import mode
from sklearn.neighbors import NearestNeighbors


def print_cknn_results(N, sorted_indices, sorted_distances, k, normalized_knn_df, target_label, file=sys.stdout):
    if file == sys.stdout:
        print()
    print(f'Instances sorted according to the Euclidean distance w.r.t. the target instance:', file=file)
    index_max_chars = str(len(str(N)))
    for i, (index, distance) in enumerate(zip(sorted_indices, sorted_distances)):
        if i == k:
            print('\t' + '-' * 32, file=file)
        print(('\tindex: {:'+index_max_chars+'d}, distance: {:.10f}').format(index, distance), file=file)

    print(f'\nThe normalized {k} nearest neighbors are:', file=file)
    print(normalized_knn_df, file=file)

    print(f'\nThe target instance label predicted is: {target_label}', file=file)


def save_cknn_log(res_dir, filename, N, sorted_indices, sorted_distances, k, normalized_knn_df, target_label):
    filepath = os.path.join(res_dir, f'{filename}.txt')
    with open(filepath, 'w') as log_file:
        print_cknn_results(N, sorted_indices, sorted_distances, k, normalized_knn_df, target_label, file=log_file)

    return filepath


def save_distances(res_dir, filename, distances_dict):
    filepath = os.path.join(res_dir, f'{filename}.csv')
    with open(filepath, 'w') as csv_file:
        csv_file.write('index,exact_estimate\n')
        for j in range(0, len(distances_dict)):
            csv_file.write('{},{:.10f}\n'.format(j, distances_dict[j]))

    return filepath


def save_dict_data_to_json_file(res_dir, filename, dict_data):
    filepath = os.path.join(res_dir, f'{filename}.json')
    with open(filepath, 'w') as json_file:
        json.dump(dict_data, json_file, ensure_ascii=False, indent=4)

    return filepath


def run_classical_knn(training_df, target_df, k, N, d, original_training_df, res_dir, verbose=True, store_results=True):
    # Prepare the data for the k-NN
    training_instances = np.array(training_df.iloc[:, :d])
    target_instance = np.array([target_df.iloc[0, :d]])

    # Fit the model to the training data
    knn_model = NearestNeighbors(n_neighbors=N, metric='euclidean', algorithm='brute')
    knn_model.fit(training_instances)

    # Determine the k nearest neighbors, and predict the target label accordingly
    knn_model_output = knn_model.kneighbors(target_instance)
    sorted_indices, sorted_distances = knn_model_output[1][0], knn_model_output[0][0]
    knn_df = original_training_df.iloc[sorted_indices[0: k], :].reset_index(drop=True)
    normalized_knn_df = training_df.iloc[sorted_indices[0: k], :].reset_index(drop=True)
    target_label = mode(normalized_knn_df.iloc[:, d]).mode[0]

    # Show the results (if needed)
    if verbose:
        print_cknn_results(N, sorted_indices, sorted_distances, k, normalized_knn_df, target_label)

    # Save the results (if needed)
    knn_indices_out_file, knn_out_file, normalized_knn_out_file, target_label_out_file = None, None, None, None
    if store_results:
        save_cknn_log(res_dir, 'knn_log', N, sorted_indices, sorted_distances, k, normalized_knn_df, target_label)

        distances_dict = {index: distance for index, distance in zip(sorted_indices, sorted_distances)}
        save_distances(res_dir, 'knn_distances', distances_dict)

        knn_indices_out_file = save_dict_data_to_json_file(res_dir, 'k_nearest_neighbors_indices',
                                                           {'exact': sorted_indices[0: k].tolist()})

        knn_out_file = os.path.join(res_dir, 'k_nearest_neighbors.csv')
        knn_df.to_csv(knn_out_file, index=False)

        normalized_knn_out_file = os.path.join(res_dir, 'normalized_k_nearest_neighbors.csv')
        normalized_knn_df.to_csv(normalized_knn_out_file, index=False)

        target_label_out_file = save_dict_data_to_json_file(
            res_dir, 'target_label', {'exact': getattr(target_label, "tolist", lambda: target_label)()}
        )

    return knn_indices_out_file, knn_out_file, normalized_knn_out_file, target_label_out_file
