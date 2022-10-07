import json
import numpy as np
import warnings

from scipy.stats import ranksums, mannwhitneyu, wilcoxon

warnings.filterwarnings('ignore', message='Sample size too small for normal approximation.')
warnings.filterwarnings('ignore', message='Exact p-value calculation does not work if there are zeros. Switching to '
                                          'normal approximation.')


def load_data(cltd_res_file, exec_type, encodings, datasets, k_values, avg_on_runs, dist_estimate, metric,
              partition_key):
    data = []

    # Open the collected results file
    with open(cltd_res_file) as cltdrf:
        # Get the results for the specified execution type
        exec_type_res = json.load(cltdrf)[exec_type]

        # Iterate over the encodings of interest
        encodings_for_exec_type = exec_type_res.keys() if len(encodings) == 0 else encodings
        for encoding in encodings_for_exec_type:
            encoding_data = []
            encoding_res = exec_type_res[encoding]

            # Iterate over the datasets of interest
            datasets_for_encoding = encoding_res.keys() if len(datasets) == 0 else datasets
            for i, dataset in enumerate(datasets_for_encoding):
                dataset_data = []
                dataset_res = encoding_res[dataset]

                # Iterate over the k values of interest
                k_values_for_dataset = dataset_res.keys() if len(k_values) == 0 else [f'k_{k}' for k in k_values]
                for j, k_value in enumerate(k_values_for_dataset):
                    k_value_data = []
                    k_value_res = dataset_res[k_value]

                    # Iterate over all runs
                    runs_for_k_value = k_value_res.keys()
                    for run in runs_for_k_value:
                        run_res = k_value_res[run]

                        # Get the distance estimate of interest
                        dist_estimate_for_run = list(run_res.keys())[0] if dist_estimate is None else dist_estimate

                        # Get the results for the metric of interest and append them to the list for the current k value
                        per_fold_key = 'value_per_fold' if metric == 'accuracy' else 'mean_per_fold'
                        metric_per_fold = run_res[dist_estimate_for_run][metric][per_fold_key]
                        k_value_data.append(metric_per_fold)

                    # Either average the "metric per fold" on runs or flatten the list
                    if avg_on_runs:
                        k_value_data = [np.mean(fold_metric_values) for fold_metric_values in zip(*k_value_data)]
                    else:
                        k_value_data = [item for sublist in k_value_data for item in sublist]

                    # Manage the data for the current k value depending on the partition key
                    if partition_key is None:
                        data += k_value_data
                    elif partition_key == 'dataset':
                        dataset_data += k_value_data
                    elif partition_key == 'encoding':
                        encoding_data += k_value_data
                    elif partition_key == 'k':
                        if j < len(data):
                            data[j] += k_value_data
                        else:
                            data.append(k_value_data)

                # If the partition key is the dataset, append the collected data to the correct list
                if partition_key == 'dataset':
                    if i < len(data):
                        data[i] += dataset_data
                    else:
                        data.append(dataset_data)

            # If the partition key is the encoding, append the data for the current encoding to the list
            if partition_key == 'encoding':
                data.append(encoding_data)

    return data


def flatten_list(input_list):
    flat_list = []

    for element in input_list:
        if isinstance(element, list):
            flat_list += flatten_list(element)
        else:
            flat_list.append(element)

    return flat_list


def get_adaptive_limits(first_data_list, second_data_list, percentage=0.05, decimals=4):
    all_data = flatten_list(first_data_list) + flatten_list(second_data_list)
    min_value, max_value = min(all_data), max(all_data)
    abs_diff = max_value - min_value

    return [round(min_value - abs_diff * percentage, decimals), round(max_value + abs_diff * percentage, decimals)]


def compute_statistic(statistical_test, first_data_list, second_data_list):
    if statistical_test == 'ranksums':
        statistic, p_value = ranksums(first_data_list, second_data_list)
    elif statistical_test == 'mannwhitneyu':
        statistic, p_value = mannwhitneyu(first_data_list, second_data_list)
    elif statistical_test == 'wilcoxon':
        if np.any(np.array(first_data_list) - np.array(second_data_list)):  # there is at least one different element
            statistic, p_value = wilcoxon(first_data_list, second_data_list)
        else:
            statistic, p_value = None, 1
    else:
        statistic, p_value = None, None

    return statistic, p_value, p_value < 0.05
