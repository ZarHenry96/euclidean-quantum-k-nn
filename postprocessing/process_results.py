import argparse
import ast
import json
import numpy as np
import os
import pandas as pd

from postprocessing.json_encoder import MyJSONEncoder


def jaccard_index(list_1, list_2):
    s1, s2 = set(list_1), set(list_2)

    return len(s1.intersection(s2)) / len(s1.union(s2))


def process_results(res_filepath, dist_estimates, eval_nearest_neighbors):
    processed_results = {}

    # Load the results
    results_df = pd.read_csv(res_filepath, sep=',')

    # Iterate over distance estimates
    column_suffix, print_prefix, jaccard_index_columns = '', '', []
    for dist_estimate in dist_estimates:
        processed_results[dist_estimate] = {}
        if len(dist_estimates) > 1:
            column_suffix = f'_{dist_estimate}'
            print_prefix = '\t'
            print(f'\nDistance estimate: \'{dist_estimate}\'')

        # Compute and show the accuracy per fold and the mean accuracy over folds (with std)
        accuracy_per_fold = results_df.groupby('fold')\
            .apply(lambda x: (x['expected_label'] == x[f'predicted_label{column_suffix}']).sum() / len(x))

        print(f'\n{print_prefix}Accuracy per fold:')
        for fold, accuracy in accuracy_per_fold.iteritems():
            print('{}\tFold: {}, accuracy: {:.5f}'.format(print_prefix, fold, accuracy))

        mean_accuracy_per_fold, accuracy_per_fold_std = np.mean(accuracy_per_fold), np.std(accuracy_per_fold)
        print('\n{}Mean accuracy per fold: {:.5f} +- {:.5f}'.format(
            print_prefix, mean_accuracy_per_fold, accuracy_per_fold_std)
        )

        # Insert the values into the output data structure
        processed_results[dist_estimate]['accuracy'] = {
            'value_per_fold': list(accuracy_per_fold.values),
            'mean': mean_accuracy_per_fold,
            'std': accuracy_per_fold_std
        }

        # Evaluate the quality of the extracted nearest neighbors (if needed)
        if eval_nearest_neighbors:
            # Compute the Jaccard index for each test instance
            jaccard_index_column = f'jaccard_index{column_suffix}'
            results_df[jaccard_index_column] = results_df.apply(
                lambda x: jaccard_index(
                    ast.literal_eval(x['expected_knn_indices'].replace(' ', ',')),
                    ast.literal_eval(x[f'predicted_knn_indices{column_suffix}'].replace(' ', ','))
                ), axis=1
            )

            # Compute and show the mean Jaccard index per fold (with std)
            jaccard_index_stats_per_fold = results_df.groupby('fold').apply(
                lambda x: (np.mean(x[jaccard_index_column]), np.std(x[jaccard_index_column]))
            )

            print(f'\n{print_prefix}Mean Jaccard index per fold:')
            for fold, (mean, std) in jaccard_index_stats_per_fold.iteritems():
                print('{}\tFold: {}, mean Jaccard index: {:.5f} +- {:.5f}'.format(print_prefix, fold, mean, std))

            #  Compute and show the overall mean Jaccard index (with std)
            mean_jaccard_index, jaccard_index_std = \
                np.mean(results_df[jaccard_index_column]), np.std(results_df[jaccard_index_column])
            print('\n{}Mean Jaccard index overall: {:.5f} +- {:.5f}'.format(
                print_prefix, mean_jaccard_index, jaccard_index_std)
            )

            # Insert the values into the output data structure
            mean_jaccard_index_per_fold, jaccard_index_per_fold_std = zip(*jaccard_index_stats_per_fold.values)
            processed_results[dist_estimate]['jaccard_index'] = {
                'mean_per_fold': mean_jaccard_index_per_fold,
                'std_per_fold': jaccard_index_per_fold_std,
                'mean': mean_jaccard_index,
                'std': jaccard_index_std
            }

            jaccard_index_columns.append(jaccard_index_column)

    # Save the processed results
    out_dir = os.path.dirname(res_filepath)
    with open(os.path.join(out_dir, 'results_processed.json'), 'w') as proc_res_file:
        proc_res_file.write(json.dumps(processed_results, cls=MyJSONEncoder, ensure_ascii=False, indent=4))

    if eval_nearest_neighbors:
        results_df.loc[:, ['fold', 'test'] + jaccard_index_columns].to_csv(
            os.path.join(out_dir, 'results_jaccard.csv'), index=False
        )
