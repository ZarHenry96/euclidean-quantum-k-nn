import argparse
import ast
import json
import numpy as np
import os
import pandas as pd


class MyJSONEncoder(json.JSONEncoder):
    def __init__(self, *args, **kwargs):
        super(MyJSONEncoder, self).__init__(*args, **kwargs)
        self.current_indent = 0
        self.current_indent_str = ""

    def encode(self, obj):
        if isinstance(obj, (list, tuple)):
            primitives_only = True
            for item in obj:
                if isinstance(item, (list, tuple, dict)):
                    primitives_only = False
                    break
            output = []
            if primitives_only:
                for item in obj:
                    output.append(json.dumps(item))
                return "[" + ", ".join(output) + "]"
            else:
                self.current_indent += self.indent
                self.current_indent_str = "".join([" " for x in range(self.current_indent)])
                for item in obj:
                    output.append(self.current_indent_str + self.encode(item))
                self.current_indent -= self.indent
                self.current_indent_str = "".join([" " for x in range(self.current_indent)])
                return "[\n" + ",\n".join(output) + "\n" + self.current_indent_str + "]"
        elif isinstance(obj, dict):
            output = []
            self.current_indent += self.indent
            self.current_indent_str = "".join([" " for x in range(self.current_indent)])
            for key, value in obj.items():
                output.append(self.current_indent_str + json.dumps(key) + ": " + self.encode(value))
            self.current_indent -= self.indent
            self.current_indent_str = "".join([" " for x in range(self.current_indent)])
            return "{\n" + ",\n".join(output) + "\n" + self.current_indent_str + "}"
        else:
            return json.dumps(obj)


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for processing the results of the experiments on the classical'
                                                 '/quantum k-NN based on the Euclidean distance.')
    parser.add_argument('res_dirs', metavar='res_dirs', type=str, nargs='+', default=None,
                        help='list of directories containing the results of the experiments.')
    args = parser.parse_args()

    last_res_dir_index = len(args.res_dirs) - 1
    for i, res_dir in enumerate(args.res_dirs):
        print('\nDirectory: \'{}\''.format(res_dir.split(os.sep)[-1]))

        # Load the experiment configuration file and process the results
        with open(os.path.join(res_dir, 'exp_config.json')) as exp_config_file:
            exp_config = json.load(exp_config_file)
            process_results(os.path.join(res_dir, 'results.csv'), exp_config['knn']['dist_estimates'],
                            exp_config['eval_nearest_neighbors'])

        print('\n' + ('=' * 67) if i != last_res_dir_index else '')
