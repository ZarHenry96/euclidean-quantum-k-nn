import argparse
import json
import os

from json_encoder import MyJSONEncoder


def collect_processed_results(root_res_dir):
    collected_res = {}

    # Iterate over 'exec_type' directories
    for exec_type in sorted(os.listdir(root_res_dir)):
        exec_type_dir = os.path.join(root_res_dir, exec_type)
        # Check if it is a directory
        if os.path.isdir(exec_type_dir):
            collected_res[exec_type] = {}

            # Iterate over 'encoding' directories
            for encoding in sorted(os.listdir(exec_type_dir)):
                encoding_dir = os.path.join(exec_type_dir, encoding)
                collected_res[exec_type][encoding] = {}

                # Iterate over 'dataset' directories
                for dataset in sorted(os.listdir(encoding_dir)):
                    dataset_dir = os.path.join(encoding_dir, dataset)
                    collected_res[exec_type][encoding][dataset] = {}

                    # Iterate over 'k value' directories
                    for k_value in sorted(os.listdir(dataset_dir)):
                        k_value_dir = os.path.join(dataset_dir, k_value)
                        collected_res[exec_type][encoding][dataset][k_value] = {}

                        # Iterate over 'round' directories
                        for round in sorted(os.listdir(k_value_dir)):
                            round_dir = os.path.join(k_value_dir, round)

                            # Load the processed results
                            exp_proc_res_filepath = os.path.join(round_dir, 'results_processed.json')
                            with open(exp_proc_res_filepath) as exp_proc_res_file:
                                exp_proc_res = json.load(exp_proc_res_file)

                                # Add the experiment results to the output dictionary
                                collected_res[exec_type][encoding][dataset][k_value][round] = exp_proc_res

    # Save the collected results to the output file
    with open(os.path.join(root_res_dir, 'collected_results.json'), 'w') as out_file:
        out_file.write(json.dumps(collected_res, cls=MyJSONEncoder, ensure_ascii=False, indent=4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for collecting the results of multiple experiments in a single'
                                                 ' file.')
    parser.add_argument('root_res_dir', metavar='root_res_dir', type=str, nargs='?', default=None,
                        help='root results directory (as defined in run_exps.sh).')
    args = parser.parse_args()

    collect_processed_results(args.root_res_dir)
