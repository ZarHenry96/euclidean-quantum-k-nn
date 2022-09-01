import argparse
import json
import os

from json_encoder import MyJSONEncoder


def collect_processed_results(root_res_dir, exec_times_too=True):
    collected_res = {}
    exec_times = {}

    # Iterate over 'exec_type' directories
    for exec_type in sorted(os.listdir(root_res_dir)):
        exec_type_dir = os.path.join(root_res_dir, exec_type)
        # Check if it is a directory
        if os.path.isdir(exec_type_dir):
            collected_res[exec_type] = {}
            exec_times[exec_type] = {}

            # Iterate over 'encoding' directories
            for encoding in sorted(os.listdir(exec_type_dir)):
                encoding_dir = os.path.join(exec_type_dir, encoding)
                collected_res[exec_type][encoding] = {}
                exec_times[exec_type][encoding] = {}

                # Iterate over 'dataset' directories
                for dataset in sorted(os.listdir(encoding_dir)):
                    dataset_dir = os.path.join(encoding_dir, dataset)
                    collected_res[exec_type][encoding][dataset] = {}
                    exec_times[exec_type][encoding][dataset] = {}

                    # Iterate over 'k value' directories
                    for k_value in sorted(os.listdir(dataset_dir)):
                        k_value_dir = os.path.join(dataset_dir, k_value)
                        collected_res[exec_type][encoding][dataset][k_value] = {}
                        exec_times[exec_type][encoding][dataset][k_value] = {}

                        # Iterate over 'run' directories
                        for run in sorted(os.listdir(k_value_dir)):
                            run_dir = os.path.join(k_value_dir, run)

                            # Load the processed results
                            exp_proc_res_filepath = os.path.join(run_dir, 'results_processed.json')
                            with open(exp_proc_res_filepath) as exp_proc_res_file:
                                exp_proc_res = json.load(exp_proc_res_file)

                                # Add the experiment results to the output dictionary
                                collected_res[exec_type][encoding][dataset][k_value][run] = exp_proc_res

                            # Load the execution time (if needed)
                            if exec_times_too:
                                exp_exec_time_filepath = os.path.join(run_dir, 'execution_time.txt')
                                with open(exp_exec_time_filepath) as exp_exec_time_file:
                                    exp_exec_time = float(exp_exec_time_file.readline().split()[0])

                                    # Add the execution time to the output dictionary
                                    exec_times[exec_type][encoding][dataset][k_value][run] = exp_exec_time

    # Save the collected results to the output file
    with open(os.path.join(root_res_dir, 'collected_results.json'), 'w') as out_file:
        out_file.write(json.dumps(collected_res, cls=MyJSONEncoder, ensure_ascii=False, indent=4))

    # Save the execution times to the output file (if needed)
    if exec_times_too:
        with open(os.path.join(root_res_dir, 'execution_times.json'), 'w') as out_file:
            out_file.write(json.dumps(exec_times, cls=MyJSONEncoder, ensure_ascii=False, indent=4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for collecting the processed results of multiple experiments '
                                                 'in a single file.')
    parser.add_argument('root_res_dir', metavar='root_res_dir', type=str, nargs='?', default=None,
                        help='root results directory (as defined in run_exps.sh).')
    parser.add_argument('--not-exec-times', dest='not_exec_times', action='store_const', const=True, default=False,
                        help='do not collect the execution times (the execution times are collected by default).')
    args = parser.parse_args()

    collect_processed_results(args.root_res_dir, exec_times_too=(not args.not_exec_times))
