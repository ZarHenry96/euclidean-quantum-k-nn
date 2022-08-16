import argparse
import os
import sys

from datetime import datetime

from algorithm.qknn import run_qknn


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Command line main file for the execution of the classical/quantum '
                                                 'k-NN based on the euclidean distance.')
    parser.add_argument('training_data_file', metavar='training_data_file', type=str, nargs='?',
                        help='file containing the training data (CSV file).')
    parser.add_argument('target_instance_file', metavar='target_instance_file', type=str, nargs='?',
                        help='file containing the target instance (CSV file); if multiple instances are present, only '
                             'the first one is considered.')
    parser.add_argument('k', metavar='k', type=int, nargs='?', default=5,
                        help='number of nearest neighbours (k-NN hyper-parameter)')
    parser.add_argument('--exec-type', metavar='exec_type', type=str, nargs='?', default='local_simulation',
                        help='type of execution, allowed values: classical, statevector, local_simulation,'
                             'online_simulation, quantum.')
    parser.add_argument('--encoding', metavar='encoding', type=str, nargs='?', default='extension',
                        help='type of samples encoding, allowed values: extension, translation.')
    parser.add_argument('--backend-name', metavar='backend_name', type=str, nargs='?',
                        default='ibmq_qasm_simulator|ibm_nairobi',
                        help='name of the online backend, either an online simulator or a quantum device.')
    parser.add_argument('--job-name', metavar='job_name', type=str, nargs='?', default=None,
                        help='name assigned to the online job (only for online executions).')
    parser.add_argument('--shots', metavar='shots', type=int, nargs='?', default=1024,
                        help='number of shots (for simulations and quantum executions).')
    parser.add_argument('--pseudocounts', metavar='pseudocounts', type=int, nargs='?', default=0,
                        help='pseudocounts (for each index value) for Laplace smoothing.')
    parser.add_argument('--seed-simulator', metavar='seed_simulator', type=int, nargs='?', default=None,
                        help='simulator sampling seed (only for local and online simulations).')
    parser.add_argument('--seed-transpiler', metavar='seed_transpiler', type=int, nargs='?', default=None,
                        help='transpiler seed (only for quantum executions).')
    parser.add_argument('--dist-estimates', metavar='dist_estimates', type=str, nargs='+',
                        default=['avg'], help='list of Euclidian distance estimates used for k nearest neighbors '
                        ' extraction, allowed values: zero, one, avg, diff. The classical exec-type provides the '
                        '\'exact\' values.')
    parser.add_argument('--res-dir', metavar='res_dir', type=str, nargs='?', default=None,
                        help='directory where to store the results.')
    parser.add_argument('--classical-expectation', dest='classical_expectation', action='store_const',
                        const=True, default=False, help='compute (classically) the expected indices and distances.')
    parser.add_argument('--not-verbose', dest='not_verbose', action='store_const', const=True, default=False,
                        help='no information is printed on the stdout.')
    parser.add_argument('--not-store', dest='not_store', action='store_const', const=True, default=False,
                        help='the results of the execution are not stored in memory.')
    parser.add_argument('--not-save-circuit-plot', dest='not_save_circuit_plot', action='store_const', const=True,
                        default=False, help='do not save the circuit plot.')
    args = parser.parse_args()

    if args.training_data_file is None or args.target_instance_file is None:
        print('You need to provide both a CSV file containing the training data and a CSV file containing the test'
              'instance', file=sys.stderr)
        exit(-1)

    exec_type = args.exec_type
    if exec_type not in ['classical', 'statevector', 'local_simulation', 'online_simulation', 'quantum']:
        print(f"Unknown exec type '{exec_type}'", file=sys.stderr)
        exit(-1)

    encoding = args.encoding
    if encoding not in ['extension', 'translation']:
        print(f"Unknown encoding '{encoding}'", file=sys.stderr)
        exit(-1)

    backend_name = args.backend_name
    if exec_type in ['online_simulation', 'quantum'] and backend_name == 'ibmq_qasm_simulator|ibm_nairobi':
        backend_name = backend_name.split('|')[0 if exec_type == 'online_simulation' else 1]

    job_name = args.job_name if args.job_name is not None else f'qknn_{exec_type}'

    dist_estimates = args.dist_estimates
    for dist_estimate in dist_estimates:
        if dist_estimate not in ['zero', 'one', 'avg', 'diff']:
            print(f"Unknown dist. estimate '{dist_estimate}'", file=sys.stderr)
            exit(-1)

    root_res_dir = args.res_dir if args.res_dir is not None \
        else os.path.join('./', os.path.dirname(sys.argv[0]), 'results')
    res_dir = os.path.join(root_res_dir, exec_type, encoding if exec_type != 'classical' else '',
                           datetime.now().strftime('%d-%m-%Y_%H-%M-%S'))

    (knn_indices_file, knn_files, normalized_knn_files, target_label_file), expected_knn_indices_file, \
    normalized_target_instance_file, (algorithm_execution_time, classical_expectation_time) = \
        run_qknn(args.training_data_file, args.target_instance_file, args.k, exec_type, encoding, backend_name,
                 job_name, args.shots, args.pseudocounts, args.seed_simulator, args.seed_transpiler, dist_estimates,
                 res_dir, args.classical_expectation, not args.not_verbose, not args.not_store,
                 not args.not_save_circuit_plot)

    if not args.not_verbose:
        print('\n\n' + '=' * 120 + '\n\n')

    print('Algorithm output files:')
    print('\tk-NN indices file: {}'.format(f'\n\t\t{knn_indices_file}' if knn_indices_file is not None else 'None'))
    print('\tk-NN files: {}'.format(('\n\t\t'+'\n\t\t'.join(knn_files)) if len(knn_files) > 0 else '[]'))
    print('\tnormalized k-NN files: {}'.format(
        ('\n\t\t'+'\n\t\t'.join(normalized_knn_files)) if len(normalized_knn_files) > 0 else '[]'
    ))
    print('\ttarget labels file: {}'.format(
        f'\n\t\t{target_label_file}' if target_label_file is not None else 'None')
    )

    print('\nExpected k-NN indices file: {}'.format(
        f'\n\t{expected_knn_indices_file}' if expected_knn_indices_file is not None else 'None'
    ))

    print('\nNormalized target instance file: {}'.format(
        f'\n\t{normalized_target_instance_file}' if normalized_target_instance_file is not None else 'None'
    ))

    print(f'\nExecution times:')
    print('\talgorithm execution time:    {:.10f} s'.format(algorithm_execution_time))
    print('\tclassical expectation time:  {:.10f} s'.format(classical_expectation_time))
