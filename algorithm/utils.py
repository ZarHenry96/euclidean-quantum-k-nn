import json
import os
import sys

from qiskit import Aer, IBMQ


def save_data_to_json_file(res_dir, filename, data, indent=None):
    filepath = os.path.join(res_dir, f'{filename}.json')
    with open(filepath, 'w') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=indent)

    return filepath


def save_exp_config(res_dir, filename, training_data_file, target_instance_file, k, exec_type, encoding,
                    backend_name, job_name, shots, pseudocounts, seed_simulator, seed_transpiler, dist_estimates,
                    classical_expectation, verbose, save_circuit_plot):
    config = {
        'training_data': training_data_file,
        'target_data': target_instance_file,
        'k': k,
        'exec_type': exec_type,
        'encoding': encoding,
        'backend_name': backend_name,
        'job_name': job_name,
        'shots': shots,
        'pseudocounts': pseudocounts,
        'seed_simulator': seed_simulator,
        'seed_transpiler': seed_transpiler,
        'dist_estimates': dist_estimates,
        'res_dir': res_dir,
        'classical_expectation': classical_expectation,
        'verbose': verbose,
        'store_results': True,
        'save_circuit_plot': save_circuit_plot
    }

    return save_data_to_json_file(res_dir, filename, config, indent=4)


def print_cknn_results(N, sorted_indices, sorted_distances, k, normalized_knn_df, target_label, file=sys.stdout):
    if file == sys.stdout:
        print()
    print(f'Instances sorted according to the Euclidean distance w.r.t. the target instance:', file=file)
    index_max_chars = str(len(str(N)))
    for i, (index, distance) in enumerate(zip(sorted_indices, sorted_distances)):
        if i == k:
            print('\t' + '-' * 71, file=file)
        print(('\tindex: {:'+index_max_chars+'d}, distance: {:.10f}').format(index, distance), file=file)

    print(f'\nThe normalized {k} nearest neighbors are:', file=file)
    print(normalized_knn_df, file=file)

    print(f'\nThe target instance label predicted is: {target_label}', file=file)


def print_cknn_expectation_results(knn_indices, knn_distances, k, file=sys.stdout):
    if file == sys.stdout:
        print()
    print(f'Classical expectation ({k} nearest neighbors):', file=file)
    index_max_chars = str(len(str(max(knn_indices))))
    for index, distance in zip(knn_indices, knn_distances):
        print(('\tindex: {:' + index_max_chars + 'd}, distance: {:.10f}').format(index, distance), file=file)
    if file == sys.stdout:
        print()


def save_cknn_log(res_dir, filename, N, sorted_indices, sorted_distances, k, normalized_knn_df, target_label):
    filepath = os.path.join(res_dir, f'{filename}.txt')
    with open(filepath, 'w') as log_file:
        print_cknn_results(N, sorted_indices, sorted_distances, k, normalized_knn_df, target_label, file=log_file)

    return filepath


def save_cknn_expectation_log(res_dir, filename, knn_indices, knn_distances, k):
    filepath = os.path.join(res_dir, f'{filename}.txt')
    with open(filepath, 'w') as expectation_log_file:
        print_cknn_expectation_results(knn_indices, knn_distances, k, file=expectation_log_file)

    return filepath


def save_cknn_distances(res_dir, filename, distances_dict):
    filepath = os.path.join(res_dir, f'{filename}.csv')
    with open(filepath, 'w') as csv_file:
        csv_file.write('index,exact_estimate\n')
        for j in range(0, len(distances_dict)):
            csv_file.write('{},{:.10f}\n'.format(j, distances_dict[j]))

    return filepath


def select_backend(exec_type, backend_name):
    if exec_type in ['local_simulation', 'statevector']:
        # Use the Aer simulator
        backend = Aer.get_backend('aer_simulator')
    else:
        # Use IBMQ backends
        if IBMQ.active_account() is not None:
            IBMQ.disable_account()
        provider = IBMQ.load_account()
        backend = provider.get_backend(backend_name)

    return backend


def save_data_to_txt_file(res_dir, filename, data, list_on_rows=False):
    filepath = os.path.join(res_dir, f'{filename}.txt')
    with open(filepath, 'w') as txt_file:
        if list_on_rows:
            for entry in data:
                txt_file.write(f'{entry}\n')
        else:
            txt_file.write(f'{data}')

    return filepath


def print_qknn_results(p0, p1, index_qubits_num, index_and_ancillary_joint_p, euclidean_distances, dist_estimates,
                       sorted_indices_lists, k, normalized_knn_dfs, target_labels, file=sys.stdout):
    if file == sys.stdout:
        print()
    print('P(ancillary_qubit_state):', file=file)
    print('\tP(0) = {:.10f}    P(1) = {:.10f}'.format(p0, p1), file=file)

    index_dec_max_chars = str(len(str(2 ** index_qubits_num - 1)))

    print('\nP(index_state,ancillary_qubit_state):', file=file)
    for index_bin_state, joint_p in index_and_ancillary_joint_p.items():
        index_dec_state = int(index_bin_state, 2)
        print(('\tindex state {:'+index_dec_max_chars+'d} (binary: {})')
              .format(index_dec_state, index_bin_state), file=file)
        print(('\t\tP({:'+index_dec_max_chars+'d},0) = {:.10f}    P({:'+index_dec_max_chars+'d},1) = {:.10f}')
              .format(index_dec_state, joint_p['0'], index_dec_state, joint_p['1']), file=file)

    print('\nEuclidean distances (w/o nonexistent index states):', file=file)
    for index_dec_state, estimated_distances in euclidean_distances.items():
        index_bin_state = ('{0:0' + str(index_qubits_num) + 'b}').format(index_dec_state)
        print(('\tindex state {:'+index_dec_max_chars+'d} (binary: {})')
              .format(index_dec_state, index_bin_state), file=file)
        formatting_strings = [f"'{dist_estimate}'" + " estimate = {:.10f}"
                              for dist_estimate in estimated_distances.keys()]
        print('\t\t' + '    '.join(formatting_strings).format(*estimated_distances.values()), file=file)

    for dist_estimate, sorted_indices, normalized_knn_df, target_label in \
            zip(dist_estimates, sorted_indices_lists, normalized_knn_dfs, target_labels):
        print(f"\n\nInstances sorted according to the '{dist_estimate}' distance estimate  (w/o nonexistent "
              f"index states):", file=file)
        for i, index_dec_state in enumerate(sorted_indices):
            estimated_distance = euclidean_distances[index_dec_state][dist_estimate]
            if i == k:
                print('\t' + '-' * 82, file=file)
            print(('\tindex state: {:'+index_dec_max_chars+'d}, distance: {:.10f}')
                  .format(index_dec_state, estimated_distance), file=file)

        print(f"\nThe normalized {k} nearest neighbors for the target instance provided (according to the "
              f"'{dist_estimate}' distance estimate) are:", file=file)
        print(normalized_knn_df, file=file)

        print(f'\nThe target instance label predicted is: {target_label}', file=file)


def save_qknn_log(res_dir, filename, p0, p1, index_qubits_num, index_and_ancillary_joint_p, euclidean_distances,
                  dist_estimates, sorted_indices_lists, k, normalized_knn_dfs, target_labels):
    filepath = os.path.join(res_dir, f'{filename}.txt')
    with open(filepath, 'w') as log_file:
        print_qknn_results(p0, p1, index_qubits_num, index_and_ancillary_joint_p, euclidean_distances, dist_estimates,
                           sorted_indices_lists, k, normalized_knn_dfs, target_labels, file=log_file)

    return filepath


def save_qknn_probabilities_and_distances(res_dir, filename, index_and_ancillary_joint_p, euclidean_distances, N):
    filepath = os.path.join(res_dir, f'{filename}.csv')
    with open(filepath, 'w') as csv_file:
        csv_file.write(
            'index,binary_index_state,P(index;0),P(index;1),'
            'zero_dist_estimate,one_dist_estimate,avg_dist_estimate,diff_dist_estimate\n'
        )
        for index_bin_state, joint_p in index_and_ancillary_joint_p.items():
            index_dec_state = int(index_bin_state, 2)
            significant_index = index_dec_state < N
            csv_file.write('{},{},{},{},{},{},{},{}\n'.format(
                index_dec_state, index_bin_state, round(joint_p['0'], 10), round(joint_p['1'], 10),
                round(euclidean_distances[index_dec_state]['zero'], 10)
                if significant_index and 'zero' in euclidean_distances[index_dec_state] else '',
                round(euclidean_distances[index_dec_state]['one'], 10)
                if significant_index and 'one' in euclidean_distances[index_dec_state] else '',
                round(euclidean_distances[index_dec_state]['avg'], 10)
                if significant_index and 'avg' in euclidean_distances[index_dec_state] else '',
                round(euclidean_distances[index_dec_state]['diff'], 10)
                if significant_index and 'diff' in euclidean_distances[index_dec_state] else ''
            ))

    return filepath
