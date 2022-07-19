import json
import os
import sys

from qiskit import Aer, IBMQ


def save_exp_config(res_dir, filename, training_data_file, target_instance_file, k, exec_type, encoding,
                    backend_name, job_name, shots, sorting_dist_estimate, verbose, save_circuit_plot):
    with open(os.path.join(res_dir, f'{filename}.json'), 'w') as json_config:
        config = {
            'training_data': training_data_file,
            'target_data': target_instance_file,
            'k': k,
            'exec_type': exec_type,
            'encoding': encoding,
            'backend_name': backend_name,
            'job_name': job_name,
            'shots': shots,
            'sorting_dist_estimate': sorting_dist_estimate,
            'res_dir': res_dir,
            'verbose': verbose,
            'store_results': True,
            'save_circuit_plot': save_circuit_plot
        }
        json.dump(config, json_config, ensure_ascii=False, indent=4)


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
    with open(os.path.join(res_dir, f'{filename}.txt'), 'w') as txt_file:
        if list_on_rows:
            for entry in data:
                txt_file.write(f'{entry}\n')
        else:
            txt_file.write(f'{data}')


def print_qknn_results(p0, p1, index_qubits, index_and_ancillary_joint_p, euclidean_distances, sorting_dist_estimate,
                       sorted_indices, k, normalized_nearest_neighbors_df, file=sys.stdout):
    if file == sys.stdout:
        print()
    print('P(ancillary_qubit_state):', file=file)
    print('\tP(0) = {:.15f}\tP(1) = {:.15f}'.format(p0, p1), file=file)

    index_decimal_max_chars = str(len(str(2 ** index_qubits - 1)))

    print('\nP(index_state,ancillary_qubit_state):', file=file)
    for index_binary_state, joint_p in index_and_ancillary_joint_p.items():
        index_decimal_state = int(index_binary_state, 2)
        print(('\tindex state {:'+index_decimal_max_chars+'d} (binary: {})')
              .format(index_decimal_state, index_binary_state), file=file)
        print(('\t\tP({:'+index_decimal_max_chars+'d},0) = {:.15f}    P({:'+index_decimal_max_chars+'d},1) = {:.15f}')
              .format(index_decimal_state, joint_p["0"], index_decimal_state, joint_p["1"]), file=file)

    print('Euclidean distances (w/o nonexistent index states):', file=file)
    for index_decimal_state, distances in euclidean_distances.items():
        index_binary_state = ('{0:0' + str(index_qubits) + 'b}').format(index_decimal_state)
        print(('\tindex state {:'+index_decimal_max_chars+'d} (binary: {})')
              .format(index_decimal_state, index_binary_state), file=file)
        print("\t\t'zero' estimate = {:.15f}    'one' estimate = {:.15f}    'avg' estimate = {:.15f}"
              .format(distances["zero"], distances["one"], distances["avg"]), file=file)

    print(f"\nInstances sorted according to the '{sorting_dist_estimate}' distance estimate  (w/o nonexistent index"
          " states):", file=file)
    for i, index_decimal_state in enumerate(sorted_indices):
        distance_value = euclidean_distances[index_decimal_state][sorting_dist_estimate]
        if i == k:
            print('\t' + '-' * 34, file=file)
        print(('\tindex state {:'+index_decimal_max_chars+'d}: {:.15f}')
              .format(index_decimal_state, distance_value), file=file)

    print(f'\nThe normalized {k} nearest neighbors for the target instance provided are:', file=file)
    print(normalized_nearest_neighbors_df, file=file)


def save_qknn_log(res_dir, filename, p0, p1, index_qubits, index_and_ancillary_joint_p, euclidean_distances,
                  sorting_dist_estimate, sorted_indices, k, normalized_nearest_neighbors_df):
    with open(os.path.join(res_dir, f'{filename}.txt'), 'w') as log_file:
        print_qknn_results(p0, p1, index_qubits, index_and_ancillary_joint_p, euclidean_distances,
                           sorting_dist_estimate, sorted_indices, k, normalized_nearest_neighbors_df, file=log_file)


def save_probabilities_and_distances(res_dir, filename, index_and_ancillary_joint_p, euclidean_distances):
    with open(os.path.join(res_dir, f'{filename}.csv'), 'w') as csv_file:
        csv_file.write(
            'index,binary_index_state,P(index;0),P(index;1),zero_dist_estimate,one_dist_estimate,avg_dist_estimate\n'
        )
        for index_binary_state, joint_p in index_and_ancillary_joint_p.items():
            index_decimal = int(index_binary_state, 2)
            distance_estimated = index_decimal in euclidean_distances
            csv_file.write('{},{},{},{},{},{},{}\n'.format(
                index_decimal, index_binary_state, joint_p['0'], joint_p['1'],
                euclidean_distances[index_decimal]['zero'] if distance_estimated else None,
                euclidean_distances[index_decimal]['one'] if distance_estimated else None,
                euclidean_distances[index_decimal]['avg'] if distance_estimated else None
            ))
