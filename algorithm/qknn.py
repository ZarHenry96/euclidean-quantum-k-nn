import math
import numpy as np
import os
import pandas as pd
import shutil
import sys
import time

from matplotlib import pyplot as plt
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit, execute
from qiskit.tools.monitor import job_monitor
from scipy.stats import mode

from algorithm.classical_knn import run_cknn
from algorithm.utils import save_exp_config, select_backend, save_data_to_txt_file, save_data_to_json_file, \
    print_qknn_results, save_qknn_log, save_qknn_probabilities_and_distances


def load_and_normalize_data(training_data_file, target_instance_file, res_input_dir, verbose=True, store=True):
    training_df = pd.read_csv(training_data_file, sep=',')
    target_df = pd.read_csv(target_instance_file, sep=',')

    # Copy the input files inside the results directory (if needed)
    if store:
        shutil.copy2(training_data_file, res_input_dir)
        shutil.copy2(target_instance_file, res_input_dir)

    if (len(training_df.columns)-1) != len(target_df.columns):
        print('Error: dimensionality of training data and target instance do not match!', file=sys.stderr)
        exit(-1)

    features_number = len(training_df.columns) - 1
    sqrt_features_number = math.sqrt(features_number)

    # Compute the min-max average value and the range for each attribute
    training_min_max_avg = \
        (training_df.iloc[:, :features_number].min() + training_df.iloc[:, :features_number].max()) / 2
    training_range = training_df.iloc[:, :features_number].max() - training_df.iloc[:, :features_number].min()

    # Replace the zero ranges with 1 in order to avoid divisions by 0
    zero_range_columns = np.nonzero(~(training_range.to_numpy() != 0))[0]
    if len(zero_range_columns) > 0:
        training_range.iloc[zero_range_columns] = [1 for _ in range(0, len(zero_range_columns))]

    # Normalize the target instance (and clip attributes outside the desired range)
    target_df.iloc[0, :features_number] = \
        (target_df.iloc[0, :features_number] - training_min_max_avg) / (training_range * sqrt_features_number)
    lower_bound, upper_bound = -1 / (2 * sqrt_features_number), 1 / (2 * sqrt_features_number)
    for attribute_index, attribute_val in enumerate(target_df.iloc[0, 0:features_number]):
        target_df.iloc[0, attribute_index] = max(min(attribute_val, upper_bound), lower_bound)

    # Normalize the training data
    training_df.iloc[:, :features_number] = \
        (training_df.iloc[:, :features_number] - training_min_max_avg) / (training_range * sqrt_features_number)

    # Show the normalized data (if needed)
    if verbose:
        print(f'Normalized training dataset:\n{training_df}')
        print(f'\nNormalized target instance:\n{target_df}\n')

    # Save the normalized data (if needed)
    normalized_target_instance_file = None
    if store:
        normalized_training_data_file = \
            os.path.join(res_input_dir, 'normalized_{}'.format(os.path.basename(training_data_file)))
        training_df.to_csv(normalized_training_data_file, index=False)

        normalized_target_instance_file = \
            os.path.join(res_input_dir, 'normalized_{}'.format(os.path.basename(target_instance_file)))
        target_df.to_csv(normalized_target_instance_file, index=False)

    return training_df, target_df, normalized_target_instance_file


def build_qknn_circuit(training_df, target_df, N, d, encoding, exec_type):
    # Compute the circuit size
    cnot_swap_circuit_qubits_num = 2
    index_qubits_num = math.ceil(math.log2(N))
    features_qubits_num = math.ceil(math.log2(2 * d + (3 if encoding == 'extension' else 4)))
    qubits_num = cnot_swap_circuit_qubits_num + index_qubits_num + features_qubits_num
    c_bits_num = 1 + index_qubits_num

    # Create a Quantum Circuit acting on the q register
    qr = QuantumRegister(qubits_num, 'q')
    cr = ClassicalRegister(c_bits_num, 'c')
    circuit = QuantumCircuit(qr, cr)

    # Data structures for the joint initialization of the target CNOT-SWAP qubit, the index and features registers
    init_qubits_num = 1 + index_qubits_num + features_qubits_num
    amplitudes = np.zeros(2 ** init_qubits_num)
    amplitudes_base_value = math.sqrt(1 / (2 * N))

    # Encoding variables
    if encoding == 'extension':
        multiplication_factor, features_offset, translation_feature_abs_value = math.sqrt(4 / 3), 0, 0
    else:
        multiplication_factor, features_offset, translation_feature_abs_value = 1, 1, 0.5

    # Training data and target instance norms
    training_norms = []
    target_norm = np.linalg.norm(target_df.iloc[0, 0:d])

    # Compute the training data amplitudes
    for instance_index, row in training_df.iterrows():
        training_norms.append(np.linalg.norm(row[0:d]))

        # Training instance features (two times)
        for i in range(0, 2 * d):
            index = 2 * instance_index + (2 ** (index_qubits_num + 1)) * i
            amplitudes[index] = amplitudes_base_value * multiplication_factor * row[i % d]

        # Training instance norm
        index = 2 * instance_index + (2 ** (index_qubits_num + 1)) * (2 * d)
        amplitudes[index] = amplitudes_base_value * multiplication_factor * training_norms[-1]

        # Translation feature (for translation encoding)
        if encoding == 'translation':
            index = 2 * instance_index + (2 ** (index_qubits_num + 1)) * (2 * d + features_offset)
            amplitudes[index] = amplitudes_base_value * translation_feature_abs_value

        # Zero value
        index = 2 * instance_index + (2 ** (index_qubits_num + 1)) * (2 * d + features_offset + 1)
        amplitudes[index] = amplitudes_base_value * 0

        # Residual (for unitary norm)
        index = 2 * instance_index + (2 ** (index_qubits_num + 1)) * (2 * d + features_offset + 2)
        amplitudes[index] = amplitudes_base_value * math.sqrt(
            1 - (3 * multiplication_factor**2 * training_norms[-1]**2 + translation_feature_abs_value**2)
        )

    # Compute the target instance amplitudes
    for j in range(0, N):
        # Target instance features with sign flipped (two times)
        for i in range(0, 2 * d):
            index = 1 + 2 * j + (2 ** (index_qubits_num + 1)) * i
            amplitudes[index] = amplitudes_base_value * multiplication_factor * (-target_df.iloc[0, i % d])

        # Corresponding training instance norm
        index = 1 + 2 * j + (2 ** (index_qubits_num+1)) * (2 * d)
        amplitudes[index] = amplitudes_base_value * multiplication_factor * training_norms[j]

        # Translation feature (for translation encoding)
        if encoding == 'translation':
            index = 1 + 2 * j + (2 ** (index_qubits_num + 1)) * (2 * d + features_offset)
            amplitudes[index] = amplitudes_base_value * (-translation_feature_abs_value)

        # Residual (for unitary norm)
        index = 1 + 2 * j + (2 ** (index_qubits_num + 1)) * (2 * d + features_offset + 1)
        amplitudes[index] = amplitudes_base_value * math.sqrt(
            1 - (2 * multiplication_factor**2 * target_norm**2 + multiplication_factor**2 * training_norms[j]**2 +
                 translation_feature_abs_value**2)
        )

        # Zero value
        index = 1 + 2 * j + (2 ** (index_qubits_num + 1)) * (2 * d + features_offset + 2)
        amplitudes[index] = amplitudes_base_value * 0

    # Set all "target CNOT-SWAP qubit + index_register + features_register" amplitudes
    target_cnot_swap_qubit = 1
    circuit.initialize(amplitudes, qr[target_cnot_swap_qubit: target_cnot_swap_qubit + init_qubits_num])

    # Add the CNOT-SWAP gates
    circuit.h(qr[0])
    circuit.cnot(qr[0], qr[1])
    circuit.h(qr[0])

    # Either save the final statevector or measure the qubits states (depending on the execution type)
    if exec_type == 'statevector':
        circuit.save_statevector()
    else:
        # Measure the control CNOT-SWAP qubit
        circuit.measure(qr[0], cr[0])
        circuit.barrier(qr[0: qubits_num])
        # Measure the index register
        first_index_qubit = 2
        circuit.measure(qr[first_index_qubit: first_index_qubit + index_qubits_num], cr[1: 1 + index_qubits_num])

    return circuit, qubits_num, index_qubits_num, features_qubits_num, target_norm


def get_probabilities_from_statevector(statevector, index_qubits_num):
    # Ancillary qubit
    p0, p1 = statevector.probabilities([0])

    # Joint index and ancillary qubits
    index_and_ancillary_joint_p = {}
    first_index_qubit = 2
    joint_p = statevector.probabilities([0] + list(range(first_index_qubit, first_index_qubit + index_qubits_num)))
    for j in range(0, (2 ** index_qubits_num)):
        index_state = ('{0:0' + str(index_qubits_num) + 'b}').format(j)
        index_and_ancillary_joint_p[index_state] = {'0': joint_p[2 * j], '1': joint_p[2 * j + 1]}

    return p0, p1, index_and_ancillary_joint_p


def get_probabilities_from_counts(counts, index_qubits_num, shots, pseudocounts, N):
    # Prepare data structures
    p0, p1, index_and_ancillary_joint_p = 0.0, 0.0, {}
    for j in range(0, (2 ** index_qubits_num)):
        index_state = ('{0:0' + str(index_qubits_num) + 'b}').format(j)
        index_and_ancillary_joint_p[index_state] = {'0': 0.0, '1': 0.0}

    # Process counts
    smoothed_total_counts = shots + pseudocounts * 2 * N
    for measured_dec_state in range(0, (2 ** (index_qubits_num + 1))):
        measured_state = ('{0:0' + str(index_qubits_num + 1) + 'b}').format(measured_dec_state)
        index_state = measured_state[0: -1]
        index_dec_state = int(index_state, 2)
        ancillary_state = measured_state[-1]

        state_counts = counts.get(measured_state, 0)
        smoothed_state_counts = state_counts + (pseudocounts if index_dec_state < N else 0)
        smoothed_state_probability = smoothed_state_counts / smoothed_total_counts

        if ancillary_state == '0':
            p0 += smoothed_state_probability
        else:
            p1 += smoothed_state_probability

        index_and_ancillary_joint_p[index_state][ancillary_state] += smoothed_state_probability

    return p0, p1, index_and_ancillary_joint_p


def get_sqrt_argument_from_scalar_product(scalar_product, squared_target_norm, encoding):
    if encoding == 'extension':
        sqrt_arg = (3 / 4) * scalar_product + squared_target_norm
    else:
        sqrt_arg = scalar_product + (1 / 4) + squared_target_norm

    return max(min(sqrt_arg, 1.0), 0.0)


def extract_euclidean_distances(index_and_ancillary_joint_p, dist_estimates, N, index_qubits_num, squared_target_norm,
                                encoding):
    euclidean_distances = {}

    avg = 'avg' in dist_estimates
    diff = 'diff' in dist_estimates
    one = 'one' in dist_estimates or avg
    zero = 'zero' in dist_estimates or avg

    for j in range(0, N):
        index_state = ('{0:0' + str(index_qubits_num) + 'b}').format(j)
        joint_j_0_p = index_and_ancillary_joint_p[index_state]['0']
        joint_j_1_p = index_and_ancillary_joint_p[index_state]['1']

        euclidean_distances[j] = {}

        if zero:
            scalar_prod_0 = 2 * N * joint_j_0_p - 1
            sqrt_arg_0 = get_sqrt_argument_from_scalar_product(scalar_prod_0, squared_target_norm, encoding)
            euclidean_distances[j]['zero'] = math.sqrt(sqrt_arg_0)

        if one:
            scalar_prod_1 = 1 - 2 * N * joint_j_1_p
            sqrt_arg_1 = get_sqrt_argument_from_scalar_product(scalar_prod_1, squared_target_norm, encoding)
            euclidean_distances[j]['one'] = math.sqrt(sqrt_arg_1)

        if avg:
            euclidean_distances[j]['avg'] = (euclidean_distances[j]['zero'] + euclidean_distances[j]['one']) / 2

        if diff:
            scalar_prod_diff = N * (joint_j_0_p - joint_j_1_p)
            sqrt_arg_diff = get_sqrt_argument_from_scalar_product(scalar_prod_diff, squared_target_norm, encoding)
            euclidean_distances[j]['diff'] = math.sqrt(sqrt_arg_diff)

    return euclidean_distances


def run_qknn(training_data_file, target_instance_file, k, exec_type, encoding, backend_name, job_name, shots,
             pseudocounts, seed_simulator, seed_transpiler, dist_estimates, res_dir, classical_expectation=False,
             verbose=True, store_results=True, save_circuit_plot=True):
    start_time = time.time()

    # Prepare the results directories
    res_c_expectation_dir = os.path.join(res_dir, 'classical_expectation')
    res_input_dir = os.path.join(res_dir, 'input')
    res_output_dir = os.path.join(res_dir, 'output')
    if store_results:
        os.makedirs(res_dir, exist_ok=True)
        if classical_expectation:
            os.makedirs(res_c_expectation_dir)
        os.makedirs(res_input_dir)
        os.makedirs(res_output_dir)

        # Save the experiment configuration
        save_exp_config(res_dir, 'config', training_data_file, target_instance_file, k, exec_type, encoding,
                        backend_name, job_name, shots, pseudocounts, seed_simulator, seed_transpiler, dist_estimates,
                        classical_expectation, verbose, save_circuit_plot)

    # Get the original training dataframe
    original_training_df = pd.read_csv(training_data_file, sep=',')

    # Load and normalize the input (training and target) data
    training_df, target_df, normalized_target_instance_file = \
        load_and_normalize_data(training_data_file, target_instance_file, res_input_dir, verbose=verbose,
                                store=store_results)
    N, d = len(training_df), len(training_df.columns) - 1

    # Compute the classical expectation (if needed)
    classical_expectation_start_time = time.time()
    expected_knn_indices_out_file = None
    if classical_expectation:
        expected_knn_indices_out_file, _, _, _ = \
            run_cknn(training_df, target_df, k, N, d, original_training_df, res_c_expectation_dir,
                     expectation=True, verbose=verbose, store_results=store_results)

    # Compute the time required by the classical expectation
    classical_expectation_execution_time = time.time() - classical_expectation_start_time

    # If it is a classical execution, run the classical k-NN and exit
    if exec_type == 'classical':
        knn_indices_out_file, knn_out_file, normalized_knn_out_file, target_label_out_file = \
            run_cknn(training_df, target_df, k, N, d, original_training_df, res_output_dir,
                     expectation=False, verbose=verbose, store_results=store_results)

        knn_out_files = [knn_out_file] if knn_out_file is not None else []
        normalized_knn_out_files = [normalized_knn_out_file] if normalized_knn_out_file is not None else []

        # Compute the execution time of the algorithm w/o the time required by the classical expectation
        algorithm_execution_time = (time.time() - start_time) - classical_expectation_execution_time

        return (knn_indices_out_file, knn_out_files, normalized_knn_out_files, target_label_out_file), \
            expected_knn_indices_out_file, normalized_target_instance_file, \
            (algorithm_execution_time, classical_expectation_execution_time)

    # Select the backend for the execution
    backend = select_backend(exec_type, backend_name)

    # Build the quantum circuit
    circuit, qubits_num, index_qubits_num, features_qubits_num, target_norm = \
        build_qknn_circuit(training_df, target_df, N, d, encoding, exec_type)

    # Draw and save the circuit (if needed)
    if store_results and save_circuit_plot:
        circuit_filepath = os.path.join(res_dir, 'qknn_circuit.png')
        circuit.draw(output='mpl', filename=circuit_filepath, fold=-1)
        plt.close()

        if verbose:
            print('\nThe circuit plot is available at: {}\n'.format(circuit_filepath))

    # Execute the job
    job = None
    if exec_type == 'statevector':
        job = execute(circuit, backend)
    elif exec_type == 'local_simulation':
        job = execute(circuit, backend, shots=shots, seed_simulator=seed_simulator)
    else:
        if exec_type == 'online_simulation':
            job = execute(circuit, backend, shots=shots, seed_simulator=seed_simulator, job_name=job_name)
        elif exec_type == 'quantum':
            job = execute(circuit, backend, shots=shots, seed_transpiler=seed_transpiler, job_name=job_name)

        # Show and save the online Job information (if needed)
        if verbose:
            print(f'\nJob ID: {job.job_id()}')
        if store_results:
            job_info_list = [job.job_id(), job_name, backend_name, shots]
            save_data_to_txt_file(res_dir, 'qknn_job_info', job_info_list, list_on_rows=True)

        # Monitor the online Job execution
        job_monitor(job)

    # Get the results
    result = job.result()

    # Process the results depending on the execution modality
    if exec_type == 'statevector':
        output_statevector = result.get_statevector(circuit, decimals=15)
        if verbose:
            print(f'\nResults\n\nOutput state vector:\n{list(output_statevector)}')
        if store_results:
            save_data_to_txt_file(res_output_dir, 'qknn_output_state_vector', list(output_statevector))

        # Process the output statevector
        p0, p1, index_and_ancillary_joint_p = \
            get_probabilities_from_statevector(output_statevector, index_qubits_num)
    else:
        counts = result.get_counts(circuit)
        sorted_counts = {i: counts[i] for i in sorted(counts.keys())}

        # Show and save the counts (if needed)
        if verbose:
            print('\nResults\n\nCircuit output counts (w/o Laplace smoothing): {}'.format(sorted_counts))
            print('\n[Shots = {}, Pseudocounts (per index state) = {}]'.format(shots, pseudocounts))
        if store_results:
            save_data_to_json_file(res_output_dir, 'qknn_counts', sorted_counts, indent=4)

        # Process counts
        p0, p1, index_and_ancillary_joint_p = \
            get_probabilities_from_counts(sorted_counts, index_qubits_num, shots, pseudocounts, N)

    # Extract the distances from the probability values
    euclidean_distances = extract_euclidean_distances(index_and_ancillary_joint_p, dist_estimates,
                                                      N, index_qubits_num, target_norm ** 2, encoding)

    # Initialize some useful variables
    sorted_indices_lists, knn_dfs, normalized_knn_dfs, target_labels = [], [], [], []

    # Get the k nearest neighbors based on the specified distance estimates, and compute the target label accordingly
    for dist_estimate in dist_estimates:
        sorted_indices = [
            index
            for index, _ in sorted(euclidean_distances.items(), key=lambda x: (round(x[1][dist_estimate], 10), x[0]))
        ]
        sorted_indices_lists.append(sorted_indices)

        knn_dfs.append(original_training_df.iloc[sorted_indices[0: k], :].reset_index(drop=True))
        normalized_knn_df = training_df.iloc[sorted_indices[0: k], :].reset_index(drop=True)
        normalized_knn_dfs.append(normalized_knn_df)

        target_labels.append(mode(normalized_knn_df.iloc[:, d], keepdims=False).mode)

    # Show the results (if needed)
    if verbose:
        print_qknn_results(p0, p1, index_qubits_num, index_and_ancillary_joint_p, euclidean_distances, dist_estimates,
                           sorted_indices_lists, k, normalized_knn_dfs, target_labels)

    # Store the results (if needed)
    knn_indices_out_file, knn_out_files, normalized_knn_out_files, target_label_out_file = None, [], [], None
    if store_results:
        save_qknn_log(res_output_dir, 'qknn_log', p0, p1, index_qubits_num, index_and_ancillary_joint_p,
                      euclidean_distances, dist_estimates, sorted_indices_lists, k, normalized_knn_dfs, target_labels)

        save_qknn_probabilities_and_distances(res_output_dir, 'qknn_probabilities_and_distances',
                                              index_and_ancillary_joint_p, euclidean_distances, N)

        knn_indices_dict = {dist_estimate: sorted_indices[0: k]
                            for dist_estimate, sorted_indices in zip(dist_estimates, sorted_indices_lists)}
        knn_indices_out_file = save_data_to_json_file(res_output_dir, 'k_nearest_neighbors_indices', knn_indices_dict)

        for i, dist_estimate in enumerate(dist_estimates):
            dfs_out_files_suffix = f'_{dist_estimate}' if len(dist_estimates) > 1 else ''

            knn_out_files.append(os.path.join(res_output_dir, 'k_nearest_neighbors{}.csv'.format(dfs_out_files_suffix)))
            knn_dfs[i].to_csv(knn_out_files[i], index=False)

            normalized_knn_out_files.append(os.path.join(
                res_output_dir, 'normalized_k_nearest_neighbors{}.csv'.format(dfs_out_files_suffix)
            ))
            normalized_knn_dfs[i].to_csv(normalized_knn_out_files[i], index=False)

        target_labels_dict = {dist_estimate: getattr(target_label, "tolist", lambda: target_label)()
                              for dist_estimate, target_label in zip(dist_estimates, target_labels)}
        target_label_out_file = save_data_to_json_file(res_output_dir, 'target_label', target_labels_dict)

    # Compute the execution time of the algorithm w/o the time required by the classical expectation
    algorithm_execution_time = (time.time() - start_time) - classical_expectation_execution_time

    return (knn_indices_out_file, knn_out_files, normalized_knn_out_files, target_label_out_file), \
        expected_knn_indices_out_file, normalized_target_instance_file, \
        (algorithm_execution_time, classical_expectation_execution_time)
