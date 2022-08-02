import copy
import math
import numpy as np
import os
import pandas as pd
import shutil
import sys

from matplotlib import pyplot as plt
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit, execute
from qiskit.tools.monitor import job_monitor

from algorithm.classical_knn import classical_knn
from algorithm.utils import save_exp_config, select_backend, save_data_to_txt_file, print_qknn_results, save_qknn_log, \
     save_probabilities_and_distances


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
        target_df.iloc[0, attribute_index] = max(min(upper_bound, attribute_val), lower_bound)

    # Normalize the training data
    training_df.iloc[:, :features_number] = \
        (training_df.iloc[:, :features_number] - training_min_max_avg) / (training_range * sqrt_features_number)

    # Show the normalized data (if needed)
    if verbose:
        print(f'Normalized training dataset:\n{training_df}')
        print(f'\nNormalized target instance:\n{target_df}\n')

    # Save the normalized data (if needed)
    normalized_data_files = [None, None]
    if store:
        normalized_training_data_file = \
            os.path.join(res_input_dir, 'normalized_{}'.format(os.path.basename(training_data_file)))
        training_df.to_csv(
            normalized_training_data_file,
            index=False
        )

        normalized_target_instance_file = \
            os.path.join(res_input_dir, 'normalized_{}'.format(os.path.basename(target_instance_file)))
        target_df.to_csv(
            normalized_target_instance_file,
            index=False
        )

        normalized_data_files = [normalized_training_data_file, normalized_target_instance_file]

    return training_df, target_df, normalized_data_files


def build_qknn_circuit(training_df, target_df, N, d, encoding, exec_type):
    # Compute the circuit size
    cnot_swap_circuit_qubits = 2
    index_qubits = math.ceil(math.log2(N))
    features_qubits = math.ceil(math.log2(2 * d + (3 if encoding == 'extension' else 4)))
    qubits_num = cnot_swap_circuit_qubits + index_qubits + features_qubits
    c_bits_num = 1 + index_qubits

    # Create a Quantum Circuit acting on the q register
    qr = QuantumRegister(qubits_num, 'q')
    cr = ClassicalRegister(c_bits_num, 'c')
    circuit = QuantumCircuit(qr, cr)

    # Data structures for the joint initialization of the target CNOT-SWAP qubit, the index and features registers
    init_qubits = 1 + index_qubits + features_qubits
    amplitudes = np.zeros(2 ** init_qubits)
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
    encoded_training_set = []
    for instance_index, row in training_df.iterrows():
        training_norms.append(np.linalg.norm(row[0:d]))
        encoded_training_set.append([])

        # Training instance features (two times)
        for i in range(0, 2 * d):
            index = 2 * instance_index + (2 ** (index_qubits + 1)) * i
            amplitudes[index] = amplitudes_base_value * multiplication_factor * row[i % d]
            encoded_training_set[-1].append(amplitudes[index] / amplitudes_base_value)

        # Training instance norm
        index = 2 * instance_index + (2 ** (index_qubits + 1)) * (2 * d)
        amplitudes[index] = amplitudes_base_value * multiplication_factor * training_norms[-1]
        encoded_training_set[-1].append(amplitudes[index] / amplitudes_base_value)

        # Translation feature (for translation encoding)
        if encoding == 'translation':
            index = 2 * instance_index + (2 ** (index_qubits + 1)) * (2 * d + features_offset)
            amplitudes[index] = amplitudes_base_value * translation_feature_abs_value
            encoded_training_set[-1].append(translation_feature_abs_value)

        # Zero value
        index = 2 * instance_index + (2 ** (index_qubits + 1)) * (2 * d + features_offset + 1)
        amplitudes[index] = amplitudes_base_value * 0
        encoded_training_set[-1].append(0)

        # Residual (for unitary norm)
        index = 2 * instance_index + (2 ** (index_qubits + 1)) * (2 * d + features_offset + 2)
        amplitudes[index] = amplitudes_base_value * math.sqrt(
            1 - (3 * multiplication_factor**2 * training_norms[-1]**2 + translation_feature_abs_value**2)
        )
        encoded_training_set[-1].append(amplitudes[index] / amplitudes_base_value)

    # Compute the target instance amplitudes
    encoded_target_instances = []
    for j in range(0, N):
        encoded_target_instances.append([])

        # Target instance features with sign flipped (two times)
        for i in range(0, 2 * d):
            index = 1 + 2 * j + (2 ** (index_qubits + 1)) * i
            amplitudes[index] = amplitudes_base_value * multiplication_factor * (-target_df.iloc[0, i % d])
            encoded_training_set[-1].append(amplitudes[index] / amplitudes_base_value)

        # Corresponding training instance norm
        index = 1 + 2 * j + (2 ** (index_qubits+1)) * (2 * d)
        amplitudes[index] = amplitudes_base_value * multiplication_factor * training_norms[j]
        encoded_target_instances[-1].append(amplitudes[index] / amplitudes_base_value)

        # Translation feature (for translation encoding)
        if encoding == 'translation':
            index = 1 + 2 * j + (2 ** (index_qubits + 1)) * (2 * d + features_offset)
            amplitudes[index] = amplitudes_base_value * (-translation_feature_abs_value)
            encoded_target_instances[-1].append(-translation_feature_abs_value)

        # Residual (for unitary norm)
        index = 1 + 2 * j + (2 ** (index_qubits + 1)) * (2 * d + features_offset + 1)
        amplitudes[index] = amplitudes_base_value * math.sqrt(
            1 - (2 * multiplication_factor**2 * target_norm**2 + multiplication_factor**2 * training_norms[j]**2 +
                 translation_feature_abs_value**2)
        )
        encoded_target_instances[-1].append(amplitudes[index] / amplitudes_base_value)

        # Zero value
        index = 1 + 2 * j + (2 ** (index_qubits + 1)) * (2 * d + features_offset + 2)
        amplitudes[index] = amplitudes_base_value * 0
        encoded_target_instances[-1].append(0)

    # Set all "target CNOT-SWAP qubit + index_register + features_register" amplitudes
    target_cnot_swap_qubit = 1
    circuit.initialize(amplitudes, qr[target_cnot_swap_qubit: target_cnot_swap_qubit + init_qubits])

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
        circuit.measure(qr[first_index_qubit: first_index_qubit + index_qubits], cr[1: 1 + index_qubits])

    return circuit, qubits_num, index_qubits, features_qubits, target_norm, \
           [encoded_training_set, encoded_target_instances]


def get_probabilities_from_statevector(statevector, qubits_num, index_qubits):
    # Prepare the data structures
    p0, p1, index_and_ancillary_joint_p = 0.0, 0.0, {}
    for j in range(0, (2 ** index_qubits)):
        index_state = ('{0:0' + str(index_qubits) + 'b}').format(j)
        index_and_ancillary_joint_p[index_state] = {'0': 0.0, '1': 0.0}

    # Process the statevector, computing the probabilities of interest
    for circuit_dec_state, state_amplitude in enumerate(statevector):
        state_probability = np.abs(state_amplitude) ** 2

        if circuit_dec_state % 2 == 0:
            p0 += state_probability
        else:
            p1 += state_probability

        circuit_state = ('{0:0' + str(qubits_num) + 'b}').format(circuit_dec_state)
        index_state = circuit_state[qubits_num - (index_qubits + 2):-2]
        ancillary_state = circuit_state[qubits_num - 1]
        index_and_ancillary_joint_p[index_state][ancillary_state] += state_probability

    return p0, p1, index_and_ancillary_joint_p


def get_probabilities_from_counts(counts, index_qubits, shots, pseudocounts, N):
    # Prepare data structures
    p0, p1, index_and_ancillary_joint_p = 0.0, 0.0, {}
    for j in range(0, (2 ** index_qubits)):
        index_state = ('{0:0' + str(index_qubits) + 'b}').format(j)
        index_and_ancillary_joint_p[index_state] = {'0': 0.0, '1': 0.0}

    # Process counts
    smoothed_total_counts = shots + pseudocounts * N
    for measured_state, state_counts in counts.items():
        index_state = measured_state[0: -1]
        index_dec_state = int(index_state, 2)
        ancillary_state = measured_state[-1]

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

    return sqrt_arg if sqrt_arg >= 0.0 else 0.0


def extract_euclidean_distances(index_and_ancillary_joint_p, N, index_qubits, encoding, squared_target_norm):
    euclidean_distances = {}

    for j in range(0, N):
        index_state = ('{0:0' + str(index_qubits) + 'b}').format(j)
        euclidean_distances[j] = {}

        joint_j_0_p = index_and_ancillary_joint_p[index_state]['0']
        scalar_prod_0 = 2 * N * joint_j_0_p - 1
        sqrt_arg_0 = get_sqrt_argument_from_scalar_product(scalar_prod_0, squared_target_norm, encoding)
        euclidean_distances[j]['zero'] = math.sqrt(sqrt_arg_0)

        joint_j_1_p = index_and_ancillary_joint_p[index_state]['1']
        scalar_prod_1 = 1 - 2 * N * joint_j_1_p
        sqrt_arg_1 = get_sqrt_argument_from_scalar_product(scalar_prod_1, squared_target_norm, encoding)
        euclidean_distances[j]['one'] = math.sqrt(sqrt_arg_1)

        euclidean_distances[j]['avg'] = (euclidean_distances[j]['zero'] + euclidean_distances[j]['one']) / 2

        scalar_prod_diff = N * (joint_j_0_p - joint_j_1_p)
        sqrt_arg_diff = get_sqrt_argument_from_scalar_product(scalar_prod_diff, squared_target_norm, encoding)
        euclidean_distances[j]['diff'] = math.sqrt(sqrt_arg_diff)

    return euclidean_distances


def run_qknn(training_data_file, target_instance_file, k, exec_type, encoding, backend_name, job_name, shots,
             pseudocounts, sorting_dist_estimate, res_dir, verbose=True, store_results=True, save_circuit_plot=True):
    # Prepare results directories
    res_input_dir = os.path.join(res_dir, 'input')
    res_output_dir = os.path.join(res_dir, 'output')
    if store_results:
        os.makedirs(res_dir, exist_ok=True)
        os.makedirs(res_input_dir, exist_ok=True)
        os.makedirs(res_output_dir, exist_ok=True)

        # Save the experiment configuration
        save_exp_config(res_dir, 'exp_config', training_data_file, target_instance_file, k, exec_type, encoding,
                        backend_name, job_name, shots, pseudocounts, sorting_dist_estimate, verbose,
                        save_circuit_plot)

    # Load and normalize the input (training and target) data
    training_df, target_df, normalized_data_files = \
        load_and_normalize_data(training_data_file, target_instance_file, res_input_dir,
                                verbose=verbose, store=store_results)
    N, d = len(training_df), len(training_df.columns) - 1

    # Get the original training dataframe
    original_training_df = pd.read_csv(training_data_file, sep=',')

    # If it is a classical execution, run the classical KNN and exit
    if exec_type == 'classical':
        normalized_nearest_neighbors_df, normalized_knn_filename = \
            classical_knn(training_df, target_df, k, original_training_df, store_results, res_dir, verbose=verbose)
        return normalized_nearest_neighbors_df, target_df, normalized_knn_filename, normalized_data_files[1]

    # Select the backend for the execution
    backend = select_backend(exec_type, backend_name)

    # Compute the expected k nearest neighbors (classical computation)
    classical_knn(training_df, target_df, k, original_training_df, store_results, res_dir,
                  expectation=True, verbose=verbose)

    # Build the quantum circuit
    circuit, qubits_num, index_qubits, features_qubits, target_norm, _ = \
        build_qknn_circuit(training_df, target_df, N, d, encoding, exec_type)

    # Draw, show, and save the circuit (if needed)
    circuit_for_drawing = copy.deepcopy(circuit)
    circuit_for_drawing.data[0][0].params = []
    if verbose:
        print('\n{}'.format(circuit_for_drawing.draw(output='text')))
    if store_results and save_circuit_plot:
        circuit_for_drawing.draw(output='mpl', filename=os.path.join(res_dir, 'qknn_circuit.png'), fold=-1)
        plt.close()

    # Execute the job
    if exec_type == 'statevector':
        job = execute(circuit, backend)
    elif exec_type == 'local_simulation':
        job = execute(circuit, backend, shots=shots)
    else:
        job = execute(circuit, backend, shots=shots, job_name=job_name)

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
            print(f'\nOutput state vector:\n{list(output_statevector)}')
        if store_results:
            save_data_to_txt_file(res_output_dir, 'qknn_output_state_vector', list(output_statevector))

        # Process the output statevector
        p0, p1, index_and_ancillary_joint_p = \
            get_probabilities_from_statevector(output_statevector, qubits_num, index_qubits)
    else:
        counts = result.get_counts(circuit)
        sorted_counts = {i: counts[i] for i in sorted(counts.keys())}

        # Show and save the counts (if needed)
        if verbose:
            print('\nResults\nCircuit output counts (w/o Laplace smoothing): {}'.format(sorted_counts))
            print('\n[Shots = {}, Pseudocounts (per index state) = {}]'.format(shots, pseudocounts))
        if store_results:
            save_data_to_txt_file(res_output_dir, 'qknn_counts', counts)

        # Process counts
        p0, p1, index_and_ancillary_joint_p = \
            get_probabilities_from_counts(counts, index_qubits, shots, pseudocounts, N)

    # Extract the distances from the probability values
    euclidean_distances = extract_euclidean_distances(index_and_ancillary_joint_p, N, index_qubits, encoding,
                                                      target_norm**2)

    # Get the k nearest neighbors based on the specified distance estimate
    sorted_indices = [
        index
        for index, _ in sorted(euclidean_distances.items(), key=lambda item: item[1][sorting_dist_estimate])
    ]
    nearest_neighbors_df = original_training_df.iloc[sorted_indices[0: k], :]
    normalized_nearest_neighbors_df = training_df.iloc[sorted_indices[0: k], :]

    # Display and store the results (if needed)
    normalized_nearest_neighbors_output_file = None
    if verbose:
        print_qknn_results(p0, p1, index_qubits, index_and_ancillary_joint_p, euclidean_distances,
                           sorting_dist_estimate, sorted_indices, k, normalized_nearest_neighbors_df)
    if store_results:
        save_qknn_log(res_output_dir, 'qknn_log', p0, p1, index_qubits, index_and_ancillary_joint_p,
                      euclidean_distances, sorting_dist_estimate, sorted_indices, k, normalized_nearest_neighbors_df)

        save_probabilities_and_distances(res_output_dir, 'qknn_probabilities_and_distances',
                                         index_and_ancillary_joint_p, euclidean_distances)

        nearest_neighbors_output_file = os.path.join(res_output_dir, 'nearest_neighbors.csv')
        nearest_neighbors_df.to_csv(nearest_neighbors_output_file, index=False)

        normalized_nearest_neighbors_output_file = \
            os.path.join(res_output_dir, 'normalized_nearest_neighbors.csv')
        normalized_nearest_neighbors_df.to_csv(normalized_nearest_neighbors_output_file, index=False)

    normalized_nearest_neighbors_df.reset_index(drop=True, inplace=True)
    return normalized_nearest_neighbors_df, target_df, \
           normalized_nearest_neighbors_output_file, normalized_data_files[1]
