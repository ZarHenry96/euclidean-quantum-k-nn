#!/usr/bin/bash

# Function to print the correct usage on terminal
function usage {
    echo "Usage: ./run_exps.sh [-e exps_dir] [-d datasets_dir] [-f folds_number] [-o k_fold_seed] [-r root_res_dir] [-v eval_nn] [-k k] [-n encoding] [-s shots] [-p pseudocounts] [-u rounds] [-i sim_seeds_seed] [-t dist_estimates] [-c save_circ_plot] [-m num_processes]"
    echo "    -e exps_dir = directory containing the .template configuration files for the experiments"
    echo "    -d datasets_dir = directory containing the .csv datasets files and the .json class mapping files"
    echo "    -f folds_number = number of folds for the k-fold cross validation"
    echo "    -o k_fold_seed = seed for the k-fold cross validation"
    echo "    -r root_res_dir = root directory where to store the results"
    echo "    -v eval_nn = whether to evaluate the quality of the extracted nearest neighbors or not"
    echo "    -k k = number of nearest neighbors to extract (for multiple values set the -k parameter multiple times)"
    echo "    -n encoding = type of data encoding used in quantum circuits, allowed values: extension, translation (for multiple values set the -n parameter multiple times)"
    echo "    -s shots = number of measurements for simulations and quantum executions"
    echo "    -p pseudocounts = pseudocounts -for each index- for Laplace smoothing (only for simulations and quantum executions)"
    echo "    -u rounds = number of execution rounds (only for simulations and quantum executions)"
    echo "    -i sim_seeds_seed = seed for generating simulator seeds (only for local and online simulations, for multiple values set the -i parameter multiple times)"
    echo "    -t dist_estimates = Euclidean distance estimate used for k nearest neighbors extraction, allowed values: zero, one, avg, diff (ignored in classical executions, for multiple values set the -t parameter multiple times)"
    echo "    -c save_circ_plot = whether to save the circuit plot or not"
    echo "    -m num_processes = number of processes for the parallel execution of circuits"
    exit 0
}

# Variables default values
exp_templates_dir="experiments/templates"
datasets_dir="data/datasets"
folds_number=5
k_fold_seed=7
root_res_dir="results"
eval_nn="true"
k_values=()
encodings=()
shots=1024
pseudocounts=0
rounds_number=1
sim_seeds_seeds=()
dist_estimates=()
save_circ_plot="true"
num_processes=5

# Parse the script arguments
while getopts 'e:d:f:o:r:v:k:n:s:p:u:i:t:c:m:h' option; do
    case "$option" in
        e) exp_templates_dir="${OPTARG}";;
        d) datasets_dir="${OPTARG}";;
        f) folds_number="${OPTARG}";;
        o) k_fold_seed="${OPTARG}";;
        r) root_res_dir="${OPTARG}";;
        v) eval_nn="${OPTARG}";;
        k) k_values+=("${OPTARG}");;
        n) encodings+=("${OPTARG}");;
        s) shots="${OPTARG}";;
        p) pseudocounts="${OPTARG}";;
        u) rounds_number="${OPTARG}";;
        i) sim_seeds_seeds+=("${OPTARG}");;
        t) dist_estimates+=("${OPTARG}");;
        c) save_circ_plot="${OPTARG}";;
        m) num_processes="${OPTARG}";;
        h) usage;;
        *) usage;;
    esac
done
shift "$((OPTIND -1))"

# Use the default values for the array variables not set by the user (except for 'sim_seeds_seeds')
if (( ${#k_values[@]} == 0 )); then
    k_values=(5)
fi
if (( ${#encodings[@]} == 0 )); then
    encodings=("extension")
fi
if (( ${#dist_estimates[@]} == 0 )); then
    dist_estimates=("avg")
fi

# Prepare the distance estimates string for the template file
dist_estimates_str=""
for dist_estimate in "${dist_estimates[@]}"; do
    dist_estimates_str="${dist_estimates_str}\"${dist_estimate}\", "
done
dist_estimates_str="${dist_estimates_str%??}"

# Create a temporary directory to store the experiments configuration files
tmp_dir="tmp"
mkdir -p "${tmp_dir}"

# Iterate over template files
for template_file in "${exp_templates_dir}"/*.template; do
    # Get the experiment type (exec_type) from the template filename
    template_filename="${template_file##*/}"
    exec_type="${template_filename%.*}"

    # Check if the execution type is 'classical' and set the encodings accordingly
    encodings_for_exec_type=("${encodings[@]}")
    if [ "${exec_type}" == "classical" ]; then
        encodings_for_exec_type=("classical")
    fi

    # Check if it is a 'classical' or a 'statevector' execution and set the number of rounds accordingly
    rounds_for_exec_type="${rounds_number}"
    if [ "${exec_type}" == "classical" ] || [ "${exec_type}" == "statevector" ]; then
        rounds_for_exec_type=1
    fi

    # Iterate over encodings
    for encoding in "${encodings_for_exec_type[@]}"; do
        # Iterate over datasets
        for dataset_file in "${datasets_dir}"/*.csv; do
            dataset_file_no_extension="${dataset_file%.*}"
            dataset_filename="${dataset_file##*/}"
            dataset_name="${dataset_filename%.*}"

            # Check if the class mapping file exists and set the corresponding variable accordingly
            class_mapping_file="null"
            if [ -f "${dataset_file_no_extension}_class_mapping.json" ]; then
                class_mapping_file="\"${dataset_file_no_extension}_class_mapping.json\""
            fi

            # Iterate over k values
            for k in "${k_values[@]}"; do
                # Iterate over rounds
                for round in $(seq 0 $((rounds_for_exec_type - 1))); do
                    # Get the "simulator seeds" seed for the current round
                    if (( round < ${#sim_seeds_seeds[@]} )); then
                        sim_seeds_seed="${sim_seeds_seeds[round]}"
                    else
                        sim_seeds_seed="${RANDOM}"
                    fi

                    # Set the results directory and the config filename for the experiment
                    exp_res_dir="${root_res_dir}/${exec_type}/${encoding}/${dataset_name}/k_${k}/round_${round}"
                    exp_config_file="${tmp_dir}/${exec_type}_${encoding}_${dataset_name}_k_${k}_round_${round}.json"

                    # Create the experiment configuration file starting from the template
                    sed -e "s@\${dataset}@${dataset_file}@" -e "s@\${class_mapping}@${class_mapping_file}@" \
                        -e "s@\${folds_number}@${folds_number}@" -e "s@\${k_fold_random_seed}@${k_fold_seed}@" \
                        -e "s@\${res_dir}@${exp_res_dir}@" -e "s@\${eval_nn}@${eval_nn}@" \
                        -e "s@\${k}@${k}@" -e "s@\${encoding}@${encoding}@" -e "s@\${shots}@${shots}@" \
                        -e "s@\${pseudocounts}@${pseudocounts}@" -e "s@\${sim_seeds_seed}@${sim_seeds_seed}@" \
                        -e "s@\${dist_estimates}@${dist_estimates_str}@" -e "s@\${save_circ_plot}@${save_circ_plot}@" \
                        -e "s@\${num_processes}@${num_processes}@" "${template_file}" > "${exp_config_file}"

                    # Run the experiment
                    python main.py "${exp_config_file}"
                    printf "\n\n"
                    printf "%.s─" $(seq 1 "$(tput cols)")
                    printf "\n\n\n"
                done
            done
        done
    done
done

# Delete the temporary directory
rm -rf "${tmp_dir}"

# Collect the results of the experiments in a single file
python postprocessing/collect_processed_results.py "${root_res_dir}"
echo "Results collected"
