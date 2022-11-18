#!/usr/bin/bash

first_res_file="results/exps_first_round/collected_results.json"
second_res_file="results/exps_first_round/collected_results.json"
plots_root_dir="results/exps_first_round/plots"
extension=".pdf"



echo "Scatter plots: execution types comparison"

cltd_res_file_x="${first_res_file}"
cltd_res_file_y="${second_res_file}"

declare -a exec_type_x=("classical"   "classical"   "statevector"      "statevector"      "statevector"      "statevector")
declare -a exec_type_y=("statevector" "statevector" "local_simulation" "local_simulation" "local_simulation" "local_simulation")

declare -a encoding_x=("classical" "classical"   "extension" "extension" "translation" "translation")
declare -a encoding_y=("extension" "translation" "extension" "extension" "translation" "translation")

declare -a dist_estimate_x=("exact" "exact" "avg" "diff" "avg" "diff")
declare -a dist_estimate_y=("avg"   "avg"   "avg" "diff" "avg" "diff")

declare -a metrics=("accuracy" "jaccard_index" "average_jaccard_index")

plots_directory="${plots_root_dir}/scatterplots/exec_types_comp"

last_exec_type_index=$((${#exec_type_x[@]} - 1))
for i in $(seq 0 ${last_exec_type_index}); do
    exec_type_name_x="${exec_type_x[i]//local_simulation/simulation}"
    exec_type_name_y="${exec_type_y[i]//local_simulation/simulation}"

    for metric in "${metrics[@]}"; do
        metric_name="${metric//_/ }"

        python visualization/generate_scatterplot.py \
               --x-cltd-res-file "${cltd_res_file_x}" --x-exec-type "${exec_type_x[i]}" --x-encodings "${encoding_x[i]}" \
               --x-kvalues 3 5 7 9 --x-avg-on-runs --x-dist-estimate "${dist_estimate_x[i]}" --x-partition-key "k" \
               --y-cltd-res-file "${cltd_res_file_y}" --y-exec-type "${exec_type_y[i]}" --y-encodings "${encoding_y[i]}" \
               --y-kvalues 3 5 7 9 --y-avg-on-runs --y-dist-estimate "${dist_estimate_y[i]}" --y-partition-key "k" \
               --metric "${metric}" \
               --legend-labels "k=3" "k=5" "k=7" "k=9" \
               --x-label "${exec_type_name_x} (${encoding_x[i]}, ${dist_estimate_x[i]})" \
               --y-label "${exec_type_name_y} (${encoding_y[i]}, ${dist_estimate_y[i]})" \
               --title "Execution types comparison in '${metric_name}'" \
               --out-file "${plots_directory}/${metric}/${exec_type_name_x}_${encoding_x[i]}_${dist_estimate_x[i]}_vs_${exec_type_name_y}_${encoding_y[i]}_${dist_estimate_y[i]}-${metric}_scatterplot${extension}" \
               --statistical-test "wilcoxon"
    done
done



echo "Scatter plots: encodings comparison"

cltd_res_file_x="${first_res_file}"
cltd_res_file_y="${second_res_file}"

declare -a exec_types=("statevector" "local_simulation")

declare -a encoding_x=("extension")
declare -a encoding_y=("translation")

declare -a dist_estimates=("avg" "diff")

declare -a metrics=("accuracy" "jaccard_index" "average_jaccard_index")

plots_directory="${plots_root_dir}/scatterplots/encodings_comp"

for exec_type in "${exec_types[@]}"; do
    exec_type_name="${exec_type//local_simulation/simulation}"

    for dist_estimate in "${dist_estimates[@]}"; do
        for metric in "${metrics[@]}"; do
            metric_name="${metric//_/ }"

            python visualization/generate_scatterplot.py \
                   --x-cltd-res-file "${cltd_res_file_x}" --x-exec-type "${exec_type}" --x-encodings "${encoding_x[0]}" \
                   --x-kvalues 3 5 7 9 --x-avg-on-runs --x-dist-estimate "${dist_estimate}" --x-partition-key "k" \
                   --y-cltd-res-file "${cltd_res_file_y}" --y-exec-type "${exec_type}" --y-encodings "${encoding_y[0]}" \
                   --y-kvalues 3 5 7 9 --y-avg-on-runs --y-dist-estimate "${dist_estimate}" --y-partition-key "k" \
                   --metric "${metric}" \
                   --legend-labels "k=3" "k=5" "k=7" "k=9" \
                   --x-label "${encoding_x[0]} (${exec_type_name}, ${dist_estimate})" \
                   --y-label "${encoding_y[0]} (${exec_type_name}, ${dist_estimate})" \
                   --title "Encodings comparison in '${metric_name}'" \
                   --out-file "${plots_directory}/${metric}/${exec_type_name}_${encoding_x[0]}_${dist_estimate}_vs_${exec_type_name}_${encoding_y[0]}_${dist_estimate}-${metric}_scatterplot${extension}" \
                   --statistical-test "wilcoxon"
        done
    done
done



echo "Scatter plots: distance estimates comparison"

cltd_res_file_x="${first_res_file}"
cltd_res_file_y="${second_res_file}"

declare -a exec_types=("statevector" "local_simulation")

declare -a encodings=("extension" "translation")

declare -a k_values=(3 5 7 9)
declare -a k_values_strings=("k=3" "k=5" "k=7" "k=9")

declare -a dist_estimate_x=("avg")
declare -a dist_estimate_y=("diff")

declare -a partition_keys=("k" "encoding")

declare -a metrics=("accuracy" "jaccard_index" "average_jaccard_index")

plots_directory="${plots_root_dir}/scatterplots/dist_estimates_comp"

for exec_type in "${exec_types[@]}"; do
    exec_type_name="${exec_type//local_simulation/simulation}"

    for metric in "${metrics[@]}"; do
        metric_name="${metric//_/ }"

        # all
        for partition_key in "${partition_keys[@]}"; do
            legend_labels=("${k_values_strings[@]}")
            axes_labels_term="all encodings"
            if [ "${partition_key}" == "encoding" ]; then
                legend_labels=("${encodings[@]}")
                axes_labels_term="all k values"
            fi

            python visualization/generate_scatterplot.py \
                   --x-cltd-res-file "${cltd_res_file_x}" --x-exec-type "${exec_type}" --x-encodings "${encodings[@]}" \
                   --x-kvalues "${k_values[@]}" --x-avg-on-runs --x-dist-estimate "${dist_estimate_x[0]}" --x-partition-key "${partition_key}" \
                   --y-cltd-res-file "${cltd_res_file_y}" --y-exec-type "${exec_type}" --y-encodings "${encodings[@]}" \
                   --y-kvalues "${k_values[@]}" --y-avg-on-runs --y-dist-estimate "${dist_estimate_y[0]}" --y-partition-key "${partition_key}" \
                   --metric "${metric}" \
                   --legend-labels "${legend_labels[@]}" \
                   --x-label "${dist_estimate_x[0]} (${exec_type_name}, ${axes_labels_term})" \
                   --y-label "${dist_estimate_y[0]} (${exec_type_name}, ${axes_labels_term})" \
                   --title "Distance estimates comparison in '${metric_name}'" \
                   --out-file "${plots_directory}/${metric}/all/${exec_type_name}_${dist_estimate_x[0]}_vs_${exec_type_name}_${dist_estimate_y[0]}-${partition_key}_split-${metric}_scatterplot${extension}" \
                   --statistical-test "wilcoxon"
        done

        # per encoding
        for encoding in "${encodings[@]}"; do
            python visualization/generate_scatterplot.py \
               --x-cltd-res-file "${cltd_res_file_x}" --x-exec-type "${exec_type}" --x-encodings "${encoding}" \
               --x-kvalues "${k_values[@]}" --x-avg-on-runs --x-dist-estimate "${dist_estimate_x[0]}" --x-partition-key "k" \
               --y-cltd-res-file "${cltd_res_file_y}" --y-exec-type "${exec_type}" --y-encodings "${encoding}" \
               --y-kvalues "${k_values[@]}" --y-avg-on-runs --y-dist-estimate "${dist_estimate_y[0]}" --y-partition-key "k" \
               --metric "${metric}" \
               --legend-labels "${k_values_strings[@]}" \
               --x-label "${dist_estimate_x[0]} (${exec_type_name}, ${encoding})" \
               --y-label "${dist_estimate_y[0]} (${exec_type_name}, ${encoding})" \
               --title "Distance estimates comparison in '${metric_name}'" \
               --out-file "${plots_directory}/${metric}/per_encoding/${exec_type_name}_${encoding}_${dist_estimate_x[0]}_vs_${exec_type_name}_${encoding}_${dist_estimate_y[0]}-${metric}_scatterplot${extension}" \
               --statistical-test "wilcoxon"
        done
    done
done



echo "Box plots: dataset-based comparisons"

first_cltd_res_file="${first_res_file}"
second_cltd_res_file="${second_res_file}"

declare -a encodings=("extension" "translation")

declare -a datasets=("01_iris_setosa_versicolor" "01_iris_setosa_virginica" "01_iris_versicolor_virginica"
                     "02_transfusion" "03_vertebral_column_2C" "04_seeds_1_2" "05_ecoli_cp_im" "06_glasses_1_2"
                     "07_breast_tissue_adi_fadmasgla" "08_breast_cancer" "09_accent_recognition_uk_us" "10_leaf_11_9")
declare -a datasets_strings=("1a" "1b" "1c" "2" "3" "4" "5" "6" "7" "8" "9" "10")

declare -a k_values=(3 5 7 9)
declare -a k_values_strings=("k=3" "k=5" "k=7" "k=9")

declare -a dist_estimates=("avg" "diff")

declare -a metrics=("jaccard_index" "average_jaccard_index")

declare -a y_limits=(0.0 0.675)

plots_directory="${plots_root_dir}/boxplots/dataset_based"

last_dataset_index=$((${#datasets[@]} - 1))
for i in $(seq 0 ${last_dataset_index}); do
    dataset="${datasets[i]}"
    dataset_string="${datasets_strings[i]}"

    for metric in "${metrics[@]}"; do
        metric_name="${metric//_/ }"

        # encodings comparison
        for dist_estimate in "${dist_estimates[@]}"; do
            python visualization/generate_boxplot.py \
                   --first-cltd-res-file "${first_cltd_res_file}" --first-exec-type "local_simulation" \
                   --first-encodings "${encodings[0]}" --first-datasets "${dataset}" --first-kvalues "${k_values[@]}" \
                   --first-avg-on-runs --first-dist-estimate "${dist_estimate}" \
                   --second-cltd-res-file "${second_cltd_res_file}" --second-exec-type "local_simulation" \
                   --second-encodings "${encodings[1]}" --second-datasets "${dataset}" --second-kvalues "${k_values[@]}" \
                   --second-avg-on-runs --second-dist-estimate "${dist_estimate}" \
                   --metric "${metric}" --x-axis-prop "k" \
                   --legend-labels "${encodings[@]}" --x-ticks-labels "${k_values_strings[@]}"  \
                   --x-label "k value" --y-label "${metric_name}" \
                   --title "'${metric_name^}' distribution for different k values (simulation, ${dist_estimate}, ${dataset_string})" \
                   --y-limits "${y_limits[@]}" \
                   --out-file "${plots_directory}/encodings_comp/${metric}/${dataset}-simulation_${encodings[0]}_${dist_estimate}_vs_simulation_${encodings[1]}_${dist_estimate}-${metric}_boxplot${extension}" \
                   --statistical-test "wilcoxon"
        done

        # distance estimates comparison
        for encoding in "${encodings[@]}"; do
            python visualization/generate_boxplot.py \
                   --first-cltd-res-file "${first_cltd_res_file}" --first-exec-type "local_simulation" \
                   --first-encodings "${encoding}" --first-datasets "${dataset}" --first-kvalues "${k_values[@]}" \
                   --first-avg-on-runs --first-dist-estimate "${dist_estimates[0]}" \
                   --second-cltd-res-file "${second_cltd_res_file}" --second-exec-type "local_simulation" \
                   --second-encodings "${encoding}" --second-datasets "${dataset}" --second-kvalues "${k_values[@]}" \
                   --second-avg-on-runs --second-dist-estimate "${dist_estimates[1]}" \
                   --metric "${metric}" --x-axis-prop "k" \
                   --legend-labels "${dist_estimates[@]}" --x-ticks-labels "${k_values_strings[@]}"  \
                   --x-label "k value" --y-label "${metric_name}" \
                   --title "'${metric_name^}' distribution for different k values (simulation, ${encoding}, ${dataset_string})" \
                   --y-limits "${y_limits[@]}" \
                   --out-file "${plots_directory}/dist_estimates_comp/${metric}/${dataset}-simulation_${encoding}_${dist_estimates[0]}_vs_simulation_${encoding}_${dist_estimates[1]}-${metric}_boxplot${extension}" \
                   --statistical-test "wilcoxon"
        done
    done
done



echo "Box plots: k-value-based comparisons"

first_cltd_res_file="${first_res_file}"
second_cltd_res_file="${second_res_file}"

declare -a encodings=("extension" "translation")

declare -a datasets=("01_iris_setosa_versicolor" "01_iris_setosa_virginica" "01_iris_versicolor_virginica"
                     "02_transfusion" "03_vertebral_column_2C" "04_seeds_1_2" "05_ecoli_cp_im" "06_glasses_1_2"
                     "07_breast_tissue_adi_fadmasgla" "08_breast_cancer" "09_accent_recognition_uk_us" "10_leaf_11_9")
declare -a datasets_strings=("1a" "1b" "1c" "2" "3" "4" "5" "6" "7" "8" "9" "10")

declare -a k_values=(3 5 7 9)

declare -a dist_estimates=("avg" "diff")

declare -a metrics=("jaccard_index" "average_jaccard_index")

declare -a y_limits=(0.0 0.675)

plots_directory="${plots_root_dir}/boxplots/k_value_based"

for k_value in "${k_values[@]}"; do
    for metric in "${metrics[@]}"; do
        metric_name="${metric//_/ }"

        # encodings comparison
        for dist_estimate in "${dist_estimates[@]}"; do
            python visualization/generate_boxplot.py \
                   --first-cltd-res-file "${first_cltd_res_file}" --first-exec-type "local_simulation" \
                   --first-encodings "${encodings[0]}" --first-datasets "${datasets[@]}" --first-kvalues "${k_value}" \
                   --first-avg-on-runs --first-dist-estimate "${dist_estimate}" \
                   --second-cltd-res-file "${second_cltd_res_file}" --second-exec-type "local_simulation" \
                   --second-encodings "${encodings[1]}" --second-datasets "${datasets[@]}" --second-kvalues "${k_value}" \
                   --second-avg-on-runs --second-dist-estimate "${dist_estimate}" \
                   --metric "${metric}" --x-axis-prop "dataset" \
                   --legend-labels "${encodings[@]}" --x-ticks-labels "${datasets_strings[@]}"  \
                   --x-label "dataset" --y-label "${metric_name}" \
                   --title "'${metric_name^}' distribution on different datasets (simulation, ${dist_estimate}, k=${k_value})" \
                   --y-limits "${y_limits[@]}" \
                   --out-file "${plots_directory}/encodings_comp/${metric}/k_${k_value}-simulation_${encodings[0]}_${dist_estimate}_vs_simulation_${encodings[1]}_${dist_estimate}-${metric}_boxplot${extension}" \
                   --statistical-test "wilcoxon"
        done

        # distance estimates comparison
        for encoding in "${encodings[@]}"; do
            python visualization/generate_boxplot.py \
                   --first-cltd-res-file "${first_cltd_res_file}" --first-exec-type "local_simulation" \
                   --first-encodings "${encoding}" --first-datasets "${datasets[@]}" --first-kvalues "${k_value}" \
                   --first-avg-on-runs --first-dist-estimate "${dist_estimates[0]}" \
                   --second-cltd-res-file "${second_cltd_res_file}" --second-exec-type "local_simulation" \
                   --second-encodings "${encoding}" --second-datasets "${datasets[@]}" --second-kvalues "${k_value}" \
                   --second-avg-on-runs --second-dist-estimate "${dist_estimates[1]}" \
                   --metric "${metric}" --x-axis-prop "dataset" \
                   --legend-labels "${dist_estimates[@]}" --x-ticks-labels "${datasets_strings[@]}"  \
                   --x-label "dataset" --y-label "${metric_name}" \
                   --title "'${metric_name^}' distribution on different datasets (simulation, ${encoding}, k=${k_value})" \
                   --y-limits "${y_limits[@]}" \
                   --out-file "${plots_directory}/dist_estimates_comp/${metric}/k_${k_value}-simulation_${encoding}_${dist_estimates[0]}_vs_simulation_${encoding}_${dist_estimates[1]}-${metric}_boxplot${extension}" \
                   --statistical-test "wilcoxon"
        done
    done
done
