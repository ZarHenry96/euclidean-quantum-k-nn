#!/usr/bin/bash

exps_res_file="results/exps_first_round/collected_results.json"
classical_baseline_res_file="results/qml_pipeline_paper_results/baseline_collected_results.json"
num_shots_res_file="results/exps_diff_num_shots/#_shots/collected_results.json"

plots_root_dir="results/exps_first_round/plots"
extension=".pdf"

exec_types_comp_scatter="true"
encodings_comp_scatter="true"
dist_estimates_comp_scatter="true"
extra_comp_scatter="true"
dataset_based_box="false"
k_value_based_box="false"
classical_baseline_qknn_comp_scatter="false"
diff_nums_shots_exec_types_comp_scatter="false"
nums_shots_comp_scatter="false"
num_shots_diff_box="false"
comp_summary_diff_box="true"


if [ "${exec_types_comp_scatter}" == "true" ]; then
    echo "Scatter plots: execution types comparisons"

    cltd_res_file_x="${exps_res_file}"
    cltd_res_file_y="${exps_res_file}"

    declare -a exec_type_x=("classical"   "classical"   "classical"   "classical"   "statevector"      "statevector"      "statevector"      "statevector")
    declare -a exec_type_y=("statevector" "statevector" "statevector" "statevector" "local_simulation" "local_simulation" "local_simulation" "local_simulation")

    declare -a encoding_x=("classical" "classical" "classical"   "classical"   "extension" "extension" "translation" "translation")
    declare -a encoding_y=("extension" "extension" "translation" "translation" "extension" "extension" "translation" "translation")

    declare -a dist_estimate_x=("exact" "exact" "exact" "exact" "avg" "diff" "avg" "diff")
    declare -a dist_estimate_y=("avg"   "diff"  "avg"   "diff"  "avg" "diff" "avg" "diff")

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
fi


if [ "${encodings_comp_scatter}" == "true" ]; then
    echo "Scatter plots: encodings comparisons"

    cltd_res_file_x="${exps_res_file}"
    cltd_res_file_y="${exps_res_file}"

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
fi


if [ "${dist_estimates_comp_scatter}" == "true" ]; then
    echo "Scatter plots: distance estimates comparisons"

    cltd_res_file_x="${exps_res_file}"
    cltd_res_file_y="${exps_res_file}"

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
fi


if [ "${extra_comp_scatter}" == "true" ]; then
    echo "Scatter plots: extra comparisons"

    cltd_res_file_x="${exps_res_file}"
    cltd_res_file_y="${exps_res_file}"

    declare -a dist_estimate_x=("avg" "diff")
    declare -a dist_estimate_y=("diff" "avg")

    declare -a metrics=("accuracy" "jaccard_index" "average_jaccard_index")

    plots_directory="${plots_root_dir}/scatterplots/extra_comp"

    last_dist_estimate_index=$((${#dist_estimate_x[@]} - 1))
    for i in $(seq 0 ${last_dist_estimate_index}); do
        for metric in "${metrics[@]}"; do
            metric_name="${metric//_/ }"

            python visualization/generate_scatterplot.py \
                --x-cltd-res-file "${cltd_res_file_x}" --x-exec-type "local_simulation" --x-encodings "extension" \
                --x-kvalues 3 5 7 9 --x-avg-on-runs --x-dist-estimate "${dist_estimate_x[i]}" --x-partition-key "k" \
                --y-cltd-res-file "${cltd_res_file_y}" --y-exec-type "local_simulation" --y-encodings "translation" \
                --y-kvalues 3 5 7 9 --y-avg-on-runs --y-dist-estimate "${dist_estimate_y[i]}" --y-partition-key "k" \
                --metric "${metric}" \
                --legend-labels "k=3" "k=5" "k=7" "k=9" \
                --x-label "extension, ${dist_estimate_x[i]} (simulation)" \
                --y-label "translation, ${dist_estimate_y[i]} (simulation)" \
                --title "Extra comparison in '${metric_name}'" \
                --out-file "${plots_directory}/${metric}/simulation_extension_${dist_estimate_x[i]}_vs_simulation_translation_${dist_estimate_y[i]}-${metric}_scatterplot${extension}" \
                --statistical-test "wilcoxon"
        done
    done
fi


if [ "${dataset_based_box}" == "true" ]; then
    echo "Box plots: dataset-based comparisons (avg on runs & no avg on runs)"

    first_cltd_res_file="${exps_res_file}"
    second_cltd_res_file="${exps_res_file}"

    declare -a encodings=("extension" "translation")

    declare -a datasets=("01_iris_setosa_versicolor" "01_iris_setosa_virginica" "01_iris_versicolor_virginica"
                         "02_transfusion" "03_vertebral_column_2C" "04_seeds_1_2" "05_ecoli_cp_im" "06_glasses_1_2"
                         "07_breast_tissue_adi_fadmasgla" "08_breast_cancer" "09_accent_recognition_uk_us" "10_leaf_11_9")
    declare -a datasets_strings=("1a" "1b" "1c" "2" "3" "4" "5" "6" "7" "8" "9" "10")

    declare -a k_values=(3 5 7 9)
    declare -a k_values_strings=("k=3" "k=5" "k=7" "k=9")

    declare -a avg_on_runs=("true" "false")

    declare -a dist_estimates=("avg" "diff")

    declare -a metrics=("accuracy" "jaccard_index" "average_jaccard_index")

    declare -a y_limits=(-0.05 1.05)

    last_dataset_index=$((${#datasets[@]} - 1))
    for i in $(seq 0 ${last_dataset_index}); do
        dataset="${datasets[i]}"
        dataset_string="${datasets_strings[i]}"

        for avg_on_runs_flag in "${avg_on_runs[@]}"; do
            if [ "${avg_on_runs_flag}" == "true" ]; then
                first_avg_on_runs=("--first-avg-on-runs")
                second_avg_on_runs=("--second-avg-on-runs")
                title_suffix=""
                plots_directory="${plots_root_dir}/boxplots/avg_on_runs/dataset_based"
                out_file_spec=""
                # statistical_test="wilcoxon"
            else
                first_avg_on_runs=()
                second_avg_on_runs=()
                title_suffix=", no avg"
                plots_directory="${plots_root_dir}/boxplots/no_avg_on_runs/dataset_based"
                out_file_spec="no_avg_on_runs-"
                # statistical_test="ranksums"
            fi

            for metric in "${metrics[@]}"; do
                metric_name="${metric//_/ }"

                legend_position="upper left"
                if [ "${metric}" == "accuracy" ]; then
                    legend_position="lower left"
                fi

                # encodings comparison
                for dist_estimate in "${dist_estimates[@]}"; do
                    python visualization/generate_boxplot.py \
                           --first-cltd-res-file "${first_cltd_res_file}" --first-exec-type "local_simulation" \
                           --first-encodings "${encodings[0]}" --first-datasets "${dataset}" --first-kvalues "${k_values[@]}" \
                           "${first_avg_on_runs[@]}" --first-dist-estimate "${dist_estimate}" \
                           --second-cltd-res-file "${second_cltd_res_file}" --second-exec-type "local_simulation" \
                           --second-encodings "${encodings[1]}" --second-datasets "${dataset}" --second-kvalues "${k_values[@]}" \
                           "${second_avg_on_runs[@]}" --second-dist-estimate "${dist_estimate}" \
                           --metric "${metric}" --x-axis-prop "k" \
                           --legend-labels "${encodings[@]}" --legend-position "${legend_position}" \
                           --x-ticks-labels "${k_values_strings[@]}"  --x-label "k value" --y-label "${metric_name}" \
                           --title "'${metric_name^}' distribution for different k values (sim., ${dist_estimate}, ${dataset_string}${title_suffix})" \
                           --y-limits "${y_limits[@]}" \
                           --out-file "${plots_directory}/encodings_comp/${metric}/${dataset}-simulation_${encodings[0]}_${dist_estimate}_vs_simulation_${encodings[1]}_${dist_estimate}-${out_file_spec}${metric}_boxplot${extension}"
                done

                # distance estimates comparison
                for encoding in "${encodings[@]}"; do
                    python visualization/generate_boxplot.py \
                           --first-cltd-res-file "${first_cltd_res_file}" --first-exec-type "local_simulation" \
                           --first-encodings "${encoding}" --first-datasets "${dataset}" --first-kvalues "${k_values[@]}" \
                           "${first_avg_on_runs[@]}" --first-dist-estimate "${dist_estimates[0]}" \
                           --second-cltd-res-file "${second_cltd_res_file}" --second-exec-type "local_simulation" \
                           --second-encodings "${encoding}" --second-datasets "${dataset}" --second-kvalues "${k_values[@]}" \
                           "${second_avg_on_runs[@]}" --second-dist-estimate "${dist_estimates[1]}" \
                           --metric "${metric}" --x-axis-prop "k" \
                           --legend-labels "${dist_estimates[@]}" --legend-position "${legend_position}"\
                           --x-ticks-labels "${k_values_strings[@]}"  --x-label "k value" --y-label "${metric_name}" \
                           --title "'${metric_name^}' distribution for different k values (sim., ${encoding}, ${dataset_string}${title_suffix})" \
                           --y-limits "${y_limits[@]}" \
                           --out-file "${plots_directory}/dist_estimates_comp/${metric}/${dataset}-simulation_${encoding}_${dist_estimates[0]}_vs_simulation_${encoding}_${dist_estimates[1]}-${out_file_spec}${metric}_boxplot${extension}"
                done
            done
        done
    done
fi


if [ "${k_value_based_box}" == "true" ]; then
    echo "Box plots: k-value-based comparisons (avg on runs & no avg on runs)"

    first_cltd_res_file="${exps_res_file}"
    second_cltd_res_file="${exps_res_file}"

    declare -a encodings=("extension" "translation")

    declare -a datasets=("01_iris_setosa_versicolor" "01_iris_setosa_virginica" "01_iris_versicolor_virginica"
                         "02_transfusion" "03_vertebral_column_2C" "04_seeds_1_2" "05_ecoli_cp_im" "06_glasses_1_2"
                         "07_breast_tissue_adi_fadmasgla" "08_breast_cancer" "09_accent_recognition_uk_us" "10_leaf_11_9")
    declare -a datasets_strings=("1a" "1b" "1c" "2" "3" "4" "5" "6" "7" "8" "9" "10")

    declare -a k_values=(3 5 7 9)

    declare -a avg_on_runs=("true" "false")

    declare -a dist_estimates=("avg" "diff")

    declare -a metrics=("accuracy" "jaccard_index" "average_jaccard_index")

    declare -a y_limits=(-0.05 1.05)

    for k_value in "${k_values[@]}"; do
        for avg_on_runs_flag in "${avg_on_runs[@]}"; do
            if [ "${avg_on_runs_flag}" == "true" ]; then
                first_avg_on_runs=("--first-avg-on-runs")
                second_avg_on_runs=("--second-avg-on-runs")
                title_suffix=""
                plots_directory="${plots_root_dir}/boxplots/avg_on_runs/k_value_based"
                out_file_spec=""
                # statistical_test="wilcoxon"
            else
                first_avg_on_runs=()
                second_avg_on_runs=()
                title_suffix=", no avg"
                plots_directory="${plots_root_dir}/boxplots/no_avg_on_runs/k_value_based"
                out_file_spec="no_avg_on_runs-"
                # statistical_test="ranksums"
            fi

            for metric in "${metrics[@]}"; do
                metric_name="${metric//_/ }"

                legend_position="upper left"
                if [ "${metric}" == "accuracy" ]; then
                    legend_position="lower left"
                fi

                # encodings comparison
                for dist_estimate in "${dist_estimates[@]}"; do
                    python visualization/generate_boxplot.py \
                           --first-cltd-res-file "${first_cltd_res_file}" --first-exec-type "local_simulation" \
                           --first-encodings "${encodings[0]}" --first-datasets "${datasets[@]}" --first-kvalues "${k_value}" \
                           "${first_avg_on_runs[@]}" --first-dist-estimate "${dist_estimate}" \
                           --second-cltd-res-file "${second_cltd_res_file}" --second-exec-type "local_simulation" \
                           --second-encodings "${encodings[1]}" --second-datasets "${datasets[@]}" --second-kvalues "${k_value}" \
                           "${second_avg_on_runs[@]}" --second-dist-estimate "${dist_estimate}" \
                           --metric "${metric}" --x-axis-prop "dataset" \
                           --legend-labels "${encodings[@]}" --legend-position "${legend_position}" \
                           --x-ticks-labels "${datasets_strings[@]}"  --x-label "dataset" --y-label "${metric_name}" \
                           --title "'${metric_name^}' distribution on different datasets (simulation, ${dist_estimate}, k=${k_value}${title_suffix})" \
                           --y-limits "${y_limits[@]}" \
                           --out-file "${plots_directory}/encodings_comp/${metric}/k_${k_value}-simulation_${encodings[0]}_${dist_estimate}_vs_simulation_${encodings[1]}_${dist_estimate}-${out_file_spec}${metric}_boxplot${extension}"
                done

                # distance estimates comparison
                for encoding in "${encodings[@]}"; do
                    python visualization/generate_boxplot.py \
                           --first-cltd-res-file "${first_cltd_res_file}" --first-exec-type "local_simulation" \
                           --first-encodings "${encoding}" --first-datasets "${datasets[@]}" --first-kvalues "${k_value}" \
                           "${first_avg_on_runs[@]}" --first-dist-estimate "${dist_estimates[0]}" \
                           --second-cltd-res-file "${second_cltd_res_file}" --second-exec-type "local_simulation" \
                           --second-encodings "${encoding}" --second-datasets "${datasets[@]}" --second-kvalues "${k_value}" \
                           "${second_avg_on_runs[@]}" --second-dist-estimate "${dist_estimates[1]}" \
                           --metric "${metric}" --x-axis-prop "dataset" \
                           --legend-labels "${dist_estimates[@]}" --legend-position "${legend_position}" \
                           --x-ticks-labels "${datasets_strings[@]}"  --x-label "dataset" --y-label "${metric_name}" \
                           --title "'${metric_name^}' distribution on different datasets (simulation, ${encoding}, k=${k_value}${title_suffix})" \
                           --y-limits "${y_limits[@]}" \
                           --out-file "${plots_directory}/dist_estimates_comp/${metric}/k_${k_value}-simulation_${encoding}_${dist_estimates[0]}_vs_simulation_${encoding}_${dist_estimates[1]}-${out_file_spec}${metric}_boxplot${extension}"
                done
            done
        done
    done
fi


if [ "${classical_baseline_qknn_comp_scatter}" == "true" ]; then
    echo "Scatter plots: comparisons of classical baseline methods and quantum k-NN"

    cltd_res_file_x="${classical_baseline_res_file}"
    cltd_res_file_y="${exps_res_file}"

    declare -a baseline_methods=("knn_cosine" "random_forest" "svm_linear" "svm_rbf")

    declare -a exec_types=("statevector" "local_simulation")
    declare -a encodings=("translation")
    declare -a dist_estimates=("avg")

    plots_directory="${plots_root_dir}/scatterplots/baseline_qknn_comp"

    for baseline_method in "${baseline_methods[@]}"; do
        baseline_method_name="${baseline_method//rbf/gaussian}"
        baseline_method_name="${baseline_method_name//_/ }"

        for exec_type in "${exec_types[@]}"; do
            exec_type_name="${exec_type//local_simulation/simulation}"

            for encoding in "${encodings[@]}"; do
                for dist_estimate in "${dist_estimates[@]}"; do
                        python visualization/generate_scatterplot.py \
                            --x-cltd-res-file "${cltd_res_file_x}" --x-exec-type "${baseline_method}" --x-encodings "_" \
                            --x-kvalues 3 5 7 9 --x-avg-on-runs --x-dist-estimate "_" --x-partition-key "k" \
                            --y-cltd-res-file "${cltd_res_file_y}" --y-exec-type "${exec_type}" --y-encodings "${encoding}" \
                            --y-kvalues 3 5 7 9 --y-avg-on-runs --y-dist-estimate "${dist_estimate}" --y-partition-key "k" \
                            --metric "accuracy" --legend-labels "k=3" "k=5" "k=7" "k=9" \
                            --x-label "${baseline_method_name}" --y-label "${exec_type_name}, ${encoding}, ${dist_estimate}" \
                            --title "Comparison with a baseline method in 'accuracy'" \
                            --out-file "${plots_directory}/accuracy/${baseline_method}_vs_${exec_type}_${encoding}_${dist_estimate}-accuracy_scatterplot${extension}" \
                            --statistical-test "wilcoxon"
                done
            done
        done
    done
fi


if [ "${diff_nums_shots_exec_types_comp_scatter}" == "true" ]; then
    echo "Scatter plots: execution types comparisons for different numbers of shots"

    cltd_res_file_x="${exps_res_file}"

    declare -a encodings=("extension")
    declare -a dist_estimates=("avg")

    declare -a num_shots_list=(512 1024 2048 4096 8192)

    declare -a metrics=("accuracy" "jaccard_index" "average_jaccard_index")

    plots_directory="${plots_root_dir}/scatterplots/diff_nums_shots_exec_types_comp"

    for encoding in "${encodings[@]}"; do
        for dist_estimate in "${dist_estimates[@]}"; do
            for num_shots in "${num_shots_list[@]}"; do
                cltd_res_file_y="${num_shots_res_file//#/${num_shots}}"
                num_shots_out_file_spec="${num_shots}"

                if (( num_shots == 512 )); then
                    num_shots_out_file_spec="0${num_shots_out_file_spec}"
                elif (( num_shots == 1024 )); then
                    cltd_res_file_y="${exps_res_file}"
                fi

                for metric in "${metrics[@]}"; do
                    metric_name="${metric//_/ }"

                    python visualization/generate_scatterplot.py \
                           --x-cltd-res-file "${cltd_res_file_x}" --x-exec-type "statevector" --x-encodings "${encoding}" \
                           --x-kvalues 3 5 7 9 --x-avg-on-runs --x-dist-estimate "${dist_estimate}" --x-partition-key "k" \
                           --y-cltd-res-file "${cltd_res_file_y}" --y-exec-type "local_simulation" --y-encodings "${encoding}" \
                           --y-kvalues 3 5 7 9 --y-avg-on-runs --y-dist-estimate "${dist_estimate}" --y-partition-key "k" \
                           --metric "${metric}" \
                           --legend-labels "k=3" "k=5" "k=7" "k=9" \
                           --x-label "statevector (${encoding}, ${dist_estimate})" \
                           --y-label "simulation, ${num_shots} shots (${encoding}, ${dist_estimate})" \
                           --title "Exec. types comp. in '${metric_name}' (${num_shots} shots)" \
                           --out-file "${plots_directory}/${metric}/statevector_${encoding}_${dist_estimate}_vs_simulation_${encoding}_${dist_estimate}-${num_shots_out_file_spec}_shots-${metric}_scatterplot${extension}" \
                           --statistical-test "wilcoxon"
                done
            done
        done
    done
fi


if [ "${nums_shots_comp_scatter}" == "true" ]; then
    echo "Scatter plots: numbers of shots comparisons"

    declare -a encodings=("extension")
    declare -a dist_estimates=("avg")

    declare -a num_shots_list=(512 1024 2048 4096 8192)
    last_num_shots_index=$((${#num_shots_list[@]} - 1))

    declare -a metrics=("accuracy" "jaccard_index" "average_jaccard_index")

    plots_directory="${plots_root_dir}/scatterplots/nums_shots_comp"

    for encoding in "${encodings[@]}"; do
        for dist_estimate in "${dist_estimates[@]}"; do
            for i in $(seq 0 ${last_num_shots_index}); do
                num_shots_x="${num_shots_list[i]}"

                cltd_res_file_x="${num_shots_res_file//#/${num_shots_x}}"
                num_shots_x_out_file_spec="${num_shots_x}"
                if (( num_shots_x == 512 )); then
                    num_shots_x_out_file_spec="0${num_shots_x_out_file_spec}"
                elif (( num_shots_x == 1024 )); then
                    cltd_res_file_x="${exps_res_file}"
                fi

                for j in $(seq $((i + 1)) ${last_num_shots_index}); do
                    num_shots_y="${num_shots_list[j]}"

                    cltd_res_file_y="${num_shots_res_file//#/${num_shots_y}}"
                    num_shots_y_out_file_spec="${num_shots_y}"
                    if (( num_shots_y == 512 )); then
                        num_shots_y_out_file_spec="0${num_shots_y_out_file_spec}"
                    elif (( num_shots_y == 1024 )); then
                        cltd_res_file_y="${exps_res_file}"
                    fi

                    for metric in "${metrics[@]}"; do
                        metric_name="${metric//_/ }"

                        python visualization/generate_scatterplot.py \
                               --x-cltd-res-file "${cltd_res_file_x}" --x-exec-type "local_simulation" --x-encodings "${encoding}" \
                               --x-kvalues 3 5 7 9 --x-avg-on-runs --x-dist-estimate "${dist_estimate}" --x-partition-key "k" \
                               --y-cltd-res-file "${cltd_res_file_y}" --y-exec-type "local_simulation" --y-encodings "${encoding}" \
                               --y-kvalues 3 5 7 9 --y-avg-on-runs --y-dist-estimate "${dist_estimate}" --y-partition-key "k" \
                               --metric "${metric}" \
                               --legend-labels "k=3" "k=5" "k=7" "k=9" \
                               --x-label "simulation, ${num_shots_x} shots (${encoding}, ${dist_estimate})" \
                               --y-label "simulation, ${num_shots_y} shots (${encoding}, ${dist_estimate})" \
                               --title "Numbers of shots comparison in '${metric_name}'" \
                               --out-file "${plots_directory}/${metric}/simulation_${encoding}_${dist_estimate}_${num_shots_x_out_file_spec}_shots_vs_simulation_${encoding}_${dist_estimate}_${num_shots_y_out_file_spec}_shots-${metric}_scatterplot${extension}" \
                               --statistical-test "wilcoxon"
                    done
                done
            done
        done
    done
fi


if [ "${num_shots_diff_box}" == "true" ]; then
    echo "Difference box plots: number-of-shots-based differences"

    baseline_num_shots=512
    baseline_cltd_res_file="${num_shots_res_file//#/${baseline_num_shots}}"
    declare -a comp_num_shots_list=(1024 2048 4096 8192)

    declare -a encodings=("extension")

    declare -a datasets=("01_iris_setosa_versicolor" "01_iris_setosa_virginica" "01_iris_versicolor_virginica"
                         "02_transfusion" "03_vertebral_column_2C" "04_seeds_1_2" "05_ecoli_cp_im" "06_glasses_1_2"
                         "07_breast_tissue_adi_fadmasgla" "08_breast_cancer" "09_accent_recognition_uk_us" "10_leaf_11_9")

    declare -a k_values=(3 5 7 9)

    declare -a dist_estimates=("avg")

    declare -a metrics=("accuracy" "jaccard_index" "average_jaccard_index")

    plots_directory="${plots_root_dir}/diff_boxplots/num_shots"

    for encoding in "${encodings[@]}"; do
        for dist_estimate in "${dist_estimates[@]}"; do
            baseline_args=()
            comp_args=()
            x_ticks_labels=()
            for comp_num_shots in "${comp_num_shots_list[@]}"; do
                comp_cltd_res_file="${num_shots_res_file//#/${comp_num_shots}}"
                if (( comp_num_shots == 1024 )); then
                    comp_cltd_res_file="${exps_res_file}"
                fi

                baseline_args+=("--baseline-cltd-res-file" "${baseline_cltd_res_file}"
                                "--baseline-exec-type" "local_simulation"
                                "--baseline-encodings" "${encoding}"
                                "--baseline-datasets" "${datasets[@]}"
                                "--baseline-kvalues" "${k_values[@]}"
                                "--baseline-avg-on-runs"
                                "--baseline-dist-estimate" "${dist_estimate}")

                comp_args+=("--cltd-res-file" "${comp_cltd_res_file}"
                            "--exec-type" "local_simulation"
                            "--encodings" "${encoding}"
                            "--datasets" "${datasets[@]}"
                            "--kvalues" "${k_values[@]}"
                            "--avg-on-runs"
                            "--dist-estimate" "${dist_estimate}")

                x_ticks_labels+=("${comp_num_shots}")
            done

            for metric in "${metrics[@]}"; do
                metric_name="${metric//_/ }"
                metric_name="${metric_name//j/J}"

                y_limits=(-0.3 0.3)
                if [ "${metric}" == "accuracy" ]; then
                    y_limits=(-0.4 0.4)
                fi

                python visualization/generate_diff_boxplot.py "${baseline_args[@]}" "${comp_args[@]}" \
                       --metric "${metric}" --show-means --x-ticks-labels "${x_ticks_labels[@]}" \
                       --x-label "Number of shots compared to ${baseline_num_shots}" \
                       --y-label "Difference in avg. '${metric_name}'" \
                       --title "Distribution of fold avg. '${metric_name}' difference w.r.t. ${baseline_num_shots}\nshots (config: simulation, ${encoding}, ${dist_estimate})" \
                       --y-limits "${y_limits[@]}" \
                       --out-file "${plots_directory}/${metric}/simulation_${encoding}_${dist_estimate}-${metric}_num_shots_diff_boxplot${extension}" \
                       --statistical-tests "ttest-1samp" "wilcoxon"
            done
        done
    done
fi


if [ "${comp_summary_diff_box}" == "true" ]; then
    echo "Difference box plots: summary of comparisons (encodings and distance estimates)"

    baseline_cltd_res_file="${exps_res_file}"
    comp_cltd_res_file="${exps_res_file}"

    declare -a baseline_encodings=("translation" "translation" "extension" "translation" "translation")
    declare -a comp_encodings=("extension" "extension" "extension" "translation" "extension")

    declare -a datasets=("01_iris_setosa_versicolor" "01_iris_setosa_virginica" "01_iris_versicolor_virginica"
                         "02_transfusion" "03_vertebral_column_2C" "04_seeds_1_2" "05_ecoli_cp_im" "06_glasses_1_2"
                         "07_breast_tissue_adi_fadmasgla" "08_breast_cancer" "09_accent_recognition_uk_us" "10_leaf_11_9")

    declare -a k_values=(3 5 7 9)

    declare -a baseline_dist_estimates=("avg" "diff" "diff" "diff" "diff")
    declare -a comp_dist_estimates=("avg" "diff" "avg" "avg" "avg")

    declare -a metrics=("accuracy" "jaccard_index" "average_jaccard_index")

    plots_directory="${plots_root_dir}/diff_boxplots/comp_summary"

    baseline_args=()
    comp_args=()
    x_ticks_labels=()
    last_comp_index=$((${#comp_encodings[@]} - 1))
    for i in $(seq 0 ${last_comp_index}); do
        baseline_encoding="${baseline_encodings[i]}"
        baseline_dist_estimate="${baseline_dist_estimates[i]}"
        comp_encoding="${comp_encodings[i]}"
        comp_dist_estimate="${comp_dist_estimates[i]}"

        baseline_args+=("--baseline-cltd-res-file" "${baseline_cltd_res_file}"
                        "--baseline-exec-type" "local_simulation"
                        "--baseline-encodings" "${baseline_encoding}"
                        "--baseline-datasets" "${datasets[@]}"
                        "--baseline-kvalues" "${k_values[@]}"
                        "--baseline-avg-on-runs"
                        "--baseline-dist-estimate" "${baseline_dist_estimate}")

        comp_args+=("--cltd-res-file" "${comp_cltd_res_file}"
                    "--exec-type" "local_simulation"
                    "--encodings" "${comp_encoding}"
                    "--datasets" "${datasets[@]}"
                    "--kvalues" "${k_values[@]}"
                    "--avg-on-runs"
                    "--dist-estimate" "${comp_dist_estimate}")

        x_ticks_labels+=("(${comp_encoding}, ${comp_dist_estimate})\n-\n(${baseline_encoding}, ${baseline_dist_estimate})")
    done

    for metric in "${metrics[@]}"; do
        metric_name="${metric//_/ }"
        metric_name="${metric_name//j/J}"

        y_limits=(-0.2 0.2)
        if [ "${metric}" == "accuracy" ]; then
            y_limits=(-0.55 0.55)
        fi

        python visualization/generate_diff_boxplot.py "${baseline_args[@]}" "${comp_args[@]}" \
               --metric "${metric}" --show-means --vertical-separation --x-ticks-labels "${x_ticks_labels[@]}" \
               --x-label "Configurations compared" --y-label "Difference in avg. '${metric_name}'" \
               --title "Distribution of fold avg. '${metric_name}' difference\nfor various configurations comparisons" \
               --y-limits "${y_limits[@]}" \
               --out-file "${plots_directory}/${metric}/encodings_and_dist_estimates_comp_summary-${metric}_diff_boxplot${extension}" \
               --statistical-tests "ttest-1samp" "wilcoxon"
    done
fi
