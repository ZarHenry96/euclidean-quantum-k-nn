import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

from utils import load_data, get_adaptive_limits


def generate_diff_boxplot(bp_data, x_ticks_labels, x_label, y_label, title, y_limits, out_file):
    matplotlib.rcParams['figure.dpi'] = 300

    # Create the figure
    width, height = 9, 8
    plt.figure(figsize=(width, height))

    # Generate the boxplot
    plt.boxplot(bp_data)

    # Plot a horizontal dashed line for zero difference
    x_left_limit, x_right_limit = 0, len(bp_data) + 1
    plt.hlines(0, x_left_limit, x_right_limit, label='Zero difference', linestyles='--', colors='grey')

    # Fontsizes
    title_fontsize = 15
    axis_label_fontsize = 13
    tick_label_fontsize = 13
    legend_fontsize = 13

    # Set other properties
    plt.xticks(np.arange(1, len(bp_data) + 1), x_ticks_labels)
    plt.tick_params(axis='both', which='major', labelsize=tick_label_fontsize)
    plt.legend(fontsize=legend_fontsize)

    plt.xlabel(x_label, fontsize=axis_label_fontsize, labelpad=16)
    plt.ylabel(y_label, fontsize=axis_label_fontsize, labelpad=16)
    plt.title(title, fontsize=title_fontsize, pad=18)

    plt.xlim(x_left_limit, x_right_limit)
    plt.ylim(y_limits[0], y_limits[1])

    # Save the figure
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


def main(baseline_cltd_res_file, baseline_exec_type, baseline_encodings, baseline_datasets, baseline_kvalues,
         baseline_avg_on_runs, baseline_dist_estimate, cltd_res_file, exec_type, encodings, datasets, kvalues,
         avg_on_runs, dist_estimate, metric, x_ticks_labels, x_label, y_label, title, adaptive_y_limits, y_limits,
         out_file):
    boxplot_data = []

    # Load data and compute the differences
    for bl_res_file, bl_exec_type, bl_encodings, bl_datasets, bl_kvalues, bl_dist_estimate, \
        comp_res_file, comp_exec_type, comp_encodings, comp_datasets, comp_kvalues, comp_dist_estimate in \
        zip(baseline_cltd_res_file, baseline_exec_type, baseline_encodings, baseline_datasets, baseline_kvalues,
            baseline_dist_estimate, cltd_res_file, exec_type, encodings, datasets, kvalues, dist_estimate):
        baseline_data = load_data(bl_res_file, bl_exec_type, bl_encodings, bl_datasets, bl_kvalues,
                                  baseline_avg_on_runs, bl_dist_estimate, metric, None)
        comparison_data = load_data(comp_res_file, comp_exec_type, comp_encodings, comp_datasets, comp_kvalues,
                                    avg_on_runs, comp_dist_estimate, metric, None)
        boxplot_data.append(list(np.array(comparison_data) - np.array(baseline_data)))

    # Pre-plotting operations
    title = title.replace('\\n', '\n')
    if adaptive_y_limits:
        y_limits = get_adaptive_limits(boxplot_data, [])
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    # Generate the plot
    generate_diff_boxplot(boxplot_data, x_ticks_labels, x_label, y_label, title, y_limits, out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for generating difference boxplots for experiments on the'
                                                 'classical/quantum k-NN based on the Euclidean distance.')
    parser.add_argument('--baseline-cltd-res-file', metavar='baseline_cltd_res_file', type=str, nargs='?',
                        action='append', default=[], help='baseline collected results file for the difference boxplot.')
    parser.add_argument('--baseline-exec-type', metavar='baseline_exec_type', type=str, nargs='?', action='append',
                        default=[], help='baseline execution type.')
    parser.add_argument('--baseline-encodings', metavar='baseline_encodings', type=str, nargs='+', action='append',
                        default=[], help='list of baseline encodings.')
    parser.add_argument('--baseline-datasets', metavar='baseline_datasets', type=str, nargs='+', action='append',
                        default=[], help='list of baseline datasets.')
    parser.add_argument('--baseline-kvalues', metavar='baseline_kvalues', type=int, nargs='+', action='append',
                        default=[], help='list of baseline k values.')
    parser.add_argument('--baseline-avg-on-runs', dest='baseline_avg_on_runs', action='store_const', const=True,
                        default=False, help='average baseline fold metric values on runs.')
    parser.add_argument('--baseline-dist-estimate', metavar='baseline_dist_estimate', type=str, nargs='?',
                        action='append', default=[], help='baseline distance estimate.')
    parser.add_argument('--cltd-res-file', metavar='cltd_res_file', type=str, nargs='?', action='append', default=[],
                        help='comparison collected results file for the difference boxplot.')
    parser.add_argument('--exec-type', metavar='exec_type', type=str, nargs='?', action='append', default=[],
                        help='comparison execution type.')
    parser.add_argument('--encodings', metavar='encodings', type=str, nargs='+', action='append', default=[],
                        help='list of comparison encodings.')
    parser.add_argument('--datasets', metavar='datasets', type=str, nargs='+', action='append', default=[],
                        help='list of comparison datasets.')
    parser.add_argument('--kvalues', metavar='kvalues', type=int, nargs='+', action='append', default=[],
                        help='list of comparison k values.')
    parser.add_argument('--avg-on-runs', dest='avg_on_runs', action='store_const', const=True, default=False,
                        help='average comparison fold metric values on runs.')
    parser.add_argument('--dist-estimate', metavar='dist_estimate', type=str, nargs='?', action='append', default=[],
                        help='comparison distance estimate.')
    parser.add_argument('--metric', metavar='metric', type=str, nargs='?', default='accuracy', help='metric to plot, '
                        'allowed values: accuracy, jaccard_index, average_jaccard_index (accuracy is used by default).')
    parser.add_argument('--x-ticks-labels', metavar='x_ticks_labels', type=str, nargs='+', default=[],
                        help='ticks labels for the x axis.')
    parser.add_argument('--x-label', metavar='x_label', type=str, nargs='?', default='', help='label for the x axis.')
    parser.add_argument('--y-label', metavar='y_label', type=str, nargs='?', default='', help='label for the y axis.')
    parser.add_argument('--title', metavar='title', type=str, nargs='?', default='', help='chart title.')
    parser.add_argument('--adaptive-y-limits', dest='adaptive_y_limits', action='store_const', const=True,
                        default=False, help='set the y axis limits automatically (based on the given values).')
    parser.add_argument('--y-limits', metavar='y_limits', type=float, nargs='+', default=[-1.05, 1.05],
                        help='limit values (lower and upper) for the y axis; the default values are -1.05 and 1.05. '
                             'This option is ignored if the adaptive-y-limits flag is enabled.')
    parser.add_argument('--out-file', metavar='out_file', type=str, default='diff_boxplot.pdf',
                        help='output file path.')
    args = parser.parse_args()

    if len(args.baseline_cltd_res_file) != 0 and len(args.baseline_exec_type) != 0 and \
            len(args.cltd_res_file) != 0 and len(args.exec_type) != 0:
        main(args.baseline_cltd_res_file, args.baseline_exec_type, args.baseline_encodings, args.baseline_datasets,
             args.baseline_kvalues, args.baseline_avg_on_runs, args.baseline_dist_estimate, args.cltd_res_file,
             args.exec_type, args.encodings, args.datasets, args.kvalues, args.avg_on_runs, args.dist_estimate,
             args.metric, args.x_ticks_labels, args.x_label, args.y_label, args.title, args.adaptive_y_limits,
             args.y_limits, args.out_file)
