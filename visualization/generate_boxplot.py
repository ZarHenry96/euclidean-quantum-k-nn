import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

from utils import load_data, get_adaptive_limits, compute_statistic


def generate_boxplot(bp_data, x_ticks_labels, x_label, y_label, title, y_limits, out_file):
    matplotlib.rcParams['figure.dpi'] = 300

    # Create the figure
    width, height = min(max(3 + len(bp_data), 9), 16), 9
    plt.figure(figsize=(width, height))

    # Generate the boxplot
    plt.boxplot(bp_data)

    # Fontsizes
    title_fontsize = 15
    axis_label_fontsize = 13
    tick_label_fontsize = 13

    # Set other properties
    plt.xticks(np.arange(1, len(bp_data) + 1), x_ticks_labels)
    plt.tick_params(axis='both', which='major', labelsize=tick_label_fontsize)

    plt.xlabel(x_label, fontsize=axis_label_fontsize, labelpad=16)
    plt.ylabel(y_label, fontsize=axis_label_fontsize, labelpad=16)
    plt.title(title, fontsize=title_fontsize, pad=18)

    plt.ylim(y_limits[0], y_limits[1])

    # Save the figure
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


def compute_boxplot_statistics(bp_data, x_ticks_labels, statistical_test, statistics_out_file):
    with open(statistics_out_file, 'w') as sof:
        sof.write('statistical_test,1st_x_tick_label,2nd_x_tick_label,statistic,p_value,is_significant\n')

        for i in range(0, len(bp_data)-1):
            first_data, first_label = bp_data[i], x_ticks_labels[i]
            for j in range(i+1, len(bp_data)):
                second_data, second_label = bp_data[j], x_ticks_labels[j]

                statistic, p_value, is_significant = compute_statistic(statistical_test, first_data, second_data)

                sof.write('{},{},{},{},{},{}\n'.format(
                    statistical_test, first_label, second_label, statistic, p_value, is_significant
                ))


def set_bp_color(bp, color, background_color, median_color):
    plt.setp(bp['boxes'], color=color, facecolor=background_color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=median_color, linewidth=1.2)
    plt.setp(bp['fliers'], markeredgecolor=color)


def generate_multi_boxplot(bp_data, legend_labels, legend_position, x_ticks_labels, x_label, y_label, title, y_limits,
                           out_file):
    matplotlib.rcParams['figure.dpi'] = 300

    boxplots_num, ticks_num = len(bp_data), len(x_ticks_labels)

    # Create the figure
    width = min(max(3 + boxplots_num * ticks_num, 9), 26)
    height = 9 if width < 24 else 12
    plt.figure(figsize=(width, height))

    # Define position properties
    box_width, inter_boxes_dist = 0.6, 0.1
    max_pos_offset = ((box_width + inter_boxes_dist) / 2) * (boxplots_num - 1)
    pos_offsets = np.linspace(-max_pos_offset, max_pos_offset, boxplots_num)

    # Generate the boxplots
    boxplots = []
    for i, data_i in enumerate(bp_data):
        bp = plt.boxplot(data_i, positions=np.array(range(len(data_i))) * (float(boxplots_num)) + pos_offsets[i],
                         widths=box_width, patch_artist=True)
        boxplots.append(bp)

    # Set the colors
    colors = ['#D7191C', '#2C7BB6', '#B6B62C', '#662CB6', '#2EB62C']
    facecolors = ['mistyrose', 'lightcyan', 'lemonchiffon', '#D2B0FF', '#C4FFC4']
    mediancolors = ['maroon', 'midnightblue', 'olive', 'indigo', 'darkgreen']
    for i, bp in enumerate(boxplots):
        set_bp_color(bp, colors[i % len(colors)], facecolors[i % len(facecolors)],
                     mediancolors[i % len(mediancolors)])

    # Fontsizes
    title_fontsize = 16
    axis_label_fontsize = 14
    tick_label_fontsize = 14
    legend_fontsize = 13

    # Set the legend
    for i in range(0, len(boxplots)):
        plt.plot([], c=colors[i], label=legend_labels[i])
    plt.legend(loc=legend_position, fontsize=legend_fontsize)

    # Set other properties
    plt.xticks(range(0, ticks_num * boxplots_num, boxplots_num), x_ticks_labels)
    plt.tick_params(axis='both', which='major', labelsize=tick_label_fontsize)

    plt.xlabel(x_label, fontsize=axis_label_fontsize, labelpad=16)
    plt.ylabel(y_label, fontsize=axis_label_fontsize, labelpad=16)
    plt.title(title, fontsize=title_fontsize, pad=18)

    plt.xlim(-boxplots_num / 2, (ticks_num - 0.5) * boxplots_num)
    plt.ylim(y_limits[0], y_limits[1])

    # Save the figure
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


def compute_multi_boxplot_statistics(bp_data, legend_labels, x_ticks_labels, statistical_test, statistics_out_file):
    with open(statistics_out_file, 'w') as sof:
        sof.write('statistical_test,1st_legend_label,2nd_legend_label,x_tick_label,statistic,p_value,is_significant\n')

        for i in range(0, len(bp_data)-1):
            first_bp_data, first_legend_label = bp_data[i], legend_labels[i]
            for j in range(i+1, len(bp_data)):
                second_bp_data, second_legend_label = bp_data[j], legend_labels[j]
                for k, x_tick_label in enumerate(x_ticks_labels):
                    first_data, second_data = first_bp_data[k], second_bp_data[k]

                    statistic, p_value, is_significant = compute_statistic(statistical_test, first_data, second_data)

                    sof.write('{},{},{},{},{},{},{}\n'.format(
                        statistical_test, first_legend_label, second_legend_label, x_tick_label,
                        statistic, p_value, is_significant
                    ))


def main(first_cltd_res_file, first_exec_type, first_encodings, first_datasets, first_kvalues, first_avg_on_runs,
         first_dist_estimate, second_cltd_res_file, second_exec_type, second_encodings, second_datasets, second_kvalues,
         second_avg_on_runs, second_dist_estimate, metric, x_axis_prop, legend_labels, legend_position, x_ticks_labels,
         x_label, y_label, title, adaptive_y_limits, y_limits, out_file, statistical_test, statistics_out_file):
    # Load results data
    first_data = load_data(first_cltd_res_file, first_exec_type, first_encodings, first_datasets, first_kvalues,
                           first_avg_on_runs, first_dist_estimate, metric, x_axis_prop)
    second_data = []
    if second_cltd_res_file is not None and second_exec_type is not None:
        second_data = load_data(second_cltd_res_file, second_exec_type, second_encodings, second_datasets,
                                second_kvalues, second_avg_on_runs, second_dist_estimate, metric, x_axis_prop)

    # Determine the y limits, if the adaptive_y_limits flag is enabled
    if adaptive_y_limits:
        y_limits = get_adaptive_limits(first_data, second_data)

    # Create output directory
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    # Generate the boxplot
    if second_cltd_res_file is None or second_exec_type is None:
        generate_boxplot(first_data, x_ticks_labels, x_label, y_label, title, y_limits, out_file)
    else:
        generate_multi_boxplot([first_data, second_data], legend_labels, legend_position, x_ticks_labels, x_label,
                               y_label, title, y_limits, out_file)

    # Run the statistical test (if needed)
    if statistical_test is not None:
        if statistics_out_file is None:
            statistics_out_file = '{}_statistics.csv'.format(os.path.splitext(out_file)[0])
        else:
            os.makedirs(os.path.dirname(statistics_out_file), exist_ok=True)

        if second_cltd_res_file is None or second_exec_type is None:
            compute_boxplot_statistics(first_data, x_ticks_labels, statistical_test, statistics_out_file)
        else:
            compute_multi_boxplot_statistics([first_data, second_data], legend_labels, x_ticks_labels,
                                             statistical_test, statistics_out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for generating boxplots for simulation experiments on '
                                                 'the classical/quantum k-NN based on the Euclidean distance.')
    parser.add_argument('--first-cltd-res-file', metavar='first_cltd_res_file', type=str, nargs='?', default=None,
                        help='collected_results.json file containing the first set of data for the boxplot.')
    parser.add_argument('--first-exec-type', metavar='first_exec_type', type=str, nargs='?', default=None,
                        help='first execution type.')
    parser.add_argument('--first-encodings', metavar='first_encodings', type=str, nargs='+', default=[],
                        help='first list of encodings; all are considered by default.')
    parser.add_argument('--first-datasets', metavar='first_datasets', type=str, nargs='+', default=[],
                        help='first list of datasets; all are considered by default.')
    parser.add_argument('--first-kvalues', metavar='first_kvalues', type=int, nargs='+', default=[],
                        help='first list of k values; all are considered by default.')
    parser.add_argument('--first-avg-on-runs', dest='first_avg_on_runs', action='store_const', const=True,
                        default=False, help='average fold metric values on runs for the first set of data.')
    parser.add_argument('--first-dist-estimate', metavar='first_dist_estimate', type=str, nargs='?', default=None,
                        help='first distance estimate; the first one is taken by default.')
    parser.add_argument('--second-cltd-res-file', metavar='second_cltd_res_file', type=str, nargs='?', default=None,
                        help='collected_results.json file containing the second set of data for the boxplot.')
    parser.add_argument('--second-exec-type', metavar='second_exec_type', type=str, nargs='?', default=None,
                        help='second execution type.')
    parser.add_argument('--second-encodings', metavar='second_encodings', type=str, nargs='+', default=[],
                        help='second list of encodings; all are considered by default.')
    parser.add_argument('--second-datasets', metavar='second_datasets', type=str, nargs='+', default=[],
                        help='second list of datasets; all are considered by default.')
    parser.add_argument('--second-kvalues', metavar='second_kvalues', type=int, nargs='+', default=[],
                        help='second list of k values; all are considered by default.')
    parser.add_argument('--second-avg-on-runs', dest='second_avg_on_runs', action='store_const', const=True,
                        default=False, help='average fold metric values on runs for the second set of data.')
    parser.add_argument('--second-dist-estimate', metavar='second_dist_estimate', type=str, nargs='?', default=None,
                        help='second distance estimate; the first one is taken by default.')
    parser.add_argument('--metric', metavar='metric', type=str, nargs='?', default='accuracy', help='metric to plot, '
                        'allowed values: accuracy, jaccard_index, average_jaccard_index (accuracy is used by default).')
    parser.add_argument('--x-axis-prop', metavar='x_axis_prop', type=str, nargs='?', default='k', help='property used '
                        'for the x axis, allowed values: dataset, encoding, k. The default value is k.')
    parser.add_argument('--legend-labels', metavar='legend_labels', type=str, nargs='+', default=[], help='legend '
                        'labels (they must be two); the legend is shown only if two sets of data are considered.')
    parser.add_argument('--legend-position', metavar='legend_position', type=str, nargs='?', default='upper left',
                        help='position of the legend in the plot; the default position is upper left.')
    parser.add_argument('--x-ticks-labels', metavar='x_ticks_labels', type=str, nargs='+', default=[], 
                        help='ticks labels for the x axis; their number must match the number of x-axis-prop values.')
    parser.add_argument('--x-label', metavar='x_label', type=str, nargs='?', default='', help='label for the x axis.')
    parser.add_argument('--y-label', metavar='y_label', type=str, nargs='?', default='', help='label for the y axis.')
    parser.add_argument('--title', metavar='title', type=str, nargs='?', default='', help='chart title.')
    parser.add_argument('--adaptive-y-limits', dest='adaptive_y_limits', action='store_const', const=True,
                        default=False, help='set the y axis limits automatically (based on the metric values).')
    parser.add_argument('--y-limits', metavar='y_limits', type=float, nargs='+', default=[-0.05, 1.05],
                        help='limit values (lower and upper) for the y axis; the default values are -0.05 and 1.05. '
                             'This option is ignored if the adaptive-y-limits flag is enabled.')
    parser.add_argument('--out-file', metavar='out_file', type=str, default='boxplot.pdf', help='output file path.')
    parser.add_argument('--statistical-test', metavar='statistical_test', type=str, nargs='?', default=None,
                        help='statistical test to run on the boxplot data, allowed values are: ranksums, mannwhitneyu, '
                             'wilcoxon (the last one is for paired data). If two sets of data have been provided, the '
                             'test is run separately for each x-axis-prop value.')
    parser.add_argument('--statistics-out-file', metavar='statistics_out_file', type=str, nargs='?', default=None,
                        help='statistics output file path; the default value is \'[out-file]\'_statistics.csv.')
    args = parser.parse_args()

    if args.first_cltd_res_file is not None and args.first_exec_type is not None:
        main(args.first_cltd_res_file, args.first_exec_type, args.first_encodings, args.first_datasets,
             args.first_kvalues, args.first_avg_on_runs, args.first_dist_estimate, args.second_cltd_res_file,
             args.second_exec_type, args.second_encodings, args.second_datasets, args.second_kvalues,
             args.second_avg_on_runs, args.second_dist_estimate, args.metric, args.x_axis_prop, args.legend_labels,
             args.legend_position, args.x_ticks_labels, args.x_label, args.y_label, args.title, args.adaptive_y_limits,
             args.y_limits, args.out_file, args.statistical_test, args.statistics_out_file)
