import argparse
import matplotlib
import matplotlib.pyplot as plt
import os

from utils import load_data, get_adaptive_limits, compute_statistic


def generate_scatterplot(x_data, y_data, legend_labels, legend_position, x_label, y_label, title, plot_limits,
                         out_file, verbose):
    if out_file.endswith('.pdf'):
        dpi = 300
        width, height = (9, 9) if '\\n' not in title else (10, 10)
    else:
        dpi = 150
        px = 1 / dpi
        width, height = 1350*px, 1350*px

    matplotlib.rcParams['figure.dpi'] = dpi
    fig, ax = plt.subplots(figsize=(width, height))

    title_fontsize = 21
    axis_label_fontsize = 20
    tick_label_fontsize = 20
    legend_fontsize = 17.5

    markers = ['o', '*', '+', '.']
    markers_base_size = 100
    markers_sizes = [markers_base_size, markers_base_size+2, markers_base_size+1, markers_base_size+1]
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    if verbose:
        print('\n{}\n{} vs {}'.format(title.capitalize(), x_label.capitalize(), y_label.capitalize()))

    for i, (x_vals, y_vals) in enumerate(zip(x_data, y_data)):
        legend_label = legend_labels[i] if len(legend_labels) != 0 else None

        ax.scatter(x_vals, y_vals, s=markers_sizes[i % len(markers_sizes)], marker=markers[i % len(markers)],
                   color=colors[i % len(colors)], label=legend_label)

        if verbose:
            x_better = [j for j in range(0, len(x_vals)) if x_vals[j] > y_vals[j]]
            y_better = [j for j in range(0, len(x_vals)) if x_vals[j] < y_vals[j]]
            equal_xy = [j for j in range(0, len(x_vals)) if x_vals[j] == y_vals[j]]
            print('')
            if legend_label is not None:
                print(legend_label.capitalize())
            print('    X better:  {}'.format(len(x_better)))
            print('    Y better:  {}'.format(len(y_better)))
            print('    X,Y equal: {}'.format(len(equal_xy)))

    l_limit, u_limit = plot_limits
    ax.plot([l_limit, u_limit], [l_limit, u_limit], ls="--", c="grey")

    if legend_labels is not None:
        plt.legend(loc=legend_position, fontsize=legend_fontsize)

    plt.xlim(l_limit, u_limit)
    plt.ylim(l_limit, u_limit)

    plt.tick_params(axis='both', which='major', labelsize=tick_label_fontsize)

    plt.xlabel(x_label, fontsize=axis_label_fontsize, labelpad=14)
    plt.ylabel(y_label, fontsize=axis_label_fontsize, labelpad=14)
    plt.title(title.replace('\\n', '\n'), fontsize=title_fontsize, pad=17)

    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()


def compute_scatter_statistics(x_data, y_data, legend_labels, statistical_test, statistics_out_file):
    with open(statistics_out_file, 'w') as sof:
        sof.write('statistical_test,legend_label,statistic,p_value,is_significant\n')

        for i, (x_vals, y_vals) in enumerate(zip(x_data, y_data)):
            legend_label = legend_labels[i] if len(legend_labels) != 0 else None

            statistic, p_value, is_significant = compute_statistic(statistical_test, x_vals, y_vals)

            sof.write('{},{},{},{},{}\n'.format(statistical_test, legend_label, statistic, p_value, is_significant))


def main(x_cltd_res_file, x_exec_type, x_encodings, x_datasets, x_kvalues, x_avg_on_runs, x_dist_estimate,
         x_partition_key, y_cltd_res_file, y_exec_type, y_encodings, y_datasets, y_kvalues, y_avg_on_runs,
         y_dist_estimate, y_partition_key, metric, legend_labels, legend_position, x_label, y_label, title,
         adaptive_plot_limits, plot_limits, out_file, statistical_test, statistics_out_file, verbose):
    # Load results data
    x_data = load_data(x_cltd_res_file, x_exec_type, x_encodings, x_datasets, x_kvalues, x_avg_on_runs, x_dist_estimate,
                       metric, x_partition_key)
    y_data = load_data(y_cltd_res_file, y_exec_type, y_encodings, y_datasets, y_kvalues, y_avg_on_runs, y_dist_estimate,
                       metric, y_partition_key)

    # Determine the plot limits, if the adaptive_plot_limits flag is enabled
    if adaptive_plot_limits:
        plot_limits = get_adaptive_limits(x_data, y_data)

    # Create output directory
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    # Generate the scatter plot
    generate_scatterplot(x_data, y_data, legend_labels, legend_position, x_label, y_label, title, plot_limits,
                         out_file, verbose)

    # Run the selected statistical test (if needed)
    if statistical_test is not None:
        if statistics_out_file is None:
            statistics_out_file = '{}_statistics.csv'.format(os.path.splitext(out_file)[0])
        else:
            os.makedirs(os.path.dirname(statistics_out_file), exist_ok=True)
        compute_scatter_statistics(x_data, y_data, legend_labels, statistical_test, statistics_out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for generating comparison scatter plots for experiments on '
                                                 'the classical/quantum k-NN based on the Euclidean distance.')
    parser.add_argument('--x-cltd-res-file', metavar='x_cltd_res_file', type=str, nargs='?', default=None,
                        help='collected_results.json file for the x axis.')
    parser.add_argument('--x-exec-type', metavar='x_exec_type', type=str, nargs='?', default=None,
                        help='execution type for the x axis results file.')
    parser.add_argument('--x-encodings', metavar='x_encodings', type=str, nargs='+', default=[],
                        help='list of encodings for the x axis results file; all are considered by default.')
    parser.add_argument('--x-datasets', metavar='x_datasets', type=str, nargs='+', default=[],
                        help='list of datasets names for the x axis results file; all are considered by default.')
    parser.add_argument('--x-kvalues', metavar='x_kvalues', type=int, nargs='+', default=[],
                        help='list of k values for the x axis results file; all are considered by default.')
    parser.add_argument('--x-avg-on-runs', dest='x_avg_on_runs', action='store_const', const=True, default=False,
                        help='average fold metric values on runs in the x axis results file.')
    parser.add_argument('--x-dist-estimate', metavar='x_dist_estimate', type=str, nargs='?', default=None,
                        help='distance estimate for the x axis results file; the first one is taken by default.')
    parser.add_argument('--x-partition-key', metavar='x_partition_key', type=str, nargs='?', default=None,
                        help='key for partitioning the results of the x axis results file; allowed values: '
                             'dataset, encoding, k.')
    parser.add_argument('--y-cltd-res-file', metavar='y_cltd_res_file', type=str, nargs='?', default=None,
                        help='collected_results.json file for the y axis.')
    parser.add_argument('--y-exec-type', metavar='y_exec_type', type=str, nargs='?', default=None,
                        help='execution type for the y axis results file.')
    parser.add_argument('--y-encodings', metavar='y_encodings', type=str, nargs='+', default=[],
                        help='list of encodings for the y axis results file; all are considered by default.')
    parser.add_argument('--y-datasets', metavar='y_datasets', type=str, nargs='+', default=[],
                        help='list of datasets names for the y axis results file; all are considered by default.')
    parser.add_argument('--y-kvalues', metavar='y_kvalues', type=int, nargs='+', default=[],
                        help='list of k values for the y axis results file; all are considered by default.')
    parser.add_argument('--y-avg-on-runs', dest='y_avg_on_runs', action='store_const', const=True, default=False,
                        help='average fold metric values on runs in the y axis results file.')
    parser.add_argument('--y-dist-estimate', metavar='y_dist_estimate', type=str, nargs='?', default=None,
                        help='distance estimate for the y axis results file; the first one is taken by default.')
    parser.add_argument('--y-partition-key', metavar='y_partition_key', type=str, nargs='?', default=None,
                        help='key for partitioning the results of the x axis results file; allowed values: '
                             'dataset, encoding, k.')
    parser.add_argument('--metric', metavar='metric', type=str, nargs='?', default='accuracy', help='metric to plot, '
                        'allowed values: accuracy, jaccard_index, average_jaccard_index (accuracy is used by default).')
    parser.add_argument('--legend-labels', metavar='legend_labels', type=str, nargs='+', default=[],
                        help='legend labels; the number of labels must match the number of partitions.')
    parser.add_argument('--legend-position', metavar='legend_position', type=str, nargs='?', default='upper left',
                        help='position of the legend in the plot.')
    parser.add_argument('--x-label', metavar='x_label', type=str, nargs='?', default='', help='label for the x axis.')
    parser.add_argument('--y-label', metavar='y_label', type=str, nargs='?', default='', help='label for the y axis.')
    parser.add_argument('--title', metavar='title', type=str, nargs='?', default='', help='chart title.')
    parser.add_argument('--adaptive-plot-limits', dest='adaptive_plot_limits', action='store_const', const=True,
                        default=False, help='set the plot limits automatically (based on the metric values).')
    parser.add_argument('--plot-limits', metavar='plot_limits', type=float, nargs='+', default=[-0.05, 1.05],
                        help='limit values (lower and upper) for the scatter plot; the default values are -0.05 and '
                             '1.05. This option is ignored if the adaptive-plot-limits flag is enabled.')
    parser.add_argument('--out-file', metavar='out_file', type=str, default='scatterplot.pdf', help='output file path.')
    parser.add_argument('--statistical-test', metavar='statistical_test', type=str, nargs='?', default=None,
                        help='statistical test to run on the scatter plot data (partition by partition), allowed '
                             'values are: ranksums, mannwhitneyu, wilcoxon. The last one is for paired data.')
    parser.add_argument('--statistics-out-file', metavar='statistics_out_file', type=str, nargs='?', default=None,
                        help='statistics output file path; the default value is \'[out-file]\'_statistics.csv.')
    parser.add_argument('--verbose', dest='verbose', action='store_const', const=True, default=False,
                        help='print some information about the scatter plot data.')
    args = parser.parse_args()

    if args.x_cltd_res_file is not None and args.x_exec_type is not None and args.y_cltd_res_file is not None \
            and args.y_exec_type is not None:
        main(args.x_cltd_res_file, args.x_exec_type, args.x_encodings, args.x_datasets, args.x_kvalues,
             args.x_avg_on_runs, args.x_dist_estimate, args.x_partition_key, args.y_cltd_res_file, args.y_exec_type,
             args.y_encodings, args.y_datasets, args.y_kvalues, args.y_avg_on_runs, args.y_dist_estimate,
             args.y_partition_key, args.metric, args.legend_labels, args.legend_position, args.x_label, args.y_label,
             args.title, args.adaptive_plot_limits, args.plot_limits, args.out_file, args.statistical_test,
             args.statistics_out_file, args.verbose)
