import sys
import argparse
import os
import json
from collections import defaultdict
import csv
import matplotlib.pyplot as plt

SUPPORTED_MODELS = ['beta-vae', 'nn-critic', 'hybrid-critic', 'self-critic']

def aggregate_results(report_dir):
    model_names = os.listdir(report_dir)
    aggregate_results = {}
    for model_name in model_names:
        if model_name not in SUPPORTED_MODELS:
            continue
        model_result = aggregate_results[model_name] = {}
        model_dir = os.path.join(report_dir, model_name)
        dataset_names = os.listdir(model_dir)
        for dataset_name in dataset_names:
            model_ds_result = model_result[dataset_name] = defaultdict(list)
            dataset_dir = os.path.join(model_dir, dataset_name)
            exp_names = os.listdir(dataset_dir)
            for exp_name in exp_names:
                exp_dir = os.path.join(dataset_dir, exp_name)
                if not os.path.isdir(exp_dir):
                    continue
                
                metrics_file = os.path.join(exp_dir, 'metrics.json')
                if not os.path.exists(metrics_file):
                    continue
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                
                for metric_name, metric_value in metrics.items():
                    model_ds_result[metric_name].append(metric_value)


    # Compute mean and std
    aggregate_results_stats = {}
    for model_name, model_result in aggregate_results.items():
        model_result_stats = aggregate_results_stats[model_name] = {}
        for dataset_name, dataset_result in model_result.items():
            dataset_result_stats = model_result_stats[dataset_name] = {}
            for metric_name, metric_values in dataset_result.items():
                mean = sum(metric_values) / len(metric_values)
                std = (sum((x - mean)**2 for x in metric_values) / len(metric_values))**0.5
                dataset_result_stats[metric_name] = {
                    'mean': round(mean, 2),
                    'std': round(std, 2)
                }
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    out_report = os.path.join(output_dir, 'agg_metrics.json')
    with open(out_report, 'w') as f:
        json.dump(aggregate_results_stats, f, indent=4)


    # Now dump in a table format

    # only work for one dataset
    dataset_names = list(aggregate_results_stats['beta-vae'].keys())
    assert len(dataset_names) == 1, 'Only work for one dataset'
    dataset_name = dataset_names[0]

    def get_table_line(model_name, dataset_name):
        line = f' '
        for metric in ['Negative log likelihood', 'Mutual information', 'Active units', 'Density', 'Coverage']:
            line += f'& {aggregate_results_stats[model_name][dataset_name][metric]["mean"]} ({aggregate_results_stats[model_name][dataset_name][metric]["std"]}) '
        return line


    latex_table_str = f'''\\begin{{table}}
\caption{{Experimental results on the {dataset_name} }}\label{{tab1}}
\\begin{{tabular}}{{|l|p{{2.2cm}}|p{{2cm}}|p{{1.5cm}}|p{{2cm}}|p{{2cm}}|}}
\hline
Model &  NLL & MI & AU & Density & Coverage \\\\
\hline
VAE  {get_table_line('beta-vae', dataset_name)} \\\\
Self critic {get_table_line('self-critic', dataset_name)} \\\\
Hybrid critic {get_table_line('hybrid-critic', dataset_name)} \\\\
Neural critic {get_table_line('nn-critic', dataset_name)} \\\\
\hline
\end{{tabular}}
\end{{table}}
'''

    out_table = os.path.join(output_dir, 'agg_metrics.tex')
    with open(out_table, 'w') as f:
        f.write(latex_table_str)





def plot_graphs(report_dir):
    graph_value_files = list(filter(lambda x: x.endswith('.csv'), os.listdir(report_dir)))
    graph_value_files = list(map(lambda x: os.path.join(report_dir, x), graph_value_files))

    graph_lines = {}
    for graph_value_file in graph_value_files:
        graph_name = os.path.basename(graph_value_file).split('.')[0]
        line_name = graph_name.split('_')[1].replace('beta-', '').replace('nn','neural').replace('-', ' ').title()
       
        with open(graph_value_file, 'r') as f:
            reader = csv.reader(f)
            data = list(reader)[1:]
            data = list(map(lambda x: (float(x[1]), float(x[2])), data))
            data.sort(key=lambda x: x[0])
            graph_lines[line_name] = data

    
    plt.figure(figsize=(10, 6))
    plt.xlabel('Epoch') 
    plt.ylabel('MI')
    

    for line_name, line_data in graph_lines.items():
        x, y = zip(*line_data)
        plt.plot(x, y, label=line_name)
    plt.legend()
    plt.savefig(os.path.join(report_dir, 'mi.svg'), format='svg', dpi=1200)



        





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--report_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--graph", type=bool, default=False)
    args = parser.parse_args(sys.argv[1:])

    if args.graph:
        plot_graphs(args.report_dir)
    else:
        aggregate_results(args.report_dir)
