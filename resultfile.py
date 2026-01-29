import re
import pandas as pd
from collections import defaultdict
import os


with open('./output/result.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()


results = defaultdict(lambda: defaultdict(dict))
current_method = None
current_label = "Overall"


method_pattern = re.compile(r'^(Improved KP|KP|GNN|MLP_GRID)')
label_pattern = re.compile(r'label: (.+)')
metric_patterns = {
    'Precision': re.compile(r'Precision\s+: ([0-9.]+)'),
    'Recall': re.compile(r'Recall\s+: ([0-9.]+)'),
    'F1 Score': re.compile(r'F1 Score\s+: ([0-9.]+)'),
    'MAE': re.compile(r'MAE \(Mean Abs\)\s+: ([0-9.]+)'),
    'MSE': re.compile(r'MSE \(Mean Sq\)\s+: ([0-9.]+)'),
    'RMSE': re.compile(r'RMSE \(Root MSE\)\s+: ([0-9.]+)')
}


for line in lines:
    method_match = method_pattern.match(line.strip())
    if method_match:
        current_method = method_match.group(1)
        current_label = "Overall"
        continue

    label_match = label_pattern.search(line)
    if label_match:
        current_label = label_match.group(1)
        continue

    for metric_name, pattern in metric_patterns.items():
        metric_match = pattern.search(line)
        if metric_match:
            results[current_label][current_method][metric_name] = float(metric_match.group(1))


all_labels = sorted(results.keys())
all_methods = ['Improved KP', 'KP', 'GNN', 'MLP_GRID']
metrics = ['Precision', 'Recall', 'F1 Score', 'MAE', 'MSE', 'RMSE']


os.makedirs("output/csv_tables", exist_ok=True)


for label in all_labels:
    data = []
    for method in all_methods:
        row = [results[label].get(method, {}).get(metric, None) for metric in metrics]
        data.append(row)
    df = pd.DataFrame(data, index=all_methods, columns=metrics)
    df.to_csv(f"output/csv_tables/{label.replace(':', '_')}.csv")
    print(f"Saved CSV for label: {label}")
