import plotly.graph_objects as go
import os
import numpy as np
import json

base_figure_folder = "./charts"


def compare_auc(item):
    return item[1]


def order_results_auc(results):
    result_list = []
    for key in results.keys():
        result_list.append((key, results[key]))
    return sorted(result_list, key=compare_auc)


def draw_bar(results, folder_name):
    folder = f'{base_figure_folder}/{folder_name}'
    if not os.path.exists(folder):
        os.makedirs(folder)

    avg_dict = dict()
    for experiment_name in results:

        use_case_names = []
        use_case_auc = []
        for use_case in results[experiment_name]:
            if use_case == "avg":
                avg_dict[experiment_name] = results[experiment_name][use_case]
            use_case_names.append(use_case)
            use_case_auc.append(results[experiment_name][use_case])

        data = [go.Bar(x=use_case_names, y=use_case_auc)]
        fig = go.Figure(data=data, layout=go.Layout(title=experiment_name))
        fig.write_html(f'{folder}/{experiment_name}.html')

    results_dictionary = order_results_auc(avg_dict)
    avg_names = []
    avg_auc = []
    for names, auc in results_dictionary:
        avg_names.append(names)
        avg_auc.append(auc)

    data = [go.Bar(x=avg_names, y=avg_auc)]
    fig = go.Figure(data=data, layout=go.Layout(title=f"averages"))
    fig.write_html(f'{folder}/averages.html')


def draw_cross(results, folder_name):
    folder = f'{base_figure_folder}/{folder_name}'
    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(f'{folder}/results.json', 'w') as file:
        file.write(json.dumps(results.copy()))

    global_figure = go.Figure()

    for experiment_name in results.keys():
        fig = go.Figure()
        avg_array = None
        for use_case in results[experiment_name]:
            fig.add_trace(
                go.Scatter(x=np.arange(len(results[experiment_name][use_case])), y=results[experiment_name][use_case],
                           mode='lines', name=use_case))
            if avg_array is None:
                avg_array = results[experiment_name][use_case]
            else:
                avg_array = [sum(x) for x in zip(avg_array, results[experiment_name][use_case])]
        use_case_no = len(results[experiment_name])
        avg_array = [x / use_case_no for x in avg_array]
        global_figure.add_trace(
            go.Scatter(x=np.arange(len(avg_array)), y=avg_array, mode='lines', name=experiment_name))
        fig.add_trace(go.Scatter(x=np.arange(len(avg_array)), y=avg_array, mode='lines', name="avg"))

        fig.update_layout(title=experiment_name)
        fig.write_html(f'{folder}/{experiment_name}.html')
    global_figure.update_layout(title="all_averages")
    global_figure.write_html(f'{folder}/all_averages.html')


def draw_drifts(results_proba):
    for experiment_name in results_proba:
        folder = f'{base_figure_folder}/{experiment_name}'
        if not os.path.exists(folder):
            os.makedirs(folder)
        for use_case in results_proba[experiment_name]:
            fig = go.Figure()
            drift_x = []
            drift_y = []
            warning_x = []
            warning_y = []
            for cross_no in results_proba[experiment_name][use_case]:
                for drift in results_proba[experiment_name][use_case][cross_no]["drifts"]:
                    drift_x.append(cross_no)
                    drift_y.append(drift)
                for warning in results_proba[experiment_name][use_case][cross_no]["warnings"]:
                    warning_x.append(cross_no)
                    warning_y.append(warning)
            fig.add_trace(go.Scatter(x=warning_x, y=warning_y, mode='markers', name="warning", color="orange"))
            fig.add_trace(go.Scatter(x=drift_x, y=drift_y, mode='markers', name="drift", color="red"))
            fig.write_html(f'{folder}/{use_case}.html')
