import pandas as pd
import numpy as np
from multiprocessing import Pool
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from skmultiflow.drift_detection import DDM
from utilities import get_dataset, RFModel
from visualization import draw_bar, draw_cross
from multiprocessing import Process, Manager
from tqdm import tqdm
import os
import json

data_folder = "./data"
datasets = ["AMZN", "credit_fraud", "elec", "noaa", "XLE", "XTN"]
train_length = [2917, 23139, 24000, 14527, 2917, 2097]
dataset_lengths = [3647, 28924, 30000, 18159, 3647, 2622]

# datasets = ["AMZN", "XTN"]
# train_length = [2917, 2097]
# dataset_lengths = [3647, 2622]

# datasets = ["XTN"]
# dataset_lengths = [2622]
# train_length = [2097]

relearn_percentage = 0.1

# train calibration test fix 60 10 20 10
train_ratio = 0.6
calibration_ratio = 0.1
test_ratio = 0.2
last_ratio = 0.1
continuous_calibration_split = 500
cross_fold = 30

settings_list = [{"name": "no calibration",
                  "relearn_percentage": relearn_percentage,
                  "train_calibration_split": 0.7,
                  "calibration_method": "sigmoid",
                  "calibrate": False,
                  "recalibrate": False,
                  "retrain": False,
                  "continuous_calibration": False},

                 {"name": "calibration once",
                  "relearn_percentage": relearn_percentage,
                  "train_calibration_split": 0.7,
                  "calibration_method": "sigmoid",
                  "calibrate": True,
                  "recalibrate": False,
                  "retrain": False,
                  "continuous_calibration": False},

                 {"name": "retrain only",
                  "relearn_percentage": relearn_percentage,
                  "train_calibration_split": 0.7,
                  "calibration_method": "sigmoid",
                  "calibrate": True,
                  "recalibrate": False,
                  "retrain": True,
                  "continuous_calibration": False},

                 {"name": "recalibration only",
                  "relearn_percentage": relearn_percentage,
                  "train_calibration_split": 0.7,
                  "calibration_method": "sigmoid",
                  "calibrate": True,
                  "recalibrate": True,
                  "retrain": False,
                  "continuous_calibration": False},

                 {"name": "retrain recalibration",
                  "relearn_percentage": relearn_percentage,
                  "train_calibration_split": 0.7,
                  "calibration_method": "sigmoid",
                  "calibrate": True,
                  "recalibrate": True,
                  "retrain": True,
                  "continuous_calibration": False},

                 {"name": "continuous calibration",
                  "relearn_percentage": relearn_percentage,
                  "train_calibration_split": 0.7,
                  "calibration_method": "sigmoid",
                  "calibrate": True,
                  "recalibrate": False,
                  "retrain": False,
                  "continuous_calibration": True}
                 ]

total = cross_fold * len(datasets)
pbar = tqdm(total=total)


def run_experiment(settings, results, folder_name):
    relearn_percentage = settings["relearn_percentage"]
    # results[settings["name"]] = manager.dict()
    avg_auc = 0
    single_result = dict()
    for data_no, dataset_name in enumerate(datasets):
        print(f"{settings['name']} - {dataset_name}")
        x, y = get_dataset(dataset_name)
        model = RFModel(x=x,
                        y=y,
                        train_cal_split=settings["train_calibration_split"],
                        calibration_method=settings["calibration_method"],
                        calibrate=settings["calibrate"])
        model.fit(0, train_length[data_no])

        ddm = DDM()
        warning_on = False
        # print(f'start {train_length[data_no]}')
        predicted_all = []
        truth_all = []
        detected_drifts = []
        for i in range(train_length[data_no], len(x)):
            predicted_y = model.predict([x[i]])
            predicted_all.append(model.predict_proba([x[i]])[:, 1])
            truth_all.append(y[i][0])
            ddm.add_element(int(predicted_y[0] != y[i][0]))

            if not warning_on and ddm.detected_warning_zone():
                # print(f'warning {i}')
                warning_on = True

            if ddm.detected_change():
                detected_drifts.append(i)
                # print(f'change {i}')
                if settings['retrain'] and settings['recalibrate']:
                    model.fit(int(i * (1 - relearn_percentage)), i)
                if settings["retrain"]:
                    model.train(int(i * (1 - relearn_percentage)), i)
                if settings['recalibrate']:
                    model.recalibrate(int(i * (1 - relearn_percentage)), i)
                warning_on = False
                ddm = DDM()

            if settings["continuous_calibration"]:
                model.recalibrate(int(i * (1 - relearn_percentage)), i)
        score = roc_auc_score(truth_all, predicted_all)
        avg_auc += score
        single_result[dataset_name] = score
    single_result["avg"] = avg_auc / len(datasets)
    results[settings["name"]] = single_result


def run_cross_validation(args):
    settings, results, results_proba, logs, folder_name = args
    # os.makedirs(f'./charts/{folder_name}/logs')
    # with open(f'./charts/{folder_name}/logs/{settings["name"].replace(" ", "_")}.log', 'a') as logfile:
    # print(f"{settings['name']} - {datasets[0]}")
    logs[settings['name']] = ""
    # logfile.write(f"{settings['name']} - {datasets[0]} \n")
    relearn_percentage = settings["relearn_percentage"]
    # results[settings["name"]] = manager.dict()
    single_result_auc = dict()
    single_result_proba = dict()
    results_proba[settings["name"]] = single_result_proba
    for data_no, dataset_name in enumerate(datasets):
        x, y = get_dataset(dataset_name)
        model = RFModel(x=x,
                        y=y,
                        train_cal_split=settings["train_calibration_split"],
                        calibration_method=settings["calibration_method"],
                        calibrate=settings["calibrate"])

        single_result_auc[dataset_name] = []
        results[settings["name"]] = single_result_auc

        data_size = len(x)
        train_size = int(data_size * train_ratio)
        calibration_size = int(data_size * calibration_ratio)
        test_start_initial = train_size + calibration_size
        last_index = int(data_size * (1 - last_ratio))
        step = data_size
        if cross_fold > 1:
            step = int((last_index - test_start_initial) / cross_fold - 1)
        indices = list(range(0, last_index - test_start_initial, step + 1))
        indices.append(last_index - test_start_initial)
        for cross_no, cross_i in enumerate(indices):
            train_size = int((test_start_initial + cross_i) * (1 - calibration_ratio / train_ratio))
            calibration_size = int((test_start_initial + cross_i) * calibration_ratio / train_ratio)
            test_start = train_size + calibration_size
            model.train(0, train_size)
            model.recalibrate(train_size, test_start)

            ddm = DDM()
            warning_on = False
            retrain_needed = False
            # print(f'start {train_length[data_no]}')
            predicted_all = []
            detected_drifts = []
            detected_warnings = []
            warning_drift_window = []

            truth_all = np.ravel(y[test_start:len(y)])

            for i in range(test_start, len(x)):
                predicted_y = model.predict([x[i]])
                predicted_all.append(model.predict_proba([x[i]])[:, 1])
                ddm.add_element(int(predicted_y[0] != y[i][0]))

                if not warning_on and ddm.detected_warning_zone():
                    warning_on = True
                    detected_warnings.append(i)

                def retrain_recalibrate():
                    if settings["retrain"] and settings["recalibrate"]:
                        retrain_size = int(i * (relearn_percentage * 0.7))
                        recalibration_size = int(i * (relearn_percentage * 0.3))
                        model.train(i - retrain_size, i - recalibration_size)
                        model.recalibrate(i - recalibration_size, i)
                    else:
                        if settings["retrain"]:
                            model.train(int(i * (1 - relearn_percentage)), i)
                        if settings['recalibrate']:
                            model.recalibrate(int(i * (1 - relearn_percentage)), i)

                if ddm.detected_change():
                    detected_drifts.append(i)
                    if len(detected_warnings) > 0:
                        warning_drift_window.append(detected_drifts[-1] - detected_warnings[-1])
                    retrain_recalibrate()
                    if len(detected_warnings) > 0 and i * (1 - relearn_percentage) < detected_warnings[-1]:
                        retrain_needed = True
                    warning_on = False
                    ddm = DDM()

                if retrain_needed and len(detected_warnings) > 0 and i * (1 - relearn_percentage) > \
                        detected_warnings[
                            -1]:
                    retrain_needed = False
                    retrain_recalibrate()

                if settings["continuous_calibration"] and i % continuous_calibration_split:
                    model.recalibrate(int(i * (1 - relearn_percentage)), i)

            single_result_proba = results_proba[settings["name"]]
            single_result_proba[cross_no] = ([int(x) for x in list(truth_all)], [float(x[0]) for x in predicted_all])
            results_proba[settings["name"]] = single_result_proba

            score = roc_auc_score(truth_all, predicted_all)
            single_result_auc = results[settings["name"]]
            single_result_auc[dataset_name].append(score)
            results[settings["name"]] = single_result_auc
            mean_window = 0
            if len(warning_drift_window) > 0:
                mean_window = int(np.average(warning_drift_window))
            print(f"{settings['name']} - {dataset_name} drifts: {len(detected_drifts)} window: {mean_window}")
            logs[settings[
                'name']] += f"\n{settings['name']} - {dataset_name} drifts: {len(detected_drifts)} window: {mean_window}"
            # logfile.write(
            #     f"{settings['name']} - {dataset_name} drifts: {len(detected_drifts)} window: {mean_window}\n")
            pbar.update(1)


if __name__ == '__main__':
    folder_name = "cross_full_run5"
    manager = Manager()
    results_auc = manager.dict()
    logs = manager.dict()
    results_proba = manager.dict()
    # processes = []
    # for settings in settings_list:
    #     proc = Process(target=run_cross_validation, args=(settings, results, folder_name))
    #     processes.append(proc)
    #     proc.start()
    #
    # for proc in processes:
    #     proc.join()

    # draw_bar(results, folder_name)

    # total = int(sum(dataset_lengths) * 0.2 / cross_fold)
    # run_cross_validation((settings_list[4], results, folder_name))
    # print(results)
    # results = {'retrain recalibration': {'AMZN': [0.8635061920793332, 0.8776990605267212], 'XTN': [0.8253328175869517, 0.8055102516309413]}}
    # draw_cross(results, folder_name)

    pool = Pool()
    args = [(x, results_auc, results_proba, logs, folder_name) for x in settings_list]
    pool.imap_unordered(run_cross_validation, args)
    pool.close()
    pool.join()
    pbar.close()
    print(results_auc)

    draw_cross(results_auc, folder_name)

    with open(f"./charts/{folder_name}/results_proba.json", 'w') as file:
        file.write(json.dumps(results_proba.copy()))

    with open(f"./charts/{folder_name}/logs.txt", 'a') as file:
        for experiment_name in logs:
            file.write(f"\n___\n{experiment_name}\n___\n")
            file.write(logs[experiment_name])
