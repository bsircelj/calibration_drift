import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from skmultiflow.drift_detection import DDM

data_folder = "./data"


def get_dataset(name):
    data_y = pd.read_csv(f"{data_folder}/{name}_y.csv", index_col="Unnamed: 0")
    data_x = pd.read_csv(f"{data_folder}/{name}_x.csv", index_col="Unnamed: 0")

    return data_x.to_numpy(), data_y.to_numpy()


class RFModel:
    def __init__(self, x, y, train_cal_split=0.7, calibrate=True, calibration_method='sigmoid'):
        self.rf_model = RandomForestClassifier(random_state=42)
        self.calibrate = calibrate
        self.calibration_model = CalibratedClassifierCV(base_estimator=self.rf_model,
                                                        method=calibration_method,
                                                        cv="prefit")
        self.train_cal_split = train_cal_split
        self.x = x
        self.y = y

    def fit(self, start, end):
        if self.calibrate:
            train_size = int((end - start) * self.train_cal_split)
            self.rf_model.fit(self.x[start:start + train_size], np.ravel(self.y[start:start + train_size]))
            self.calibration_model.fit(self.x[start + train_size:end], self.y[start + train_size:end])
        else:
            self.rf_model.fit(self.x[start:end], np.ravel(self.y[start:end]))

    def train(self, start, end):
        self.rf_model.fit(self.x[start:end], np.ravel(self.y[start:end]))

    def recalibrate(self, start, end):
        self.calibration_model.fit(self.x[start:end], self.y[start:end])

    def predict(self, x):
        if self.calibrate:
            return self.calibration_model.predict(x)
        return self.rf_model.predict(x)

    def predict_proba(self, x):
        if self.calibrate:
            return self.calibration_model.predict_proba(x)
        return self.rf_model.predict_proba(x)
