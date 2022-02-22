import numpy as np
from TaxiFareModel.data import get_data
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LinearRegression
from TaxiFareModel.encoders import DistanceTransformer,TimeFeaturesEncoder
from sklearn.compose import ColumnTransformer


class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def clean_data(df, test=False):
        '''returns a DataFrame without outliers and missing values'''
        df = df.dropna(how='any')
        df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0)]
        df = df[(df.pickup_latitude != 0) | (df.pickup_longitude != 0)]
        if "fare_amount" in list(df):
            df = df[df.fare_amount.between(0, 4000)]
        df = df[df.passenger_count < 8]
        df = df[df.passenger_count >= 0]
        df = df[df["pickup_latitude"].between(left=40, right=42)]
        df = df[df["pickup_longitude"].between(left=-74.3, right=-72.9)]
        df = df[df["dropoff_latitude"].between(left=40, right=42)]
        df = df[df["dropoff_longitude"].between(left=-74, right=-72.9)]
        return df

    def split(self):
        return train_test_split(self.X, self.y, test_size=0.15)

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([
        ('dist_trans', DistanceTransformer()),
        ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        pipe = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
        ])
        return pipe

    def compute_rmse(self,y_pred, y_true):
        return np.sqrt(((y_pred - y_true)**2).mean())

    def run(self):
        """set and train the pipeline"""
        self.pipeline = self.set_pipeline()
        self.pipeline.fit(self.X, self.y)
        return self.pipeline

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = self.compute_rmse(y_pred, y_test)
        return rmse


if __name__ == "__main__":
    # get data
    df = get_data()

    # clean data
    df_clean = Trainer.clean_data(df)

    # set X and y
    y = df_clean["fare_amount"]
    X = df_clean.drop("fare_amount", axis=1)
    train = Trainer(X,y)

    # hold out
    X_train, X_val, y_train, y_val = train.split()

    # train
    pipeline = train.run()

    # evaluate
    eval = train.evaluate(X_val,y_val)
    print(eval)
