import pandas as pd 
import mlflow

## NOTE: You can use Microsoft Azure Machine Learning Studio for experiment tracking. 
# Follow assignment description and uncomment below for that (
# you might also need to pip azureml (pip install azureml-core):


#from azureml.core import Workspace
#ws = Workspace.from_config() 
#mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())


## NOTE: Optionally, you can use the public tracking server. 
# Do not use it for data you cannot afford to lose. See note in assignment text. 
# If you leave this line as a comment, mlflow will save the runs to your local filesystem.

#mlflow.set_tracking_uri("http://training.itu.dk:5000/") 

# TODO: Set the experiment name
mlflow.set_experiment("<bjmi> - <templateTest>")

# Import some of the sklearn modules you are likely to use.
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor, LocalOutlierFactor
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error , mean_absolute_error , r2_score
from sklearn.impute import SimpleImputer

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer

# Align the data frames


class windDirectionToInt(BaseEstimator,TransformerMixin):
    def fit(self, X, y):
        return self
    
    def transform(self, X):
        dict = {"N": 0, "NNE": 0.3926990817, "NE": 0.7853981634, "ENE": 1.1780972451, "E": 1.5707963268, "ESE": 1.9634954085, 
                "SE": 2.3561944902, "SSE": 2.7488935719, "S":  3.1415926536 , "SSW": 3.5342917353, "SW": 3.926990817, 
                "WSW": 4.3196898987, "W": 4.7123889804, "WNW": 5.1050880621, "NW": 5.4977871438, "NNW": 5.8904862255
               }
        X['Direction'] = X['Direction'].map(dict)
        return X





# Start a run
# TODO: Set a descriptive name. This is optional, but makes it easier to keep track of your runs.
with mlflow.start_run(run_name="testRun"):
# TODO: Insert path to dataset

    df = pd.read_json("./dataset.json", orient="split")
    # TODO: Handle missing data
    
    pipeline = Pipeline([
    ("windDirection", windDirectionToInt()),
    ("FillNA", ColumnTransformer([
        ("imputer", SimpleImputer(strategy="median"), ["Source_time","Speed","Direction"]),
        ], remainder="passthrough")),
    ("scale", MinMaxScaler()),
    ("linRegres", LinearRegression()),
    ])
    
    # TODO: Currently the only metric is MAE. You should add more. What other metrics could you use? Why?
    
    metrics = [
    ("MAE", mean_absolute_error ,[]),]
    
    X = df[["Speed","Direction"]]
    y = df["Total"]
    
    number_of_splits = 5
    
    #TODO: Log your parameters. What parameters are important to log?
    #HINT: You can get access to the transformers in your pipeline using ‘pipeline.steps‘

    for train, test in TimeSeriesSplit(number_of_splits).split(X,y): 
        pipeline.fit(X.iloc[train],y.iloc[train])
        predictions = pipeline.predict(X.iloc[test])
        truth = y.iloc[test]
        from matplotlib import pyplot as plt 
        plt.plot(truth.index , truth.values , label="Truth") 
        plt.plot(truth.index, predictions, label="Predictions") 
        plt.show()
        
        # Calculate and save the metrics for this fold
        for name, func, scores in metrics: 
            score = func(truth, predictions) 
            scores.append(score)
        
    # Log a summary of the metrics
    for name, _, scores in metrics:
    # NOTE: Here we just log the mean of the scores.
    # Are there other summarizations that could be interesting? 
        mean_score = sum(scores)/number_of_splits 
        mlflow.log_metric(f"mean_{name}", mean_score)