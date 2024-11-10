from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
import mlflow
import mlflow.sklearn
import numpy as np

from ucimlrepo import fetch_ucirepo
Air_Quality = fetch_ucirepo(id=360)
Base_Calidad_Aire = Air_Quality.data.original
Base_Calidad_Aire = Base_Calidad_Aire[['PT08.S1(CO)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'C6H6(GT)']]
Base_Calidad_Aire.columns = ['PT08_S1_CO', 'PT08_S3_NOx', 'PT08_S4_NO2', 'PT08_S5_O3', 'C6H6_GT']

X = Base_Calidad_Aire.drop(columns=['C6H6_GT'])
y = Base_Calidad_Aire['C6H6_GT']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {'n_estimators': [50, 100, 200],
              'max_depth': [None, 10, 20],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 2, 4],
              'max_features': ['sqrt', 'log2']}

experiment = mlflow.set_experiment("ExtraTrees_AirQuality")

with mlflow.start_run(experiment_id=experiment.experiment_id):
    model = ExtraTreesRegressor(random_state=42)
    grid_search = GridSearchCV( estimator=model,
                                param_grid=param_grid,
                                scoring=make_scorer(mean_squared_error, greater_is_better=False, squared=False),
                                cv=5,
                                n_jobs=-1,
                                verbose=2,
                                error_score="raise")
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    for param, value in best_params.items():
        mlflow.log_param(param, value)
    

    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    
    mlflow.sklearn.log_model(best_model, "extra-trees-model", input_example=X_test.iloc[:1])

    print("Mejores Hiperparámetros:", best_params)
    print("Mean Squared Error (MSE):", mse)
    print("Root Mean Squared Error (RMSE):", rmse)
    print("R-squared:", r2)