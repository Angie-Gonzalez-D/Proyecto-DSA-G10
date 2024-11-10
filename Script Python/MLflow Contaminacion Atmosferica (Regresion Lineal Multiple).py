from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.statsmodels
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

experiment = mlflow.set_experiment("LinearRegression_AirQuality")

from ucimlrepo import fetch_ucirepo
Air_Quality = fetch_ucirepo(id=360)
Base_Calidad_Aire = Air_Quality.data.original
Base_Calidad_Aire = Base_Calidad_Aire[['Date', 'Time', 'PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH', 'C6H6(GT)']]
Base_Calidad_Aire.columns = ['Date', 'Time', 'PT08_S1_CO', 'PT08_S2_NMHC', 'PT08_S3_NOx', 'PT08_S4_NO2', 'PT08_S5_O3', 'T', 'RH', 'AH', 'C6H6_GT']
Base_Calidad_Aire['Date'] = pd.to_datetime(Base_Calidad_Aire['Date'], errors='coerce')
Base_Calidad_Aire['Day'] = Base_Calidad_Aire['Date'].dt.day
Base_Calidad_Aire['Hour'] = Base_Calidad_Aire['Time'].str.split(':', expand=True)[0].astype(int)

X = Base_Calidad_Aire[['PT08_S1_CO', 'PT08_S2_NMHC', 'PT08_S3_NOx', 'PT08_S4_NO2', 'PT08_S5_O3', 'T', 'RH', 'AH', 'Hour', 'Day']]
y = Base_Calidad_Aire['C6H6_GT']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run(experiment_id=experiment.experiment_id):
    formula = 'C6H6_GT ~ PT08_S1_CO + PT08_S2_NMHC + PT08_S3_NOx + PT08_S4_NO2 + PT08_S5_O3 + T + RH + AH + Hour + Day'
    model = smf.ols(formula=formula, data=pd.concat([X_train, y_train], axis=1))
    results = model.fit()
    
    mlflow.statsmodels.log_model(results, artifact_path="linear-regression-model", input_example=X_test.iloc[:1])

    y_pred = results.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Registrar métricas en MLflow
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)

    print("Mean Squared Error (MSE):", mse)
    print("Root Mean Squared Error (RMSE):", rmse)
    print("R-squared:", r2)