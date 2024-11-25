import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import statsmodels.formula.api as smf
import plotly.graph_objects as go 

# Preparar los datos
from ucimlrepo import fetch_ucirepo

# Cargar los datos desde el repositorio
Air_Quality = fetch_ucirepo(id=360)
Base_Calidad_Aire = Air_Quality.data.original
Base_Calidad_Aire.drop(columns=['CO(GT)', 'NMHC(GT)', 'NOx(GT)', 'NO2(GT)'], inplace=True)

Base = Base_Calidad_Aire[['Date', 'Time', 'PT08.S1(CO)', 'PT08.S2(NMHC)', 
                          'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 
                          'T', 'RH', 'AH', 'C6H6(GT)']].copy()

# Renombrar columnas y agregar variables derivadas
Base.columns = ['Date', 'Time', 'PT08_S1_CO', 'PT08_S2_NMHC', 'PT08_S3_NOx', 
                'PT08_S4_NO2', 'PT08_S5_O3', 'T', 'RH', 'AH', 'C6H6_GT']

Base['Date'] = pd.to_datetime(Base['Date'], errors='coerce')
Base['Month'] = Base['Date'].dt.month_name()
Base['Week_Day'] = Base['Date'].dt.day_name()
Base['Day'] = Base['Date'].dt.day
Base['Hour'] = Base['Time'].str.split(':', expand=True)[0].astype(int)

# Calcular el índice de calidad del aire
def calcular_indice_calidad(C6H6_GT):
    if C6H6_GT <= 0.17:
        return 0.5
    elif 0.17 < C6H6_GT <= 1.0:
        return 1
    elif 1.0 < C6H6_GT <= 3.0:
        return 1.5
    elif 3.0 < C6H6_GT <= 5.0:
        return 2
    elif 5.0 < C6H6_GT <= 7.0:
        return 2.5
    elif 7.0 < C6H6_GT <= 10.0:
        return 3
    elif 10.0 < C6H6_GT <= 15.0:
        return 3.5
    elif 15.0 < C6H6_GT <= 20.0:
        return 4
    elif 20.0 < C6H6_GT <= 25.0:
        return 4.5
    elif C6H6_GT > 25.0:
        return 5
    else:
        return None

Base['Air_Quality_Index'] = Base['C6H6_GT'].apply(calcular_indice_calidad)

def clasificar_calidad_aire(promedio_C6H6):
    if promedio_C6H6 < 1.1:
        return "EXCELENTE"
    elif 1.1 <= promedio_C6H6 < 3:
        return "BUENA"
    elif 3 <= promedio_C6H6 < 7.1:
        return "MODERADA"
    else:
        return "MALA"

# Definir variables para el modelo
X = Base[['PT08_S1_CO', 'PT08_S2_NMHC', 'PT08_S3_NOx', 'PT08_S4_NO2', 'PT08_S5_O3', 
          'T', 'RH', 'AH', 'Hour', 'Day', 'Month']]
y = Base['C6H6_GT']

# Entrenar el modelo de regresión lineal
formula = 'C6H6_GT ~ PT08_S1_CO + PT08_S2_NMHC + PT08_S3_NOx + PT08_S4_NO2 + PT08_S5_O3 + T + RH + AH + Hour + Day + Month'
model = smf.ols(formula=formula, data=pd.concat([X, y], axis=1))
results = model.fit()


# Crear la aplicación de Dash
app = dash.Dash(__name__)
app.title = "Dashboard de Calidad del Aire"

# Layout del dashboard
app.layout = html.Div([
    html.H1("Predicción de la Concentración de Benceno (C6H6_GT)", style={'textAlign': 'center'}),
    
    html.Div([
        # Sección de variables de entrada (izquierda)
        html.Div([
            html.Label("PT08_S1_CO (Sensor CO):"),
            dcc.Input(id='input-PT08_S1_CO', type='number', value=200, step=1),
            
            html.Label("PT08_S2_NMHC (Sensor NMHC):"),
            dcc.Input(id='input-PT08_S2_NMHC', type='number', value=400, step=1),
            
            html.Label("PT08_S3_NOx (Sensor NOx):"),
            dcc.Input(id='input-PT08_S3_NOx', type='number', value=800, step=1),
            
            html.Label("PT08_S4_NO2 (Sensor NO2):"),
            dcc.Input(id='input-PT08_S4_NO2', type='number', value=600, step=1),
            
            html.Label("PT08_S5_O3 (Sensor O3):"),
            dcc.Input(id='input-PT08_S5_O3', type='number', value=300, step=1),
            
            html.Label("Temperatura (T):"),
            dcc.Input(id='input-T', type='number', value=25, step=0.1),
            
            html.Label("Humedad Relativa (RH):"),
            dcc.Input(id='input-RH', type='number', value=50, step=0.1),
            
            html.Label("Humedad Absoluta (AH):"),
            dcc.Input(id='input-AH', type='number', value=1.5, step=0.01),
            
            html.Label("Hora del Día (Hour):"),
            dcc.Dropdown(
                id='input-Hour',
                options=[{'label': f'{hour}:00', 'value': hour} for hour in range(24)],
                value=12,
                placeholder="Selecciona la hora"
            ),
            
            html.Label("Día del Mes (Day):"),
            dcc.Dropdown(
                id='input-Day',
                options=[{'label': str(day), 'value': day} for day in range(1, 32)],
                value=15,
                placeholder="Selecciona el día"
            ),

            html.Label("Mes:"),
            dcc.Dropdown(
                id='input-Month',
                options=[{'label': str(month), 'value': month} for month in Base['Month'].unique()],
                value="Selecciona el mes",
                placeholder="Selecciona el mes"
            ),
            
            html.Button('Predecir', id='predict-button', n_clicks=0)
        ], style={'display': 'grid', 'gridTemplateColumns': '1fr', 'gap': '10px'}),

        # Sección de resultados (derecha)
        html.Div([
            # Resultado de la predicción
            html.Div(id='output-prediction', style={'textAlign': 'center', 'marginTop': '20px'}),

            # Gráfico medidor (gauge) para calidad del aire
            dcc.Graph(id='gauge-chart'),
        ], style={'display': 'grid', 'gridTemplateColumns': '1fr', 'gap': '20px', 'marginTop': '20px'})

    ], style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '30px'}),  # Dos columnas (izquierda y derecha)
    
])
# Callback para manejar la predicción
@app.callback(
    [Output('output-prediction', 'children'),
     Output('gauge-chart', 'figure')],    
    [Input('predict-button', 'n_clicks')],
    [
        Input('input-PT08_S1_CO', 'value'),
        Input('input-PT08_S2_NMHC', 'value'),
        Input('input-PT08_S3_NOx', 'value'),
        Input('input-PT08_S4_NO2', 'value'),
        Input('input-PT08_S5_O3', 'value'),
        Input('input-T', 'value'),
        Input('input-RH', 'value'),
        Input('input-AH', 'value'),
        Input('input-Hour', 'value'),
        Input('input-Day', 'value'),
	Input('input-Month', 'value')
    ]
)
def predict(n_clicks, PT08_S1_CO, PT08_S2_NMHC, PT08_S3_NOx, PT08_S4_NO2, PT08_S5_O3, T, RH, AH, Hour, Day, Month):
    if n_clicks > 0:
        input_data = pd.DataFrame({
            'PT08_S1_CO': [PT08_S1_CO],
            'PT08_S2_NMHC': [PT08_S2_NMHC],
            'PT08_S3_NOx': [PT08_S3_NOx],
            'PT08_S4_NO2': [PT08_S4_NO2],
            'PT08_S5_O3': [PT08_S5_O3],
            'T': [T],
            'RH': [RH],
            'AH': [AH],
            'Hour': [Hour],
            'Day': [Day],
	    'Month': [Month]
        })
        prediction = results.predict(input_data)[0]
        air_quality_index = calcular_indice_calidad(prediction)
        categoria = clasificar_calidad_aire(prediction)
        # Crear el gráfico medidor
        gauge_figure = go.Figure(go.Indicator(
            mode="gauge+number",
            value=air_quality_index,
            title={'text': "Índice de Calidad del Aire"},
            gauge={
                'axis': {'range': [0, 5]},
                'steps': [
                    {'range': [0, 1], 'color': "green"},
                    {'range': [1, 2.5], 'color': "yellow"},
                    {'range': [2.5, 4], 'color': "orange"},
                    {'range': [4, 5], 'color': "red"}
                ],
	    'bar': {'color':"black", 'thickness':0.2}

            }
        ))

        return [
            html.Div([
                html.H3(f"Predicción de Concentración de Benceno: {prediction:.2f} µg/m³", style={'color': 'blue'}),
                html.H4([f"Índice de Calidad del Aire: {air_quality_index}", html.Br(), f"Categoría: {categoria}"], style={'color': 'green'})
            ]),
            gauge_figure
        ]
    else:
        # Si no se ha hecho clic en el botón, devolver una tupla con un mensaje y gráfico vacío
        return [
            html.Div("Ingresa los valores y presiona Predecir.", style={'textAlign': 'center'}),
            go.Figure()  # Vacío
        ]
# Ejecutar el servidor
if __name__ == '__main__':
    app.run_server(debug=True)
