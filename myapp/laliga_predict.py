from keras.models import load_model
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from laliga_preprocess import *
from laliga_model import *
from collections import deque
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from pathlib import Path

# Obtener la ruta base del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent

# Configuración del modelo y los datos
n_outputs = 2
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = 'mse'
batch_size = 137
sync_steps = 100
discount_factor = 0.99
episodes = 100

# Lista de archivos de datos históricos
lista_archivos = [
    BASE_DIR / 'myapp' / 'data' / '2022_2023.csv',
    BASE_DIR / 'myapp' / 'data' / '2023_2024.csv'
]

# Archivos individuales
archivo = BASE_DIR / 'myapp' / 'data' / 'jugadores_ordenados.csv'
archivo_data = BASE_DIR / 'myapp' / 'data' / 'data_players.csv'

# Cargar y preprocesar datos de entrenamiento
datos = cargar_datos_por_jugador(lista_archivos)
X_train, y_train = preparar_datos_secuenciales(datos, weight_scheme='exponential')
X_train, encoder, scaler = procesar_datos_con_onehot_y_minmax(X_train)
y_train = pad_sequences(y_train, padding='post', dtype='float32', value=0.0)

# Crear buffer de replay
replay_buffer = deque(maxlen=1000)

@tf.keras.utils.register_keras_serializable()
class DuelingLayer(layers.Layer):
    def call(self, inputs):
        value, advantage = inputs
        return value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))

# Cargar el modelo
model_path = BASE_DIR / 'myapp' / 'data' / 'checkpoint.keras'  # Ruta del modelo
model = load_model(model_path, custom_objects={'DuelingLayer': DuelingLayer})

# Función para cargar y preprocesar el último registro de cada jugador desde un archivo CSV
def cargar_y_preprocesar_ultimos_jugadores(ruta_archivo, encoder, scaler):
    df = pd.read_csv(ruta_archivo, header=None)
    df.columns = ['Name', 'Shirtnumber', 'Nationality', 'Position', 'Age', 'Minutes', 'Goals', 'Assists', 
                  'Pens Made', 'Pens Att', 'Shots', 'Shots on Target', 'Cards Yellow', 'Cards Red', 'Touches', 
                  'Tackles', 'Interceptions', 'Blocks', 'XG', 'NPXG', 'XG Assist', 'SCA', 'GCA', 
                  'Passes Completed', 'Passes', 'Passes %', 'Progressive Passes', 'Carries', 
                  'Progressive Carries', 'Take Ons', 'Take Ons Won', 'GC', 'Salvadas', 'PSxG', 
                  'Opponent Team', 'Match Date', 'Puntuation', 'Team Style', 'Year', 'Month', 'Day', 'Reward']

    ultimos_registros = df.groupby('Name').tail(1)
    columnas_deseadas = ultimos_registros[['Position', 'Minutes', 'Goals', 'Assists', 
                  'Pens Made', 'Pens Att', 'Shots', 'Shots on Target', 'Cards Yellow', 'Cards Red', 'Touches', 
                  'Tackles', 'Interceptions', 'Blocks', 'XG', 'NPXG', 'XG Assist', 'SCA', 'GCA', 
                  'Passes Completed', 'Passes', 'Passes %', 'Progressive Passes', 'Carries', 
                  'Progressive Carries', 'Take Ons', 'Take Ons Won', 'GC', 'Salvadas', 'PSxG', 
                  'Opponent Team','Team Style']].apply(pd.to_numeric, errors='coerce')

    categorias = ultimos_registros[['Position', 'Team Style']].fillna("Unknown").values
    numericas = columnas_deseadas.drop(columns=['Position', 'Team Style']).values

    categorias = np.array([
        [cat if cat in encoder.categories_[i] else "Unknown" for i, cat in enumerate(row)]
        for row in categorias
    ])

    categorias_encoded = encoder.transform(categorias)
    numericas_scaled = scaler.transform(numericas)

    datos_procesados = np.hstack([categorias_encoded, numericas_scaled])

    df_procesado = pd.DataFrame(datos_procesados)
    df_procesado["Name"] = ultimos_registros["Name"].values
    return df_procesado

# Función para hacer predicciones para cada jugador y añadirlas al DataFrame
def predecir_para_jugadores(df, model):
    predicciones = []
    for index, row in df.iterrows():
        input_data = np.array(row.drop("Name"), dtype=np.float32).reshape(1, -1)

        if np.isnan(input_data).any():
            print(f"Valores NaN en la entrada para el jugador {row['Name']}. Saltando predicción.")
            predicciones.append(np.nan)
            continue

        q_values = model.predict(input_data)
        estadistica = q_values[0][1] - q_values[0][0]
        predicciones.append(estadistica)

    df["Prediccion"] = predicciones
    return df

# Procesar archivo de jugadores, predecir, añadir predicciones a data_players y guardar
import pandas as pd

def procesar_jugadores_con_probabilidad(ruta_archivo_jugadores, ruta_archivo_data_players, model, encoder, scaler):
    df_jugadores = cargar_y_preprocesar_ultimos_jugadores(ruta_archivo_jugadores, encoder, scaler)
    df_jugadores = predecir_para_jugadores(df_jugadores, model)

    df_data_players = pd.read_csv(ruta_archivo_data_players)

    # Eliminar filas donde "Nombre" contenga "Player" o "Players" (sin distinguir mayúsculas/minúsculas)
    df_data_players = df_data_players[~df_data_players["Nombre"].str.contains(r'Player[s]?', case=False, na=False)]

    df_data_players["Prediccion"] = df_data_players["Nombre"].map(
        df_jugadores.set_index("Name")["Prediccion"]
    )

    df_data_players.to_csv(ruta_archivo_data_players, index=False)
    print(f"Archivo sobrescrito con éxito en {ruta_archivo_data_players}")

import pandas as pd
import numpy as np

# Función para calcular la distancia relativa de la estadística a un rango dado
def distancia_relativa(estadistica, limite_inferior, limite_superior):
    if estadistica < limite_inferior:
        return abs(limite_inferior - estadistica)
    elif estadistica > limite_superior:
        return abs(estadistica - limite_superior)
    else:
        return 0  # Si está dentro del rango, la distancia es cero

# Función para calcular las probabilidades de pertenecer a cada color
def calcular_probabilidades(estadistica):
    # Definir los rangos para cada color
    rango_rojo = (-np.inf, -5)
    rango_naranja = (-5, 15)
    rango_verde = (15, 20)
    rango_azul = (20, np.inf)

    # Calcular las distancias relativas de la estadística a cada rango
    dist_rojo = distancia_relativa(estadistica, *rango_rojo)
    dist_naranja = distancia_relativa(estadistica, *rango_naranja)
    dist_verde = distancia_relativa(estadistica, *rango_verde)
    dist_azul = distancia_relativa(estadistica, *rango_azul)

    # Convertir las distancias en probabilidades inversamente proporcionales
    pesos = {
        "Rojo": 1 / (dist_rojo + 1),     # +1 para evitar división por cero
        "Naranja": 1 / (dist_naranja + 1),
        "Verde": 1 / (dist_verde + 1),
        "Azul": 1 / (dist_azul + 1)
    }

    # Calcular la suma total de los pesos
    total_pesos = sum(pesos.values())

    # Normalizar los pesos para que sumen 100
    probabilidades = {color: (peso / total_pesos) * 100 for color, peso in pesos.items()}

    return {color: round(prob, 2) for color, prob in probabilidades.items()}

# Función para calcular el color principal y si alinear o no
def calcular_color_probabilidad_y_alinear(estadistica):
    probabilidades = calcular_probabilidades(estadistica)
    color = max(probabilidades, key=probabilidades.get)  # El color con la mayor probabilidad
    alinear = "Sí" if estadistica >= 0 else "No"  # Alinear solo si la estadística es positiva
    return color, probabilidades, alinear

# Función principal que procesa el archivo CSV y lo sobrescribe con las columnas añadidas
def procesar_jugadores_con_porcentages(ruta_archivo):
    # Cargar los datos desde el archivo CSV
    df = pd.read_csv(ruta_archivo)
    df = df.drop_duplicates(subset=['Nombre'], keep='first')
    # Aplicar la función a cada fila del DataFrame
    resultados = []
    for index, row in df.iterrows():
        estadistica = row['Prediccion']
        color, probabilidades, alinear = calcular_color_probabilidad_y_alinear(estadistica)

        # Añadir los resultados en el DataFrame
        resultados.append({
            "Nombre": row["Nombre"],
            "Posicion": row["Posicion"],
            "Numero": row["Numero"],
            "Equipo": row["Equipo"],
            "Prediccion": estadistica,
            "Color": color,
            "Probabilidad": max(probabilidades.values()),  # Probabilidad máxima
            "Alinear": alinear,
            "Rojo": probabilidades["Rojo"],
            "Naranja": probabilidades["Naranja"],
            "Verde": probabilidades["Verde"],
            "Azul": probabilidades["Azul"]
        })

    # Crear un DataFrame con los resultados
    df_resultados = pd.DataFrame(resultados)

    # Guardar el DataFrame en el archivo CSV, sobrescribiendo el archivo original
    df_resultados.to_csv(ruta_archivo, index=False)

    print(f"Predicciones guardadas en {ruta_archivo}")

def eliminar_jugadores_probabilidad_equilibrada(ruta_archivo):
    # Cargar los datos desde el archivo
    df = pd.read_csv(ruta_archivo)

    # Filtrar jugadores que NO tienen todas las probabilidades en 25%
    df_filtrado = df[~(
        (df["Rojo"] == 25.0) & 
        (df["Naranja"] == 25.0) & 
        (df["Verde"] == 25.0) & 
        (df["Azul"] == 25.0)
    )]

    # Sobrescribir el archivo con los datos filtrados
    df_filtrado.to_csv(ruta_archivo, index=False)
    print(f"Archivo sobrescrito con éxito en {ruta_archivo}. Jugadores eliminados: {len(df) - len(df_filtrado)}")


# Ejemplo de uso

procesar_jugadores_con_probabilidad(archivo, archivo_data, model, encoder, scaler)
procesar_jugadores_con_porcentages(archivo_data)
eliminar_jugadores_probabilidad_equilibrada(archivo_data)