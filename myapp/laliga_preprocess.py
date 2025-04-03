import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from keras.preprocessing.sequence import pad_sequences

def cargar_datos_por_jugador(lista_archivos):
    """
    Carga datos de varios archivos y combina los datos de los jugadores en un solo diccionario.

    Args:
        lista_archivos (list): Lista de rutas a archivos CSV con datos de jugadores.

    Returns:
        dict: Diccionario con datos de jugadores, donde la clave es el nombre del jugador y la temporada, y el valor es una lista de lotes.
    """
    datos_jugadores = {}

    for ruta_archivo in lista_archivos:
        with open(ruta_archivo, 'r', encoding='utf-8') as archivo:
            jugador_actual = None
            temporada_actual = str(ruta_archivo).split('/')[-1].split('.')[0]
            lote_actual = []

            for linea in archivo:
                linea = linea.strip()
                if linea:
                    partes = linea.split(',')
                    nombre_jugador = partes[0].strip()
                    nombre_temporada = f"{nombre_jugador}_{temporada_actual}"  # Usar nombre + temporada como clave

                    if jugador_actual and jugador_actual != nombre_jugador:
                        # Si el jugador ya tiene datos, agrega los nuevos lotes
                        if nombre_temporada not in datos_jugadores:
                            datos_jugadores[nombre_temporada] = []  # Si no existe, crear lista
                        datos_jugadores[nombre_temporada].append(lote_actual)  # Agregar lote
                        lote_actual = []  # Reiniciar lote

                    jugador_actual = nombre_jugador
                    # Ajustar las columnas deseadas
                    columnas_deseadas = partes[3:4] + partes[5:35] + partes[38:39] + partes[42:43]  # Ajustamos las columnas
                    lote_actual.append(columnas_deseadas)

            # Guarda el último lote al finalizar la lectura del archivo
            if jugador_actual:
                nombre_temporada = f"{jugador_actual}_{temporada_actual}"
                if nombre_temporada not in datos_jugadores:
                    datos_jugadores[nombre_temporada] = []
                datos_jugadores[nombre_temporada].append(lote_actual)

    return datos_jugadores


def preparar_datos_secuenciales(datos_jugadores, weight_scheme='linear'):
    """
    Prepara los datos en secuencias para el modelo, obteniendo medias ponderadas por característica numérica
    e incluyendo las características categóricas. Si no hay partidos anteriores, usa el partido actual como referencia.

    Args:
        datos_jugadores (dict): Diccionario donde la clave es el ID del jugador (nombre + temporada) y el valor es una lista de partidos.
        weight_scheme (str): Especifica el tipo de esquema de peso. 'linear' para pesos lineales, 'exponential' para decaimiento exponencial.

    Returns:
        tuple: x_data (secuencias de medias ponderadas de características anteriores, incluyendo categóricas) y y_data (rendimientos futuros).
    """
    x_data = []
    y_data = []

    for jugador_temporada, partidos in datos_jugadores.items():
        for lote in partidos:  # Por cada lote de partidos
            matches = []  # Lista que contendrá las medias ponderadas por cada característica numérica y categórica
            matches_puntuation = []  # Lista que contendrá los valores de 'y' (rendimiento futuro)
            previous_matches = []  # Para almacenar las características numéricas de partidos anteriores

            for partido_procesado in lote:
                # Extraer características categóricas
                cat_primera = partido_procesado[0]  # Primera característica categórica (nombre del jugador con temporada)
                cat_penultima = partido_procesado[-2]  # Penúltima característica categórica

                # 'y_current' es la última columna
                y_current = partido_procesado[-1]

                # 'x_numeric' es un array que contiene todas las características numéricas, excluyendo las categóricas
                try:
                    # Excluimos la primera y penúltima columna
                    x_numeric = np.array([float(valor) for valor in partido_procesado[1:-2]])
                except ValueError as e:
                    print(f"Error de conversión: {e}. Partido omitido.")
                    continue  # O maneja el error según se necesite

                if not previous_matches:
                    # Si no hay partidos anteriores, usamos las características del partido actual como "media"
                    x_avg = x_numeric
                else:
                    # Incluir el partido actual en el cálculo de la media ponderada
                    all_matches = previous_matches + [x_numeric]

                    # Calcular la media ponderada por cada característica numérica
                    if weight_scheme == 'linear':
                        # Pesos lineales: 1, 2, 3, ..., para dar más peso a los partidos recientes, incluyendo el actual
                        weights = np.arange(1, len(all_matches) + 1)
                    elif weight_scheme == 'exponential':
                        # Pesos exponenciales, donde los más recientes tienen mucho más peso
                        alpha = 0.5  # Ajustable según se necesite
                        weights = np.array([alpha**i for i in reversed(range(len(all_matches)))] )
                    else:
                        raise ValueError("Esquema de peso no soportado.")

                    # Normalizar los pesos
                    weights = weights / weights.sum()

                    # Convertir todos los partidos anteriores más el actual a un array y calcular la media ponderada
                    all_matches_array = np.array(all_matches)
                    x_avg = np.average(all_matches_array, axis=0, weights=weights)

                # Combinar las características categóricas con las medias ponderadas de las numéricas
                x_combined = [cat_primera] + x_avg.tolist() + [cat_penultima]

                # Guardar la combinación de categóricas y numéricas en 'matches' y el rendimiento actual en 'matches_puntuation'
                matches.append(x_combined)
                matches_puntuation.append(y_current)

                # Añadir el partido actual (solo características numéricas) a la lista de partidos anteriores
                previous_matches.append(x_numeric)

            # Añadir las secuencias de medias y las puntuaciones correspondientes a los datos finales
            x_data.append(matches)
            y_data.append(matches_puntuation)

    return x_data, y_data


def procesar_datos_con_onehot_y_minmax(x, fit=True, encoder=None, scaler=None):
    """
    Aplica OneHotEncoder a las características categóricas y MinMaxScaler a las numéricas.
    Normaliza los datos y aplica padding si es necesario. Durante inferencia, reutiliza encoder y scaler ajustados.

    Args:
        x (list): Lista de listas de datos, que incluyen características numéricas y categóricas.
        fit (bool): Indica si ajustar (fit) o reutilizar transformadores ya ajustados (para inferencia).
        encoder: El OneHotEncoder ajustado para inferencia.
        scaler: El MinMaxScaler ajustado para inferencia.

    Returns:
        np.array: Arreglo de numpy con los datos procesados y transformados.
        encoder, scaler: Los objetos OneHotEncoder y MinMaxScaler ajustados durante el entrenamiento.
    """
    # Separar las características categóricas y numéricas
    categoricas = []
    numericas = []

    for partidos in x:
        for partido in partidos:
            # Separar las categóricas (1ra y penúltima columna)
            categoricas.append([partido[0], partido[-1]])  # Penúltima columna ya no existe

            # Las numéricas son las intermedias (excluyendo la primera y última columna)
            numericas.append(partido[1:-1])

    if fit:
        # Ajustar y transformar en entrenamiento
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        categoricas_encoded = encoder.fit_transform(categoricas)

        scaler = MinMaxScaler()  # Usamos MinMaxScaler para normalizar entre 0 y 1
        numericas_scaled = scaler.fit_transform(numericas)
    else:
        # Solo transformar en inferencia
        categoricas_encoded = encoder.transform(categoricas)
        numericas_scaled = scaler.transform(numericas)

    # Combinar categóricas codificadas y numéricas escaladas
    datos_procesados = np.hstack([categoricas_encoded, numericas_scaled])

    # Rehacer la estructura original (listas dentro de listas) para cada partido
    index = 0
    x_normalizado = []
    for partidos in x:
        num_partidos = len(partidos)
        x_normalizado.append(datos_procesados[index:index + num_partidos])
        index += num_partidos

    # Convertir a numpy y aplicar padding
    x_final_padded = pad_sequences(x_normalizado, padding='post', dtype='float32', value=0.0)

    return np.array(x_final_padded), encoder, scaler

import pandas as pd
import numpy as np

def extraer_nombres_ultimos(csv_file): 
    df = pd.read_csv(csv_file, skip_blank_lines=False, header=None)
    nombres = []
    bloque_actual = []

    for index, fila in df.iterrows():
        if pd.isnull(fila[0]):
            if bloque_actual:
                nombres.append(bloque_actual[-1][0])
                bloque_actual = []
        else:
            bloque_actual.append(fila)

    if bloque_actual:
        nombres.append(bloque_actual[-1][0])

    nombres_array = np.array(nombres)
    return nombres_array


