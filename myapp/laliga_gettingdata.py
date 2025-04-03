import requests
from bs4 import BeautifulSoup 
import pandas as pd
import re
import time
import os
import csv
import locale
import re
from copy import deepcopy 
import shutil
from pathlib import Path

# Obtener la ruta base del proyecto dinámicamente
BASE_DIR = Path(__file__).resolve().parent.parent

months_mapping = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12,
    'Enero': 1, 'Febrero': 2, 'Marzo': 3, 'Abril': 4, 'Mayo': 5, 'Junio': 6, 'Julio': 7, 'Agosto': 8,
    'Septiembre': 9, 'Octubre': 10, 'Noviembre': 11, 'Diciembre': 12
}

# Mapeo de equipos a su posición
teams_mapping = {
    'Barcelona': 1,
    'Real Madrid': 2,
    'Atlético Madrid': 3,
    'Real Sociedad': 4,
    'Villarreal': 5,
    'Real Betis': 6,
    'Osasuna': 7,
    'Athletic Club': 8,
    'Mallorca': 9,
    'Girona': 10,
    'Rayo Vallecano': 11,
    'Sevilla': 12,
    'Celta Vigo': 13,
    'Alavés': 14,
    'Getafe': 15,
    'Valencia': 16,
    'Las Palmas': 17,
    'Valladolid': 18,
    'Espanyol': 19,
    'Leganés': 20
}

team_style_mapping = {
    'Barcelona': 'Ofensivo',
    'Real Madrid': 'Ofensivo',
    'Atlético Madrid': 'Defensivo',
    'Real Sociedad': 'Ofensivo',
    'Villarreal': 'Ofensivo',
    'Real Betis': 'Ofensivo',
    'Osasuna': 'Defensivo',
    'Athletic Club': 'Ofensivo',
    'Mallorca': 'Defensivo',
    'Girona': 'Ofensivo',
    'Rayo Vallecano': 'Contragolpe',
    'Sevilla': 'Ofensivo',
    'Celta Vigo': 'Posesión',
    'Alavés': 'Defensivo',
    'Getafe': 'Defensivo',
    'Valencia': 'Contragolpe',
    'Almería': 'Ofensivo',
    'Valladolid': 'Contragolpe',
    'Espanyol': 'Contragolpe',
    'Leganés': 'Defensivo'
}
def backup_data_players(data_file):
    """
    Guarda una copia del data_players.csv en data_players_antiguo.csv.
    Si data_players_antiguo.csv ya existe, lo sobrescribe.
    """
    # Cambia el nombre del archivo de forma correcta usando with_name
    backup_file = data_file.with_name("data_players_antiguo.csv")

    # Si `data_players.csv` existe, hacer el backup
    if data_file.exists():
        print(" Guardando backup en 'data_players_antiguo.csv'...")
        shutil.copy(data_file, backup_file)
        print(" Backup realizado correctamente.")
    else:
        print(" 'data_players.csv' no existe. No se creó backup.")




print("✅ 'data_players.csv' actualizado correctamente.")
# Función para realizar la solicitud HTTP con reintentos y manejo de errores
import cloudscraper
import time

def fetch_page(url, retries=5, delay=5):
    scraper = cloudscraper.create_scraper()

    for attempt in range(retries):
        try:
            print(f"Intento {attempt + 1}: Accediendo a {url}...")
            response = scraper.get(url)

            if response.status_code == 200:
                print("✅ Página obtenida con éxito")
                return response  # ✅ Devuelve el objeto Response en lugar de response.text
            elif response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', delay))
                print(f"⚠️ Error 429. Esperando {retry_after} segundos antes del próximo intento...")
                time.sleep(retry_after)
                delay *= 2
            elif response.status_code == 403:
                print("⚠️ Error 403. Acceso prohibido. Posible bloqueo.")
                return None
            else:
                print(f"⚠️ Error {response.status_code}: No se pudo acceder.")
                return None

        except Exception as e:
            print(f"⚠️ Excepción durante la solicitud: {e}")
            time.sleep(delay)
            delay *= 2

    return None

# Función para extraer enlaces de equipos
from bs4 import BeautifulSoup

import requests
from bs4 import BeautifulSoup

def extract_team_links(url, filename):
    # Descargar la página HTML
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Patrón para los enlaces de equipos
    link_pattern = re.compile(r'^/en/squads/[a-zA-Z0-9]+/.+-Stats$')

    # Extraer todos los enlaces 'a' de la página
    links = soup.find_all('a')

    # Filtrar y construir lista de enlaces válidos
    valid_team_links = []
    seen_links = set()  # Usamos un set para evitar duplicados

    for link in links:
        href = link.get('href', '')
        
        # Verifica si coincide con el patrón y excluye enlaces que contengan "vs"
        if link_pattern.match(href) and 'vs' not in link.text:
            full_link = f'https://fbref.com{href}'
            if full_link not in seen_links:  # Verifica si ya hemos agregado el link
                valid_team_links.append(full_link)
                seen_links.add(full_link)  # Añadimos al set para evitar duplicados

        # Parar cuando tengamos 20 enlaces
        if len(valid_team_links) >= 20:
            break

    # Abrir el archivo en modo escritura, lo que borra su contenido
    with open(filename, 'w') as f:
        # Escribir los enlaces en el archivo
        for link in valid_team_links:
            f.write(link + '\n')

# URL de la página
url = 'https://fbref.com/en/comps/12/La-Liga-Stats'

# Llama a la función para extraer los enlaces de los equipos
#extract_team_links(url, 'team_links.txt')





def extract_match_links(team_url):
    response = fetch_page(team_url)
    if response:
        soup = BeautifulSoup(response.content, 'html.parser')
        match_links = set()
        # Lista de nombres de meses en inglés y español
        months = [
            'January', 'February', 'March', 'April', 'May', 'June', 
            'July', 'August', 'September', 'October', 'November', 'December',
            'Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
            'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'
        ]
        # Buscar todos los enlaces con href que contengan un mes y terminen con "-La-Liga"
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            if href and any(month in href for month in months):
                # Verifica que el último segmento del enlace contenga "La-Liga" y algo más
                if 'La-Liga' in href.split('/')[-1]:
                    full_match_link = 'https://fbref.com' + href
                    match_links.add(full_match_link)
        return list(match_links)
    else:
        print(f"Error al acceder al enlace del equipo.")
        return []




def guardar_partidos_unicos(input_file, output_file):
    """
    Lee los enlaces de partidos desde el archivo `input_file`,
    elimina los duplicados y guarda solo los enlaces únicos en `output_file`.
    """
    try:
        # Leer los enlaces de partidos desde el archivo de entrada
        with open(input_file, 'r', encoding='utf-8') as file:
            enlaces = file.readlines()

        # Eliminar espacios en blanco y duplicados
        enlaces_unicos = list(set(link.strip() for link in enlaces if link.strip()))

        # Guardar los enlaces únicos en el archivo de salida
        with open(output_file, 'w', encoding='utf-8') as file:
            for enlace in sorted(enlaces_unicos):  # Ordenar para tener consistencia
                file.write(f"{enlace}\n")

        print(f"Se han guardado {len(enlaces_unicos)} enlaces únicos en {output_file}")

    except FileNotFoundError:
        print(f"El archivo {input_file} no existe.")


def extract_goalkeeper_stats(soup):
    """
    Extrae las estadísticas de los porteros de la página.
    Devuelve una lista de diccionarios con los datos de cada portero encontrado.
    """
    # Encontrar todos los divs que contienen estadísticas de porteros
    keeper_tables = soup.find_all('div', id=re.compile(r'div_keeper_stats_.*'))
    
    goalkeeper_stats = []
    
    if keeper_tables:
        for keeper_table in keeper_tables:
            table = keeper_table.find('table', {'id': re.compile(r'keeper_stats_.*')})
            if table:
                for row in table.find_all('tr'):
                    cells = row.find_all('td')
                    if len(cells) > 0:
                        player_header = row.find('th', {'data-stat': 'player'})
                        if player_header:
                            player_name = player_header.get_text(strip=True)
                            goals_against = cells[4].get_text(strip=True) or '0'
                            saves = cells[5].get_text(strip=True) or '0'
                            psxg = cells[7].get_text(strip=True) or '0'
                            
                            goalkeeper_data = {
                                'Name': player_name,
                                'GC (Goles en contra)': goals_against,
                                'Salvadas': saves,
                                'PSxG': psxg
                            }
                            
                            goalkeeper_stats.append(goalkeeper_data)
    
    return goalkeeper_stats

def preprocess_player_position(player_stats):
    """
    Preprocesa la posición de un jugador para tomar solo la primera posición si es que tiene varias.
    """
    if ',' in player_stats['Position']:
        return player_stats['Position'].split(',')[0].strip()
    return player_stats['Position']


import os
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re

def extract_player_stats(url):
    """
    Dado un URL de un partido, extrae las estadísticas de todos los jugadores y porteros.
    Añade el nombre del equipo como una característica en los datos del jugador.
    """
    scraper = cloudscraper.create_scraper()  # Inicializa CloudScraper
    player_stats_list = []
    player_info = []
    i = 0
     
    try:
        response = scraper.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')

            # Buscar el título del informe del partido y la fecha
            content_div = soup.find('div', id='content')
            match_title = ''
            match_date = ''
            team1_name = ''
            team2_name = ''
            
            if content_div:
                match_title_element = content_div.find('h1')
                if match_title_element:
                    match_title = match_title_element.get_text(strip=True)

                    # Separar el título y la fecha
                    if ' – ' in match_title:
                        title_part, date_part = match_title.split(' – ', 1)

                        # Extraer nombres de equipos y fecha
                        if ' vs. ' in title_part:
                            team1_name, rest = title_part.split(' vs. ', 1)
                            team2_name = rest.split(' Match Report')[0].strip()
                            team1_name = team1_name.strip()
                            team2_name = team2_name.strip()

                            match = re.search(r'(\b[A-Za-z]+ \d{1,2}, 202\d)', date_part)
                            
                            if match:
                                match_date = match.group(0).strip()  # Extrae la parte relevante de la fecha
                            else:
                                match_date = ''  # Si no encuentra la fecha en el formato esperado

            # Buscar estadísticas de porteros
            goalkeeper_stats_list = extract_goalkeeper_stats(soup)
            keeper_stats_dict = {keeper['Name']: keeper for keeper in goalkeeper_stats_list}

            # Buscar todos los div cuyo id contenga tanto "stats" como "summary"
            containers = soup.find_all('div', id=re.compile(r'stats.*summary|summary.*stats'))

            if containers:
                for container in containers:
                    tables = container.find_all('table', {'class': 'stats_table'})

                    if tables:
                        for table in tables:
                            i = i + 1
                            for row in table.find_all('tr'):
                                cells = row.find_all('td')

                                if len(cells) > 0:
                                    player_header = row.find('th', {'data-stat': 'player'})
                                    if player_header:
                                        player_name = player_header.get_text(strip=True)
                                        if 'jugadores' not in player_name:
                                            shirtnumber = cells[0].get_text(strip=True) or '0'
                                            nationality = cells[1].get_text(strip=True) or '0'
                                            position = cells[2].get_text(strip=True) or '0'
                                            age = cells[3].get_text(strip=True) or '0'
                                            minutes = cells[4].get_text(strip=True) or '0'
                                            goals = cells[5].get_text(strip=True) or '0'
                                            assists = cells[6].get_text(strip=True) or '0'
                                            pens_made = cells[7].get_text(strip=True) or '0'
                                            pens_att = cells[8].get_text(strip=True) or '0'
                                            shots = cells[9].get_text(strip=True) or '0'
                                            shots_on_target = cells[10].get_text(strip=True) or '0'
                                            cards_yellow = cells[11].get_text(strip=True) or '0'
                                            cards_red = cells[12].get_text(strip=True) or '0'
                                            touches = cells[13].get_text(strip=True) or '0'
                                            tackles = cells[14].get_text(strip=True) or '0'
                                            interceptions = cells[15].get_text(strip=True) or '0'
                                            blocks = cells[16].get_text(strip=True) or '0'
                                            xg = cells[17].get_text(strip=True) or '0'
                                            npxg = cells[18].get_text(strip=True) or '0'
                                            xg_assist = cells[19].get_text(strip=True) or '0'
                                            sca = cells[20].get_text(strip=True) or '0'
                                            gca = cells[21].get_text(strip=True) or '0'
                                            passes_completed = cells[22].get_text(strip=True) or '0'
                                            passes = cells[23].get_text(strip=True) or '0'
                                            passes_pct = cells[24].get_text(strip=True) or '0'
                                            progressive_passes = cells[25].get_text(strip=True) or '0'
                                            carries = cells[26].get_text(strip=True) or '0'
                                            progressive_carries = cells[27].get_text(strip=True) or '0'
                                            take_ons = cells[28].get_text(strip=True) or '0'
                                            take_ons_won = cells[29].get_text(strip=True) or '0'

                                            # Alternar el nombre del equipo basado en el índice del contenedor
                                            opponent_team = team2_name if i == 1 else team1_name
                                            self_team = team1_name if i == 1 else team2_name
                                            # Agregar estadísticas del jugador
                                            player_stats = {
                                                'Name': player_name,
                                                'Shirtnumber': shirtnumber,
                                                'Nationality': nationality,
                                                'Position': position,
                                                'Age': age,
                                                'Minutes': minutes,
                                                'Goals': goals,
                                                'Assists': assists,
                                                'Pens Made': pens_made,
                                                'Pens Att': pens_att,
                                                'Shots': shots,
                                                'Shots on Target': shots_on_target,
                                                'Cards Yellow': cards_yellow,
                                                'Cards Red': cards_red,
                                                'Touches': touches,
                                                'Tackles': tackles,
                                                'Interceptions': interceptions,
                                                'Blocks': blocks,
                                                'XG': xg,
                                                'NPXG': npxg,
                                                'XG Assist': xg_assist,
                                                'SCA': sca,
                                                'GCA': gca,
                                                'Passes Completed': passes_completed,
                                                'Passes': passes,
                                                'Passes %': passes_pct,
                                                'Progressive Passes': progressive_passes,
                                                'Carries': carries,
                                                'Progressive Carries': progressive_carries,
                                                'Take Ons': take_ons,
                                                'Take Ons Won': take_ons_won,
                                                'GC (Goles en contra)': '0',
                                                'Salvadas': '0',
                                                'PSxG': '0',
                                                'Opponent Team': opponent_team,
                                                'Match Date': match_date,
                                                'Puntuation': 0
                                            }

                                            # Preprocesar la posición
                                            player_stats['Position'] = preprocess_player_position(player_stats)
                                            player_stats_list.append(player_stats)
                                            player_info.append([player_name, player_stats['Position'],shirtnumber, self_team])
                                            

            # Actualizar estadísticas de porteros
            for player in player_stats_list:
                player_name = player['Name']

                if player_name in keeper_stats_dict:
                    keeper_stats = keeper_stats_dict[player_name]
                    player['GC (Goles en contra)'] = keeper_stats['GC (Goles en contra)']
                    player['Salvadas'] = keeper_stats['Salvadas']
                    player['PSxG'] = keeper_stats['PSxG']

                    for jug in player_stats_list:
                        if jug['Opponent Team'] == player['Opponent Team']:
                            jug['GC (Goles en contra)'] = player['GC (Goles en contra)']
        else:
            print(f"Error al acceder a la página. Código de estado: {response.status_code}")

    except Exception as e:
        print(f"Error al procesar el enlace {url}: {str(e)}")

    # Actualizar las puntuaciones de los jugadores
    update_player_stats_with_score(player_stats_list)
    return player_stats_list, player_info

def process_links(file_path, output_file, player_info_file, delay_seconds=5):
    """
    Procesa cada enlace en el archivo y añade un retraso entre cada solicitud para evitar bloqueos.
    Los resultados se guardan en dos archivos: uno para estadísticas y otro para la información de los jugadores.
    
    :param file_path: Ruta del archivo que contiene los enlaces.
    :param output_file: Ruta del archivo de salida donde se guardarán las estadísticas de los jugadores.
    :param player_info_file: Ruta del archivo de salida donde se guardará la información de los jugadores.
    :param delay_seconds: Número de segundos de retraso entre cada solicitud.
    """
    # Si el archivo de salida no existe, escribir los encabezados personalizados
    if not os.path.isfile(output_file):
        with open(output_file, 'w', encoding='utf-8') as f:
            headers = "Name,Shirtnumber,Nationality,Position,Age,Minutes,Goals,Assists,Pens Made,Pens Att,Shots,Shots on Target,Cards Yellow,Cards Red,Touches,Tackles,Interceptions,Blocks,XG,NPXG,XG Assist,SCA,GCA,Passes Completed,Passes,Passes %,Progressive Passes,Carries,Progressive Carries,Take Ons,Take Ons Won,GC (Goles en contra),Salvadas,PSxG,Opponent Team,Match Date,Puntuation,Team Style,Year,Month,Day,Reward\n"
            f.write(headers)

    if not os.path.isfile(player_info_file):
        with open(player_info_file, 'w', encoding='utf-8') as f:
            headers = "Player Name,Position,Shirtnumber,Team\n"
            f.write(headers)

    i = 1
    with open(file_path, 'r', encoding='utf-8') as file:
        links = file.readlines()

    for link in links:
        url = link.strip()  # Elimina espacios en blanco o saltos de línea
        if url:
            print(f"Procesando el enlace: {i}")
            i = i + 1

            try:
                # Llama a la función que procesa el enlace y asegura que retorne el resultado
                result, player_info = extract_player_stats(url)

                if result:  # Verifica que la función retornó datos válidos
                    # Guardar las estadísticas en el archivo de salida
                    df = pd.DataFrame(result)
                    df.to_csv(output_file, mode='a', header=False, index=False)

                    # Guardar la información de los jugadores en el archivo correspondiente
                    df_info = pd.DataFrame(player_info, columns=['Player Name','Position','Shirtnumber', 'Team'])
                    df_info.to_csv(player_info_file, mode='a', header=False, index=False)

                else:
                    print(f"No se encontraron datos para el enlace: {url}")

            except Exception as e:
                print(f"Error al procesar el enlace {url}: {str(e)}")
                # Continúa con el siguiente enlace en caso de error

            # Añade el retraso entre solicitudes
            time.sleep(delay_seconds)

        
# Llama a la función con el archivo de enlaces, archivo de salida y un retraso de 5 segundos
def calculate_player_score(player_stats):
    """
    Calcula la puntuación de un jugador basado en sus estadísticas y su posición, incluyendo nuevas características.
    
    :param player_stats: Diccionario con las estadísticas del jugador.
    :return: Puntuación total del jugador.
    """
    # Obtener datos del diccionario de estadísticas
    position = player_stats['Position']
    minutes_played = int(player_stats['Minutes'])
    goals = int(player_stats['Goals'])
    assists = int(player_stats['Assists'])
    shots_on_target = int(player_stats['Shots on Target'])
    yellow_cards = int(player_stats['Cards Yellow'])
    red_cards = int(player_stats['Cards Red'])
    touches = int(player_stats['Touches'])
    tackles = int(player_stats['Tackles'])
    interceptions = int(player_stats['Interceptions'])
    blocks = int(player_stats['Blocks'])
    goals_conceded = int(player_stats.get('GC (Goles en contra)', 0))
    saves = int(player_stats.get('Salvadas', 0))
    take_ons_won = int(player_stats['Take Ons Won'])
    pens_made = int(player_stats.get('Pens Made', 0))
    pens_att = int(player_stats.get('Pens Att', 0))
    progressive_carries = float(player_stats.get('Progressive Carries', 0))
    
    # Nuevas estadísticas
    progressive_passes = float(player_stats.get('Progressive Passes', 0))
    sca = float(player_stats.get('SCA', 0))  # Shot-Creating Actions
    gca = float(player_stats.get('GCA', 0))  # Goal-Creating Actions
    passes_pct = float(player_stats.get('Passes %', 0))  # Porcentaje de pases completados

    # Inicializar la puntuación
    score = 0

    # Puntuación por minutos jugados
    if minutes_played >= 60:
        score += 2
    else:
        score += 1

    # Puntuación por goles según la posición
    if position in ['GK', 'DF', 'FB', 'LB', 'RB', 'CB']:
        score += goals * 6
    elif position in ['MF', 'DM', 'CM', 'LM', 'RM', 'WM', 'AM']:
        score += goals * 5
    elif position in ['FW', 'LW', 'RW']:
        score += goals * 4
    
    # Puntuación por asistencias
    score += assists * 3

    # Penalti fallado
    pens_missed = pens_att - pens_made
    score -= pens_missed * 2

    # Penalti convertido
    score += pens_made * 3

    # Puntuación por portería a cero (solo para porteros y defensas que juegan más de 60 minutos)
    if position == 'GK' and minutes_played >= 60 and goals_conceded == 0:
        score += 4
    elif position in ['DF', 'FB', 'LB', 'RB', 'CB'] and minutes_played >= 60 and goals_conceded == 0:
        score += 4
    elif position in ['MF', 'DM', 'CM', 'LM', 'RM', 'WM', 'AM'] and minutes_played >= 60 and goals_conceded == 0:
        score += 2
    elif position in ['FW', 'LW', 'RW'] and minutes_played >= 60 and goals_conceded == 0:
        score += 1

    # Puntuación por goles recibidos (solo para porteros y defensas)
    if position in ['GK', 'DF', 'FB', 'LB', 'RB', 'CB']:
        score -= (goals_conceded // 2) * 2
    else:
        score -= (goals_conceded // 2) * 1

    # Puntuación por tarjetas
    score -= yellow_cards * 1  # Penaliza 1 punto por tarjeta amarilla
    score -= red_cards * 3  # Penaliza 3 puntos por tarjeta roja directa

    # Puntuación por paradas (solo para porteros, cada dos paradas)
    if position == 'GK':
        score += (saves // 2) * 1

    # Acciones adicionales ofensivas y defensivas
    score += (progressive_carries // 3) * 1
    score += (shots_on_target // 3) * 1
    score += (take_ons_won // 2) * 1
    score += (tackles // 4) * 1  # Cada 4 tackles suma 1 punto
    score += (interceptions // 4) * 1  # Cada 4 intercepciones suma 1 punto
    score += (blocks // 4) * 1  # Cada 4 bloqueos suma 1 punto
    score += (touches // 20) * 1  # Cada 20 toques suma 1 punto

    # Nuevas características añadidas

    # 1. Progressive Passes - Mejora en pases progresivos
    if position in ['MF', 'DF']:
        score += (progressive_passes // 3) * 1  # Cada 3 pases progresivos suma 1 punto

    # 2. Shot-Creating Actions (SCA) - Creación de oportunidades de gol
    if position in ['FW', 'MF', 'AM', 'LW', 'RW']:
        score += (sca // 3) * 1  # Cada 3 SCA suma 1 punto

    # 3. Goal-Creating Actions (GCA) - Creación de goles
    if position in ['FW', 'AM', 'LW', 'RW']:
        score += (gca // 4) * 2  # Cada GCA suma 2 puntos

    # 4. Pass Completion % - Precisión en el pase
    if passes_pct >= 85 and position in ['MF', 'DF', 'DM', 'CM']:
        score += 2  # Suma 2 puntos si el porcentaje de pases completados es >= 85%

    return score

def update_player_stats_with_score(player_stats_list):
    for player in player_stats_list:
        player['Puntuation'] = calculate_player_score(player) 

def count_empty_lines_in_csv(file_path):
    empty_line_count = 0
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                # Si todas las celdas en la fila están vacías, contamos la línea como vacía
                if all(not cell.strip() for cell in row):
                    empty_line_count += 1
    except Exception as e:
        print(f"Error al procesar el archivo: {str(e)}")
    
    return empty_line_count


def clean_data_string(data_str):
    cleaned_str = re.sub(r'[^\w\s,]', '', data_str)
    return cleaned_str.strip()

# Función para extraer componentes de la fecha (año, mes, día)
def extract_date_components(date_str):
    cleaned_date = re.sub(r'\s+', ' ', date_str).strip()
    parts = cleaned_date.split()

    # Verificar si es en español (día de la semana en español) o en inglés
    if len(parts) >= 4:
        # Si hay un día de la semana en la fecha, lo eliminamos
        parts.pop(0)  # Remover "Sábado", "Monday", etc.
    
    if len(parts) < 3:
        return None, None, None
    
    month_str = parts[0]  # Mes en español o inglés
    day = int(re.sub(r'[^\d]', '', parts[1]))  # Eliminar cualquier carácter no numérico del día
    year = int(re.sub(r'\D', '', parts[2]))  # Eliminar cualquier caracter no numérico del año
    
    # Buscar el mes en el diccionario
    month = months_mapping.get(month_str, None)
    if month is None:
        print(f"Mes no válido: {month_str} en la fecha {date_str}")
        return None, None, None
    
    return year, month, day


def process_and_sort_player_stats(input_file_path, output_file_path):
    if not os.path.isfile(input_file_path) or os.path.getsize(input_file_path) == 0:
        print(f"Error: El archivo '{input_file_path}' no existe o está vacío.")
        return

    # Leer el archivo CSV
    df = pd.read_csv(input_file_path, header=0)

    # Limpiar y convertir la columna 'Puntuation' a numérico
    df['Puntuation'] = pd.to_numeric(df['Puntuation'], errors='coerce')

    # Verificar si hay valores que no se pudieron convertir
    if df['Puntuation'].isna().sum() > 0:
        print(f"Advertencia: Se encontraron {df['Puntuation'].isna().sum()} valores no numéricos en 'Puntuation'.")
        print(df[df['Puntuation'].isna()])

    # Intentar convertir las celdas erróneas o valores NaN a cero
    df['Puntuation'] = df['Puntuation'].fillna(0)

    # Limpiar la columna de fechas
    df['Match Date'] = df['Match Date'].apply(lambda x: clean_data_string(x))

    # Aplicar la función para extraer año, mes y día de la fecha
    df[['Year', 'Month', 'Day']] = df['Match Date'].apply(lambda x: pd.Series(extract_date_components(x)))

    # Verificar las filas con fechas inválidas
    if df[['Year', 'Month', 'Day']].isna().sum().sum() > 0:
        print(f"Advertencia: Se encontraron filas con fechas inválidas.")

    

    # Filtrar filas con fechas inválidas
    df = df.dropna(subset=['Year', 'Month', 'Day'])

    # Reemplazar los nombres de los equipos por sus posiciones
    df['Team Style'] = df['Opponent Team'].map(team_style_mapping)
    df['Opponent Team'] = df['Opponent Team'].map(teams_mapping)
    # Ordenar los datos por nombre, luego por año, mes y día
    sorted_df = df.sort_values(by=['Name', 'Year', 'Month', 'Day'])

    # Calcular la recompensa (Restar 4 a la puntuación)
    sorted_df['Reward'] = sorted_df['Puntuation'] - 4

    # Guardar el DataFrame ordenado en un nuevo archivo CSV
    sorted_df.to_csv(output_file_path, index=False)
    print(f"Datos procesados y guardados correctamente en {output_file_path}")

# Ejemplo de uso
input_file_path = 'player_stats.csv'
output_file_path = 'sorted_players.csv'
                
def desplaçar_columnes_i_eliminar_primera_fila(archivo_entrada, archivo_salida):
    # Leer el archivo CSV
    with open(archivo_entrada, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        datos = list(reader)

    # Verificar si hay datos
    if not datos:
        print("El archivo de entrada está vacío.")
        return

    # Lista auxiliar para almacenar los datos procesados
    datos_procesados = []

    i = 0  # índice para recorrer las filas
    while i < len(datos):
        fila_actual = datos[i]

        # Si la fila está vacía, la añadimos tal cual y seguimos
        if len(fila_actual) == 0:
            datos_procesados.append(fila_actual)
            i += 1
            continue

        nombre_actual = fila_actual[0]  # Columna 0 es el nombre del jugador

        # Asegurarse de que la fila tiene al menos 1 columna (la del nombre)
        if len(fila_actual) < 1:
            i += 1
            continue

        # Encontrar todas las filas del mismo jugador
        filas_jugador = []
        while i < len(datos) and len(datos[i]) > 0 and datos[i][0] == nombre_actual:
            filas_jugador.append(datos[i])
            i += 1

        # Desplazar valores de cada fila a la siguiente fila del mismo jugador
        for j in range(len(filas_jugador) - 1):  # Procesamos hasta la penúltima fila
            if j == 0:
                fila_actual = filas_jugador[j]
            else:
                fila_actual = aux

            aux = tuple(deepcopy(filas_jugador[j + 1]))
            fila_siguiente = filas_jugador[j + 1]

            
            for col in list(range(5, 35)) + [37, 38]:
                if len(fila_actual) > col and fila_actual[col].strip():  # Comprobar si la columna no está vacía
                    # Comprobar que la fila siguiente tiene la misma cantidad de columnas
                    if len(fila_siguiente) > col:
                        # Desplazar el valor a la fila siguiente
                        fila_siguiente[col] = fila_actual[col]

        # Añadir todas las filas del jugador, excluyendo la primera fila
        datos_procesados.extend(filas_jugador[1:])  # Excluye la primera fila del jugador

    # Escribir el archivo de salida
    with open(archivo_salida, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(datos_procesados)


def agregar_espacio_blanco_por_cambio_de_nombre(input_file):
    """
    Lee un archivo línea por línea y agrega un espacio en blanco cada vez que el nombre del jugador cambia,
    modificando el archivo en el lugar (sin crear un archivo de salida).
    
    Args:
        input_file (str): Ruta del archivo de entrada y salida.
    """
    # Leer todas las líneas del archivo original
    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
    
    # Procesar las líneas y agregar los espacios en blanco
    processed_lines = []
    last_name = None

    for line in lines:
        line = line.strip()
        if not line:
            continue  # Ignorar líneas vacías
        
        # Extraer el nombre del jugador (se asume que el nombre es la primera columna)
        current_name = line.split(',')[0].strip()

        # Comparar con el nombre anterior
        if last_name is not None and current_name != last_name:
            # Si el nombre cambia, agregar una línea en blanco
            processed_lines.append('\n')
        
        # Agregar la línea actual al resultado procesado
        processed_lines.append(line + '\n')

        # Actualizar el último nombre procesado
        last_name = current_name

    # Sobrescribir el archivo original con las líneas procesadas
    with open(input_file, 'w', encoding='utf-8') as outfile:
        outfile.writelines(processed_lines)

    print("El archivo ha sido modificado con espacios en blanco añadidos.")


def detectar_bloques_de_37_lineas(ruta_archivo_entrada):
    """
    Detecta y devuelve las posiciones de los bloques de exactamente 37 líneas consecutivas en el archivo.

    Args:
        ruta_archivo_entrada (str): Ruta al archivo original.

    Returns:
        list: Lista con las posiciones iniciales y finales de los bloques de 37 líneas consecutivas.
    """
    with open(ruta_archivo_entrada, 'r', encoding='utf-8') as archivo_entrada:
        lineas = archivo_entrada.readlines()

    bloques_37_lineas = []
    bloque_actual = []
    linea_actual = 1
    inicio_bloque = None

    for index, linea in enumerate(lineas):
        if linea.strip():  # Si la línea no está vacía
            if inicio_bloque is None:
                inicio_bloque = index + 1  # Marca el inicio del bloque
            bloque_actual.append(linea)
        else:
            if len(bloque_actual) > 37:  # Si encontramos un bloque de 37 líneas
                bloques_37_lineas.append((inicio_bloque, index))  # Guardamos la posición inicial y final del bloque
            bloque_actual = []  # Reiniciamos el bloque actual
            inicio_bloque = None

    # Verificar el último bloque
    if len(bloque_actual) == 37:
        bloques_37_lineas.append((inicio_bloque, len(lineas)))

    return bloques_37_lineas
def eliminar_lineas_excedentes(ruta_archivo_entrada, ruta_archivo_salida):
    """
    Elimina líneas de bloques consecutivos de más de 37 líneas en un archivo de entrada y guarda el resultado en un archivo de salida.

    Args:
        ruta_archivo_entrada (str): Ruta al archivo original.
        ruta_archivo_salida (str): Ruta donde se guardará el archivo con las líneas eliminadas.
    """
    with open(ruta_archivo_entrada, 'r', encoding='utf-8') as archivo_entrada:
        lineas = archivo_entrada.readlines()

    # Listas para almacenar el nuevo contenido y el bloque actual
    nuevo_contenido = []
    bloque_actual = []
    
    for linea in lineas:
        if linea.strip():  # Si la línea no está vacía
            bloque_actual.append(linea)  # Añadir la línea al bloque actual
        else:
            if len(bloque_actual) > 37:  # Si el bloque excede 37 líneas, eliminar las excedentes
                bloque_actual = bloque_actual[:37]  # Cortar a 37 líneas

            nuevo_contenido.extend(bloque_actual)  # Añadir el bloque al nuevo contenido
            nuevo_contenido.append(linea)  # Añadir el salto de línea
            bloque_actual = []  # Reiniciar el bloque

    # Verificar el último bloque al final del archivo
    if len(bloque_actual) > 37:
        bloque_actual = bloque_actual[:37]  # Cortar a 37 líneas
    nuevo_contenido.extend(bloque_actual)  # Añadir el último bloque

    # Escribir el nuevo contenido en el archivo de salida
    with open(ruta_archivo_salida, 'w', encoding='utf-8') as archivo_salida:
        archivo_salida.writelines(nuevo_contenido)

    print(f"Archivo procesado guardado en {ruta_archivo_salida}")

def filter_unique_rows(input_csv, output_csv):
    """
    Filtra un archivo CSV, eliminando filas donde la primera columna contenga 'Players' o un número,
    y guarda las filas únicas en un nuevo archivo CSV.
    
    :param input_csv: Ruta al archivo CSV de entrada.
    :param output_csv: Ruta al archivo CSV de salida donde se guardarán las filas filtradas y únicas.
    """
    # Leer el archivo CSV
    df = pd.read_csv(input_csv)

    # Filtrar filas donde la primera columna contenga 'Players' o un número
    df_filtered = df[~df.iloc[:, 0].str.contains("Players|Name", na=False)]
    df_filtered = df_filtered[~df_filtered.iloc[:, 0].str.match(r'^\d+$', na=False)]


    # Obtener filas únicas
    df_unique = df_filtered.drop_duplicates()

    # Guardar el resultado en un nuevo archivo CSV
    df_unique.to_csv(output_csv, index=False)

    print(f"Filtrado completado. Se han guardado las filas únicas en '{output_csv}'.")
def agregar_encabezado_csv(file_path):
    """
    Agrega un encabezado al archivo CSV especificado. Sobrescribe el archivo colocando el encabezado al principio.

    Args:
        file_path (str): Ruta del archivo CSV.
    """
    # Definir el encabezado
    encabezado = "Nombre,Shirtnumber,Nationality,Position,Age,Minutes,Goals,Assists,Pens Made,Pens Att,Shots,Shots on Target,Cards Yellow,Cards Red,Touches,Tackles,Interceptions,Blocks,XG,NPXG,XG Assist,SCA,GCA,Passes Completed,Passes,Passes %,Progressive Passes,Carries,Progressive Carries,Take Ons,Take Ons Won,GC (Goles en contra),Salvadas,PSxG,Opponent Team,Match Date,Puntuation,Team Style,Year,Month,Day,Reward\n"

    # Leer las líneas existentes del archivo
    with open(file_path, 'r', encoding='utf-8') as file:
        contenido = file.readlines()

    # Sobrescribir el archivo con el encabezado seguido por el contenido existente
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(encabezado)  # Escribir el encabezado
        file.writelines(contenido)  # Escribir el contenido original

    print("Encabezado agregado correctamente.")

def main_process(file_path, output_file, data_file, player_stats_file, sorted_players_file):
    """
    Ejecuta el proceso principal utilizando los enlaces y archivos proporcionados.

    :param team_url: Enlace a la página de equipos.
    :param file_path: Ruta al archivo donde se guardarán los enlaces de equipos.
    :param output_file: Ruta al archivo de salida donde se guardarán los enlaces únicos de partidos.
    :param player_stats_file: Archivo de salida para las estadísticas de jugadores.
    :param sorted_players_file: Archivo de salida con las estadísticas de jugadores ordenadas.
    """
    backup_data_players(data_file)
    # 1. Cargar partidos previamente guardados en el archivo output_file
    existing_links = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_links = set(line.strip() for line in f.readlines())
    
    # 2. Extraer enlaces de equipos y partidos
    print("Extrayendo enlaces de equipos...")
    
    
    all_match_links = []
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, 'r', encoding='utf-8') as f:
            team_links = f.readlines()

        for team_link in team_links:
            print(f"Extrayendo partidos del equipo: {team_link.strip()}")
            match_links = extract_match_links(team_link.strip())
            if match_links:
                all_match_links.extend(match_links)

    if not all_match_links:
        print(f"No se encontraron partidos para los equipos extraídos de: {team_url}")
        return  # Salimos de la función si no hay partidos

    # 3. Comparar y agregar nuevos partidos
    new_links = set(all_match_links) - existing_links  # Solo los nuevos enlaces
    
    if new_links:
        print(f"Se encontraron {len(new_links)} partidos nuevos.")

        # Guardar los nuevos partidos en el archivo partidos_unicos.txt
        with open(output_file, 'a', encoding='utf-8') as f:
            for link in sorted(new_links):  # Ordenamos para mantener consistencia
                f.write(f"{link}\n")

        print(f"Se han añadido {len(new_links)} nuevos enlaces a {output_file}.")

        # Abre el archivo original en modo lectura para leer la cabecera
        with open(data_file, 'r', encoding='utf-8') as archivo:
            cabecera = archivo.readline()
        # Reabre el mismo archivo en modo escritura para sobrescribir
        with open(data_file, 'w', encoding='utf-8') as archivo:
            archivo.write(cabecera)

        # 4. Guardar nuevos enlaces en un archivo temporal para procesarlos
        temp_file = 'temp_new_links.txt'
        with open(temp_file, 'w', encoding='utf-8') as temp_f:
            for link in new_links:
                temp_f.write(f"{link}\n")

        with open(data_file, 'r+',encoding='utf-8') as archivo:
            # Leer solo la primera línea (cabecera)
            cabecera = archivo.readline()
            # Mover el puntero al inicio y sobrescribir solo la cabecera
            archivo.seek(0)
            archivo.write(cabecera)
            # Truncar el resto del archivo para eliminar las demás líneas
            archivo.truncate()

        print("Se han eliminado todas las líneas excepto la primera en el mismo archivo.")
        
        # 5. Procesar los nuevos enlaces y extraer estadísticas de jugadores
        print("Extrayendo estadísticas de jugadores...")
        process_links(temp_file, player_stats_file, data_file)

        # Eliminar archivo temporal
        os.remove(temp_file)

        # 6. Procesar y ordenar estadísticas de jugadores
        print("Procesando y ordenando estadísticas de jugadores...")
        process_and_sort_player_stats(player_stats_file, sorted_players_file)

        # 7. Contar líneas vacías en el archivo CSV
        empty_lines_count = count_empty_lines_in_csv(sorted_players_file)
        print(f"El archivo '{sorted_players_file}' contiene {empty_lines_count} líneas vacías.")

        # 8. Desplazar columnas y eliminar primera fila
        print("Desplazando columnas y eliminando primera fila...")
        desplaçar_columnes_i_eliminar_primera_fila(sorted_players_file, sorted_players_file)
        filter_unique_rows(sorted_players_file,sorted_players_file)
        filter_unique_rows(data_file,data_file)
        agregar_espacio_blanco_por_cambio_de_nombre(sorted_players_file)
        
    else:
        print("No se encontraron nuevos partidos.")

    print("Proceso completado.")


team_url = "https://fbref.com/en/comps/12/La-Liga-Stats"


# Rutas de los archivos dentro de la carpeta 'data'
file_path = BASE_DIR / 'myapp' / 'data' / 'enlaces_equipos.txt'
output_file = BASE_DIR / 'myapp' / 'data' / 'partidos_unicos.txt'
data_file = BASE_DIR / 'myapp' / 'data' / 'data_players.csv'
player_stats_file = BASE_DIR / 'myapp' / 'data' / 'estadisticas_jugadores.csv'
sorted_players_file = BASE_DIR / 'myapp' / 'data' / 'jugadores_ordenados.csv'
team_links= BASE_DIR / 'myapp' / 'data' / 'enlaces_equipos.txt'

main_process(file_path, output_file, data_file, player_stats_file, sorted_players_file)



        