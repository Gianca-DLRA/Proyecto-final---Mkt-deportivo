#!/usr/bin/env python
# coding: utf-8

# # 00 — Limpieza de datos
# ### ISAC Set-Piece Challenge — Club América
# 
# **Este notebook corre primero, antes que cualquier análisis.**
# 
# Toma el `../data/events.parquet` original (733 MB, 4.4M filas) y produce `corners_america.parquet` con solo lo que necesitamos.
# 
# **Filtros aplicados:**
# - Solo partidos donde juega el Club América (local o visitante)
# - Todas las temporadas disponibles (2021–2025)
# - Solo eventos de posesiones que nacen de córner (`play_pattern == 'From Corner'`)
# - 31 columnas relevantes (de 142 originales)
# 
# **Output:** `corners_america.parquet`

# ## 1. Imports

# In[1]:


import pandas as pd
import os

# Verificar que los archivos existen
archivos = ['../data/../data/events.parquet', '../data/../data/matches.parquet']
for archivo in archivos:
    existe = os.path.exists(archivo)
    size = os.path.getsize(archivo) / 1e6 if existe else 0
    status = f'OK  {size:.1f} MB' if existe else 'NO ENCONTRADO'
    print(f'{archivo:25s}  {status}')


# ## 2. Cargar matches y filtrar partidos del América
# 
# `../data/matches.parquet` es pequeño (0.1 MB), lo cargamos completo.
# 
# Las columnas `home_team` y `away_team` a veces vienen como diccionarios `{'id': ..., 'name': 'Club América'}` en lugar de string plano. Usamos `str(t)` para cubrir ambos casos sin error.

# In[2]:


matches = pd.read_parquet('../data/matches.parquet')
print(f'Total de partidos en la base: {len(matches):,}')
print(f'\nTemporadas disponibles:')
print(matches['season'].value_counts().sort_index())

# Filtrar partidos donde juega América (local o visitante)
es_america = (
    matches['home_team'].apply(lambda t: 'América' in str(t)) |
    matches['away_team'].apply(lambda t: 'América' in str(t))
)
matches_america = matches[es_america].copy()

print(f'\nPartidos del América encontrados: {len(matches_america):,}')
print(f'\nDesglose por temporada:')
print(matches_america['season'].value_counts().sort_index())

# Guardar los IDs para filtrar events
ids_america = set(matches_america['match_id'])
print(f'\nTotal de match_ids del América: {len(ids_america)}')


# ## 3. Cargar ../data/events.parquet y aplicar filtros
# 
# Cargamos **solo las 31 columnas** que definimos. Esto hace la carga significativamente más rápida porque Parquet permite leer columnas selectivas sin cargar todo el archivo.
# 
# Luego aplicamos los dos filtros en orden:
# 1. Solo partidos del América
# 2. Solo posesiones que nacen de córner

# In[3]:


COLUMNAS = [
    # Identificación
    'match_id', 'minute', 'period', 'possession',

    # Equipo y jugador
    'team', 'player', 'player_id', 'position',

    # Tipo de evento
    'type', 'play_pattern', 'pass_type',

    # Características del pase / córner
    'pass_inswinging', 'pass_outswinging', 'pass_length',
    'pass_cross', 'pass_height', 'pass_body_part',
    'pass_end_location', 'pass_outcome', 'pass_angle',
    'pass_shot_assist', 'pass_goal_assist',

    # Ubicación en el campo
    'location',

    # Remates
    'shot_statsbomb_xg', 'shot_outcome', 'shot_body_part', 'shot_technique',

    # OBV y presión
    'obv_total_net', 'obv_for_net', 'under_pressure', 'duration'
]

print(f'Columnas a cargar: {len(COLUMNAS)} de 142')
print('Cargando ../data/events.parquet... (puede tardar ~30 seg)')

events = pd.read_parquet('../data/events.parquet', columns=COLUMNAS)
print(f'Eventos cargados: {len(events):,}')

# Filtro 1: solo partidos del América
events_america = events[events['match_id'].isin(ids_america)].copy()
pct1 = len(events_america) / len(events) * 100
print(f'\nTras filtro América:  {len(events_america):,} eventos  ({pct1:.1f}% del total)')

# Filtro 2: solo posesiones de córner
corners = events_america[events_america['play_pattern'] == 'From Corner'].copy()
pct2 = len(corners) / len(events_america) * 100
print(f'Tras filtro córner:   {len(corners):,} eventos  ({pct2:.1f}% de partidos América)')

n_posesiones = corners.groupby(['match_id', 'possession']).ngroups
print(f'\nPosesiones de córner únicas: {n_posesiones:,}')


# ## 4. Verificación de la base limpia
# 
# Antes de guardar revisamos que todo tenga sentido: equipos correctos, goles presentes, sin columnas completamente vacías.

# In[5]:


print('=== RESUMEN DE LA BASE LIMPIA ===')
print(f'Filas:    {len(corners):,}')
print(f'Columnas: {len(corners.columns)}')

print(f'\n--- Tipos de eventos (top 10) ---')
print(corners['type'].value_counts().head(10))

print(f'\n--- Equipos presentes (América + sus rivales) ---')
print(corners['team'].value_counts().head(20))

print(f'\n--- Goles en posesiones de córner ---')
goles = corners[
    (corners['type'] == 'Shot') &
    (corners['shot_outcome'] == 'Goal')
]
print(f'Total goles: {len(goles)}')
if len(goles) > 0:
    print(goles[['match_id', 'minute', 'team', 'player', 'shot_technique']].to_string(index=False))

print(f'\n--- Nulos en columnas clave ---')
cols_clave = ['location', 'type', 'team', 'obv_total_net', 'shot_outcome', 'pass_inswinging']
nulos = corners[cols_clave].isnull().sum()
print(nulos.to_string())
print('(Los nulos en shot_outcome y pass_inswinging son normales: no todos los eventos son remates o córners)')


# ## 5. Guardar la base limpia
# 
# Guardamos como `.parquet` con compresión `snappy` — rápida de leer y escribe en segundos. Es el formato estándar para análisis en Python.

# In[6]:


OUTPUT = 'corners_america.parquet'

corners.to_parquet(OUTPUT, compression='snappy', index=False)

size_original = 733  # MB del ../data/events.parquet original
size_nuevo = os.path.getsize(OUTPUT) / 1e6
reduccion = (1 - size_nuevo / size_original) * 100

print(f'Guardado: {OUTPUT}')
print(f'Tamaño original (../data/events.parquet): {size_original} MB')
print(f'Tamaño nuevo:                     {size_nuevo:.1f} MB')
print(f'Reduccion:                        {reduccion:.0f}%')
print(f'Filas:    {len(corners):,}')
print(f'Columnas: {len(corners.columns)}')
print()
print('A partir de aqui, todos los notebooks cargan con:')
print("    df = pd.read_parquet('corners_america.parquet')")


# In[ ]:




