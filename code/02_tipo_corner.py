#!/usr/bin/env python
# coding: utf-8

# # 02 — Tipo de Córner vs. Toques en Área
# ### ISAC Set-Piece Challenge — Club América
# 
# **Pregunta:** ¿Qué tipo de córner (inswinging, outswinging, corto, directo) genera más posesiones con 3+ toques en el área y mayor tasa de gol?
# 
# **Requiere:** `corners_america.parquet` (de `00_limpiar_datos.ipynb`)
# 
# ### Tipos de córner en StatsBomb
# - **Inswinging:** el balón entra curveado hacia la portería (`pass_inswinging == True`)
# - **Outswinging:** el balón sale curveado de la portería (`pass_outswinging == True`)
# - **Corto:** pase corto, `pass_length < 10`, sin swing definido
# - **Directo:** el resto — centros sin efecto definido

# ## 1. Imports y carga

# In[1]:


import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mplsoccer import Pitch

df = pd.read_parquet('corners_america.parquet')
print(f'Base cargada: {len(df):,} eventos')


# ## 2. Filtrar córners a favor del América y parsear coordenadas

# In[2]:


# Córners a favor del América
primer_evento = (
    df.sort_values(['match_id', 'possession', 'minute'])
    .groupby(['match_id', 'possession'])
    .first()
    .reset_index()
)
posesiones_america = primer_evento[
    primer_evento['team'] == 'América'
][['match_id', 'possession']]

corners_ataque = df.merge(posesiones_america, on=['match_id', 'possession'], how='inner').copy()

# Parsear coordenadas
def parse_xy(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan, np.nan
    try:
        if isinstance(val, str):
            val = ast.literal_eval(val)
        return float(val[0]), float(val[1])
    except Exception:
        return np.nan, np.nan

coords = corners_ataque['location'].apply(parse_xy)
corners_ataque[['x', 'y']] = pd.DataFrame(coords.tolist(), index=corners_ataque.index)

print(f'Posesiones a favor del América: {len(posesiones_america):,}')


# ## 3. Clasificar el tipo de córner de cada posesión
# 
# El tipo se determina a partir del **evento del saque** — el primer `Pass` con `pass_type == 'Corner'` de cada posesión.
# 
# Jerarquía de clasificación:
# 1. Si `pass_length < 10` → **Corto** (independientemente del swing)
# 2. Si `pass_inswinging == True` → **Inswinging**
# 3. Si `pass_outswinging == True` → **Outswinging**
# 4. El resto → **Directo**

# In[3]:


# Aislar el saque de cada posesión (primer Pass con pass_type == Corner)
saques = corners_ataque[
    corners_ataque['pass_type'] == 'Corner'
].sort_values(['match_id', 'possession', 'minute'])

saques = saques.groupby(['match_id', 'possession']).first().reset_index()

def clasificar_corner(row):
    length = row.get('pass_length', np.nan)
    if pd.notna(length) and length < 10:
        return 'Corto'
    elif row.get('pass_inswinging') == True:
        return 'Inswinging'
    elif row.get('pass_outswinging') == True:
        return 'Outswinging'
    else:
        return 'Directo'

saques['tipo_corner'] = saques.apply(clasificar_corner, axis=1)

print('Distribución de tipos de córner del América:')
print(saques['tipo_corner'].value_counts())
print(f'\nTotal saques clasificados: {len(saques)}')
print(f'Posesiones sin saque identificado: {len(posesiones_america) - len(saques)}')


# ## 4. Contar toques en área y detectar goles por posesión

# In[4]:


TIPOS_TOQUE = {
    'Pass', 'Shot', 'Ball Receipt*', 'Clearance',
    'Miscontrol', 'Dribble', 'Interception', 'Block',
    'Goal Keeper', 'Foul Won', 'Foul Committed'
}

# Toques dentro del área por posesión
en_area = (
    (corners_ataque['x'] > 102) &
    (corners_ataque['y'] > 18) &
    (corners_ataque['y'] < 62) &
    (corners_ataque['type'].isin(TIPOS_TOQUE))
)
toques_area = (
    corners_ataque[en_area]
    .groupby(['match_id', 'possession'])
    .size()
    .reset_index(name='toques_en_area')
)

# Goles por posesión
goles = corners_ataque[
    (corners_ataque['type'] == 'Shot') &
    (corners_ataque['shot_outcome'] == 'Goal')
][['match_id', 'possession']].drop_duplicates()
goles['gol'] = True

# OBV por posesión
obv = (
    corners_ataque
    .groupby(['match_id', 'possession'])['obv_total_net']
    .sum()
    .reset_index(name='obv_total')
)

# Unir todo con el tipo de córner
tabla_base = (
    saques[['match_id', 'possession', 'tipo_corner']]
    .merge(toques_area, on=['match_id', 'possession'], how='left')
    .merge(goles,       on=['match_id', 'possession'], how='left')
    .merge(obv,         on=['match_id', 'possession'], how='left')
)
tabla_base['toques_en_area'] = tabla_base['toques_en_area'].fillna(0).astype(int)
tabla_base['gol']            = tabla_base['gol'].fillna(False)

tabla_base['categoria'] = tabla_base['toques_en_area'].apply(
    lambda n: '0 toques' if n == 0
    else      '1 toque'  if n == 1
    else      '2 toques' if n == 2
    else      '3+ toques'
)

print('Vista previa:')
print(tabla_base.head(8).to_string(index=False))


# ## 5. Tabla principal: tipo de córner × tasa de gol

# In[5]:


ORDEN_TIPO = ['Inswinging', 'Outswinging', 'Corto', 'Directo']

tabla_tipo = (
    tabla_base
    .groupby('tipo_corner')
    .agg(
        posesiones      = ('gol', 'count'),
        goles           = ('gol', 'sum'),
        obv_promedio    = ('obv_total', 'mean'),
        pct_3mas_toques = ('toques_en_area', lambda x: (x >= 3).mean() * 100)
    )
    .reindex(ORDEN_TIPO)
)
tabla_tipo['tasa_gol_%']      = (tabla_tipo['goles'] / tabla_tipo['posesiones'] * 100).round(2)
tabla_tipo['obv_promedio']    = tabla_tipo['obv_promedio'].round(4)
tabla_tipo['pct_3mas_toques'] = tabla_tipo['pct_3mas_toques'].round(1)

print('================================================================')
print('  TIPO DE CÓRNER — TASA DE GOL, OBV Y % CON 3+ TOQUES EN ÁREA')
print('================================================================')
print(tabla_tipo[['posesiones','goles','tasa_gol_%','obv_promedio','pct_3mas_toques']].to_string())
print()
print('pct_3mas_toques = % de córners de ese tipo que generaron 3+ toques en el área')


# ## 6. Tabla cruzada: tipo de córner × categoría de toques

# In[6]:


ORDEN_CAT = ['0 toques', '1 toque', '2 toques', '3+ toques']

# Distribución de categorías por tipo (en %)
cruzada = pd.crosstab(
    tabla_base['tipo_corner'],
    tabla_base['categoria'],
    normalize='index'
).reindex(index=ORDEN_TIPO, columns=ORDEN_CAT) * 100

print('============================================================')
print('  DISTRIBUCIÓN DE TOQUES EN ÁREA POR TIPO DE CÓRNER (%)')
print('============================================================')
print(cruzada.round(1).to_string())
print()
print('Cada fila suma 100%. Muestra qué % de córners de cada tipo')
print('generan 0, 1, 2 o 3+ toques dentro del área.')


# ## 7. Gráfica: tasa de gol y % de 3+ toques por tipo de córner

# In[7]:


fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('#0d1117')

COLORES_TIPO = {
    'Inswinging':  '#e84242',
    'Outswinging': '#1a78cf',
    'Corto':       '#e8a838',
    'Directo':     '#6a6a6a'
}
colores = [COLORES_TIPO[t] for t in ORDEN_TIPO if t in tabla_tipo.index]
tipos_validos = [t for t in ORDEN_TIPO if t in tabla_tipo.index]

for ax in axes:
    ax.set_facecolor('#0d1117')
    ax.spines[['top','right','left']].set_visible(False)
    ax.spines['bottom'].set_color('#444')
    ax.tick_params(colors='white', labelsize=11)
    ax.yaxis.grid(True, color='#333', zorder=0)
    ax.set_axisbelow(True)

# Gráfica 1: tasa de gol
vals1 = tabla_tipo.loc[tipos_validos, 'tasa_gol_%']
barras1 = axes[0].bar(tipos_validos, vals1, color=colores, edgecolor='#ffffff22', zorder=3)
for b, v, t in zip(barras1, vals1, tipos_validos):
    n = int(tabla_tipo.loc[t, 'posesiones'])
    axes[0].text(b.get_x() + b.get_width()/2, b.get_height() + 0.05,
                 f'{v:.2f}%\n(n={n})', ha='center', va='bottom',
                 color='white', fontsize=9)
axes[0].set_ylabel('Tasa de gol (%)', color='white')
axes[0].set_title('Tasa de gol por tipo de córner', color='white', fontsize=12, pad=12)

# Gráfica 2: % con 3+ toques en área
vals2 = tabla_tipo.loc[tipos_validos, 'pct_3mas_toques']
barras2 = axes[1].bar(tipos_validos, vals2, color=colores, edgecolor='#ffffff22', zorder=3)
for b, v, t in zip(barras2, vals2, tipos_validos):
    n = int(tabla_tipo.loc[t, 'posesiones'])
    axes[1].text(b.get_x() + b.get_width()/2, b.get_height() + 0.2,
                 f'{v:.1f}%\n(n={n})', ha='center', va='bottom',
                 color='white', fontsize=9)
axes[1].set_ylabel('% posesiones con 3+ toques en área', color='white')
axes[1].set_title('% que genera 3+ toques en área', color='white', fontsize=12, pad=12)

fig.suptitle('Tipo de córner del América — Efectividad y generación de segundas pelotas\nLiga MX 2021–2025',
             color='white', fontsize=13, y=1.03)
plt.tight_layout()
plt.savefig('tipo_corner_efectividad.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print('Guardado: tipo_corner_efectividad.png')


# ## 8. Pitch map por tipo de córner
# 
# Dónde caen los centros de cada tipo — solo los saques (primer evento de la posesión).

# In[8]:


# Parsear coordenadas del destino del saque (pass_end_location)
def parse_xy_end(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan, np.nan
    try:
        if isinstance(val, str):
            val = ast.literal_eval(val)
        return float(val[0]), float(val[1])
    except Exception:
        return np.nan, np.nan

saques_full = corners_ataque[
    corners_ataque['pass_type'] == 'Corner'
].sort_values(['match_id','possession','minute']).groupby(['match_id','possession']).first().reset_index()

saques_full['tipo_corner'] = saques_full.apply(clasificar_corner, axis=1)

end_coords = saques_full['pass_end_location'].apply(parse_xy_end)
saques_full[['end_x', 'end_y']] = pd.DataFrame(end_coords.tolist(), index=saques_full.index)

# 4 pitch maps, uno por tipo
pitch = Pitch(pitch_type='statsbomb', pitch_color='#0d1117', line_color='#555', half=True)
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.patch.set_facecolor('#0d1117')
axes = axes.flatten()

for ax, tipo in zip(axes, ORDEN_TIPO):
    pitch.draw(ax=ax)
    datos = saques_full[saques_full['tipo_corner'] == tipo]
    xs = datos['end_x'].dropna()
    ys = datos['end_y'].dropna()
    if len(xs) > 2:
        bin_stat = pitch.bin_statistic(xs, ys, statistic='count', bins=(10, 7))
        pitch.heatmap(bin_stat, ax=ax, cmap='Reds', edgecolors='#111', alpha=0.85)
        pitch.scatter(xs, ys, ax=ax, s=18, color='white', alpha=0.4, zorder=3)
    tasa = tabla_tipo.loc[tipo, 'tasa_gol_%'] if tipo in tabla_tipo.index else 0
    n    = int(tabla_tipo.loc[tipo, 'posesiones']) if tipo in tabla_tipo.index else 0
    ax.set_title(f'{tipo}  |  tasa gol: {tasa:.2f}%  |  n={n}',
                 color='white', fontsize=11, pad=8)

fig.suptitle('Dónde cae el centro según tipo de córner — Club América',
             color='white', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('pitch_map_tipo_corner.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print('Guardado: pitch_map_tipo_corner.png')


# ## 9. Resumen ejecutivo

# In[9]:


print('================================================')
print('  RESUMEN EJECUTIVO — Tipo de córner')
print('================================================')
print(tabla_tipo[['posesiones','goles','tasa_gol_%','obv_promedio','pct_3mas_toques']].to_string())
print()
print('Tabla cruzada (% distribución de toques por tipo):')
print(cruzada.round(1).to_string())
print()
print('Archivos generados:')
print('  tipo_corner_efectividad.png')
print('  pitch_map_tipo_corner.png')

