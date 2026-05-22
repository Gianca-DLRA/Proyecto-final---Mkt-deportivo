#!/usr/bin/env python
# coding: utf-8

# # 01b — Análisis de Multi-Toques (versión revisada)
# ### ISAC Set-Piece Challenge — Club América
# 
# **Diferencia respecto al notebook anterior:**
# 
# En la versión anterior exigíamos que **todos** los toques estuvieran dentro del área.
# Eso era demasiado estricto — un córner puede generar una jugada peligrosa donde el primer contacto es en el borde y el remate es dentro.
# 
# **Nueva lógica (Opción A):**
# - Una posesión "entra al área" si tuvo **al menos 1 toque dentro del área grande**
# - El conteo de toques es de **todos los toques de esa posesión**, no solo los del área
# - Posesiones sin ningún toque en el área se quedan en categoría '0 toques en área'
# 
# Esto nos permite capturar jugadas donde el primer contacto es afuera pero la jugada termina dentro.

# ## 1. Imports y carga

# In[1]:


import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
from mplsoccer import Pitch

df = pd.read_parquet('corners_america.parquet')
print(f'Base cargada: {len(df):,} eventos, {len(df.columns)} columnas')


# ── Helper: guardar DataFrame como imagen PNG con encabezado verde oscuro ────
def save_table_as_image(df, title, filename):
    DARK_GREEN  = '#1B5E20'
    BG_DARK     = '#12121f'
    ROW_ODD     = '#1e1e30'
    ROW_EVEN    = '#16162a'
    EDGE_COLOR  = '#2e2e4a'

    index_label = df.index.name if df.index.name else 'Categoría'
    col_labels  = [index_label] + list(df.columns)
    cell_data   = [[str(idx)] + [str(v) for v in row] for idx, row in df.iterrows()]

    n_rows = len(cell_data)
    n_cols = len(col_labels)
    fig_w  = max(10, n_cols * 2.4)
    fig_h  = max(3,  (n_rows + 2) * 0.55)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(BG_DARK)
    ax.set_facecolor(BG_DARK)
    ax.axis('off')

    ax.set_title(title, color='white', fontsize=13, fontweight='bold', pad=14)

    tbl = ax.table(
        cellText=cell_data,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.15, 1.9)

    for j in range(n_cols):
        cell = tbl[(0, j)]
        cell.set_facecolor(DARK_GREEN)
        cell.set_text_props(color='white', fontweight='bold')
        cell.set_edgecolor(EDGE_COLOR)

    for i in range(1, n_rows + 1):
        bg = ROW_EVEN if i % 2 == 0 else ROW_ODD
        for j in range(n_cols):
            cell = tbl[(i, j)]
            cell.set_facecolor(bg)
            cell.set_text_props(color='#e0e0e0')
            cell.set_edgecolor(EDGE_COLOR)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor=BG_DARK)
    plt.close()
    print(f'Guardado: {filename}')


# ## 2. Filtrar córners a favor del América

# In[2]:


primer_evento = (
    df
    .sort_values(['match_id', 'possession', 'minute'])
    .groupby(['match_id', 'possession'])
    .first()
    .reset_index()
)

posesiones_america = primer_evento[
    primer_evento['team'] == 'América'
][['match_id', 'possession']]

corners_ataque = df.merge(posesiones_america, on=['match_id', 'possession'], how='inner')
print(f'Posesiones a favor del América: {len(posesiones_america):,}')
print(f'Eventos en esas posesiones:     {len(corners_ataque):,}')


# ## 3. Parsear coordenadas

# In[3]:


def parse_xy(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan, np.nan
    try:
        if isinstance(val, str):
            val = ast.literal_eval(val)
        return float(val[0]), float(val[1])
    except Exception:
        return np.nan, np.nan

corners_ataque = corners_ataque.copy()
coords = corners_ataque['location'].apply(parse_xy)
corners_ataque[['x', 'y']] = pd.DataFrame(coords.tolist(), index=corners_ataque.index)
print(f'Coordenadas parseadas. Válidas: {corners_ataque["x"].notna().sum():,}')


# ## 4. Nueva lógica de toques — Opción A
# 
# **Paso 1:** Identificar qué posesiones tuvieron al menos 1 toque dentro del área grande.
# 
# **Paso 2:** Para esas posesiones, contar el total de toques de toda la jugada (no solo los del área).
# 
# **Paso 3:** Las posesiones sin ningún toque en el área quedan en categoría '0 toques en área' — esas son las que el córner se despejó sin que nadie lo tocara adentro.

# In[4]:


TIPOS_TOQUE = {
    'Pass', 'Shot', 'Ball Receipt*', 'Clearance',
    'Miscontrol', 'Dribble', 'Interception', 'Block',
    'Goal Keeper', 'Foul Won', 'Foul Committed'
}

# ── Paso 1: ¿La posesión tuvo al menos 1 toque dentro del área? ──────────────
en_area = (
    (corners_ataque['x'] > 102) &
    (corners_ataque['y'] > 18) &
    (corners_ataque['y'] < 62) &
    (corners_ataque['type'].isin(TIPOS_TOQUE))
)

posesiones_con_area = (
    corners_ataque[en_area]
    .groupby(['match_id', 'possession'])
    .size()
    .reset_index(name='toques_dentro_area')  # cuántos toques hubo DENTRO del área
)

print(f'Posesiones que tuvieron al menos 1 toque dentro del área: {len(posesiones_con_area):,}')
print(f'Posesiones donde el córner se fue sin tocar el área:       {len(posesiones_america) - len(posesiones_con_area):,}')

# ── Paso 2: Contar toques TOTALES de cada posesión (sin restricción de zona) ─
toques_totales = (
    corners_ataque[corners_ataque['type'].isin(TIPOS_TOQUE)]
    .groupby(['match_id', 'possession'])
    .size()
    .reset_index(name='toques_totales')
)

print(f'\nDistribución de toques totales por posesión:')
print(toques_totales['toques_totales'].describe().round(2))


# ## 5. Construir tabla central con la nueva lógica

# In[5]:


# Detectar goles
goles = corners_ataque[
    (corners_ataque['type'] == 'Shot') &
    (corners_ataque['shot_outcome'] == 'Goal')
][['match_id', 'possession']].drop_duplicates()
goles['gol'] = True

# Unir todo
tabla_base = (
    posesiones_america
    .merge(posesiones_con_area[['match_id', 'possession', 'toques_dentro_area']],
           on=['match_id', 'possession'], how='left')
    .merge(toques_totales, on=['match_id', 'possession'], how='left')
    .merge(goles, on=['match_id', 'possession'], how='left')
)

tabla_base['toques_dentro_area'] = tabla_base['toques_dentro_area'].fillna(0).astype(int)
tabla_base['toques_totales']     = tabla_base['toques_totales'].fillna(0).astype(int)
tabla_base['gol']                = tabla_base['gol'].fillna(False)

# Categoría basada en toques DENTRO del área (para comparar con versión anterior)
# pero el conteo que usamos es toques_totales para las posesiones que sí entraron al área
tabla_base['categoria'] = tabla_base['toques_dentro_area'].apply(
    lambda n: '0 — no entró al área' if n == 0
    else      '1 toque en área'       if n == 1
    else      '2 toques en área'      if n == 2
    else      '3+ toques en área'
)

ORDEN = ['0 — no entró al área', '1 toque en área', '2 toques en área', '3+ toques en área']

print('Vista previa de la tabla:')
print(tabla_base.head(10).to_string(index=False))


# ## 6. Comparación: versión anterior vs. versión nueva
# 
# Aquí vemos cuántas posesiones **cambian de categoría** con el nuevo criterio.

# In[6]:


tabla = (
    tabla_base
    .groupby('categoria')
    .agg(
        posesiones=('gol', 'count'),
        goles=('gol', 'sum'),
        toques_totales_promedio=('toques_totales', 'mean')
    )
    .reindex(ORDEN)
)
tabla['tasa_gol_%'] = (tabla['goles'] / tabla['posesiones'] * 100).round(2)
tabla['toques_totales_promedio'] = tabla['toques_totales_promedio'].round(1)

print('=================================================================')
print('  TASA DE GOL — OPCIÓN A (toques dentro del área, jugada completa)')
print('=================================================================')
print(tabla.to_string())
print()
print(f'Total posesiones: {tabla["posesiones"].sum()}')
print(f'Total goles:      {int(tabla["goles"].sum())}')
print(f'Tasa global:      {tabla["goles"].sum()/tabla["posesiones"].sum()*100:.2f}%')
print()
print('--- REFERENCIA: versión anterior (todos los toques debían ser en área) ---')
print('0 toques:  365 posesiones, 0 goles,  0.00%')
print('1 toque:   164 posesiones, 1 gol,    0.61%')
print('2 toques:  287 posesiones, 12 goles, 4.18%')
print('3+ toques:  92 posesiones, 11 goles, 11.96%')

save_table_as_image(
    tabla,
    'Tasa de gol por categoría de toques en el área\nClub América — Opción A',
    'tabla_tasa_gol_v2.png',
)


# ## 7. Pitch map — toques de posesiones con gol vs sin gol
# 
# Ahora pintamos **todos** los toques de la posesión (no solo los del área), para ver el patrón completo de la jugada.

# In[7]:


# Posesiones con 2+ toques dentro del área
posesiones_multi = tabla_base[
    tabla_base['toques_dentro_area'] >= 2
][['match_id', 'possession', 'gol']]

# Todos los eventos de esas posesiones (con coordenadas)
todos_toques_multi = corners_ataque[
    corners_ataque['type'].isin(TIPOS_TOQUE)
].merge(posesiones_multi, on=['match_id', 'possession'], how='inner')

toques_gol    = todos_toques_multi[todos_toques_multi['gol'] == True]
toques_no_gol = todos_toques_multi[todos_toques_multi['gol'] == False]

print(f'Posesiones con 2+ toques en área: {len(posesiones_multi)}')
print(f'  Con gol:    {posesiones_multi["gol"].sum()}')
print(f'  Sin gol:    {(~posesiones_multi["gol"]).sum()}')


# In[8]:


pitch = Pitch(
    pitch_type='statsbomb',
    pitch_color='#0d1117',
    line_color='#555555',
    half=True
)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.patch.set_facecolor('#0d1117')

configs = [
    (toques_gol,    'Con gol ⚽',  'Reds',  '#e84242'),
    (toques_no_gol, 'Sin gol',     'Blues', '#1a78cf'),
]

for ax, (datos, titulo, cmap, color) in zip(axes, configs):
    pitch.draw(ax=ax)
    xs = datos['x'].dropna()
    ys = datos['y'].dropna()
    if len(xs) > 0:
        bin_stat = pitch.bin_statistic(xs, ys, statistic='count', bins=(12, 8))
        pitch.heatmap(bin_stat, ax=ax, cmap=cmap, edgecolors='#111111', alpha=0.85)
        pitch.scatter(xs, ys, ax=ax, s=15, color=color, alpha=0.4, zorder=3)
    ax.set_title(
        f'Córners América — 2+ toques en área\n{titulo}  ({len(datos):,} toques totales)',
        color='white', fontsize=12, pad=10
    )

fig.suptitle(
    'Todos los toques de la jugada (dentro y fuera del área)\nClub América | Liga MX 2021–2025',
    color='white', fontsize=14, y=1.02
)
plt.tight_layout()
plt.savefig('pitch_map_toques_v2.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print('Guardado: pitch_map_toques_v2.png')


# ## 8. OBV por categoría

# In[9]:


obv_por_posesion = (
    corners_ataque
    .groupby(['match_id', 'possession'])['obv_total_net']
    .sum()
    .reset_index(name='obv_total')
)

tabla_obv = (
    tabla_base
    .merge(obv_por_posesion, on=['match_id', 'possession'], how='left')
    .groupby('categoria')['obv_total']
    .agg(['mean', 'median', 'count'])
    .reindex(ORDEN)
    .round(4)
)
tabla_obv.columns = ['OBV promedio', 'OBV mediana', 'n']

print('=======================================')
print('  OBV PROMEDIO POR TOQUES EN EL ÁREA')
print('=======================================')
print(tabla_obv.to_string())
print()
print('OBV > 0 → posesión generó valor para América')
print('OBV < 0 → posesión fue favorable para el rival')

save_table_as_image(
    tabla_obv,
    'OBV promedio por categoría de toques en el área\nClub América — Opción A',
    'tabla_obv_v2.png',
)


# ## 9. Resumen ejecutivo

# In[10]:


print('==============================================')
print('  RESUMEN EJECUTIVO — Versión A')
print('==============================================')
print(f'\nCórners a favor del América: {tabla["posesiones"].sum()}')
print(f'Goles:                       {int(tabla["goles"].sum())}')
print(f'Tasa de gol global:          {tabla["goles"].sum()/tabla["posesiones"].sum()*100:.2f}%')
print()
print('--- Tasa de gol ---')
print(tabla[['posesiones','goles','tasa_gol_%']].to_string())
print()
print('--- OBV promedio ---')
print(tabla_obv.to_string())
print()
print('Archivos generados:')
print('  pitch_map_toques_v2.png')
print('  tabla_tasa_gol_v2.png')
print('  tabla_obv_v2.png')

