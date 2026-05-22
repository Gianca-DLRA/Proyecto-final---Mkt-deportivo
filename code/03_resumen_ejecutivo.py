#!/usr/bin/env python
# coding: utf-8

# # 03 — Resumen Ejecutivo Visual
# ### ISAC Set-Piece Challenge — Club América
# 
# Genera una sola figura con 4 paneles que cuenta la historia completa:
# 
# 1. **Tasa de gol por toques en área** — el hallazgo principal
# 2. **Tasa de gol por tipo de córner** — qué tipo funciona mejor
# 3. **OBV por tipo de córner** — valor más allá del gol
# 4. **Pitch map** — dónde caen los centros de córners cortos vs inswingers
# 
# **Requiere:** `corners_america.parquet`

# ## 1. Imports, carga y preparación de datos

# In[1]:


import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from mplsoccer import Pitch

# ── Paleta y estilo ───────────────────────────────────────────────────────────
BG        = '#0d1117'
BG_PANEL  = '#161b22'
AMARILLO  = '#f5c518'   # América
AZUL      = '#1a78cf'
ROJO      = '#e84242'
GRIS      = '#6a6a6a'
NARANJA   = '#e8a838'
BLANCO    = '#e6edf3'
GRIS_GRID = '#21262d'

COLORES_TIPO = {
    'Inswinging':  ROJO,
    'Outswinging': AZUL,
    'Corto':       AMARILLO,
    'Directo':     GRIS
}
ORDEN_TIPO = ['Inswinging', 'Outswinging', 'Corto', 'Directo']
ORDEN_CAT  = ['0 toques', '1 toque', '2 toques', '3+ toques']
COLORES_CAT = ['#3a3a3a', AZUL, NARANJA, ROJO]

plt.rcParams.update({
    'text.color': BLANCO,
    'axes.labelcolor': BLANCO,
    'xtick.color': BLANCO,
    'ytick.color': BLANCO,
    'font.family': 'DejaVu Sans',
})

# ── Carga ─────────────────────────────────────────────────────────────────────
df = pd.read_parquet('corners_america.parquet')
print(f'Base cargada: {len(df):,} eventos')


# ## 2. Preparar todos los datos

# In[2]:


# ── Helpers ───────────────────────────────────────────────────────────────────
def parse_xy(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan, np.nan
    try:
        if isinstance(val, str):
            val = ast.literal_eval(val)
        return float(val[0]), float(val[1])
    except Exception:
        return np.nan, np.nan

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

TIPOS_TOQUE = {
    'Pass', 'Shot', 'Ball Receipt*', 'Clearance',
    'Miscontrol', 'Dribble', 'Interception', 'Block',
    'Goal Keeper', 'Foul Won', 'Foul Committed'
}

# ── Filtrar córners del América ───────────────────────────────────────────────
primer_evento = (
    df.sort_values(['match_id', 'possession', 'minute'])
    .groupby(['match_id', 'possession']).first().reset_index()
)
pos_america = primer_evento[primer_evento['team'] == 'América'][['match_id', 'possession']]
ca = df.merge(pos_america, on=['match_id', 'possession'], how='inner').copy()

# ── Coordenadas ───────────────────────────────────────────────────────────────
coords = ca['location'].apply(parse_xy)
ca[['x', 'y']] = pd.DataFrame(coords.tolist(), index=ca.index)

# ── Tipo de córner ────────────────────────────────────────────────────────────
saques = (
    ca[ca['pass_type'] == 'Corner']
    .sort_values(['match_id', 'possession', 'minute'])
    .groupby(['match_id', 'possession']).first().reset_index()
)
saques['tipo_corner'] = saques.apply(clasificar_corner, axis=1)

end_coords = saques['pass_end_location'].apply(parse_xy)
saques[['end_x', 'end_y']] = pd.DataFrame(end_coords.tolist(), index=saques.index)

# ── Toques en área ────────────────────────────────────────────────────────────
en_area = (
    (ca['x'] > 102) & (ca['y'] > 18) & (ca['y'] < 62) &
    (ca['type'].isin(TIPOS_TOQUE))
)
toques_area = (
    ca[en_area].groupby(['match_id', 'possession'])
    .size().reset_index(name='toques_en_area')
)

# ── Goles y OBV ──────────────────────────────────────────────────────────────
goles = ca[(ca['type'] == 'Shot') & (ca['shot_outcome'] == 'Goal')] \
    [['match_id', 'possession']].drop_duplicates()
goles['gol'] = True

obv = ca.groupby(['match_id', 'possession'])['obv_total_net'].sum().reset_index(name='obv_total')

# ── Tabla base ────────────────────────────────────────────────────────────────
tb = (
    saques[['match_id', 'possession', 'tipo_corner']]
    .merge(toques_area, on=['match_id', 'possession'], how='left')
    .merge(goles,       on=['match_id', 'possession'], how='left')
    .merge(obv,         on=['match_id', 'possession'], how='left')
)
tb['toques_en_area'] = tb['toques_en_area'].fillna(0).astype(int)
tb['gol']            = tb['gol'].fillna(False)
tb['categoria']      = tb['toques_en_area'].apply(
    lambda n: '0 toques' if n == 0 else '1 toque' if n == 1
    else '2 toques' if n == 2 else '3+ toques'
)

# ── Tabla toques (para panel 1) ───────────────────────────────────────────────
t_toques = (
    tb.groupby('categoria')
    .agg(posesiones=('gol','count'), goles=('gol','sum'))
    .reindex(ORDEN_CAT)
)
t_toques['tasa'] = (t_toques['goles'] / t_toques['posesiones'] * 100).round(2)

# ── Tabla tipo (para paneles 2 y 3) ──────────────────────────────────────────
t_tipo = (
    tb.groupby('tipo_corner')
    .agg(posesiones=('gol','count'), goles=('gol','sum'), obv=('obv_total','mean'))
    .reindex(ORDEN_TIPO)
)
t_tipo['tasa'] = (t_tipo['goles'] / t_tipo['posesiones'] * 100).round(2)

print('Datos preparados.')
print(f'Posesiones del América: {len(tb)}')
print(f'Goles totales: {int(tb["gol"].sum())}')


# ## 3. Figura final — 4 paneles

# In[3]:


fig = plt.figure(figsize=(20, 14), facecolor=BG)
gs  = gridspec.GridSpec(
    2, 3,
    figure=fig,
    left=0.06, right=0.97,
    top=0.88,  bottom=0.08,
    hspace=0.45, wspace=0.35
)

ax1 = fig.add_subplot(gs[0, 0])      # tasa de gol por toques
ax2 = fig.add_subplot(gs[0, 1])      # tasa de gol por tipo
ax3 = fig.add_subplot(gs[0, 2])      # OBV por tipo
ax4 = fig.add_subplot(gs[1, :])      # pitch map (fila entera)

def estilizar(ax, titulo):
    ax.set_facecolor(BG_PANEL)
    ax.spines[['top','right','left']].set_visible(False)
    ax.spines['bottom'].set_color('#30363d')
    ax.tick_params(colors=BLANCO, labelsize=10)
    ax.yaxis.grid(True, color=GRIS_GRID, zorder=0, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(titulo, color=BLANCO, fontsize=12, pad=10, fontweight='bold')

# ─────────────────────────────────────────────────────────────────────────────
# PANEL 1 — Tasa de gol por toques en área
# ─────────────────────────────────────────────────────────────────────────────
estilizar(ax1, 'Tasa de gol según toques en el área')

barras1 = ax1.bar(
    ORDEN_CAT, t_toques['tasa'],
    color=COLORES_CAT, edgecolor='#ffffff11', zorder=3, width=0.6
)
for b, (cat, row) in zip(barras1, t_toques.iterrows()):
    ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.15,
             f"{row['tasa']:.1f}%\nn={int(row['posesiones'])}",
             ha='center', va='bottom', color=BLANCO, fontsize=9)
ax1.set_ylabel('Tasa de gol (%)', color=BLANCO, fontsize=10)
ax1.tick_params(axis='x', labelsize=9)

# Anotación clave
ax1.annotate('19x más\npeligroso', xy=(3, t_toques.loc['3+ toques','tasa']),
             xytext=(2.3, t_toques.loc['3+ toques','tasa'] * 0.6),
             color=ROJO, fontsize=9, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=ROJO, lw=1.5))

# ─────────────────────────────────────────────────────────────────────────────
# PANEL 2 — Tasa de gol por tipo de córner
# ─────────────────────────────────────────────────────────────────────────────
estilizar(ax2, 'Tasa de gol por tipo de córner')

colores2 = [COLORES_TIPO[t] for t in ORDEN_TIPO]
barras2  = ax2.bar(
    ORDEN_TIPO, t_tipo['tasa'],
    color=colores2, edgecolor='#ffffff11', zorder=3, width=0.6
)
for b, (tipo, row) in zip(barras2, t_tipo.iterrows()):
    ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.05,
             f"{row['tasa']:.1f}%\nn={int(row['posesiones'])}",
             ha='center', va='bottom', color=BLANCO, fontsize=9)
ax2.set_ylabel('Tasa de gol (%)', color=BLANCO, fontsize=10)
ax2.tick_params(axis='x', labelsize=9)

# ─────────────────────────────────────────────────────────────────────────────
# PANEL 3 — OBV promedio por tipo de córner
# ─────────────────────────────────────────────────────────────────────────────
estilizar(ax3, 'OBV promedio por tipo de córner')

obv_vals = t_tipo['obv'].values
colores3 = [AMARILLO if v > 0 else GRIS for v in obv_vals]
barras3  = ax3.bar(
    ORDEN_TIPO, obv_vals,
    color=colores3, edgecolor='#ffffff11', zorder=3, width=0.6
)
ax3.axhline(0, color='#555', linewidth=1, zorder=2)
for b, v in zip(barras3, obv_vals):
    offset = 0.0008 if v >= 0 else -0.0015
    ax3.text(b.get_x() + b.get_width()/2, v + offset,
             f'{v:+.4f}', ha='center', va='bottom', color=BLANCO, fontsize=9)
ax3.set_ylabel('OBV promedio', color=BLANCO, fontsize=10)
ax3.tick_params(axis='x', labelsize=9)

# Anotación
ax3.text(0.5, 0.92, '↑ valor para América\n↓ valor para el rival',
         transform=ax3.transAxes, ha='center', color='#888', fontsize=8)

# ─────────────────────────────────────────────────────────────────────────────
# PANEL 4 — Pitch map: dónde cae el córner corto vs inswinging
# ─────────────────────────────────────────────────────────────────────────────
pitch = Pitch(
    pitch_type='statsbomb', pitch_color=BG_PANEL,
    line_color='#444', half=True
)

# Dos pitch maps dentro de ax4
ax4.remove()
ax_p1 = fig.add_subplot(gs[1, 0:2])
ax_p2 = fig.add_subplot(gs[1, 2])

# Inswinging — dónde cae el centro
pitch.draw(ax=ax_p1)
ins = saques[saques['tipo_corner'] == 'Inswinging']
xs_i, ys_i = ins['end_x'].dropna(), ins['end_y'].dropna()
if len(xs_i) > 2:
    bs = pitch.bin_statistic(xs_i, ys_i, statistic='count', bins=(12, 8))
    pitch.heatmap(bs, ax=ax_p1, cmap='Reds', edgecolors='#111', alpha=0.85)
    pitch.scatter(xs_i, ys_i, ax=ax_p1, s=15, color=ROJO, alpha=0.35, zorder=3)
ax_p1.set_title(
    f'Inswinging — dónde cae el centro  |  tasa gol: {t_tipo.loc["Inswinging","tasa"]:.1f}%',
    color=BLANCO, fontsize=11, pad=8
)

# Corto — dónde cae el pase corto
pitch.draw(ax=ax_p2)
cor = saques[saques['tipo_corner'] == 'Corto']
xs_c, ys_c = cor['end_x'].dropna(), cor['end_y'].dropna()
if len(xs_c) > 2:
    bs2 = pitch.bin_statistic(xs_c, ys_c, statistic='count', bins=(12, 8))
    pitch.heatmap(bs2, ax=ax_p2, cmap='YlOrBr', edgecolors='#111', alpha=0.85)
    pitch.scatter(xs_c, ys_c, ax=ax_p2, s=15, color=AMARILLO, alpha=0.35, zorder=3)
ax_p2.set_title(
    f'Corto — dónde cae el pase  |  tasa gol: {t_tipo.loc["Corto","tasa"]:.1f}%',
    color=BLANCO, fontsize=11, pad=8
)

for ax in [ax_p1, ax_p2]:
    ax.set_facecolor(BG_PANEL)

# ─────────────────────────────────────────────────────────────────────────────
# Título principal y subtítulo
# ─────────────────────────────────────────────────────────────────────────────
fig.text(0.5, 0.95,
         'Análisis de Córners — Club América',
         ha='center', color=AMARILLO, fontsize=20, fontweight='bold')
fig.text(0.5, 0.915,
         'Liga MX 2021–2025  |  908 posesiones de córner  |  24 goles  |  Datos: StatsBomb 360',
         ha='center', color='#8b949e', fontsize=11)

plt.savefig('resumen_ejecutivo.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()
print('Guardado: resumen_ejecutivo.png')


# ## 4. Imprimir hallazgos clave

# In[4]:


tasa_global = tb['gol'].sum() / len(tb) * 100
tasa_3mas   = t_toques.loc['3+ toques', 'tasa']
tasa_corto  = t_tipo.loc['Corto', 'tasa']
tasa_out    = t_tipo.loc['Outswinging', 'tasa']
pct_corto   = len(tb[tb['tipo_corner'] == 'Corto']) / len(tb) * 100

print('==================================================')
print('  HALLAZGOS CLAVE — Para la propuesta del viernes')
print('==================================================')
print(f'''
1. El {tb["categoria"].value_counts(normalize=True).get("0 toques", 0)*100:.0f}% de los córners del América
   no genera ningún toque dentro del área. Tasa de gol: 0%.

2. Cuando la jugada genera 3+ toques en el área,
   la tasa de gol sube a {tasa_3mas:.1f}% — {tasa_3mas/tasa_global:.0f}x la tasa global ({tasa_global:.1f}%).

3. El córner corto es el más efectivo:
   - Tasa de gol: {tasa_corto:.1f}% (vs {tasa_out:.1f}% del outswinging)
   - OBV positivo: único tipo que genera valor neto para América
   - Solo representa el {pct_corto:.0f}% de los córners del América

4. Recomendación táctica:
   Aumentar la proporción de córners cortos y diseñar
   jugadas que generen segundas y terceras pelotas dentro
   del área, no optimizar solo el primer centro.
''')

