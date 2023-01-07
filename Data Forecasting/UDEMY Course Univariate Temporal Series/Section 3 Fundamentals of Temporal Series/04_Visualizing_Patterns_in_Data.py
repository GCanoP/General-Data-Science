"""
=====================================================================================================================================
VISUALIZING PATTERNS IN PYTHON DATA.
author : Gerardo Cano Perea.
date : May 22, 2021.
=====================================================================================================================================
"""
# Main Packages.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')

url = 'https://raw.githubusercontent.com/ecabestadistica/curso-series-temporales/master/3.%20Caracter%C3%ADsticas%20fundamentales/Python/Ti_data.csv'
df = pd.read_csv(url, index_col = 0, parse_dates = True)

# Plot.
fig, ax = plt.subplots(1, 1, figsize = (10, 6))
df.plot(ax = ax, color = 'teal', legend = None)
ax.set_ylim(17.75, 22)
ax.set_ylabel('$°C$')
ax.set_title('Internal Temperature $(T_i)$')

# Visualizing Diurnal Patterns.
# Split in days and hours.
df['date'] = df.index.normalize()
df['time'] = df.index.time.astype(str)
# Transform the dataset.
df_pivot = df.pivot(index = 'date', columns = 'time', values = 'Ti')

# Plot.
fig, ax = plt.subplots(1, 1, figsize = (10, 6))
df_pivot.T.plot(ax = ax, color = 'teal', legend = False)
ax.set_xlim([0, 47])
ax.set_ylim([17.75, 22])
ax.set_ylabel('$°C$')
ax.set_title('Internal Temperature $(T_i)$')

fig, ax = plt.subplots(1, 1, figsize = (10, 6))
df_pivot.T.plot(ax = ax, color = 'teal', alpha = 0.1, legend = False)
ax.set_xlim([0, 47])
ax.set_ylim([17.75, 22])
ax.set_ylabel('$°C$')
ax.set_title('Internal Temperature $(T_i)$')

# Heatmap.
import seaborn as sns

ax = sns.heatmap(df_pivot, cmap = "YlGnBu")

# Check what happens after 20:00 hrs.
df_roll = pd.DataFrame(index = df_pivot.index,
                       columns = np.roll(df_pivot.columns, -36),
                       data = np.roll(df_pivot.values, -36))
fig, ax = plt.subplots(1, 1, figsize = (10, 6))
df_roll.T.plot(ax = ax, color = 'teal', alpha = 0.1, legend = False)
ax.set_xlim([0, 47])
ax.set_ylim([17.75, 22])
ax.set_ylabel('$°C$')
ax.set_title('Internal Temperature $(T_i)$')

ax = sns.heatmap(df_roll, cmap = "YlGnBu")

day_colors = {'Monday': 'C0',
              'Tuesday': 'C1',
              'Wednesday': 'C1',
              'Thursday': 'C1',
              'Friday': 'C1',
              'Saturday': 'C2',
              'Sunday': 'C3'}

# Plot
fig, ax = plt.subplots(1, 1, figsize = (10, 6))

for day_name, color in day_colors.items():
    # NOTE: we are plotting the transposed DataFrame
    df_roll[df_roll.index.day_name() == day_name].T.plot(ax = ax, color = color, alpha = 0.1, legend = None)
ax.set_xlim([0, 47])
ax.set_ylim(17.75, 22)

ax.axvline(30, alpha = 0.25, color = 'black', linestyle = '--')
ax.axvline(40, alpha = 0.25, color = 'black', linestyle = '--')
ax.axvline(42, alpha = 0.25, color = 'black', linestyle = '--')

ax.set_ylabel('$˚C$')
ax.set_title('Internal Temperature $(T_i)$');

from matplotlib.lines import Line2D

custom_lines = [Line2D([0], [0], color = 'C0', lw = 2),
                Line2D([0], [0], color = 'C1', lw = 2),
                Line2D([0], [0], color = 'C2', lw = 2),
                Line2D([0], [0], color = 'C3', lw = 2)]

ax.legend(custom_lines, ['Mon', 'Tue-Fri', 'Sat', 'Sun'])
