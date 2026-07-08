
import pandas as pd
import numpy as np

file_path = r'c:\vscode\diplomarbeit\AGPT_2025-09-10T22_00_00Z_2025-09-11T22_00_00Z_60M_de_2026-01-28T20_29_10Z.csv'

try:
    df = pd.read_csv(file_path, sep=';', decimal=',', encoding='utf-8')
    df = df.iloc[:24] 
    
    cols_data = df.columns[2:]
    
    # Calculate Load (Sum of all Generation columns is roughly the Load + Export/Import, usually Load is covered by Gen)
    # The columns are generation types. Summing them gives Total Generation which equals Total Load in a balanced system (ignoring cross-border).
    total_gen = df[cols_data].sum(axis=1)
    
    wind_curve = df['Wind [MW]']
    solar_curve = df['Solar [MW]']
    
    max_load = total_gen.max()
    max_wind = wind_curve.max()
    max_solar = solar_curve.max()
    
    print(f"Max System Load (approx): {max_load:.2f} MW")
    print(f"Max Wind: {max_wind:.2f} MW")
    print(f"Max Solar: {max_solar:.2f} MW")
    
    print(f"Ratio Wind/Load: {max_wind/max_load:.3f}")
    print(f"Ratio Solar/Load: {max_solar/max_load:.3f}")

except Exception as e:
    print(f"Error: {e}")
