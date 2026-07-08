
import pandas as pd
import numpy as np

file_path = r'c:\vscode\diplomarbeit\AGPT_2025-09-10T22_00_00Z_2025-09-11T22_00_00Z_60M_de_2026-01-28T20_29_10Z.csv'

try:
    df = pd.read_csv(file_path, sep=';', decimal=',', encoding='utf-8')
    # Filter for the relevant 24 hours (The file seems to have 25 or 26 rows based on previous `view_file`)
    # The first row is 00:00 to 01:00. We need 24 rows.
    df = df.iloc[:24] 
    
    # Columns
    # Wind is col 2
    # Solar is col 3
    # Load Proxy is sum of all numeric columns starting from col 2
    
    cols_data = df.columns[2:]
    # Clean column names (remove [MW])
    
    # Convert to numeric
    for c in cols_data:
        # thousands separator might be '.' if decimal is ',' -> handled by read_csv decimal=',' usually, but typical German CSV might not use thousands sep or iterate.
        # Check raw strings first?
        pass

    # Calculate Load (Sum of all Generation)
    load_curve = df[cols_data].sum(axis=1)
    
    wind_curve = df['Wind [MW]']
    solar_curve = df['Solar [MW]']
    
    def norm(s):
        mx = s.max()
        if mx == 0: return s
        return s / mx
        
    p_load = norm(load_curve).round(3).tolist()
    p_wind = norm(wind_curve).round(3).tolist()
    p_solar = norm(solar_curve).round(3).tolist()
    
    print("daily_pattern = " + str(p_load))
    print("wind_pattern = " + str(p_wind))
    print("pv_pattern = " + str(p_solar))

except Exception as e:
    print(f"Error: {e}")
