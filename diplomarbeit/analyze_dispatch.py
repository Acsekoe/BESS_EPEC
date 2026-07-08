import pandas as pd

try:
    df = pd.read_csv('results_final.csv')
    
    # Analyze investor I1 operations
    print("=== Höchste Lade-Stunden I1 ===")
    top_charge = df.nlargest(10, 'I1_Ch_MW')[['Time', 'Node', 'I1_Ch_MW', 'I1_Dis_MW', 'Price_EUR', 'Load_MW', 'RES_MW', 'Conv_MW']]
    print(top_charge.to_string(index=False))
    
    print("\n=== Höchste Entlade-Stunden I1 ===")
    top_dis = df.nlargest(10, 'I1_Dis_MW')[['Time', 'Node', 'I1_Ch_MW', 'I1_Dis_MW', 'Price_EUR', 'Load_MW', 'RES_MW', 'Conv_MW']]
    print(top_dis.to_string(index=False))

    print("\n=== Stunden mit dem höchsten Spotpreis ===")
    top_price = df.nlargest(5, 'Price_EUR')[['Time', 'Node', 'Price_EUR', 'Load_MW', 'Conv_MW', 'I1_Ch_MW', 'I1_Dis_MW', 'Total_BESS_Power_MW']]
    print(top_price.to_string(index=False))
    
    print("\n=== Market Balance in Hour 18 (Typischer Peak) ===")
    h18 = df[df['Time'] == 18]
    print(h18[['Node', 'Load_MW', 'RES_MW', 'Conv_MW', 'Total_BESS_Power_MW', 'Price_EUR']].to_string(index=False))
except Exception as e:
    print(f"Error: {e}")
