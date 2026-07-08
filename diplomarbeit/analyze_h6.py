import pandas as pd
df = pd.read_csv('results_final.csv')
h6 = df[df['Time']==6]
print("HOUR 6 CONDITIONS:")
print(h6[['Node', 'Load_MW', 'RES_MW', 'RES_Curtail_MW', 'Conv_MW', 'Total_BESS_Power_MW', 'Price_EUR']].to_string())
