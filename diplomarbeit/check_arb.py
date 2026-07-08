import pandas as pd

df = pd.read_csv('results_final.csv')
i = 'I1'

df['Arb_Hour'] = df[f'{i}_Dis_MW'] * df['Price_EUR'] - df[f'{i}_Ch_MW'] * df['Price_EUR']
loss_hours = df[df['Arb_Hour'] < -10].sort_values('Arb_Hour')
profit_hours = df[df['Arb_Hour'] > 10].sort_values('Arb_Hour', ascending=False)

print(f"Total Arbitrage sum for {i}: {df['Arb_Hour'].sum()}")

print("\n--- GRÖßTE VERLUST-STUNDEN ---")
print(loss_hours[['Time', 'Node', 'Arb_Hour', 'Price_EUR', f'{i}_Ch_MW', f'{i}_Dis_MW', 'Total_BESS_Power_MW', 'Load_MW', 'Conv_MW']].head(10).to_string(index=False))

print("\n--- GRÖßTE GEWINN-STUNDEN ---")
print(profit_hours[['Time', 'Node', 'Arb_Hour', 'Price_EUR', f'{i}_Ch_MW', f'{i}_Dis_MW', 'Total_BESS_Power_MW', 'Load_MW', 'Conv_MW']].head(10).to_string(index=False))
