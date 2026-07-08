import pandas as pd
from bess_epec_model import NODES, TIME, RES_PROFILE, LOAD_PROFILE

res = sum(RES_PROFILE[n,t] for n in NODES for t in TIME)
load = sum(LOAD_PROFILE[n,t] for n in NODES for t in TIME)
print(f'Total RES: {res}, Total LOAD: {load}')

for t in TIME:
    res_t = sum(RES_PROFILE[n,t] for n in NODES)
    load_t = sum(LOAD_PROFILE[n,t] for n in NODES)
    if res_t > load_t:
        print(f'Hour {t}: RES ({res_t}) > LOAD ({load_t}) - EXCESS: {res_t - load_t}')
