import numpy as np
import random
from scipy.stats import loguniform
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
# Source - https://stackoverflow.com/a/1482316
# Posted by Mark Rushakoff, modified by community. See post 'Timeline' for change history
# Retrieved 2026-04-30, License - CC BY-SA 4.0

from itertools import chain, combinations

N=7 #number of people
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
def str_keys(list_of_dicts):
    return [{','.join(map(str, k)): v for k, v in d.items()} for d in list_of_dicts]

P=[p for p in powerset(range(N)) if len(p)>0]# trying out singletons??

delta = 0.5

G={}

for n in tqdm(range(N)):
    G[n]=[p for p in P if n in p]


opinions=[{} for n in range(N)]
gifts=[{} for n in range(N)]
for n in tqdm(range(N)):
    for g in G[n]:
        opinions[n][g]=1/len(G[n])
        gifts[n][g]=0.0
T=10000
print("STARTING")
for t in tqdm(range(T)):
    n = random.choice(range(N))
    ops_n = opinions[n]
    g = random.choices(G[n],weights=[ops_n[g] for g in G[n]],k=1)[0]
    epsilon=random.uniform(-1,1)
    gifts[n][g]+=epsilon
    opinions[n][g]*=delta ** epsilon
    summation=sum(opinions[n][h] for h in G[n])        
    for h in G[n]:
        opinions[n][h]/=summation
    for m in g:
        if m!=n:
            opinions[m][g]*=delta ** -epsilon
            summation=sum(opinions[m][h] for h in G[m])
            for h in G[m]:
                opinions[m][h]/=summation        

    if t%10 == 0:
        with open(f'data_{t}.json', 'w') as f:
            packet = {'t':t,'op':str_keys(opinions),'gf':str_keys(gifts)}
            json.dump(packet, f)


