# できるできないのシンプルな判断のため使用する

from math import sqrt, floor, ceil
import csv
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from functools import partial
from dwave.embedding.chain_strength import uniform_torque_compensation
import time
import dwave.inspector

# timeP = 1.1
numruns = 1000 #1000
chain_prm = 1.75

Q = dict()
with open(f"qubo.csv", "r") as Qreader:
    reader = csv.reader(Qreader)
    i, j = 0, 0
    for l in reader:
        j = 0
        if i == 0:
            n = len(l)
        for data in l:
            Q[i,j] = float(data)
            j += 1
        i += 1

Q_abs = {(i,j): abs(Q[i,j]) for i,j in Q}
chain_stgth = max(Q_abs.values())
chain_stgth = int(chain_stgth*chain_prm)

chain_strength = partial(uniform_torque_compensation, prefactor=chain_prm)


solver = "Advantage_system6.3"
sampler=EmbeddingComposite(DWaveSampler(solver = solver))
time1 = time.time()
response = sampler.sample_qubo(Q, chain_strength=chain_strength, num_reads=numruns, label='test', return_embedding = True, annealing_time = 50)
print(f"solve time (s) = {time.time() - time1}")
for s,e,o,chain_break in response.data(['sample', 'energy', 'num_occurrences', 'chain_break_fraction']):
    x_sol = {i for i in range(n) if s[i] > 0.5}
    print(f"energy={e}, occurrence={o}, chain_break={chain_break}\n")
    print(f"sol = {x_sol}")
    break

t = response.info['timing']
print(f"t = {t}")

dwave.inspector.show(response)

    
