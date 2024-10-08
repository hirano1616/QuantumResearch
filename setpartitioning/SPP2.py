# できるできないのシンプルな判断のため使用する

from math import sqrt, floor, ceil
import csv
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from functools import partial
from dwave.embedding.chain_strength import uniform_torque_compensation

import dwave.inspector

def makekrlist(n,m,A,A_list):
    k = [0 for i in range(n)]
    r = [[0 for j in range(n)] for i in range(n)]
    for i in range(n):
        k[i] = len(A[i])

    for i in range(n):
        for j in range(n):
            if i != j and i < j:
                for l in range(m):
                    if A_list[l][i] == A_list[l][j] == 1:
                        r[i][j] += 1
            else:
                r[i][j] = r[j][i]
    return k, r

def makeQList(n,c,k,r,P):
    Q = [[0 for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                Q[i][j] = c[i] - (P*k[i])
            elif i<j:
                Q[i][j] = P*r[i][j]
            else:
                Q[i][j] = Q[j][i]
    return Q

f1 = 99
m_list = [i for i in range(int(f1/10), f1, int(f1/10))]
l_list = [i for i in range(10)]

timeP = 1.1
numruns = 1000 #1000
chain_prm = 1.78

with open(f"./output/setpartitioning_mip_n={f1}_out.csv", "r") as defineP:
    reader = csv.reader(defineP)
    for row in reader:
        P_list = row
    # print(P_list)

for f2 in m_list:
    f4 = int(sqrt(f2))
    for f3 in l_list:
        with open(f"./dataset/example{f1}_{f2}_{f3}_{f4}.csv", "r") as f:
            reader = csv.reader(f)
            input_list = [row for row in reader]
            m = int(input_list[0][0])
            n = int(input_list[0][1])
            c = [int(input_list[1][i]) for i in range(n+1)]
            A = [[int(i) for i in input_list[j]] for j in range(2,n+3)]
            A_list = [[0 for i in range(n+1)] for j in range(m)]
            for i in range(n+1):
                for j in A[i]:
                    A_list[j][i] = 1
            
            k,r = makekrlist(n+1,m,A,A_list)

        opvalue = int(P_list.pop(0))
        P = int(opvalue*timeP)
               
        # print(f"P={P}, numruns={numruns}, chain_prm={chain_prm}\n")
        print(f"n={f1}, m={f2}, l={f3}\n")
        with open(f"./output/output_{f1}_{f2}_{f3}_{f4}.txt", "a") as op1:
            op1.write("######################################\n")
            op1.write(f"P={P}, numruns={numruns}, chain_prm={chain_prm}\n")
            Q_list = makeQList(n+1,c,k,r,P)
            print(len(Q_list))
            Q = {(i,j): Q_list[i][j] for i in range(n+1) for j in range(n+1) if Q_list[i][j] >= 0.1 or Q_list[i][j] <= -0.1}    
            print(len(Q))
            chain_stgth = 0
            Q_list = [[abs(Q_list[j][i]) for i in range(n+1)] for j in range(n+1)]
            for i in range(n+1):
                max_value = max(Q_list[i])
                if chain_stgth <= max_value:
                    chain_stgth = max_value
            chain_stgth = int(chain_stgth*chain_prm)
            # print(f"chain_stgth = {chain_stgth}")

            chain_strength = partial(uniform_torque_compensation, prefactor=chain_prm)

            # solver = "Advantage2_prototype1.1"
            solver = "Advantage_system6.3"

            sampler=EmbeddingComposite(DWaveSampler(solver = solver))

            response = sampler.sample_qubo(Q, chain_strength=chain_strength, num_reads=numruns, label=f'set-partition P={P}', return_embedding = True, annealing_time = 50)
            # print("準備完了")

            min_cost = 1000000
            min_sol = {}
            min_check = []
            feasible_count = 0
            for s,e,o,chain_break in response.data(['sample', 'energy', 'num_occurrences', 'chain_break_fraction']):
                x_sol = {i for i in range(n+1) if s[i] > 0.5}
                calc_cost = 0
                for i in x_sol:
                    calc_cost = calc_cost + c[i]

                check_item = [0 for i in range(m)]
                flag = 0
                for i in x_sol:
                    for j in A[i]:
                        check_item[j] += 1
        
                op1.write(f"{x_sol}\n")
                op1.write(f"energy={e}, occurrence={o}, chain_break={chain_break}\n")
                op1.write(f"元問題の最適値？: {e+m*P}\n")
                op1.write(f"実際のコスト: {calc_cost}\n")
                op1.write(f"{check_item}\n")

                for i in range(len(check_item)):
                    if check_item[i] != 1:
                        op1.write("実行不可能!\n")
                        flag = 1
                        break
                if flag == 0:
                    feasible_count += 1
                    op1.write(f"実行可能! 元問題の最適値は{e+m*P}\n")
                    # print(f"実行可能! 元問題の最適値は{e+m*P}\n")
                    # print(f"解{x_sol}\n")
                    # print(f"解から得られる関数値は{calc_cost}\n")
                    if min_cost >= calc_cost:
                        min_cost = calc_cost
                        min_sol = x_sol
                        min_check = check_item

            t = response.info['timing']
            op1.write(f"{t}\n")
            op1.write(f"実際最小の値を持つ解:{min_sol} 値:{min_cost}\n")
            op1.write(f"解の分割状況{min_check}\n")
            feasible_prm = feasible_count/numruns * 100
            op1.write(f"実行可能解/試行回数 = {feasible_prm}\n")
            op1.write("######################################\n")

            
            # print(f"実際最小の値を持つ解:{min_sol} 値:{min_cost}\n")
            # print(f"解の分割状況{min_check}\n")
            if(int(min_cost) == opvalue):
                print("できる")
            else:
                print("できない")
            # print(f"実行可能解/試行回数 = {feasible_prm}\n")
            # calc_chain = response.info['embedding_context']['chain_strength']
            # print(f"calc_chain = {calc_chain}\n")
            # dwave.inspector.show(response)
