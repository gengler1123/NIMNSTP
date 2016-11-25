# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 11:28:13 2016

@author: Gary
"""

import networkx as nx
import numpy as np
from numpy.random import exponential, randint

def FiringTime(rate):
    hold = exponential(rate)
    while (hold < 5):
        hold = exponential(rate)
    if hold%1 < 0.25:
        return round(hold)
    elif hold%1 < .5:
        return round(hold)+.25
    elif hold%1 < .75:
        return round(hold) - .5
    elif hold%1 < 1:
        return round(hold) - .25

def GetIndex(value):
    return int(value/0.125)

def PrintSpikeTrainYard(STY):
    for ST in STY:
        print(ST)

TIME = 250
CUTOFF = 60

Net = nx.DiGraph()

Net.add_nodes_from(range(5))

Net.add_weighted_edges_from([(0,1,100.0),(0,2,200.0),(1,2,-300),(1,3,-400),(2,3,300),(2,4,350.0),(3,4,-350.0),(3,0,-250.0),(4,0,100),(4,1,300)])

normalizer = 0
for e in Net.edges():
    if abs(Net[e[0]][e[1]]['weight']) > normalizer:
        normalizer = abs(Net[e[0]][e[1]]['weight'])

Net[0][1]['delay'] = 10
Net[0][2]['delay'] = 5

Net[1][2]['delay'] = 7
Net[1][3]['delay'] = 8

Net[2][3]['delay'] = 10
Net[2][4]['delay'] = 15

Net[3][4]['delay'] = 3
Net[3][0]['delay'] = 4

Net[4][0]['delay'] = 5
Net[4][1]['delay'] = 6

FiringRates = [1000/randint(45,70) for i in range(5)]

STY1 = []
STY2 = []

FirstSpike = 1000
LastSpike = 0


for i in range(5):
    STY1.append([])
    STY1[-1].append(FiringTime(FiringRates[i]))
    if (STY1[-1][-1] < FirstSpike):
        FirstSpike = STY1[-1][-1]
    while (STY1[-1][-1] < TIME):
        STY1[-1].append(FiringTime(FiringRates[i]) + STY1[-1][-1])
    STY1[-1].pop()
    if (STY1[-1][-1] > LastSpike):
        LastSpike = STY1[-1][-1]

for i,x in enumerate(STY1):
    print("Number of Spikes for Neuron " + str(i) + " is: " + str(len(x)))

for i in range(5):
    STY2.append([])
    STY2[-1].append(FiringTime(FiringRates[i]))
    if (STY2[-1][-1] < FirstSpike):
        FirstSpike = STY2[-1][-1]
    while (STY2[-1][-1] < TIME):
        STY2[-1].append(FiringTime(FiringRates[i]) + STY2[-1][-1])
    STY2[-1].pop()
    if (STY2[-1][-1] > LastSpike):
        LastSpike = STY2[-1][-1]

for i,x in enumerate(STY2):
    print("Number of Spikes for Neuron " + str(i) + " is: " + str(len(x)))

print("\n")

print("Spike TrainYard One")
PrintSpikeTrainYard(STY1)


print("\nSpike TrainYard Two")
PrintSpikeTrainYard(STY2)


'''
Calculate Kernels
'''


Times = np.arange(0,LastSpike+30, 0.125)

Kernels1 = []
Kernels2 = []

for i in range(5):
    Kernels1.append(np.zeros(len(Times)))
    Kernels2.append(np.zeros(len(Times)))



for i in range(5):
    for ft in STY1[i]:
        for m in Net.successors(i):
            weight = Net[i][m]['weight']/normalizer
            for dt in range(GetIndex(CUTOFF)):
                if (GetIndex(ft) + GetIndex(Net[i][m]['delay']) + dt < len(Times)):
                    Kernels1[m][GetIndex(ft) + GetIndex(Net[i][m]['delay']) + dt] += weight * np.exp(-dt*0.125/15) 


for i in range(5):
    for ft in STY2[i]:
        for m in Net.successors(i):
            weight = Net[i][m]['weight']/normalizer
            for dt in range(GetIndex(CUTOFF)):
                if GetIndex(ft) + GetIndex(Net[i][m]['delay']) + dt < len(Times):
                    Kernels2[m][GetIndex(ft) + GetIndex(Net[i][m]['delay']) + dt] += weight * np.exp(-dt*0.125/15) 

Kernels1 = np.array(Kernels1)
Kernels2 = np.array(Kernels2)

DiffSqu = (Kernels1 - Kernels2) ** 2

NISTYDIST = np.sqrt(sum(sum(DiffSqu * 0.125/15)))

print(NISTYDIST)


vrKernel1 = []
vrKernel2 = []


for i in range(5):
    vrKernel1.append(np.zeros(len(Times)))
    for ft in STY1[i]:
        for dt in range(GetIndex(CUTOFF)):
            if (GetIndex(ft) + dt < len(Times)):
                vrKernel1[i][GetIndex(ft) + dt] += np.exp(-dt*0.125/15)
    vrKernel2.append(np.zeros(len(Times)))
    for ft in STY2[i]:
        for dt in range(GetIndex(CUTOFF)):
            if (GetIndex(ft) + dt < len(Times)):
                vrKernel2[i][GetIndex(ft) + dt] += np.exp(-dt*0.125/15)

vrKernel1 = np.array(vrKernel1)
vrKernel2 = np.array(vrKernel2)


vrDiffSqu = (vrKernel1 - vrKernel2) ** 2


vanRosDist = np.sqrt(sum(sum(vrDiffSqu * 0.125/15)))

print(vanRosDist)



'''
Write Data
'''

f = open('STY1.csv','w')

for i in range(5):
    for ft in STY1[i]:
        f.write(str(ft) +"," + str(i) +"\n")
        
f.close()

f = open('STY2.csv','w')

for i in range(5):
    for ft in STY2[i]:
        f.write(str(ft) + "," + str(i) + "\n")

f.close()


f = open('nistyKernel1.csv', 'w')
g = open('nistyKernel2.csv', 'w')

d = open('vanRosKernel1.csv','w')
e = open('vanRosKernel2.csv','w')

f.write("t, 1, 2, 3, 4, 5\n")
g.write("t, 1, 2, 3, 4, 5\n")
d.write("t, 1, 2, 3, 4, 5\n")
e.write("t, 1, 2, 3, 4, 5\n")

for i in range(len(Times)):
    f.write(str(Times[i]))
    g.write(str(Times[i]))
    d.write(str(Times[i]))
    e.write(str(Times[i]))
    for j in range(5):
        f.write(","+str(Kernels1[j][i]))
        g.write(","+str(Kernels2[j][i]))
        d.write(","+str(vrKernel1[j][i]))
        e.write(","+str(vrKernel2[j][i]))
    f.write("\n")
    g.write("\n")
    d.write("\n")
    e.write("\n")

f.close()
g.close()
d.close()
e.close()











