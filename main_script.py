# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 13:42:23 2016

@author: Gary
"""

import Neuron
import NetworkClass as NC
import NetworkGeneration as NG
import NeuronParameters as NP
from random import random
from Structure.NetworkStimulation import NetworkStimulation
import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go

NetGen = NG.NetworkGeneration()

Net = NC.Network()

n_excit=800
p_excit=.1
w_excitmax=150
d_excitmax=15
n_total=1000
p_inhib=.2
w_inhibmax=-200
d_inhibmax=10
TypeE = 'RS'
TypeI = 'LTS'
xmax=3
ymax=3
Pmax=.6
tau_excit=.7
tau_inhib=.7
dv_excit=.2
dv_inhib=.2

P=NP.NeuronParameters()
NeuronList ={'RS':P.set_RS,
                     'LTS':P.set_LTS,
                     'CP':P.set_CP,
                     'IB':P.set_IB,
                     'FS':P.set_FS}

NE = NeuronList[TypeE]
NI = NeuronList[TypeI]

param={'NT':Net,            #SpatialNet
       'N':Neuron.Neuron,
       'neuron_excit':NE,
       'neuron_inhib':NI,
       'xmax':xmax,
       'ymax':ymax,
       'Pmax':Pmax,
       'tau_excit':tau_excit,
       'tau_inhib':tau_inhib,
       'dv_excit':dv_excit,
       'dv_inhib':dv_inhib,
       'n_excit':n_excit,
       'p_excit':p_excit,
       'w_excitmax':w_excitmax,
       'd_excitmax':d_excitmax,
       'n_total':n_total,
       'p_inhib':p_inhib,
       'w_inhibmax':w_inhibmax,
       'd_inhibmax':d_inhibmax
       } 

NetGen.SpatialNet(**param)

X = []

maxDelay =0

for i in Net.nodes():
    for j in Net.successors(i):
        if Net[i][j]['delay'] > maxDelay:
            maxDelay = Net[i][j]['delay']

print(maxDelay)

RUNS = 60

for RUN in range(RUNS):
    input_neurons = NetworkStimulation(Net, random()*xmax, random()*ymax, .4)
    I_clamp=300
              
    T=500
    
    
    for t in range(T):
        if t>50:
            for n in input_neurons.getNodes():
                Net.node[n]['model'].clamp_input(I_clamp)
            Net.NetTimestep()
    
    #Net.print_Firings()
    X.append(Net.Firings)
    Net.Firings=np.array([[0,0,None,None]])
    Net.t = 0

PSPF = []
VON = []
PM = []


tau = 5

for I in X:
    PSPF.append([])
    VON.append([])
    for n in Net.nodes():
        PSPF[-1].append(np.zeros(T))
        VON[-1].append(np.zeros(T))
    for i,x in enumerate(I):
        if i != 0:
            t = int(x[0])
            n = int(x[1])
            for z in range(20):
                if t + z < T:
                    VON[-1][n][t+z] += np.exp(-z/tau)
                pass
            for m in Net.successors(n):
                w = Net[n][m]['weight']
                d = Net[n][m]['delay']
                if t + d < T:
                    for z in range(20):
                        if t + d + z < T:
                            PSPF[-1][m][t+d+z] += w * np.exp(-z/tau)/w_inhibmax

DATA = {}
DISTANCE = {}
NAIVE = {}
NAIVEDIST={}

for R1 in range(RUNS):
    for R2 in range(RUNS):
        if R2 > R1:
            PM = []
            NM = []
            for n in Net.nodes():
                PM.append((PSPF[R1][n] - PSPF[R2][n])**2)
                NM.append((VON[R1][n] - VON[R2][n])**2)
                pass
            try:
                DATA[R1][R2] = PM
                NAIVE[R1][R2] = NM
                DISTANCE[R1][R2] = 0
                NAIVEDIST[R1][R2] = 0
                VALUE = 0
                NVAL = 0
                for N in PM:
                    VALUE += sum(N)
                DISTANCE[R1][R2] = np.sqrt(VALUE)
                for N in NM:
                    NVAL += sum(N)
                NAIVEDIST[R1][R2] = np.sqrt(NVAL)
                
            except:
                DATA[R1] = {}
                DATA[R1][R2] = PM
                
                DISTANCE[R1] = {}
                DISTANCE[R1][R2] = 0
                
                NAIVE[R1] = {}
                NAIVE[R1][R2] = NM
                
                NAIVEDIST[R1] = {}
                NAIVEDIST[R1][R2] = 0
                
                VALUE = 0
                NVAL = 0
                
                for N in PM:
                    VALUE += sum(N)
                DISTANCE[R1][R2] = np.sqrt(VALUE)
                
                for N in NM:
                    NVAL += sum(N)
                NAIVEDIST[R1][R2] = np.sqrt(NVAL)
                
            pass

RANGE = np.linspace(0,T,T)

print("Pseudometric Distances")
A = []
B = []


for i in range(RUNS):
    for j in range(RUNS):
        if j > i:
            print(DISTANCE[i][j])
            A.append(DISTANCE[i][j])

print("Naive Distances")

for i in range(RUNS):
    for j in range(RUNS):
        if j > i:
            print(NAIVEDIST[i][j])
            B.append(NAIVEDIST[i][j])




