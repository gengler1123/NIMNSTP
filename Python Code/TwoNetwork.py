# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 10:57:16 2016

@author: Gary
"""

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

Net0 = NC.Network()
Net1 = NC.Network()
Net2 = NC.Network()

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

param0={'NT':Net0,            #SpatialNet
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

param1={'NT':Net1,            #SpatialNet
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

param2={'NT':Net2,            #SpatialNet
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

NetGen.SpatialNet(**param0)
NetGen.SpatialNet(**param1)
NetGen.SpatialNet(**param2)

X0 = []
X1 = []
X2 = []

RUNS = 20

for RUN in range(RUNS):
    input_neurons0 = NetworkStimulation(Net0, random()*xmax, random()*ymax, .4)
    input_neurons1 = NetworkStimulation(Net1, random()*xmax, random()*ymax, .4)
    input_neurons2 = NetworkStimulation(Net2, random()*xmax, random()*ymax, .4)
    I_clamp=300
              
    T=500
    
    
    for t in range(T):
        if t>50:
            for n in input_neurons0.getNodes():
                Net0.node[n]['model'].clamp_input(I_clamp)
            Net0.NetTimestep()
    
    #Net.print_Firings()
    X0.append(Net0.Firings)
    Net0.Firings=np.array([[0,0,None,None]])
    Net0.t = 0
    
    for t in range(T):
        if t>50:
            for n in input_neurons1.getNodes():
                Net1.node[n]['model'].clamp_input(I_clamp)
            Net1.NetTimestep()
    
    #Net.print_Firings()
    X1.append(Net1.Firings)
    Net1.Firings=np.array([[0,0,None,None]])
    Net1.t = 0
    
    for t in range(T):
        if t>50:
            for n in input_neurons2.getNodes():
                Net2.node[n]['model'].clamp_input(I_clamp)
            Net2.NetTimestep()
    
    #Net.print_Firings()
    X2.append(Net2.Firings)
    Net2.Firings=np.array([[0,0,None,None]])
    Net2.t = 0

PSPF0 = []
VON0 = []
PM0 = []

PSPF1 = []
VON1 = []
PM1 = []

PSPF2 = []
VON2 = []
PM2 = []

tau = 5

for I in X0:
    PSPF0.append([])
    VON0.append([])
    for n in Net0.nodes():
        PSPF0[-1].append(np.zeros(T))
        VON0[-1].append(np.zeros(T))
    for i,x in enumerate(I):
        if i != 0:
            t = int(x[0])
            n = int(x[1])
            for z in range(20):
                if t + z < T:
                    VON0[-1][n][t+z] += np.exp(-z/tau)
                pass
            for m in Net0.successors(n):
                w = Net0[n][m]['weight']
                d = Net0[n][m]['delay']
                if t + d < T:
                    for z in range(20):
                        if t + d + z < T:
                            PSPF0[-1][m][t+d+z] += w * np.exp(-z/tau)/w_inhibmax
                            
for I in X1:
    PSPF1.append([])
    VON1.append([])
    for n in Net1.nodes():
        PSPF1[-1].append(np.zeros(T))
        VON1[-1].append(np.zeros(T))
    for i,x in enumerate(I):
        if i != 0:
            t = int(x[0])
            n = int(x[1])
            for z in range(20):
                if t + z < T:
                    VON1[-1][n][t+z] += np.exp(-z/tau)
                pass
            for m in Net1.successors(n):
                w = Net1[n][m]['weight']
                d = Net1[n][m]['delay']
                if t + d < T:
                    for z in range(20):
                        if t + d + z < T:
                            PSPF1[-1][m][t+d+z] += w * np.exp(-z/tau)/w_inhibmax                            

for I in X2:
    PSPF2.append([])
    VON2.append([])
    for n in Net2.nodes():
        PSPF2[-1].append(np.zeros(T))
        VON2[-1].append(np.zeros(T))
    for i,x in enumerate(I):
        if i != 0:
            t = int(x[0])
            n = int(x[1])
            for z in range(20):
                if t + z < T:
                    VON2[-1][n][t+z] += np.exp(-z/tau)
                pass
            for m in Net2.successors(n):
                w = Net2[n][m]['weight']
                d = Net2[n][m]['delay']
                if t + d < T:
                    for z in range(20):
                        if t + d + z < T:
                            PSPF2[-1][m][t+d+z] += w * np.exp(-z/tau)/w_inhibmax



DATA01 = {}
DISTANCE01 = {}
NAIVE01 = {}
NAIVEDIST01={}

DATA02 = {}
DISTANCE02 = {}
NAIVE02 = {}
NAIVEDIST02={}

DATA12 = {}
DISTANCE12 = {}
NAIVE12 = {}
NAIVEDIST12={}


for R1 in range(RUNS):
    for R2 in range(RUNS):
            PM = []
            NM = []
            for n in Net0.nodes():
                PM.append((PSPF0[R1][n] - PSPF1[R2][n])**2)
                NM.append((VON0[R1][n] - VON1[R2][n])**2)
                pass
            try:
                DATA01[R1][R2] = PM
                NAIVE01[R1][R2] = NM
                DISTANCE01[R1][R2] = 0
                NAIVEDIST01[R1][R2] = 0
                VALUE = 0
                NVAL = 0
                for N in PM:
                    VALUE += sum(N)
                DISTANCE01[R1][R2] = np.sqrt(VALUE)
                for N in NM:
                    NVAL += sum(N)
                NAIVEDIST01[R1][R2] = np.sqrt(NVAL)
                
            except:
                DATA01[R1] = {}
                DATA01[R1][R2] = PM
                
                DISTANCE01[R1] = {}
                DISTANCE01[R1][R2] = 0
                
                NAIVE01[R1] = {}
                NAIVE01[R1][R2] = NM
                
                NAIVEDIST01[R1] = {}
                NAIVEDIST01[R1][R2] = 0
                
                VALUE = 0
                NVAL = 0
                
                for N in PM:
                    VALUE += sum(N)
                DISTANCE01[R1][R2] = np.sqrt(VALUE)
                
                for N in NM:
                    NVAL += sum(N)
                NAIVEDIST01[R1][R2] = np.sqrt(NVAL)
                
for R1 in range(RUNS):
    for R2 in range(RUNS):
            PM = []
            NM = []
            for n in Net0.nodes():
                PM.append((PSPF2[R1][n] - PSPF1[R2][n])**2)
                NM.append((VON2[R1][n] - VON1[R2][n])**2)
                pass
            try:
                DATA12[R1][R2] = PM
                NAIVE12[R1][R2] = NM
                DISTANCE12[R1][R2] = 0
                NAIVEDIST12[R1][R2] = 0
                VALUE = 0
                NVAL = 0
                for N in PM:
                    VALUE += sum(N)
                DISTANCE12[R1][R2] = np.sqrt(VALUE)
                for N in NM:
                    NVAL += sum(N)
                NAIVEDIST12[R1][R2] = np.sqrt(NVAL)
                
            except:
                DATA12[R1] = {}
                DATA12[R1][R2] = PM
                
                DISTANCE12[R1] = {}
                DISTANCE12[R1][R2] = 0
                
                NAIVE12[R1] = {}
                NAIVE12[R1][R2] = NM
                
                NAIVEDIST12[R1] = {}
                NAIVEDIST12[R1][R2] = 0
                
                VALUE = 0
                NVAL = 0
                
                for N in PM:
                    VALUE += sum(N)
                DISTANCE12[R1][R2] = np.sqrt(VALUE)
                
                for N in NM:
                    NVAL += sum(N)
                NAIVEDIST12[R1][R2] = np.sqrt(NVAL)



for R1 in range(RUNS):
    for R2 in range(RUNS):
            PM = []
            NM = []
            for n in Net0.nodes():
                PM.append((PSPF0[R1][n] - PSPF2[R2][n])**2)
                NM.append((VON0[R1][n] - VON2[R2][n])**2)
                pass
            try:
                DATA02[R1][R2] = PM
                NAIVE02[R1][R2] = NM
                DISTANCE02[R1][R2] = 0
                NAIVEDIST02[R1][R2] = 0
                VALUE = 0
                NVAL = 0
                for N in PM:
                    VALUE += sum(N)
                DISTANCE02[R1][R2] = np.sqrt(VALUE)
                for N in NM:
                    NVAL += sum(N)
                NAIVEDIST02[R1][R2] = np.sqrt(NVAL)
                
            except:
                DATA02[R1] = {}
                DATA02[R1][R2] = PM
                
                DISTANCE02[R1] = {}
                DISTANCE02[R1][R2] = 0
                
                NAIVE02[R1] = {}
                NAIVE02[R1][R2] = NM
                
                NAIVEDIST02[R1] = {}
                NAIVEDIST02[R1][R2] = 0
                
                VALUE = 0
                NVAL = 0
                
                for N in PM:
                    VALUE += sum(N)
                DISTANCE02[R1][R2] = np.sqrt(VALUE)
                
                for N in NM:
                    NVAL += sum(N)
                NAIVEDIST02[R1][R2] = np.sqrt(NVAL)

RANGE = np.linspace(0,T,T)

print("Pseudometric Distances")
A01 = []
B01 = []
A02 = []
B02 = []
A12 = []
B12 = []


for i in range(RUNS):
    for j in range(RUNS):
            A01.append(DISTANCE01[i][j])

print("Naive Distances")

for i in range(RUNS):
    for j in range(RUNS):
            B01.append(NAIVEDIST01[i][j])


for i in range(RUNS):
    for j in range(RUNS):
            A02.append(DISTANCE02[i][j])

print("Naive Distances")

for i in range(RUNS):
    for j in range(RUNS):
            B02.append(NAIVEDIST02[i][j])

for i in range(RUNS):
    for j in range(RUNS):
            A12.append(DISTANCE12[i][j])

print("Naive Distances")

for i in range(RUNS):
    for j in range(RUNS):
            B12.append(NAIVEDIST12[i][j])
            
            
            
            
t01 = go.Scatter(x=A01,y=B01,mode="markers",name="01")
t02 = go.Scatter(x=A02,y=B02,mode="markers",name="02")
t12 = go.Scatter(x=A12,y=B12,mode="markers",name="12")
data=[t01,t02,t12]
plot_url = py.plot(data,filename="Different Networks 0001")