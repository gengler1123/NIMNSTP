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


for RUN in range(10):
    input_neurons = NetworkStimulation(Net, random()*xmax, random()*ymax, .4)
    I_clamp=300
              
    T=500
    
    
    for t in range(T):
        if t>50:
            for n in input_neurons.getNodes():
                Net.node[n]['model'].clamp_input(I_clamp)
            Net.NetTimestep()
    
    Net.print_Firings()
    X.append(Net.Firings)
