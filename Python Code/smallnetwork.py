# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 13:38:45 2016

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


Net = NC.Network()
P = NP.NeuronParameters()
NeuronList ={'RS':P.set_RS,
                     'LTS':P.set_LTS,
                     'CP':P.set_CP,
                     'IB':P.set_IB,
                     'FS':P.set_FS}
                     
                     

Net.add_nodes_from(range(12))



