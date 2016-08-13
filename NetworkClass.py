# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 23:09:13 2016

@author: Guanhao Wu
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class Network(nx.DiGraph):
    def __init__(self):
        nx.DiGraph.__init__(self)
        self.t=0
        self.Firings=np.array([[0,0,None,None]])

    def NetTimestep(self):
        currentfire = []
    #timestep phase    
        for n in self.nodes():
            self.node[n]['model'].timestep()
            if self.node[n]['model'].fired == True:
                currentfire.append(n)
                self.Firings=np.vstack([self.Firings,[int(self.t),int(n),self.node[n]['x'],self.node[n]['y']]])
    #communication phase        
        for n in currentfire:
            for m in self.successors(n):
                self.node[m]['model'].update_delayI(
                    self[n][m]['weight'],
                    self[n][m]['delay']
                    )                 
        self.t+=1
        print(len(currentfire))
        
    def print_Firings(self):
        plt.figure()
        plt.plot(self.Firings[:,0],self.Firings[:,1],'.')
        plt.show()