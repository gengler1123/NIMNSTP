# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 22:53:41 2016

@author: Guanhao Wu
"""

from random import random,randint,choice
from math import exp,ceil,sqrt

class NetworkGeneration():
    
    def RegNet(self,NT,N,neuron_excit,neuron_inhib,
               n_excit,p_excit,w_excitmax,d_excitmax,
               n_total,p_inhib,w_inhibmax,d_inhibmax):
                   
        for n in range(n_excit):
            NT.add_node(n)
            NT.node[n]['model']=N()    
            neuron_excit(NT.node[n]['model'])
            
        for n in range(n_excit,n_total):
            NT.add_node(n)
            NT.node[n]['model']=N()
            neuron_inhib(NT.node[n]['model'])
                   
        for n in NT.nodes():
            if n<n_excit:
                for m in NT.nodes():
                    if n!=m and random()<p_excit:
                        NT.add_edge(n,m,weight=w_excitmax*random())
                        NT[n][m]['delay']=randint(1,d_excitmax)
            else:
                for m in NT.nodes():
                    if n!=m and random()<p_inhib:
                        NT.add_edge(n,m,weight=w_inhibmax*random())
                        NT[n][m]['delay']=randint(1,d_inhibmax)
                        
        for n in NT.nodes():
            for m in NT.predecessors(n):
                NT.node[n]['model'].adjustMaxD(NT[m][n]['delay'])
                NT.node[n]['model'].delaygen()
                
    def SmallWorld(self,NT,N,k,p_rewire,neuron_excit,neuron_inhib,
                   n_excit,p_excit,w_excitmax,d_excitmax,
                   n_total,p_inhib,w_inhibmax,d_inhibmax):
        
        for n in range(n_excit):
            NT.add_node(n)
            NT.node[n]['model']=N()    
            neuron_excit(NT.node[n]['model'])
            
        for n in range(n_excit,n_total):
            NT.add_node(n)
            NT.node[n]['model']=N()
            neuron_inhib(NT.node[n]['model'])
        
        for n in NT.nodes():
            if n<n_excit:
                for i in range(k//2):
                    if n!=(n+i+1)%n_excit and n!=(n-i-1)%n_excit:
                        NT.add_edge(n,(n+i+1)%n_excit,weight=w_excitmax*random())
                        NT[n][(n+i+1)%n_excit]['delay']=randint(1,d_excitmax)
                        NT.add_edge(n,(n-i-1)%n_excit,weight=w_excitmax*random())
                        NT[n][(n-i-1)%n_excit]['delay']=randint(1,d_excitmax)
                        
#                for m in range(n_excit,n_total):
#                    if random()<p_inhib:
#                        NT.add_edge(n,m,weight=w_inhibmax*random())
#                        NT[n][m]['delay']=randint(1,d_inhibmax)
                                                    
            else:
                for m in NT.nodes():
                    if n!=m and random()<p_inhib:
                        NT.add_edge(n,m,weight=w_inhibmax*random())
                        NT[n][m]['delay']=randint(1,d_inhibmax)
                    
        e=NT.edges()
        for (u,v) in e:
            if random()<=p_rewire and u<n_excit and v<n_excit:
                w=choice(NT.nodes())
                while w == u or NT.has_edge(u,w):
                    w=choice(NT.nodes())
                NT.add_edge(u,w,weight=w_excitmax*random())
                NT[u][w]['delay']=randint(1,d_excitmax)
                NT.remove_edge(u,v)
                    
        for n in NT.nodes():
            for m in NT.predecessors(n):
                NT.node[n]['model'].adjustMaxD(NT[m][n]['delay'])
                NT.node[n]['model'].delaygen()
                
    def SpatialNet(self,NT,N,neuron_excit,neuron_inhib,
                   xmax,ymax,Pmax,tau_excit,tau_inhib,dv_excit,dv_inhib,
                   n_excit,p_excit,w_excitmax,d_excitmax,
                   n_total,p_inhib,w_inhibmax,d_inhibmax):
        
        for n in range(n_excit):
            NT.add_node(n)
            NT.node[n]['model']=N()    
            neuron_excit(NT.node[n]['model'])
                     
        for n in range(n_excit,n_total):
            NT.add_node(n)
            NT.node[n]['model']=N()
            neuron_inhib(NT.node[n]['model'])
                     
        for n in range(n_total):
            NT.node[n]['x']=random()*xmax
            NT.node[n]['y']=random()*ymax
            
        for n in NT.nodes():
            if n<n_excit:
                for m in NT.nodes():
                    Pe=Pmax*exp(-tau_excit*(sqrt((NT.node[n]['x']-NT.node[m]['x'])**2+
                        (NT.node[n]['y']-NT.node[m]['y'])**2)))
                    Dve=ceil(d_excitmax*(sqrt((NT.node[n]['x']-NT.node[m]['x'])**2+
                        (NT.node[n]['y']-NT.node[m]['y'])**2)))
                    if n!=m and Pe<p_excit:
                        NT.add_edge(n,m,weight=w_excitmax*random())
                        NT[n][m]['delay']=min(Dve,d_excitmax)
            else:
                for m in NT.nodes():
                    Pi=Pmax*exp(-tau_inhib*(sqrt((NT.node[n]['x']-NT.node[m]['x'])**2+
                        (NT.node[n]['y']-NT.node[m]['y'])**2)))
                    Dvi=ceil(d_inhibmax*(sqrt((NT.node[n]['x']-NT.node[m]['x'])**2+
                        (NT.node[n]['y']-NT.node[m]['y'])**2)))
                    if n!=m and Pi<p_inhib:
                        NT.add_edge(n,m,weight=w_inhibmax*random())
                        NT[n][m]['delay']=min(Dvi,d_inhibmax)
                        
        for n in NT.nodes():
            for m in NT.predecessors(n):
                NT.node[n]['model'].adjustMaxD(NT[m][n]['delay'])
                NT.node[n]['model'].delaygen()
            
            
        
            
                    
        