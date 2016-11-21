# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 22:11:34 2016

@author: Guanhao Wu
"""

class Neuron():
    def __init__(self):
        self.k=.7                           #somthing related to potassium
        self.vr=-60                         #some parameter in the differential eq
        self.vt=-40                         #some parameter in the differential eq
        self.vp=35                          #peak voltage 
        self.I=0                            #input current
        self.md=1                           #maximum delay 
        self.delayI=[0]*self.md             #list for delayI
        self.C=100                          #capacitance
        self.a=.03                          #some parameter in the differential eq
        self.b=-2                           #some parameter in the differential eq
        self.c=-50                          #some parameter in the differential eq
        self.d=100                          #some parameter in the differential eq
        self.fired=False                    #state of firing PUBLIC for now
        self.V=-50                          #voltage
        self.U=0                            #gating variable
        self.dt=1                           #timestep for numeric solver
        self.t=0                            #internal clock
        self.V_E=0                          #equilibrium voltage
        self.U_E=0                          #equilibrium gating variable
#timestep phase        
    def updateI(self):
        self.I+=self.delayI[self.t%self.md]
        self.delayI[self.t%self.md]=0
        
    def dV(self):
        return (self.k*(self.V-self.vr)*(self.V-self.vt)-self.U+self.I)/self.C
        
    def dU(self):
        return self.a*(self.b*(self.V-self.vr)-self.U)
        
    def timestep(self): #called externally (public function)
        self.fired=False
        self.updateI()
        DV=self.dV()
        DU=self.dU()
        
        self.V+=self.dt*DV
        self.U+=self.dt*DU
        
        if self.V>=self.peak():
            self.fired=True
            self.v_reset()
            self.u_reset()
            
        self.reset_input()
        self.t+=1
            
    def peak(self):
        return self.vp        
        
    def v_reset(self):
        self.V=self.c
        
    def u_reset(self):
        self.U+=self.d
        
    def reset_input(self):
        self.I=0
#communication phase        
    def update_delayI(self,w,d): 
        self.delayI[(self.t-1+d)%self.md]+=w 
#network generation        
    def adjustMaxD(self,delay):
        self.md=max(self.md,delay)
        
    def delaygen(self):
        self.delayI=[0 for x in range(self.md)]
#helper function    
    def clamp_input(self,i):
        self.I+=i
        
    def give_V(self):
        return self.V
        
    def give_U(self):
        return self.U
        
    def set_equil(self):
        for t in range(200):
            self.timestep()
        self.V_E=self.V
        self.U_E=self.U
        self.t=0
    
    def return_equil(self):
        self.V=self.V_E
        self.U=self.U_E
        self.t=0
