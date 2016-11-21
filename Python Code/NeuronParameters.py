# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 23:38:59 2016

@author: Guanhao Wu
"""

import types

class NeuronParameters():
    
    def set_LTS(self,N): #Low Threshold Spiking
        N.C=100
        N.k=1
        N.vr=-56
        N.vt=-42
        N.a=.03
        N.b=8
    
        def peak(N):
            return 40-.1*N.U
            
        def v_reset(N):
            N.V=-53+.04*N.U
            
        def u_reset(N):
            N.U=min(N.U+20,670)
            
        N.peak=types.MethodType(peak,N)
        N.v_reset=types.MethodType(v_reset,N)
        N.u_reset=types.MethodType(u_reset,N)
    
    def set_FS(self,N): #Fast Spiking
        N.C=20
        N.k=1
        N.vr=-55
        N.vt=-40
        N.a=.2
        N.c=-45
        N.vb=-55

        def Uv(N,V):
            if V<N.vb:
                return 0
            else:
                return .025*(V-N.vb)**3

        def dU(N):
            return N.a*(N.Uv(N.V)-N.U)
            
        N.Uv=types.MethodType(Uv,N)
        N.dU=types.MethodType(dU,N)
        
    def set_RS(self,N): #Regular Spiking
        N.C=100
        N.k=.7
        N.vr=-60
        N.vt=-40
        N.a=.03
        N.b=-2
        N.c=-50
        N.d=100
        N.vp=35
        
    def set_CP(self,N): #Chattering Spiking
        N.C=50
        N.k=1.5
        N.vr=-60
        N.vt=-40
        N.a=.03
        N.b=1
        N.c=-40
        N.d=150
        N.vp=25
        
    def set_IB(self,N): #Intrinsically Bursting
        N.C=150
        N.k=1.2
        N.vr=-75
        N.vt=-45
        N.a=.01
        N.b=5
        N.c=-56
        N.d=130
        N.vp=50
        