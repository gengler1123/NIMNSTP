# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 10:54:45 2016

@author: Gary
"""

import numpy as np
import glob
import os # for getting path of script
import csv

def polyfit(x, y, degree):
    results = {}

    coeffs = np.polyfit(x, y, degree)

     # Polynomial Coefficients
    results['polynomial'] = coeffs.tolist()

    # r-squared
    p = np.poly1d(coeffs)
    # fit values, and mean
    yhat = p(x)                         # or [p(z) for z in x]
    ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    results['determination'] = ssreg / sstot

    return results


hP = 8

PATH = os.path.dirname(os.path.abspath(__file__))

CSVFILES = glob.glob(PATH+"\\*.csv")


DATA = []

for path in CSVFILES:
    DATA.append([])
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            DATA[-1].append([float(row[0]), float(row[1])])
            
Fittings = []
Determination = {}
     
for i in range(hP):
    Fittings.append([])
    Determination[i+1] = []
    for data in DATA:
        X = np.array(data)
        Y = polyfit(X[:,0],X[:,1],i+1)
        Fittings[-1].append(Y['polynomial'])
        Determination[i+1].append(Y['determination'])


for j in range(hP):
    X = np.array(Fittings[j])
    Display = "\n\nStatistics for Entire Network Comparison Between \nPopulation Van Rossum and Pseudometric\n"
    for i in range(len(X[0])):
        mAvg = np.average(X[:,i])
        mStd = np.std(X[:,i])
        Display += "Coefficient " + str(i) + " || power = " + str(round(mAvg,4)) + " || Standard Deviation = " + str(round(mStd,4)) + "\n"
    Display += "R-Squared = " + str(np.average(Determination[j+1])) +"\n"
    print(Display)

print("Now testing log fit by fitting to Y = A + B log(X)")


Fittings = []
Determination = {}
     
Fittings.append([])
Determination[1] = []
for data in DATA:
        X = np.array(data)
        Y = polyfit(np.log(X[:,0]),X[:,1],1)
        Fittings[-1].append(Y['polynomial'])
        Determination[1].append(Y['determination'])



X = np.array(Fittings[0])
Display = "\n\nStatistics for Entire Network Comparison Between \nPopulation Van Rossum and Pseudometric\n"
for i in range(len(X[0])):
    mAvg = np.average(X[:,i])
    mStd = np.std(X[:,i])
    Display += "Coefficient " + str(i) + " || power = " + str(round(mAvg,4)) + " || Standard Deviation = " + str(round(mStd,4)) + "\n"
Display += "R-Squared = " + str(np.average(Determination[1])) +"\n"

print(Display)

D = np.array(DATA[-1])
import matplotlib.pyplot as plt
xp = np.linspace(np.log(100),np.log(200),1000)
p = np.poly1d(Y['polynomial'])
plt.plot(np.exp(xp),p(xp),D[:,0],D[:,1],'.')
plt.show()


'''
mAverage = np.average(X[:,0])
bAverage = np.average(X[:,1])
mSTDev = np.std(X[:,0])
bSTDev = np.std(X[:,1])

Display += "Slope Average = " + str(round(mAverage,4)) + " - Standard Deviation = " + str(round(mSTDev,4)) + "\n"
Display += "Intercept Avg = " + str(round(bAverage,4)) + " - Standard Deviation = " + str(round(bSTDev,4)) + "\n"

print(Display)
'''