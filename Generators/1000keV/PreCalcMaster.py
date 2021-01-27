print('Starting Imports')
import numpy as np
from scipy.special import gamma
# from MechanismsOneDimension import *
# from MechanismsStripped import *
from MechanismPreCalc import *
import sys
from time import time
import resource
import os
from scipy.signal import argrelmin
from datetime import datetime
proc_number =int(sys.argv[1])

print('Defining Isotope')
me = .511
gA = 1.254

Ms = ['MGT','MF','MGTW','MFW','MGTQ','MFQ','MT','MR','MP']
SeNME=dict.fromkeys(Ms)
SeNME['MGT'] = 3.003
SeNME['MF'] = -.606/(gA**2)
SeNME['MGTQ'] = 3.004
SeNME['MFQ'] = -0.487/(gA**2)
SeNME['MGTW'] = 2.835
SeNME['MFW'] = -.617/(gA**2)
SeNME['MT'] = 0.012
SeNME['MR']=3.252
SeNME['MP'] = .5*SeNME['MGT']

SeNME['A'] = 82
SeNME['Z'] = 36
SeNME['Sf']=100
SeNME['Qbb']=2.99512


starttime = time()
Nu = [-12,-6.154902]
Eta = [-12,-8.154902]
Lambda = [-12,-6.154902]

#for root,dirs,files in os.walk('./'):
#    for file in files:
#        if 'MechEnv' not in root:
#            print(os.path.join(root,file))

Test = bbMechanismsStats(SeNME,Eta,Nu,Lambda,0,0,bins=50)
Test.InitializeIsotope()
np.random.seed(int(datetime.now().timestamp()))
print('Beginning Sweeps')

tol = .1
Test.MultipleSweeps(iter_number=proc_number,n=1,expcounts=[100,1000,25000],Res=[1000],Input_HL='HalflivesLin50.npy',tol=tol)


print('Finished')
print('Time Taken: ',str(round((time()-starttime)/60,3)),' Min')
print(str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000)+' Mb of Memory Used Peak')
