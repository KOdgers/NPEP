import numpy as np
from scipy.special import gamma
from scipy.integrate import trapz, quad, dblquad, simps

from cmath import sqrt,polar,exp,pi
from time import time
import os
import itertools
from scipy.stats import chisquare
import pandas as pd
from scipy.signal import argrelmin

me = .511



class bbMechanismsStats:
    def __init__(self,IsoDict, EtaLim, NuLim, LambdaLim, Phi, Phi2,bins):
        self.bins=bins
        self.me = .511
        self.gA = 1.254
        self.A = IsoDict['A']  # (number of nucleons)(temp value)
        self.Qbb = IsoDict['Qbb']  # Energy released in decay
        self.hbarc = 197.3269602
        self.R = self.me * 1.2 * (self.A ** (1 / 3)) / self.hbarc
        self.T = self.Qbb /self.me
        self.a = 1 / 137  # finestructure constant
        self.Z = IsoDict['Z']  # (number of protons? maybe?)
        self.Sf = IsoDict['Sf']  # (Screening factor) this will change for different elements
        self.Zs = self.Z * self.Sf / 100

        self.Gf = 1.1663787 * 10 ** -5  # Fermis Constant
        self.cosc = 0.9749  # Cabibo Angle

        self.phi1 = Phi
        self.phi2 = Phi2
        self.Etas = EtaLim
        self.Nus = NuLim
        self.Lambdas = LambdaLim

        self.ParamList =  list(itertools.product(*[self.Etas,self.Nus,self.Lambdas]))

        self.MGT = IsoDict['MGT']
        self.MF = IsoDict['MF']
        self.MGTW = IsoDict['MGTW']
        self.MFW = IsoDict['MFW']
        self.MGTQ = IsoDict['MGTQ']
        self.MFQ = IsoDict['MFQ']
        self.MT = IsoDict['MT']
        self.MR = IsoDict['MR']
        self.MP = IsoDict['MP']

        self.G1 = 2.31*10**(-14)

        # self a0v = (gA**4)*((Gf*cosc)**4)*(me**4)/(32*np.pi**

    def InitializeIsotope(self):

        self.CalcEpsilon()
        self.Calcg0()
        self.CalcXs()
        # self.CalcZs()


    def GetKineticDist(self):
        x = np.linspace(0,self.Qbb-.01,100)
        y =[]
        for i in x:
            y.append(self.KineticDist(i/me))
        return x,y

    def GetAngularDist(self):
        x = np.linspace(0,pi,100)
        y = []
        for i in x:
            y.append(self.AngularDist(i))
        return x,y

    def GetAng(self):
        return self.Ang

    def KineticDist(self,i):
        x = .5*(self.T + 2 + i)
        x2 = .5*(self.T+2-i)

        AA = self.CalcAA(x,x2)
        return np.log(2) * (10 ** 23) * self.g0 * self.Calcw0v(x,x2) * AA / (self.R ** 2)  ## 10**27


    def AIFunc(self,i):
        x = i
        x2 = self.T+2-x
        AA = self.CalcAA(x,x2)

        return self.g0 * self.Calcw0v(x,x2) * AA / (self.R ** 2)

    def BIFunc(self,i):
        x = i
        x2 = self.T+2-x
        BB = self.CalcBB(x,x2)

        return self.g0 * self.Calcw0v(x,x2) * BB / (self.R ** 2)

    def AI(self):
        AI = quad(self.AIFunc, 1, self.T+1)
        return AI

    def BI(self):
        BI = quad(self.BIFunc, 1, self.T+1)
        return BI

    def AngularDist(self,theta):
        return (np.log(2)*10**27)*(self.AI()[0]+np.cos(theta)*self.BI()[0])/(2*pi)

    def Calcg0(self):
        self.g0 = 2.8 * (10 ** (-22)) * (self.gA ** 4)

    def Calcw0v(self,x,x2):

        return np.sqrt(x ** 2 - 1) * np.sqrt(x2 ** 2 - 1) * x * x2


    def CalcAA(self,x,x2):
        N1, N2, N3, N4 = self.CalcN(x,x2)
        return (polar(N1)[0]) ** 2 + (polar(N2)[0]) ** 2 + (polar(N3)[0]) ** 2 + (polar(N4)[0]) ** 2

    def CalcBB(self,x,x2):
        N1, N2, N3, N4 = self.CalcN(x,x2)
        return -2*np.real(np.conj(N1)*N2+np.conj(N3)*N4)

    def CalcN(self,x,x2):
        eps12 = self.CalcEps12(x,x2)
        Amm, Amp, Apm, App = self.CalcAs(x,x2)
        N1 = np.conj(Amm) * (
                    (self.Z1 - 4 * self.Z6 / 3) - (4 / self.R) * (self.Z4 - self.epsilon * self.Z6 / 6))
        N2 = np.conj(App) * (
                    (self.Z1 - 4 * self.Z6 / 3) + (4 / self.R) * (self.Z4 - self.epsilon * self.Z6 / 6))
        N3 = np.conj(Apm) * (
                    (self.Z1 - 2 * self.Z5 / 3) - eps12 * (self.Z3 + self.epsilon * self.Z5 / 3))
        N4 = np.conj(Amp) * (
                    (self.Z1 - 2 * self.Z5 / 3) + eps12 * (self.Z3 + self.epsilon * self.Z5 / 3))
        return N1, N2, N3, N4

    def CalcAs(self, x,x2):
        Am1,Am2,Ap1,Ap2 = self.CalcAmAp(x,x2)
        Amm = Am1 * Am2
        Amp = Am1 *Ap2
        Apm = Ap1 * Am2
        App = Ap1 * Ap2
        return Amm, Amp, Apm, App

    def CalcAmAp(self, x,x2):
        # x2 = self.CalcEps2(x)
        Am1 = sqrt((x + 1) / (2 * x)) * sqrt(self.F(x, 1))
        # print(x,x2)
        Am2 = sqrt((x2 + 1) / (2 * x2)) * sqrt(self.F(x2, 1))
        Ap1 = sqrt((x - 1) / (2 * x)) * sqrt(self.F(x, 1))
        Ap2 = sqrt((x2 - 1) / (2 * x2)) * sqrt(self.F(x2, 1))

        return (Am1,Am2,Ap1,Ap2)

    def CalcEps12(self, x,x2):
        return x - x2

    def CalcEpsilon(self):
        self.epsilon = 3 * self.a * self.Zs + (self.T + 2) * self.R

    def CalcP(self, x):
        return sqrt(x ** 2 - 1)

    def CalcEps2(self, x):
        return self.T+2-x

    def F(self, x, k):
        return ((gamma(2 * k + 1) / (gamma(k) * gamma(2 * self.gammak(k) + 1))) ** 2) * ((2 * self.CalcP(x) * self.R) ** (2 * self.gammak(k) - 2 * k)) * ((polar(gamma(self.gammak(k) + 1j * self.y(x)))[0]) ** 2) * exp(pi * self.y(x))

    def gammak(self, k):
        return k ** 2 - (self.a * self.Z) ** 2

    def y(self, x):
        return self.a * self.Zs * x / self.CalcP(x)

    def CalcZs(self):
        self.Z1 = self.Nu * (self.XF - 1) * self.MGT
        self.Z3 = self.MGT * ((-self.Lambda) * (self.XGTW - self.XFW) * exp(1j * self.phi1) + self.Eta * (
                    self.XGTW + self.XFW) * exp(1j * self.phi2))
        self.Z4 = self.Eta * self.XR * exp(1j * self.phi2) * self.MGT
        self.Z5 = (1 / 3) * (
                    self.Lambda * self.X1P * exp(1j * self.phi1) - self.Eta * self.X1M * exp(1j * self.phi2)) * self.MGT
        self.Z6 = self.Eta * self.XP * exp(1j * self.phi2) * self.MGT

    def CalcXs(self):
        self.XGTQ = self.MGTQ / self.MGT
        self.XGTW = self.MGTW / self.MGT
        self.XF = self.MF / self.MGT
        self.XFQ = self.MFQ / self.MGT
        self.XFW = self.MFW / self.MGT
        self.XT = self.MT / self.MGT
        self.XR = self.MR / self.MGT
        self.XP = self.MP / self.MGT

        self.X1P = (self.XGTQ + 3 * self.XFQ - 6 * self.XT)
        self.X1M = (self.XGTQ - 3 * self.XFQ - 6 * self.XT)
        self.X2P = (self.XGTW + self.XFW - self.X1P / 9)
        self.X2M = (self.XGTW - self.XFW - self.X1M / 9)





    def CalcHl(self):
        x = np.linspace(0,self.Qbb-.000001,1000)
        y = []
        for i in x:
            y.append(self.KineticDist(i/me))
        hl = simps(y,x)
        return 1/hl



    def GenerateSample(self,numberOfSamples,config,printing=False):


        self.RandomDist = config

        self.Nu = config[1]
        self.Eta = config[0]
        self.Lambda = config[2]
        self.CalcZs()
        tau = self.CalcHl()
        samples = []
        numberofLoops =0
  
        if not os.path.exists(str(config[0])+'_'+str(config[1])+'_'+str(config[2])+'cdf.npy'):
            enInd, cInd = self.GetReverse()
            cdf = np.stack([enInd,cInd],axis=1)
            np.save(str(config[0])+'_'+str(config[1])+'_'+str(config[2])+'cdf',cdf)

        cdf = np.load(str(config[0])+'_'+str(config[1])+'_'+str(config[2])+'cdf.npy')
        enInd, cInd = np.split(cdf,2,axis=1)

        while len(samples)<numberOfSamples:
            dart = np.random.uniform(0,1)
            ind = np.abs(cInd-dart).argmin()
            samples.append(int(enInd[ind][0]*1000))

  

        tsamples = []
        numberofLoops = 0

        darts = np.random.rand(1,numberOfSamples)
        darts = np.reshape(darts,(darts.shape[1]))
        tsamples = -1*np.log(1-darts)    

        sampleslist = np.stack([samples,[item*tau for item in tsamples]],axis=1)
        return sampleslist

    def GetReverse(self,n=5000):
        en = np.linspace(0,self.Qbb-.00001,n)
        y=[0]
        for l in en[1:]:
            y.append(self.KineticDist(l/me))
        hl=simps(y,en)
        ysum = [np.sum(y[:lm])/np.sum(y) if lm>0 else 0 for lm in range(0,n)]

        return np.asarray(en),np.asarray(ysum)

    def HalfLives(self,printing = False,worker = 'All'):
        Nus = np.linspace(10**self.Nus[0], 10**self.Nus[1], self.bins)
        Etas = np.linspace(10**self.Etas[0], 10**self.Etas[1], self.bins)
        Lambdas = np.linspace(10**self.Lambdas[0], 10**self.Lambdas[1], self.bins)


        self.EtaGrid, self.NuGrid, self.LambdaGrid = np.meshgrid(Etas, Nus, Lambdas)
#         print(len(NusNew),len(EtasNew),len(LambdasNew))
        total_time = time()
        self.hls = np.zeros_like(self.EtaGrid)
        # overal_start_time = time()
        
        if worker == 'All':
            for l1 in range(0, self.bins):
                # print(l1)
                nu_start_time = time()
                for l2 in range(0, self.bins):
                    start_time = time()
                    for l3 in range(0, self.bins):
                        print(l1, '.', l2, '.', l3)
                        self.Nu = Nus[l2]
                        self.Eta = Etas[l1]
                        self.Lambda = Lambdas[l3]
                        self.CalcZs()
                        tempval = self.CalcHl()
                        self.hls[l1, l2, l3] = tempval
                    print('time to complete 1 LambdaGrid[s]: ' + str(time() - start_time))
                print('time to complete 1 Nu Grid[s]: ' + str(time() - nu_start_time))

            print('time to complete: ' + str(round((time() - total_time) / 60, 2)) + ' Minutes')

            np.save('Halflives'+str(bins)+'Ext', self.hls,allow_pickle=True)
        else:
            
            for l1 in range(int(worker*len(self.EtaGrid[0, :, 0])/6),int((worker+1)*len(self.EtaGrid[0, :, 0])/6)):
                # print(l1)
                nu_start_time = time()
                for l2 in range(0, self.bins):
                    start_time = time()
                    for l3 in range(0, self.bins):
#                         print(l1, '.', l2, '.', l3)
                        self.Nu = Nus[l2]
                        self.Eta = Etas[l1]
                        self.Lambda = Lambdas[l3]
                        self.CalcZs()
                        tempval = self.CalcHl()
                        self.hls[l1, l2, l3] = tempval
                    print('time to complete 1 LambdaGrid[s]: ' + str(time() - start_time))
                print('time to complete 1 Nu Grid[s]: ' + str(time() - nu_start_time))
                        
        return self.hls



    def MultipleSweeps(self, iter_number=0, n = 10, expcounts=[5,10],Res = [0,50],Input_HL='Halflives48.npy',tol=.1):

        config_number = int(np.floor(iter_number/200))

        res = Res[0]
        edges = [12,24,36]
        a = list(itertools.product(edges, edges, edges))
#        a = [[24,24,24],[40,24,24]]
        Nus = np.linspace(10**self.Nus[0], 10**self.Nus[1], self.bins)
        Etas = np.linspace(10**self.Etas[0], 10**self.Etas[1], self.bins)
        Lambdas = np.linspace(10**self.Lambdas[0], 10**self.Lambdas[1], self.bins)
#        a=[[12,12,36]]
        configl = list(a[config_number])
        #config = [10,10,38]
#        configl = [3,7,20]
        config = [Etas[configl[0]], Nus[configl[1]], Lambdas[configl[2]]]
        if not os.path.exists('Vals_'+str(res)+'_'+str(iter_number)+'.h5'):
                print('First Time Running')
                iters_finished=0

        else:
                print('Job Was halted, starting where left off')
                f = pd.read_pickle('Vals_'+str(res)+'_'+str(iter_number)+'.h5')
                f_l = f.shape[0]
                iters_finished = f_l
                del f

        start=iters_finished
        clist=[]
        
        for j1 in expcounts:
            for j2 in Res:
                for j3 in range(0,n):
                    clist.append(str(j1)+','+str(j2))

        for j in range(start,len(clist)):
                # print(j)
                samp = int(clist[j].split(',')[0])
                res = int(clist[j].split(',')[1])
                TrueE, KTE, TE,KE,MinKT,DefKT, MinT,DefT, KTL,TL,KL, KTX, KTY, TX,TY,VarKT,VarT,minKTloc,minTloc = self.BandCalcCompV2( config,configl, NSamplesPerCase=samp,Res=res,Input_Halflives = Input_HL,Tol=tol)
                TempDict = {}
                TempDict['Samps']= [samp]
                TempDict['Res'] =[res]
                TempDict['Tol']=[tol]
                TempDict['TrueE'] = [TrueE]
                TempDict['KTE'] = [KTE]
                TempDict['TE'] = [TE]
                TempDict['KE']=[KE]
                TempDict['MinKT']=[MinKT]
                TempDict['MinT']=[MinT]
                TempDict['KTL']=[KTL]
                TempDict['TL']=[TL]
                TempDict['KL']=[KL]
                TempDict['KTX'] = [KTX]
                TempDict['KTY'] = [KTY]
                TempDict['TX'] = [TX]
                TempDict['TY'] = [TY]
                TempDict['DefKT']=[DefKT]
                TempDict['DefT']=[DefT]
                TempDict['VarKT']=[VarKT]
                TempDict['VarT']=[VarT]
                TempDict['MinKTLoc']=[minKTloc]
                TempDict['MinTLoc']=[minTloc]
                tempdf = pd.DataFrame(TempDict)
                if iters_finished+j <1:
                    tempdf.to_pickle('Vals_'+str(res)+'_'+str(iter_number)+'.h5')
                    print('Original df')
#                    del tempdf
#                    del TempDict
                else:
                    print('H5 File size[b]:', os.stat('Vals_'+str(res)+'_'+str(iter_number)+'.h5').st_size)
                    PastDf = pd.read_pickle('Vals_'+str(res)+'_'+str(iter_number)+'.h5')
                    CurrDf = pd.concat([PastDf,tempdf])
                    print('Concatenated: '+str(CurrDf.shape))
                    CurrDf.to_pickle('Vals_'+str(res)+'_'+str(iter_number)+'.h5')


    def BandCalcCompV2(self,TrueConfig,ind3,NSamplesPerCase=5, Res = 0,File_dir='./',Input_Halflives='Halflives48.npy',Tol=.1):
        print(TrueConfig)
        Res = Res/1000  # Res is given in KeV, Convert to MeV
        taus = np.load(Input_Halflives)


        Nus = np.linspace(10**self.Nus[0], 10**self.Nus[1], self.bins)
        Etas = np.linspace(10**self.Etas[0], 10**self.Etas[1], self.bins)
        Lambdas = np.linspace(10**self.Lambdas[0], 10**self.Lambdas[1], self.bins)

        self.EtaGrid, self.NuGrid, self.LambdaGrid = np.meshgrid(Etas, Nus, Lambdas)


        start_time = time()
        samples = self.GenerateSample(NSamplesPerCase, TrueConfig)

        self.MLEGrid = np.zeros_like(self.EtaGrid)

        self.KMLE = np.zeros_like(self.EtaGrid)
        self.TauMLE = np.zeros_like(self.EtaGrid)

        KTvals = np.zeros_like(self.MLEGrid)
        Tvals = np.zeros_like(self.TauMLE)
        Kvals = np.zeros_like(self.KMLE)

#                     tau = halflives[l1,l2,l3]
        for samplenum in range(0, len(samples)):
            sample = samples[samplenum]
            x, t = sample[0], sample[1]
#             print(x,t,type(x))
            ###########################################
            if int(x)>=2990:
                if Res*1000==10:
                    print(x)
                    x = 2980
            if int(x)>=2900 and Res*1000 == 100:
                x = 2850
            if Res!=1:
                tempK = np.load('KE_Linear_'+str(int(Res*1000))+'_'+str(int(x/100))+'_setsV3.npy')
            
            modx = int(100/(Res*1000))
#             print(modx,Res)
            if Res*1000 == 100:
                tempK = np.log(tempK)
            elif Res ==1:
                tempK = np.log(np.load('KE_Lin_1000_'+str(int(x/1000))+'_setsV3.npy'))
            else:
                tempK = np.log(tempK[:,:,:,int((x/(Res*1000))%modx)])
            

            tempT = np.log(np.exp(-t/taus)-np.exp(-(t+.01)/taus))
#            tempT = tempT
            Kvals = Kvals-tempK
            KTvals = KTvals - (tempK+tempT)
            Tvals = Tvals-tempT

        ind = np.unravel_index(self.MLEGrid.argmin(), self.MLEGrid.shape)

        KTvals = Kvals+Tvals
        self.MLEGrid= KTvals
        self.TauMLE = Tvals
        self.KMLE = Kvals
        del KTvals
        del Kvals
        del Tvals
                



        ind = np.unravel_index(self.MLEGrid.argmin(), self.MLEGrid.shape)

        precalcarr=[]
        for samplenum in range(0,len(samples)):
            sample = samples[samplenum]
            x,t = sample[0],sample[1]
            if Res!=1:
                tempK = np.load('KE_Linear_'+str(int(Res*1000))+'_'+str(int(x/100))+'_setsV3.npy')[ind[0],ind[1],ind[2],int(x%100)]
            else:
                tempK = np.load('KE_Lin_1000_'+str(int(x/1000))+'_setsV3.npy')[ind[0],ind[1],ind[2]]
            precalcarr.append(np.log(tempK))

        localKT = self.minFinder(self.MLEGrid)
#         print(len(localKT))
        localT = self.minFinder(self.TauMLE)
#         print(len(localT))
        
        ######################################################################
#         print(self.MLEGrid[ind],np.min(self.MLEGrid))

        ind2 = np.unravel_index(self.TauMLE.argmin(), self.TauMLE.shape)
        indK = np.unravel_index(self.KMLE.argmin(),self.KMLE.shape)
        varKT = self.VarGet(self.MLEGrid,ind)
        varT = self.VarGet(self.TauMLE,ind)
        
        print('########## KT Contenders #############')
        if len(localKT)>1:
            KTContenders = [item for item in localKT if 
                            self.MLEGrid[item[0],item[1],item[2]]
                                         -self.MLEGrid[ind[0],ind[1],ind[2]]<varKT]
            KTContVals = [self.MLEGrid[item[0],item[1],item[2]] for item in KTContenders]
            KTContenders = [x for _, x in sorted(zip(KTContVals,KTContenders), key=lambda pair: pair[0])]
            if len(KTContenders)>10:
                KTContenders=KTContenders[:10]
            print(KTContenders)
            print([self.MLEGrid[item[0],item[1],item[2]] for item in KTContenders])
        else:
            KTContenders=[ind]
        print('########## T Contenders #############')

        if len(localT)>1:
            TContenders = [item for item in localT if 
                            self.TauMLE[item[0],item[1],item[2]]
                                         -self.TauMLE[ind2[0],ind2[1],ind2[2]]<varT]
            TContVals = [self.TauMLE[item[0],item[1],item[2]] for item in TContenders]
            TContenders = [x for _, x in sorted(zip(TContVals,TContenders), key=lambda pair: pair[0])]
#             if len(TContenders)>10:
#                 TContenders=TContenders[:10]
#             print(TContenders)
#             print([self.TauMLE[item[0],item[1],item[2]] for item in TContenders])
        else:
            KTContenders=[ind]



#         print('VarKT:',varKT)
        print('Default min KT'+str(self.MLEGrid.min()))
        print('Default Loc: ',Etas[ind[0]],Nus[ind[1]],Lambdas[ind[2]],ind)
        if Tol >=1:
            minKT = np.min(self.MLEGrid)
            minKTloc = [Etas[ind[0]],Nus[ind[1]],Lambdas[ind[2]]]
            minT = np.min(self.TauMLE)
            minTloc = [Etas[ind2[0]],Nus[ind2[1]],Lambdas[ind2[2]]]
        else:
#             print('Def Tau: ',taus[ind])
            for ik in range(0,len(KTContenders)):
                if ik ==0:
                    minKTtemp , minKTloctemp = self.GetBestKT(samples,KTContenders[ik],res=Res,tol=Tol,oldarr=precalcarr)
                else:
                    minKTtemp , minKTloctemp = self.GetBestKT(samples,KTContenders[ik],res=Res,tol=Tol)
                print('MinKT',ik,minKTtemp)
                if ik ==0:
                    minKT=minKTtemp
                    minKTloc=minKTloctemp
                else:
                    if minKTtemp<minKT:
                        minKT=minKTtemp
                        minKTloc=minKTloctemp
            for ik in range(0,len(TContenders)):
                minTtemp , minTloctemp = self.GetBestT(samples,TContenders[ik],res=Res,tol=Tol)
                print('MinT ',ik,minTtemp)

                if ik ==0:
                    minT=minTtemp
                    minTloc=minTloctemp
                else:
                    if minTtemp<minT:
                        minT=minTtemp
                        minTloc=minTloctemp
#             minT, minTloc = self.GetBestT(samples, ind2, res=Res,tol=Tol)
#             minT = np.min(self.TauMLE)
#             minTloc = ind2

        KTY = np.subtract(self.MLEGrid,np.ones_like(self.MLEGrid)*minKT)
        TY = np.subtract(self.TauMLE,np.ones_like(self.TauMLE)*minT)

        print('Smallest 10 Elements: ', sorted(self.MLEGrid.flatten())[:5])
        print('Smallest after Sub: ', sorted(KTY.flatten())[:5])
        KTX = np.logspace(-2,np.log10(np.max(KTY)),400)
        TX = np.logspace(-2,np.log10(np.max(TY)),400)
 #       print('Time to finish MLE Calc: ' + str(time() - start_time) + '[s]')

        KTY =[len([item for item in KTY.flatten() if item<il]) for il in KTX]
        TY =[len([item for item in TY.flatten() if item<il]) for il in TX]

        print('Opt Loc ',str(minKTloc),'Default Loc',str([Etas[ind[0]],Nus[ind[1]],Lambdas[ind[2]]]))
        print('Opt Min', minKT, 'Default Min',np.min(self.MLEGrid))
        print('Opt Loc ',str(minTloc),'Default Loc',str([Etas[ind2[0]],Nus[ind2[1]],Lambdas[ind2[2]]]))
        print('OptMin T',minT,'Defualt MinT' ,np.min(self.TauMLE))

        print('Time to finish Optimal Calc: ' + str(time() - start_time) + '[s]')

        print('=============================================================')
        print('')
	
        KTDiff = minKT-self.MLEGrid[ind3[0],ind3[1],ind3[2]]
        TDiff = minT-self.TauMLE[ind3[0],ind3[1],ind3[2]]

        KDiff = np.min(self.KMLE)-self.KMLE[ind3[0],ind3[1],ind3[2]]


        return ind3, ind, ind2,indK,minKT,np.min(self.MLEGrid),minT,np.min(self.TauMLE), KTDiff, TDiff,KDiff, KTX, KTY, TX,TY,varKT,varT,minKTloc,minTloc

    def VarGet(self,dat,ind):
        ndim = len(ind)
        shape = dat.shape
        offsets = list(itertools.product([-1,0,1],[-1,0,1],[-1,0,1]))
        offsets = np.asarray([item for item in offsets if item.count(0)==2])
        neighbours = ind + offsets    # apply offsets to p
#         print(neighbours)
        neighbours = [ item for item in neighbours if all(it >= 0 for it in item)]
        neighbours = [ item for item in neighbours if all(it<=self.bins-1 for it in item)]
#         print(len(neighbours))
        var = [np.abs(dat[ind[0],ind[1],ind[2]]-dat[ne[0],ne[1],ne[2]]) for ne in neighbours]
#         for iv in range(0,len(neighbours)):
#             print(neighbours[iv],var[iv])
        
        return(np.max(var))

    def ParamSlice(self,loct,param):
      #  print(param,loct)
        if loct ==0:
            start = np.log10(param[0])
            end = np.log10(param[2])
        elif loct==len(param)-1:
            start = np.log10(param[-3])
            end = np.log10(param[-1])
        else:
            start =np.log10(param[loct-1])
            end = np.log10(param[loct+1])

        outparam = [10**m for m in np.linspace(start,end,5)]
       # print(outparam, param[loct]-outparam[2])
        return outparam


    def Cube(self,ind,b1,b2,b3):
        newind=[0,0,0]
        a1=b1
        if ind[0]==0:
            a1new = [a1[0],a1[0]+.6*(a1[1]-a1[0]),a1[0]+1.2*(a1[1]-a1[0])]
            
        elif ind[0] == len(a1)-1:
            newind[0]=2
            tempind=len(a1)-1
            a1new = [a1[tempind]-1.2*(a1[tempind]-a1[tempind-1]),a1[tempind]-.6*(a1[tempind]-a1[tempind-1]),a1[tempind]]
        else:
            newind[0]=1
            a1new = [a1[ind[0]]-.6*(a1[ind[0]]-a1[ind[0]-1]),a1[ind[0]],a1[ind[0]]+.6*(a1[ind[0]+1]-a1[ind[0]])]
        out1 = a1new
        a1=b2
        if ind[1]==0:
            a1new = [a1[0],a1[0]+.6*(a1[1]-a1[0]),a1[0]+1.2*(a1[1]-a1[0])]
        elif ind[1] == len(a1)-1:
            newind[1]=2
            tempind=len(a1)-1
            a1new = [a1[tempind]-1.2*(a1[tempind]-a1[tempind-1]),a1[tempind]-.6*(a1[tempind]-a1[tempind-1]),a1[tempind]]
        else:
            newind[1]=1
            a1new = [a1[ind[1]]-.6*(a1[ind[1]]-a1[ind[1]-1]),a1[ind[1]],a1[ind[1]]+.6*(a1[ind[1]+1]-a1[ind[1]])]
        out2 = a1new
        a1=b3
        if ind[2]==0:
            newind[2]=0
            a1new = [a1[0],a1[0]+.6*(a1[1]-a1[0]),a1[0]+1.2*(a1[1]-a1[0])]
        elif ind[2] == len(a1)-1:
            newind[2]=2
            tempind=len(a1)-1
            a1new = [a1[tempind]-1.2*(a1[tempind]-a1[tempind-1]),a1[tempind]-.6*(a1[tempind]-a1[tempind-1]),a1[tempind]]
        else:
            newind[2]=1
            a1new = [a1[ind[2]]-.6*(a1[ind[2]]-a1[ind[2]-1]),a1[ind[2]],a1[ind[2]]+.6*(a1[ind[2]+1]-a1[ind[2]])]
        out3 = a1new

        return out1,out2,out3,newind
        
    def GetBestKT(self,data,loc,res,var=10,tol=.01,oldarr=False):

        Nus = np.linspace(10**self.Nus[0], 10**self.Nus[1], self.bins)
        Etas = np.linspace(10**self.Etas[0], 10**self.Etas[1], self.bins)
        Lambdas = np.linspace(10**self.Lambdas[0], 10**self.Lambdas[1], self.bins)
        iters = 1
        Res = res
        while var>=tol and iters <=10:
            

            Etas,Nus,Lambdas,newloc = self.Cube(loc,Etas,Nus,Lambdas)
            KTVals=np.zeros((3,3,3))
            tauArr = np.zeros((3,3,3))
            newarr=[]
            KTPre=np.zeros((3,3,3,2996))
            
            for l1 in range(0,len(Etas)):
                self.Eta=Etas[l1]
                for l2 in range(0,len(Nus)):
                    self.Nu=Nus[l2]
                    for l3 in range(0,len(Lambdas)):

                        self.Lambda=Lambdas[l3]
                        self.CalcZs()
                        tau = self.CalcHl()
                        tauArr[l1,l2,l3]=tau
                        tempK = []
                        for x in range(0,2996):
                            if Res*1000 == 1:
                                if x ==0 or x ==1 :
                                    x0 = 0.00001
                                    xarr = np.linspace(0, .0015, 50)
                                elif x==2994 or x ==2995:
                                    x0 = self.Qbb - Res
                                    xarr = np.linspace(2.9935, self.Qbb-.0001, 50)
                                else:
                                    x0 = (x/1000)-Res/2
                                    xarr = np.linspace(x0, x0+Res, 50)
                                y = []
                                for xiter in xarr:
                                    y.append( self.KineticDist(xiter/me))
                                tempK.append(simps(y,xarr))
                        KTPre[l1,l2,l3,:]=np.asarray(tempK)/np.sum(tempK[1:2995])
                        
                        
                        
                        
            TempK = np.zeros((3,3,3))
            for sample in data:
                x, t = sample[0], sample[1]
                x = int(x)
                tempKK = np.log(KTPre[:,:,:,x])
                tempTT = np.log(np.exp(-t/tauArr)-np.exp(-(t+.01)/tauArr))
                TempK = TempK-tempKK-tempTT

                                
            KTVals=TempK
            
            if iters==1 and oldarr:
                newarr = []
                for sample in data:
                    x = int(sample[0])
                    newarr.append(np.log(KTPre[newloc[0],newloc[1],newloc[2],x]))
                print('Lens',len(newarr),len(oldarr))
                print([newarr[lo]-oldarr[lo] for lo in range(0,len(oldarr))])
            loc = np.unravel_index(KTVals.argmin(), KTVals.shape)
            iters+=1
            var = np.max([np.abs(item-KTVals[loc]) for item in KTVals.flatten()])
#             print('Best Loc:',Etas[loc[0]],Nus[loc[1]],Lambdas[loc[2]])
#             print('Old Loc',Etas[newloc[0]],Nus[newloc[1]],Lambdas[newloc[2]],'Val: ',KTVals[newloc[0],newloc[1],newloc[2]])
#             print('Middle: '+str(KTVals[1,1,1]),'Best: '+str(np.min(KTVals)),'Loc: '+str(loc),'Var: '+str(var))
#             print('======================================================== \n \n')
        return np.min(KTVals), [Etas[loc[0]],Nus[loc[1]],Lambdas[loc[2]]]
    
    
    def GetBestKT1000(self,data,loc,res,var=10,tol=.01,oldarr=False):

        Nus = np.linspace(10**self.Nus[0], 10**self.Nus[1], self.bins)
        Etas = np.linspace(10**self.Etas[0], 10**self.Etas[1], self.bins)
        Lambdas = np.linspace(10**self.Lambdas[0], 10**self.Lambdas[1], self.bins)
        iters = 1
        Res = res
        while var>=tol and iters <=10:
            

            Etas,Nus,Lambdas,newloc = self.Cube(loc,Etas,Nus,Lambdas)
            KTVals=np.zeros((3,3,3))
            tauArr = np.zeros((3,3,3))
            newarr=[]
            KTPre=np.zeros((3,3,3,3))
            
            for l1 in range(0,len(Etas)):
                self.Eta=Etas[l1]
                for l2 in range(0,len(Nus)):
                    self.Nu=Nus[l2]
                    for l3 in range(0,len(Lambdas)):

                        self.Lambda=Lambdas[l3]
                        self.CalcZs()
                        tau = self.CalcHl()
                        tauArr[l1,l2,l3]=tau
                        tempK = []
                        for x in range(0,3):
                            if Res*1000 == 1000:
                                if x ==0 :
                                    x0 = 0.00001
                                    xarr = np.linspace(0, 1, 50)
                                elif x==2:
                                    x0 = self.Qbb - Res
                                    xarr = np.linspace(2, self.Qbb-.0001, 50)
                                else:
                                    x0 = (x/1000)-Res/2
                                    xarr = np.linspace(1, 2, 50)
                                y = []
                                for xiter in xarr:
                                    y.append( self.KineticDist(xiter/me))
                                tempK.append(simps(y,xarr))
                        KTPre[l1,l2,l3,:]=np.asarray(tempK)/np.sum(tempK)
                        
                        
                        
                        
            TempK = np.zeros((3,3,3))
            for sample in data:
                x, t = sample[0], sample[1]
                x = int(x/1000)
                tempKK = np.log(KTPre[:,:,:,x])
                tempTT = np.log(np.exp(-t/tauArr)-np.exp(-(t+.01)/tauArr))
                TempK = TempK-tempKK-tempTT

                                
            KTVals=TempK
            
#             if iters==1 and oldarr:
#                 newarr = []
#                 for sample in data:
#                     x = int(sample[0])
#                     newarr.append(np.log(KTPre[newloc[0],newloc[1],newloc[2],x]))
#                 print('Lens',len(newarr),len(oldarr))
#                 print([newarr[lo]-oldarr[lo] for lo in range(0,len(oldarr))])
            loc = np.unravel_index(KTVals.argmin(), KTVals.shape)
            iters+=1
            var = np.max([np.abs(item-KTVals[loc]) for item in KTVals.flatten()])
            print('Best Loc:',Etas[loc[0]],Nus[loc[1]],Lambdas[loc[2]])
            print('Old Loc',Etas[newloc[0]],Nus[newloc[1]],Lambdas[newloc[2]],'Val: ',KTVals[newloc[0],newloc[1],newloc[2]])
            print('Middle: '+str(KTVals[1,1,1]),'Best: '+str(np.min(KTVals)),'Loc: '+str(loc),'Var: '+str(var))
            print('======================================================== \n \n')
        return np.min(KTVals), [Etas[loc[0]],Nus[loc[1]],Lambdas[loc[2]]]
    
    
    
    
    def GetBestT(self,data,loc,res,var=10,tol=.01):

        Nus = np.linspace(10**self.Nus[0], 10**self.Nus[1], self.bins)
        Etas = np.linspace(10**self.Etas[0], 10**self.Etas[1], self.bins)
        Lambdas = np.linspace(10**self.Lambdas[0], 10**self.Lambdas[1], self.bins)
        iters = 1
        Res = res
        while var>=tol and iters <=10:
            
#             print(Res)
    #         Etas = self.ParamSlice(loc[0],Etas)
    #         Nus = self.ParamSlice(loc[1],Nus)
    #         Lambdas = self.ParamSlice(loc[2],Lambdas)
            Etas,Nus,Lambdas,newloc = self.Cube(loc,Etas,Nus,Lambdas)
#             print(Etas)
#             print(Nus)
#             print(Lambdas)
            TVals=np.zeros((3,3,3))
            tauArr = np.zeros((3,3,3))
            for l1 in range(0,len(Etas)):
                self.Eta=Etas[l1]
                for l2 in range(0,len(Nus)):
                    self.Nu=Nus[l2]
                    for l3 in range(0,len(Lambdas)):

                        self.Lambda=Lambdas[l3]
                        self.CalcZs()
                        tau = self.CalcHl()
#                         if l1 == newloc[0] and l2 == newloc[1] and l3==newloc[2]:
#                             print('New Tau: ',tau)
    #                     tauArr[l1,l2,l3]=tau
                        
#                         print(round(self.Eta,12),round(self.Nu,12),round(self.Lambda,12))
                        tempT = 0
                        for sample in data:
                            x, t = sample[0], sample[1]
                            
                            tempTT = np.log(np.exp(-t/tau)-np.exp(-(t+.01)/tau))
                            tempT =tempT-tempTT
                        TVals[l1,l2,l3]=tempT



            loc = np.unravel_index(TVals.argmin(), TVals.shape)
            iters+=1
            var = np.max([np.abs(item-TVals[loc]) for item in TVals.flatten()])
#             print('Best Loc:',Etas[loc[0]],Nus[loc[1]],Lambdas[loc[2]])
#             print('Old Loc',Etas[newloc[0]],Nus[newloc[1]],Lambdas[newloc[2]],'Val: ',TVals[newloc[0],newloc[1],newloc[2]])
#             print('Middle: '+str(TVals[1,1,1]),'Best: '+str(np.min(TVals)),'Loc: '+str(loc),'Var: '+str(var))
        return np.min(TVals), [Etas[loc[0]],Nus[loc[1]],Lambdas[loc[2]]]
    
    def minFinder(self,Y):
        ind = argrelmin(Y, axis=0,mode='wrap')
        ind0 = [(ind[0][i],ind[1][i],ind[2][i])for i in range(0,len(ind[0]))]
        ind = argrelmin(Y, axis=1,mode='wrap')
        ind1 = [(ind[0][i],ind[1][i],ind[2][i]) for i in range(0,len(ind[0]))]
        ind = argrelmin(Y, axis=2,mode='wrap')
        ind2 = [(ind[0][i],ind[1][i],ind[2][i]) for i in range(0,len(ind[0]))]
        mins = list(set(ind0) & set(ind1) & set(ind2))
        return mins

