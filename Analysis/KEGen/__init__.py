import numpy as np
from scipy.special import gamma
from scipy.integrate import trapz, quad, dblquad, simps

from cmath import sqrt,polar,exp,pi
from time import time
import os
import itertools
from scipy.stats import chisquare
import pandas as pd


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
            print(x,t,type(x))
            ###########################################
            tempK = np.load('KE_Linear_'+str(int(Res*1000))+'_'+str(int(round(x)/100))+'_sets.npy')

            tempK = np.log(tempK[:,:,:,int(x%100)]*taus)
            
            #############################################

            tempK = tempK
#            tempT = -(t/taus)+np.log(taus)
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
        print(self.MLEGrid[ind],np.min(self.MLEGrid))

        ind2 = np.unravel_index(self.TauMLE.argmin(), self.TauMLE.shape)
        indK = np.unravel_index(self.KMLE.argmin(),self.KMLE.shape)

        if Tol ==1:
            minKT = np.min(self.MLEGrid)
            minKTloc = ind
            minT = np.min(self.TauMLE)
            minTloc = ind2
        else:
            print(taus[ind])
            minKT , minKTloc = self.GetBestKT(samples,ind,res=Res,tol=Tol)
            minT, minTloc = self.GetBestT(samples, ind2, res=Res,tol=Tol)

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


        return ind3, ind, ind2,indK,minKT,np.min(self.MLEGrid),minT,np.min(self.TauMLE), KTDiff, TDiff,KDiff, KTX, KTY, TX,TY

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






    def CalcKELM(self,Energy,Res = 0,proc_number=0,filename='KE_',halflives='Hl'):
#         print(TrueConfig)
#        halflives = np.load(halflives)

        Res = Res/1000  # Res is given in KeV, Convert to MeV
        Nus = np.linspace(10**self.Nus[0], 10**self.Nus[1], self.bins)
        Etas = np.linspace(10**self.Etas[0], 10**self.Etas[1], self.bins)
        Lambdas = np.linspace(10**self.Lambdas[0], 10**self.Lambdas[1], self.bins)

        self.EtaGrid, self.NuGrid, self.LambdaGrid = np.meshgrid(Etas, Nus, Lambdas)


        start_time = time()

        Kvals = np.zeros_like(self.EtaGrid)
        for l1 in range(0,self.bins):
            if l1%12==0:
                print(str(round((l1+1)/self.bins,3)*100),' % Done with Current Experiment')
            for l2 in range(0,self.bins):
                for l3 in range(0,self.bins):
                    self.Nu = Nus[l2]
                    self.Eta = Etas[l1]
                    self.Lambda = Lambdas[l3]

                    self.CalcZs()
 #                   tau = halflives[l1,l2,l3]

                    x=Energy

                    tempK = 0

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
                    tempK = simps(y,xarr)

                    Kvals[l1,l2,l3] = tempK

        np.save(filename+str(proc_number),Kvals,allow_pickle=True)
        print('Time to run: '+str((time()-start_time)/60)+' Min')
        print('Energy:' +str(Energy))
        
        


    def CalcKELM10(self,Energy,Res = 10,proc_number=0,filename='KE_',halflives='Hl'):
#         print(TrueConfig)
#        halflives = np.load(halflives)

        Res = Res/1000  # Res is given in KeV, Convert to MeV
        Nus = np.linspace(10**self.Nus[0], 10**self.Nus[1], self.bins)
        Etas = np.linspace(10**self.Etas[0], 10**self.Etas[1], self.bins)
        Lambdas = np.linspace(10**self.Lambdas[0], 10**self.Lambdas[1], self.bins)

        self.EtaGrid, self.NuGrid, self.LambdaGrid = np.meshgrid(Etas, Nus, Lambdas)


        start_time = time()

        Kvals = np.zeros_like(self.EtaGrid)
        for l1 in range(0,self.bins):
            if l1%12==0:
                print(str(round((l1+1)/self.bins,3)*100),' % Done with Current Experiment')
            for l2 in range(0,self.bins):
                for l3 in range(0,self.bins):
                    self.Nu = Nus[l2]
                    self.Eta = Etas[l1]
                    self.Lambda = Lambdas[l3]

                    self.CalcZs()
 #                   tau = halflives[l1,l2,l3]

                    x=Energy

                    tempK = 0

                    if x ==0 :
                        x0 = 0.00001
                        xarr = np.linspace(0, .01, 50)
                    elif x>=298:
                        x0 = self.Qbb - Res
                        xarr = np.linspace(2.980, self.Qbb-.0001, 50)
                    else:
#                         xint = 10*int(x/10)
                        x0 = x/100
                        xarr = np.linspace(x0, x0+Res, 50)
                    y = []
                    for xiter in xarr:
                        y.append( self.KineticDist(xiter/me))
                    tempK = simps(y,xarr)

                    Kvals[l1,l2,l3] = tempK

        np.save(filename+str(proc_number)+'_10',Kvals,allow_pickle=True)
        print('Time to run: '+str((time()-start_time)/60)+' Min')
        print('Energy:' +str(Energy))
        
        
    def CalcKELM100(self,Energy,Res = 10,proc_number=0,filename='KE_',halflives='Hl'):
#         print(TrueConfig)
#        halflives = np.load(halflives)

        Res = Res/1000  # Res is given in KeV, Convert to MeV
        Nus = np.linspace(10**self.Nus[0], 10**self.Nus[1], self.bins)
        Etas = np.linspace(10**self.Etas[0], 10**self.Etas[1], self.bins)
        Lambdas = np.linspace(10**self.Lambdas[0], 10**self.Lambdas[1], self.bins)

        self.EtaGrid, self.NuGrid, self.LambdaGrid = np.meshgrid(Etas, Nus, Lambdas)


        start_time = time()

        Kvals = np.zeros_like(self.EtaGrid)
        for l1 in range(0,self.bins):
            if l1%12==0:
                print(str(round((l1+1)/self.bins,3)*100),' % Done with Current Experiment')
            for l2 in range(0,self.bins):
                for l3 in range(0,self.bins):
                    self.Nu = Nus[l2]
                    self.Eta = Etas[l1]
                    self.Lambda = Lambdas[l3]

                    self.CalcZs()
 #                   tau = halflives[l1,l2,l3]

                    x=Energy

                    tempK = 0

                    if x ==0 :
                        x0 = 0.00001
                        xarr = np.linspace(0, .01, 50)
                    elif x==29:
                        x0 = self.Qbb - Res
                        xarr = np.linspace(2.900, self.Qbb-.0001, 50)
                    else:
#                         xint = 10*int(x/10)
                        x0 = x/10
                        xarr = np.linspace(x0, x0+Res, 50)
                    y = []
                    for xiter in xarr:
                        y.append( self.KineticDist(xiter/me))
                    tempK = simps(y,xarr)

                    Kvals[l1,l2,l3] = tempK

        np.save(filename+str(proc_number)+'_100',Kvals,allow_pickle=True)
        print('Time to run: '+str((time()-start_time)/60)+' Min')
        print('Energy:' +str(Energy))
        
    def CalcKELM1000(self,Energy,Res = 10,proc_number=0,filename='KE_',halflives='Hl'):


        Res = Res/1000  # Res is given in KeV, Convert to MeV
        Nus = np.linspace(10**self.Nus[0], 10**self.Nus[1], self.bins)
        Etas = np.linspace(10**self.Etas[0], 10**self.Etas[1], self.bins)
        Lambdas = np.linspace(10**self.Lambdas[0], 10**self.Lambdas[1], self.bins)

        self.EtaGrid, self.NuGrid, self.LambdaGrid = np.meshgrid(Etas, Nus, Lambdas)


        start_time = time()

        Kvals = np.zeros_like(self.EtaGrid)
        for l1 in range(0,self.bins):
            if l1%12==0:
                print(str(round((l1+1)/self.bins,3)*100),' % Done with Current Experiment')
            for l2 in range(0,self.bins):
                for l3 in range(0,self.bins):
                    self.Nu = Nus[l2]
                    self.Eta = Etas[l1]
                    self.Lambda = Lambdas[l3]

                    self.CalcZs()
 #                   tau = halflives[l1,l2,l3]

                    x=Energy

                    tempK = 0

                    if x ==0 :
                        x0 = 0.00001
                        xarr = np.linspace(0, 1, 100)
                    elif x==1:
                        x0 = self.Qbb - Res
                        xarr = np.linspace(1, 2, 100)
                    elif x==2:
#                         xint = 10*int(x/10)
                        x0 = x/10
                        xarr = np.linspace(2, self.Qbb-.0001, 100)
                    y = []
                    for xiter in xarr:
                        y.append( self.KineticDist(xiter/me))
                    tempK = simps(y,xarr)

                    Kvals[l1,l2,l3] = tempK

        np.save(filename+str(proc_number)+'_1000',Kvals,allow_pickle=True)
        print('Time to run: '+str((time()-start_time)/60)+' Min')
        print('Energy:' +str(Energy))
        
        
    def GenerateSample(self,numberOfSamples,config,printing=False):

        self.RandomDist = config

        Ens = np.linspace(0, self.Qbb - .01, 50)
        self.Nu = config[1]
        self.Eta = config[0]
        self.Lambda = config[2]
        self.CalcZs()
        tau = self.CalcHl()
        samples = []
        numberofLoops =0
#         cdf = self.GetReverse()
        darts=[]
        while len(samples)<numberOfSamples:
            numberofLoops+=1
            dart = np.random.uniform(0,2)
            xdart = np.random.uniform(0,self.Qbb-.000001)


            dartval = (self.KineticDist(xdart/me))*tau
            darts.append(dartval)
            if dartval>dart:
                samples.append(round(xdart*1000))
        print(str(numberofLoops))
        print(np.max(darts))
        tsamples = []
        numberofLoops = 0
        while len(tsamples) < numberOfSamples:
            numberofLoops += 1
            dart = np.random.uniform(0, 1)
            tdart = np.random.uniform(0, 5)

            dartval = (np.exp(-(tdart*tau)/tau))
            if dartval > dart:
                tsamples.append(tdart)


        sampleslist = np.stack([samples,[item*tau for item in tsamples]],axis=1)
        return sampleslist,tau
    
