# econ.py
# Prototype economics model in python
#
# Omkar H. Ramachandran
# omkar.ramachandran@colorado.edu
#
# Refer to econ_sim.pdf for method
#

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as scp
import simplex
import math
import subprocess

NCITIES = 3
TAU = 30
dTideal = 100
NC = 4
MAX = 20000
dTmax = 50

safety = np.zeros([NCITIES,NCITIES])

def time_weight(time):
#    return 1.0-0.5*(time/TAU)
    return 1.0

class Merchant:
    def __init__(self,ID,wealth,stock,current):
        self.ID = ID
        self.wealth = wealth
        self.stock = stock
        self.Next = current
        self.timestep = 0
        self.current = 0
        self.new_arrival = 1
        self.flag = 0
        self.stayed = 0

    def optimize_for_city(self,C_current,C_next):
        ii = np.where(np.array(C_current.neighbours) == C_next.ID)
        dist = C_current.distance[ii[0][0]]
        C = np.zeros(NC+1)
        C[:-1] = - (C_next.sell - C_current.buy)[:]
        C[-1] = dist
        A = np.zeros([2*NC+4,NC+1])
        A[:NC,:-1] = -np.identity(NC)
        A[NC:2*NC,:-1] = np.identity(NC)
        A[-2,:-1] = np.ones(NC)
        for i in range(NC):
            A[-4,i] = C_current.buy[i]
            A[-3,i] = C_next.sell[i]
        A[-1,-1] = 1.0
        b = np.zeros(2*NC+4)
        b[NC:2*NC] = C_current.stock
        b[-4] = self.wealth
        b[-3] = C_next.wealth
        b[-2] = 5
        b[-1] = -safety[C_current.ID,C_next.ID]
        b = b[NC:]
        A = A[NC:]
        res = simplex.simplex(C,A_ub=A,b_ub=b)
    #    print(res)
        profile = np.append(-res['fun'],res['x'])
        if(len(profile) == NC+2):
            return profile
        else:
            return np.zeros(NC+2)

    def choose_destination(self,C_list,C_current):
        profiles = np.zeros([len(C_list),NC+2])
        for i in range(len(C_list)):
            if(C_list[i].ID != C_current.ID):
                profiles[i] = self.optimize_for_city(C_current,C_list[i])
        ii = np.where(profiles[:,0] == max(profiles[:,0]))
        if(len(profiles[ii]) == 3 or len(profiles[ii]) == 2 or ii[0][0] == C_current.ID):
            m.new_arrival = 0
            return -1
        self.Next = ii[0][0]
        self.wealth -= np.sum(profiles[ii[0][0],1:-1]*C_current.buy)
        C_current.wealth += np.sum(profiles[ii[0][0],1:-1]*C_current.buy)
        self.stock = profiles[ii[0][0],1:-1]
        C_current.stock -= profiles[ii[0][0],1:-1]

    #    print(C_current.ID,ii)
        jj = np.where(C_current.neighbours == ii[0][0])
        self.timestep = C_current.distance[jj[0][0]]
        m.new_arrival = 1
        return 0

    def sell(self,C_current):
        self.wealth += np.sum(self.stock*C_current.sell)
        C_current.stock += self.stock
        C_current.wealth -= np.sum(self.stock*C_current.sell)
        self.stock = np.zeros(NC)

    def advance_timestep(self):
        self.timestep -= 1

class City:
    def __init__(self,ID,stock,sell,buy,wealth,prodrate,consrate,population,neighbours,distance):
        self.ID = ID
        self.stock = stock
        self.sell = sell
        self.buy = buy
        self.wealth = wealth
        self.prodrate = prodrate
        self.consrate = consrate
        self.population = population
        self.neighbours = neighbours
        self.distance = distance
        self.num_merchants = 0
        self.tax = 0

    def compute_buy_price(self,quantity):
        Sum = self.buy*quantity
        return np.sum(Sum)

    def estimated_selling_price(self,quantity,time):
        Sum = self.sell*quantity
        return np.sum(Sum)*time_weight(time)

    def buy_price(self,Commodity_ID,Nm):
        """ Computes the price that the merchant will get if we wants to SELL commodity COMMODITY_ID at City self"""
        if(self.consrate[Commodity_ID] <= self.prodrate[Commodity_ID]):
        # If the city produces more than it consumes, it will never buy from the
        # merchant
            return 0
        dTi = self.stock[Commodity_ID]/(self.consrate[Commodity_ID]-self.prodrate[Commodity_ID])
        cofactor = 0
        if(dTi/dTideal > 0.9):
            cofactor = 0.1
        elif(dTi/dTideal > 0.1):
            cofactor = 1-dTi/dTideal
        elif(dTi/dTideal > 0.06):
            cofactor = np.exp(0.1/(dTi/dTideal)-1)
        else:
            cofactor = np.exp(0.1/0.06-1)

        # If the city is wealthier per capita, it will be willing to tolerate a higher price
        coefficient = (self.wealth/self.population)
        if(self.num_merchants == 0):
            coefficient *= 2.0
        else:
            # If there are other merchants in the city, the price drops as sqrt(num_merchants)
            coefficient *= 1/np.sqrt(self.num_merchants)

        return coefficient*cofactor

    def sell_price(self,Commodity_ID):
        """ Computes the price that a merchant will get if he wants to BUY COMMODITY_ID from City self"""
        if(self.prodrate[Commodity_ID] >= self.consrate[Commodity_ID]):
            # Richer cities per capita will accept a lower price
            coefficient = (self.population/self.wealth)
            if(self.num_merchants == 0):
                coefficient *= 0.5
            else:
                coefficient *= np.sqrt(self.num_merchants)

            cofactor = 1.0
            dT = self.prodrate[Commodity_ID] - self.consrate[Commodity_ID]
            if(dT > dTmax):
                cofactor = 0.1
            else:
                cofactor = (1-dT/dTmax)

            return coefficient*cofactor

        else:
            coefficient = (self.population/self.wealth)
            if(self.num_merchants == 0):
                coefficient *= 0.5
            else:
                coefficient *= np.sqrt(self.num_merchants)

            cofactor = 1.0
            dT = self.stock[Commodity_ID]/(self.consrate[Commodity_ID] - self.prodrate[Commodity_ID])
            if(dT/dTideal > 0.9):
                cofactor = 0.1
            elif(dT/dTideal > 0.1):
                cofactor = (1-dT/dTideal)
            elif(dT/dTideal > 0.06):
                cofactor = np.exp(0.1/(dT/dTideal) - 1)
            else:
                cofactor = np.exp(0.1/0.06 - 1)

            return coefficient*cofactor

    def add_merchant(self,M):
        self.num_merchants += 1

    def remove_merchant(self,M):
        self.num_merchants -= 1

# Time parameters
Nt = 3000
Nruns = 100
C_wealth = np.zeros([Nruns,Nt-1,NCITIES])
comm = np.zeros([Nruns,NCITIES,Nt-1,NC])

result = subprocess.run(['git','rev-parse','HEAD'],stdout=subprocess.PIPE)

print("---------------------------------")
print("*******      EconSIM      *******")
print("---------------------------------")
print("(c) 2017 Omkar H. Ramachandran")
print("Revision: ",result.stdout)
print("Parameters: ")
print("Nruns:",Nruns,"\tNCITIES:",NCITIES)
print("NC:   ",NC,"\tNt:     ",Nt)

for r in range(Nruns):
    Nm = 5
    NM = np.zeros([Nruns,Nt])
    Nalive = Nm
    M_W = np.zeros([Nt,2])
    cum_tax=0

    Net_wealth = -np.ones(Nt-1)*10003
    merchants = []
    # Start with two merchants
    for i in range(Nm):
        start = int(np.random.uniform(low=0.0,high=NCITIES -0.01))
        M1 = Merchant(0,3,np.zeros(NC),start)
        print(start)
        merchants.append(M1)
    cons1 = np.zeros(NC)
    C1 = City(0,50*np.ones(NC),np.zeros(NC),np.zeros(NC),5000,np.zeros(NC),np.ones(NC),1000,[1,2],[5,5])

    C2 = City(1,50*np.ones(NC),np.zeros(NC),np.zeros(NC),5000,np.zeros(NC),np.ones(NC),1000,[0,2],[5,5])

    C3 = City(2,50*np.ones(NC),np.zeros(NC),np.zeros(NC),5000,np.zeros(NC),np.ones(NC),1000,[0,1],[5,5])
    safety[C1.ID,C2.ID] = 0.0
    safety[C2.ID,C1.ID] = 0.0

    # C1 produces commodity 0,3 and 4
    #C1.prodrate = np.array([0,2,0,0,2,2])
    C1.prodrate = np.array([2,0,2,0])
    # C2 produces commodity 1,2 and 5

    C2.prodrate = np.array([0,2,0,2])
    C3.prodrate = np.array([0,0,2,0])
    C_list = [C1,C2,C3]

    print("Running realization",r+1)
    for time in range(1,Nt):
        for C in C_list:
            for i in range(NC):
                C.sell[i] = C.buy_price(i,C.num_merchants)
                C.buy[i] = C.sell_price(i)
                C.stock[i] = C.stock[i] + C.prodrate[i] - C.consrate[i]
                comm[r,C.ID,time-1,i] = C.stock[i]
                if(C.stock[i] < 0):
                    C.stock[i] = 0
            C_wealth[r,time-1,C.ID] = C.wealth
            # Tax rate
            C.tax = 1e-3*C.population
            cum_tax += C.tax
            C.wealth += C.tax
            Net_wealth[time-1] += C.wealth
    #    print(C1.num_merchants,C2.num_merchants)
        for m in merchants:
            timestep = m.timestep
            m.wealth -= 0.1
            if(timestep != 0):
                m.advance_timestep()
                ii = np.where(np.array(C_list[m.current].neighbours) == m.Next)
                if(m.timestep == C_list[m.current].distance[ii[0][0]]-1):
                    C_list[m.current].remove_merchant(m)
    #                print("Removing merchant")
                continue
            else:
            #    print("Optimizing for",m.ID)
                # Merchant's at a city
                if(m.new_arrival == 1):
                    C_list[m.Next].add_merchant(m)
                    m.sell(C_list[m.Next])
                    m.current = m.Next
            #    C_list[m.current].add_merchant(m)
            #    C_list[m.current].add_merchant(m)
                # Sell off what you have
                # Choose destination
                res = m.choose_destination(C_list,C_list[m.current])
                if(res == -1 and m.stayed > 5):
                    choose = int(np.random.uniform(low=0.0,high=NCITIES -1 - 0.01))
                    m.Next = C_list[m.current].neighbours[choose]
                    m.timestep = C_list[m.current].distance[choose]
                    m.stayed = 0
                    m.new_arrival = 1
                elif(res == -1):
             #       print(time,"Merchant",m.ID,"is waiting at City",m.current)
                    m.stayed += 1
    #            elif(res == 0):
    #                print(time,"Merchant",m.ID,"is going to City",m.Next)
            if(m.wealth < 0):
#                print("Oh no... Merchant",m.ID,"is dead :-P")
                Nalive -= 1
                merchants.remove(m)
            Net_wealth[time-1] += m.wealth+0.1
        Net_wealth[time-1] -= (cum_tax)
        if(time % 43 == 0):
            start = int(np.random.uniform(low=0.0,high=1.99))
            merchants.append(Merchant(Nm,3,np.zeros(NC),start))
            C_list[start].wealth -= 3
            Nm += 1
            Nalive += 1
        NM[r,time] = Nalive
    #    for i in range(2):
    #        M_W[time,i] += merchants[i].wealth
    for m in merchants:
        print(m.wealth)

    for C in C_list:
        print(C.wealth)
