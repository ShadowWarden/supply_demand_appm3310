import numpy as np
import scipy.optimize as scp

def isIdentityColumn(column):
    c = np.zeros(column.shape)
    m = np.argmax(column)
    c[m] = 1
    return np.allclose(c,column),m

def simplexSolve(cost,A,b):
    #pricing out
    identColumns = np.array(A.shape[0])
    for i in range(A.shape[1]):
        temp = isIdentityColumn(A[:,i])
        if(temp[0]):
            cost[0] += -cost[0,i]*A[temp[1]]
    #pivoting
    while(True):
        entering = np.argmax(cost[0])
        if(cost[0,entering]<=1e-12):
            break
        leaving = np.argmin(b/np.ma.masked_less(A[:,entering],0))
        b[leaving] /= A[leaving,entering]
        A[leaving] /= A[leaving,entering]
        mult = -A[:,entering]
        mult[leaving]=0
        A += np.outer(mult,A[leaving])
        b += mult*b[leaving]
        mult = -cost[:,entering]
        cost += np.outer(mult,A[leaving])

def simplex(cost,A_ub=None,b_ub=np.empty([0]),A_eq=None,b_eq=np.empty([0])): #https://en.wikipedia.org/wiki/Revised_simplex_method
    if(A_ub is None): A_ub = np.empty([0,A_eq.shape[1]]) #prevents errors later
    if(A_eq is None): A_eq = np.empty([0,A_ub.shape[1]]) #prevents errors later
    A_ub = A_ub.astype(np.float64)
    b_ub = b_ub.astype(np.float64)
    A_eq = A_eq.astype(np.float64)
    b_eq = b_eq.astype(np.float64)
    cost = cost.astype(np.float64)
    slack = np.identity(b_ub.shape[0])
    a = b_eq.shape[0]
    for i in range(b_ub.shape[0]):
        if(b_ub[i]<0): #equivalent of > constraint, so convert it
            a+=1
            b_ub[i] *= -1
            A_ub[i] *= -1
            slack[i,i] = -1
    for i in range(b_eq.shape[0]):
        if(b_eq[i]<0):
            b_eq[i] *= -1
            A_eq[i] *= -1
    artificials = np.zeros((b_ub.shape[0]+b_eq.shape[0],a))
    c = 0
    for i in range(artificials.shape[0]):
        if((i<slack.shape[0] and slack[i,i]==-1) or i>=slack.shape[0]):
            artificials[i,c] = 1
            c+=1
    c = cost
    A = np.concatenate((np.concatenate((A_ub, A_eq), axis=0),
                        np.concatenate((slack, np.zeros((A_eq.shape[0],slack.shape[0]))), axis=0),
                        artificials), axis=1)
    b = np.concatenate((b_ub,b_eq))
    if(a!=0): #Phase 1
        cost = np.stack((np.concatenate((np.zeros(cost.shape[0]),np.zeros(slack.shape[0]),-np.ones(a)),axis=0),
                         np.concatenate((-cost,np.zeros(slack.shape[0]+a)),axis=0)))

        simplexSolve(cost,A,b)
        A = A[:,0:A.shape[1]-a]
        cost = np.reshape(cost[1,0:cost.shape[1]-a],(1,cost.shape[1]-a))
    else:
        cost = np.reshape(np.concatenate((-cost,np.zeros(slack.shape[0])),axis=0),(1,cost.shape[0]+slack.shape[0]))
    simplexSolve(cost,A,b) #Phase 2
    x = np.zeros(c.shape[0])
    for i in range(c.shape[0]):
        temp = isIdentityColumn(A[:,i])
        if(temp[0]): x[i] = b[temp[1]]
    fun = np.dot(c,x)
    return {'x': x, 'fun': fun}


# Buy coefficients
C_final = np.array([0.0,0.0,2.0,1.5,2.0,0.0])
C_initial = np.array([1.0,1.5,1.5,0.5,1.0,1.0])

# Overall money = 20

# minimize C_final*stock - C_initial*stock
# (C_final-C_initial)*stock = f
C = - (C_final-C_initial)
print("f=",C)
# Constraints: Each commodity > 0
A = np.zeros([7,6])
A[:-1,:] = -np.identity(6)
A[-1] = np.ones(6)
b = np.zeros(7)
b[-1] = 20

res = scp.linprog(C,A_ub=A,b_ub=b)
