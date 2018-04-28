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
    iters = 0
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
        iters+=1
    return iters

def primalSimplex(cost,A,b):
    iters = 0
    while(True):
        entering = np.argmax(cost)
        if(cost[entering]<=1e-12):
            break
        leaving = np.argmin(b/np.ma.masked_less(A[:,entering],0))
        b[leaving] /= A[leaving,entering]
        A[leaving] /= A[leaving,entering]
        mult = -A[:,entering]
        mult[leaving]=0
        A += np.outer(mult,A[leaving])
        b += mult*b[leaving]
        cost += -cost[entering]*A[leaving]
        iters+=1
    return iters

def dualSimplex(cost,A,b):
    iters = 0
    while(True):
        leaving = np.argmin(b)
        if(b[leaving]>=-1e-12):
            break
        entering = np.argmax(cost/np.ma.masked_greater_equal(A[leaving],0))
        b[leaving] /= A[leaving,entering]
        A[leaving] /= A[leaving,entering]
        mult = -A[:,entering]
        mult[leaving]=0
        A += np.outer(mult,A[leaving])
        b += mult*b[leaving]
        cost += -cost[entering]*A[leaving]
        iters+=1
    return iters

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
    iters = 0
    if(a!=0): #Phase 1
        cost = np.stack((np.concatenate((np.zeros(cost.shape[0]),np.zeros(slack.shape[0]),-np.ones(a)),axis=0),
                         np.concatenate((-cost,np.zeros(slack.shape[0]+a)),axis=0)))

        iters += simplexSolve(cost,A,b)
        A = A[:,0:A.shape[1]-a]
        cost = np.reshape(cost[1,0:cost.shape[1]-a],(1,cost.shape[1]-a))
    else:
        cost = np.reshape(np.concatenate((-cost,np.zeros(slack.shape[0])),axis=0),(1,cost.shape[0]+slack.shape[0]))
    iters += simplexSolve(cost,A,b) #Phase 2
    x = np.zeros(c.shape[0])
    for i in range(c.shape[0]):
        temp = isIdentityColumn(A[:,i])
        if(temp[0]): x[i] = b[temp[1]]
    fun = np.dot(c,x)
    return {'x': x, 'fun': fun, 'iters': iters}

def improvedSimplex(cost,A_ub=None,b_ub=np.empty([0]),A_eq=None,b_eq=np.empty([0])):
    if(A_ub is None): A_ub = np.empty([0,A_eq.shape[1]]) #prevents errors later
    if(A_eq is None): A_eq = np.empty([0,A_ub.shape[1]]) #prevents errors later
    A_ub = A_ub.astype(np.float64)
    b_ub = b_ub.astype(np.float64)
    A_eq = A_eq.astype(np.float64)
    b_eq = b_eq.astype(np.float64)
    origCost = cost.astype(np.float64)
    cost = -origCost
    A = np.concatenate((A_ub,A_eq,-A_eq),axis=0)
    b = np.concatenate((b_ub,b_eq,-b_eq),axis=0)
    origB = b
    b = cost
    cost = origB
    origA = A
    A = np.concatenate((np.transpose(A),-np.identity(A.shape[1])),axis=1)
    cost = np.concatenate((cost,np.zeros(A.shape[0])),axis=0)
    angles = np.arccos(np.matmul(np.transpose(A),b)/(np.linalg.norm(A,axis=0)*np.linalg.norm(b)))
    argpart = np.argpartition(angles,A.shape[0])
    try:
        A_B_inv = np.linalg.inv(A[:,argpart[:A.shape[0]]])
    except:
        print("{0}\n{1}\n{2}".format(origCost,A_ub,b_ub))
        print(argpart)
        print(A)
        print(A[:,argpart[:A.shape[0]]])
        raise
    A = np.matmul(A_B_inv,A)
    b = np.matmul(A_B_inv,b)
    cost += np.matmul(-cost[argpart[:A.shape[0]]],A)
    d = np.min(b)
    p = np.argmin(cost)
    iters = 0
    if(d<-1e-12 and cost[p]<-1e-12): #big M
        M = -cost[p]*100 if -cost[p]*100>100 else 100
        A = np.concatenate((A,np.zeros((A.shape[0],1))),axis=1)
        cost = np.concatenate((cost,[0]))
        newrow = np.zeros((1,A.shape[1]))
        newrow[0,argpart[A.shape[0]:]] = 1
        newrow[0,-1-A.shape[0]:-1] = 0
        newrow[0,-1] = 1
        A = np.concatenate((A,newrow),axis=0)
        b = np.concatenate((b,[M]))
        mult = -A[:,p]
        mult[-1]=0
        A += np.outer(mult,newrow)
        b += mult*M
        cost += -cost[p]*newrow[0,:]
        iters = dualSimplex(cost,A,b)
        A = A[:,:-1]
    elif(d<-1e-12):
        iters = dualSimplex(cost,A,b)
    elif(cost[p]<-1e-12):
        iters = primalSimplex(cost,A,b)
    #convert to primal solution and return
    constraints = []
    nonZeroVars = []
    for i in range(0,A.shape[1]):
        temp = isIdentityColumn(A[:,i])
        if(i<origA.shape[0] and temp[0]):
            constraints.append(i)
        elif(i>=origA.shape[0] and (not temp[0] or (b[temp[1]]>-1e-12 and b[temp[1]]<1e-12))):
            nonZeroVars.append(i-origA.shape[0])
    ans = np.zeros(origA.shape[1])
    print("\n\n{0}\n\n".format(origA[constraints][:,nonZeroVars]))
    ans[nonZeroVars] = np.matmul(np.linalg.inv(origA[constraints][:,nonZeroVars]),origB[constraints])
    return {'x': ans, 'fun': np.dot(ans,origCost), 'iters': iters}

np.set_printoptions(linewidth=1000)
'''
print(simplex(np.array([-2,-3,-4]),np.array([[3,2,1],[2,5,3]]),np.array([10,15])))
print(scp.linprog(np.array([-2,-3,-4]),np.array([[3,2,1],[2,5,3]]),np.array([10,15])))
print(improvedSimplex(np.array([-2,-3,-4]),np.array([[3,2,1],[2,5,3]]),np.array([10,15])))
print(simplex(np.array([-6,-5]),np.array([[-1,2],[3,-1],[6,6],[5.5,7],[-1,-1]]),np.array([4,6,36,38.5,-1])))
print(scp.linprog(np.array([-6,-5]),np.array([[-1,2],[3,-1],[6,6],[5.5,7],[-1,-1]]),np.array([4,6,36,38.5,-1])))
print(improvedSimplex(np.array([-6,-5]),np.array([[-1,2],[3,-1],[6,6],[5.5,7],[-1,-1]]),np.array([4,6,36,38.5,-1])))
print(simplex(np.array([-2,-3,-4]),A_eq=np.array([[3,2,1],[2,5,3]]),b_eq=np.array([10,15])))
print(scp.linprog(np.array([-2,-3,-4]),A_eq=np.array([[3,2,1],[2,5,3]]),b_eq=np.array([10,15])))
print(improvedSimplex(np.array([-2,-3,-4]),A_eq=np.array([[3,2,1],[2,5,3]]),b_eq=np.array([10,15])))
print(simplex(np.array([-10,-24,-20,-20,-25]),np.array([[1,1,2,3,5],[2,4,3,2,1]]),np.array([19,57])))
print(scp.linprog(np.array([-10,-24,-20,-20,-25]),np.array([[1,1,2,3,5],[2,4,3,2,1]]),np.array([19,57])))
print(improvedSimplex(np.array([-10,-24,-20,-20,-25]),np.array([[1,1,2,3,5],[2,4,3,2,1]]),np.array([19,57])))


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
'''
