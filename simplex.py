import numpy as np
import scipy.optimize as scp

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
