# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 09:08:23 2023

@author: cp01330
"""
#---------------------------------Part 1-----------------------------------------
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
import statsmodels.api as sm


#set random seed
np.random.seed(100)
seeded_gen = default_rng(100)

#define parameters
B0 = 1
B1 = 0.5
B2 = -0.3
P = 0.75
sigma2 = 1

#set sample sizes
sample_sizes = [10, 50, 100, 200, 500, 1000, 2000, 5000, 10000]

#List for B1 and B2 standard errors
se_B1_list = []
se_B2_list = []
B1_incorrect_list = []
#Create OLS Loop
for N in sample_sizes:
    # Generate correlated random variables xi and zi
    cov_matrix = [[1, P], [P, 1]]
    data = np.random.multivariate_normal([0, 0], cov_matrix, size=N)
    xi = data[:, 0]
    zi = data[:, 1]

#Generate random errors
    e = np.random.normal(0, np.sqrt(sigma2), size=N)

#Generate yi
    yi = B0 + B1 * xi + B2 * zi + e

#Add constant to the predictor variables
    X = sm.add_constant(np.column_stack((xi, zi)))
#Create the Model
    model = sm.OLS(yi, X)
    results = model.fit()
    
#Collect standard errors for beta1 and beta2
    se_B1_list.append(results.bse[1])
    se_B2_list.append(results.bse[2])

#Suppose Incorrect specification: yi = β0 + β1xi + ϵi
#Estimate β1 for the incorrect specification
    X_incorrect = sm.add_constant(xi)
    model_incorrect = sm.OLS(yi, X_incorrect)
    results_incorrect = model_incorrect.fit()
    B1_incorrect = results_incorrect.params[1]
    B1_incorrect_list.append(B1_incorrect)

print(f"Estimate for β1 in the incorrect specification: {B1_incorrect}")


#Plot standard errors as a function of the sample size
plt.figure(figsize=(10, 6))
plt.plot(sample_sizes, se_B1_list, label=r'$B1$')
plt.plot(sample_sizes, se_B2_list, label=r'$B2$')
#Put X in Log scale for better visualization
plt.xlabel('Sample Size (N)')
plt.ylabel('Standard Errors')
plt.title('Standard Errors of Model Parameters vs. Sample Size')
plt.legend()
plt.show()
#-------------------------------------PART 2------------------------------------

#This is in same file as previous question so I am changing the variable letters
# x = a
#Define the function F(a) 
def F(a):
    return a**2 - 2.5*a + 3
# Define Derivative
def dF(a):
    return 2*a - 2.5
# Define second derivative
def d2F(a):
    return 2

#Plot the function and its derivative
a_values = np.linspace(-5, 5, 100)

F_values = F(a_values)
dF_values = dF(a_values)

plt.figure(figsize=(8,6))
plt.plot(a_values, F_values, label='F(a) = x^2 - 2.5x +3')
plt.plot(a_values, dF_values, label='dF(a) = $2x - 2.5$', linestyle='--')
plt.xlabel('a')
plt.ylabel('Function Value')
plt.title("Plot of the F(x) and F Prime(x)")
plt.legend()
plt.grid(True)
plt.show()



# Find the Analytical Derivative
def newtons_method_analytical(a0, crit=1e-6, max_iter=10000000):
    a = a0
    for i in range(max_iter):
        a_new = a - dF(a) / d2F(a)
        dist = abs(a_new - a)
        if dist < crit:
            return a_new
        a = a_new
    return a
#Printing the Answer
Analytical_Derivative = newtons_method_analytical(a0=0.0)
print("The Minimum a(x) (Analytical Derivative):", Analytical_Derivative)



    
epsilon = 1e-5
F_prime = lambda a: (F(a+epsilon) - F(a-epsilon))/(2*epsilon)      
F_prime2 = lambda a: (F_prime(a+epsilon) - F_prime(a-epsilon))/(2*epsilon) 
      
dist = 20
crit = 1e-5

a0 = 2 # random guess

aold = a0

while dist>crit:
      anew = aold - F_prime(aold)/F_prime2(aold) 
      dist = abs(anew-aold)
      aold = anew
      
Finite_Difference_Result = anew
print("The Minimum a(x) (Finite Difference Method):", Finite_Difference_Result)
 




#----------------------------------Optimisation---------------------------------

# Again, since this is in the same file I will replace the letters of x and y
# x = b
# y = c


# Root
import numpy as np
from scipy.optimize import root

# Define the system of equations
def equations(vars):
    b, c = vars
    eq1 = c - 3*b**2 + 3
    eq2 = b**3 + 0.1*c - 2
    return [eq1, eq2]

# Initial guess
initial_guess = [1, 1]

# Solve the system of equations using root
result = root(equations, initial_guess)

# Extract the solution from the result object
Optimisation_Solution = result.x

# Print the solution
print("Optimisation Solution:", Optimisation_Solution)






















