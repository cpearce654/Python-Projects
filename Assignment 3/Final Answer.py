#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 17:56:37 2023

@author: craigpearce
"""

import numpy as np
import matplotlib.pyplot as plt

#Parameters
etack = 0.4562
etakk = 0.9712
etacz = 0.5629
etakz = 0.0675
p = 0.979
sigma2 = 0.0072
beta = 0.984
delta = 0.025
a = 0.33
sigma = np.sqrt(sigma2)

#Length of sequence
T = 1000

k_bar = (((1-beta)/beta + delta)*1/a)**(1/(a-1))
z_bar = 1
c_bar = (((1-beta)/beta + delta)*1/a)**(a/(a-1)) - delta * k_bar
R_bar = a * z_bar * k_bar**(a-1)
w_bar = (1-a) * (((1-beta)/beta + delta)*1/a)**(a/(a-1))
r_bar = R_bar - delta
i_bar = delta * k_bar
y_bar = z_bar * k_bar**a
n_bar = 1
I_bar = 0




#Print the steady-state values
print(f"z_bar: {z_bar}")
print(f"k_bar: {k_bar}")
print(f"c_bar: {c_bar}")
print(f"w_bar: {w_bar}")
print(f"R_bar: {R_bar}")
print(f"y_bar: {y_bar}")
print(f"i_bar: {i_bar}")
print(f"r_bar: {r_bar}")


#Arrays to store business cycle moments
y_hat = np.zeros(T)  # Array to store output
c_hat = np.zeros(T)  # Array to store consumption
i_hat = np.zeros(T)  # Array to store investment
r_hat = np.zeros(T)  # Array to store interest rate
w_hat = np.zeros(T)  # Array to store wage
k_hat = np.zeros(T+1)  # Array to store capital
z_hat = np.zeros(T+1)  # Array to store technology process
R_hat = np.zeros(T)  

#Set a seed
np.random.seed(0)
#Generate a sequence of random normal shocks εt
e = np.random.normal(0, np.sqrt(sigma2), T+1)
k_hat[0] = 0.0  # Initial capital
z_hat[0] = 0.0  # Initial technology process

#Simulate the sequences with policy functions
for t in range(1, T):  # Change the loop range to T-1
    z_hat[t+1] = p * z_hat[t] + e[t+1]
    k_hat[t+1] = etakk * k_hat[t] + etakz * z_hat[t]
    i_hat[t] = (k_bar * k_hat[t+1] - k_bar * (1 - delta) * k_hat[t]) / i_bar
    R_hat[t] = z_hat[t] + (a - 1) * k_hat[t]
    r_hat[t] = R_hat[t] + R_hat[t] * delta / r_bar
    w_hat[t] = z_hat[t] + a * k_hat[t]
    y_hat[t] = z_hat[t] + a * k_hat[t]
    c_hat[t] = etack * k_hat[t] + etacz * z_hat[t]


#Calculate business cycle moments
std_y_hat = np.std(y_hat)
std_c_hat = np.std(c_hat)
std_i_hat = np.std(i_hat)
corr_r_y = np.corrcoef(r_hat, y_hat)[0, 1]
corr_w_y = np.corrcoef(w_hat, y_hat)[0, 1]

#Report results
print("\nBusiness Cycle Moments:")
print(f"Standard Deviation of Output (y_hat): {std_y_hat:.4f}")
print(f"Standard Deviation of Consumption (c_hat): {std_c_hat:.4f}")
print(f"Standard Deviation of Investment (i_hat): {std_i_hat:.4f}")
print(f"Standard Deviation of Consumption Relative to Standard Deviation of Output (y_hat): {std_c_hat/std_y_hat:.4f}")
print(f"Standard Deviation of Investment Relative to Standard Deviation of Output (y_hat): {std_i_hat/std_y_hat:.4f}")
print(f"Correlation between Interest Rate (r_hat) and Output (y_hat): {corr_r_y:.4f}")
print(f"Correlation between Wage (w_hat) and Output (y_hat): {corr_w_y:.4f}")

#Plot the sequences
plt.figure(figsize=(12, 6))

plt.plot(y_hat, label='Output')
plt.plot(r_hat, label='Interest Rate')
plt.plot(w_hat, label='Wage')
plt.title('Output, Interest Rate, and Wage over Time')
plt.legend()
plt.tight_layout()
plt.show()

#Plot the sequences
plt.figure(figsize=(12, 6))

plt.plot(y_hat, label='Output')
plt.plot(c_hat, label='Consumption')
plt.plot(i_hat, label='Investment')
plt.title('Output, Consumption and Investment Over Time')
plt.legend()
plt.tight_layout()
plt.show()

###############################################################################
##################################Part B#######################################
###############################################################################

#Change time period to 300, 1000 is not needed to visualise the IRF's
T = 300

#Reset arrays
y_hat = np.zeros(T)  # Array to store output
c_hat = np.zeros(T)  # Array to store consumption
i_hat = np.zeros(T)  # Array to store investment
r_hat = np.zeros(T)  # Array to store interest rate
w_hat = np.zeros(T)  # Array to store wage
k_hat = np.zeros(T+1)  # Array to store capital
z_hat = np.zeros(T+1)  # Array to store technology process
R_hat = np.zeros(T) 


#Generate a sequence of positive TFP shock
tfp_shock_period = 2  #Shock Period at 2 due to the nature of e in the equations
e = np.zeros(T+1)
e[tfp_shock_period] = 1*sigma  # 1 standard deviation positive TFP shock
k_hat[0] = 0.0  # Initial capital
z_hat[0] = 0.0  # Initial technology process


#Simulate the sequences with policy functions and positive TFP shock
for t in range(1, T):  # Change the loop range to T-1
    z_hat[t+1] = p * z_hat[t] + e[t+1]
    k_hat[t+1] = etakk * k_hat[t] + etakz * z_hat[t]
    i_hat[t] = (k_bar * k_hat[t+1] - k_bar * (1 - delta) * k_hat[t]) / i_bar
    R_hat[t] = z_hat[t] + (a - 1) * k_hat[t]
    r_hat[t] = R_hat[t] + R_hat[t] * delta / r_bar
    w_hat[t] = z_hat[t] + a * k_hat[t]
    y_hat[t] = z_hat[t] + a * k_hat[t]
    c_hat[t] = etack * k_hat[t] + etacz * z_hat[t]

#Create the first figure with three subplots
plt.figure(figsize=(15, 6))

#Subplot 1: Technology Shock
plt.subplot(2, 3, 1)
plt.plot(z_hat, label='Technology Shock (z_hat)')
plt.title('Technology Shock - TFP Shock')

#Subplot 2: Output
plt.subplot(2, 3, 2)
plt.plot(y_hat, label='Output (y_hat)')
plt.title('Output - TFP Shock')

#Subplot 3: Consumption
plt.subplot(2, 3, 3)
plt.plot(c_hat, label='Consumption (c_hat)')
plt.title('Consumption - TFP Shock')

plt.tight_layout()
plt.show()

#Create the second figure with three subplots
plt.figure(figsize=(15, 6))

#Subplot 4: Investment
plt.subplot(2, 3, 4)
plt.plot(i_hat, label='Investment (i_hat)')
plt.title('Investment - TFP Shock')

#Subplot 5: Interest Rate
plt.subplot(2, 3, 5)
plt.plot(r_hat, label='Interest Rate (r_hat)')
plt.title('Interest Rate - TFP Shock')

#Subplot 6: Wage
plt.subplot(2, 3, 6)
plt.plot(w_hat, label='Wage (w_hat)')
plt.title('Wage - TFP Shock')

plt.tight_layout()
plt.show()

###############################################################################
##################################Part C#######################################
###############################################################################


#Reset arrays for negative TFP shock
y_hat = np.zeros(T)
c_hat = np.zeros(T)
i_hat = np.zeros(T)
r_hat = np.zeros(T)
w_hat = np.zeros(T)
k_hat = np.zeros(T+1)
z_hat = np.zeros(T+1)
R_hat = np.zeros(T)


#Generate a sequence of negative TFP shock
tfp_shock_period = 2 
e = np.zeros(T+1)
e[tfp_shock_period] = -1 * sigma # 1 standard deviation negative TFP shock
k_hat[0] = 0.0  # Initial capital
z_hat[0] = 0.0  # Initial technology process

#Simulate the sequences with policy functions
#Simulate the sequences with policy functions
for t in range(1, T):  # Change the loop range to T-1
    z_hat[t+1] = p * z_hat[t] + e[t+1]
    k_hat[t+1] = etakk * k_hat[t] + etakz * z_hat[t]
    i_hat[t] = (k_bar * k_hat[t+1] - k_bar * (1 - delta) * k_hat[t]) / i_bar
    R_hat[t] = z_hat[t] + (a - 1) * k_hat[t]
    r_hat[t] = R_hat[t] + R_hat[t] * delta / r_bar
    w_hat[t] = z_hat[t] + a * k_hat[t]
    y_hat[t] = z_hat[t] + a * k_hat[t]
    c_hat[t] = etack * k_hat[t] + etacz * z_hat[t]



#Plot the IRFs to negative TFP shock
plt.figure(figsize=(15, 6))

#Subplot 1: Technology Shock (Negative)
plt.subplot(2, 3, 1)
plt.plot(z_hat, label='Technology Shock (z_hat)')
plt.title('IRF to Negative TFP Shock - Technology Shock')
plt.legend()

#Subplot 2: Output (Negative)
plt.subplot(2, 3, 2)
plt.plot(y_hat, label='Output (y_hat)')
plt.title('IRF to Negative TFP Shock - Output')
plt.legend()

#Subplot 3: Consumption (Negative)
plt.subplot(2, 3, 3)
plt.plot(c_hat, label='Consumption (c_hat)')
plt.title('IRF to Negative TFP Shock - Consumption')
plt.legend()

plt.tight_layout()
plt.show()

#Create the second figure with three subplots
plt.figure(figsize=(15, 6))
#Subplot 4: Investment (Negative)
plt.subplot(2, 3, 4)
plt.plot(i_hat, label='Investment (i_hat)')
plt.title('IRF to Negative TFP Shock - Investment')
plt.legend()

#Subplot 5: Interest Rate (Negative)
plt.subplot(2, 3, 5)
plt.plot(r_hat, label='Interest Rate (r_hat)')
plt.title('IRF to Negative TFP Shock - Interest Rate')
plt.legend()

#Subplot 6: Wage (Negative)
plt.subplot(2, 3, 6)
plt.plot(w_hat, label='Wage (w_hat)')
plt.title('IRF to Negative TFP Shock - Wage')
plt.legend()

plt.tight_layout()
plt.show()


