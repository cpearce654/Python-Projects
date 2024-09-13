#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 17:38:11 2023

@author: craigpearce
"""
#Question 1

# Import Packages and read excel file
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_excel('Insert Data File', sheet_name='Data')
#Input File Name

# Change year to 2011
df = df[df.year == 2011]

#Create Subset with only these variables
df1 = df[['country','rgdpe','avh','cn','pop','ctfp']]
type(df1)

df1=df1.dropna()

#Question 2

# Find Real GDP per-capita and capital stock per-capita
df1['rgdppc'] = df1['rgdpe'] / df1['pop']
df1['cnpc'] = df1['cn'] / df1['pop']

#Create Scatterplot between Real GDP per-capita and Total Factor Production
plt.scatter(df1['ctfp'], df1['rgdppc']), plt.title('Relationship Between Real GDP per-capita and Total Factor Productivity'), plt.ylabel('Real GDP per-capita'), plt.xlabel('Total Factor Productivity')
plt.show()
#Create Scatterplot between Real GDP per-capita and Capital Stock per-capita
plt.scatter(df1['cnpc'], df1['rgdppc']), plt.title('Relationship Between Real GDP per-capita and Capital Stock per-capita'), plt.ylabel('Real GDP per-capita'), plt.xlabel('Capital Stock per-capita')
plt.show()
#Create Scatterplot between Real GDP per-capita and Average Hours Worked
plt.scatter(df1['avh'], df1['rgdppc']), plt.title('Relationship Between Real GDP per-capita and Average Hours Worked'), plt.ylabel('Real GDP per-capita'), plt.xlabel('Average Hours Worked')
plt.show()

#Question 3

#Create Variables

y = np.array(df1.rgdppc)
x1 = np.array(df1.ctfp)
x2 = np.array(df1.cnpc)
x3 = np.array(df1.avh)

#Convert to Logs
y = np.log(y)
x1 = np.log(x1)
x2 = np.log(x2)
x3 = np.log(x3)

#Reshape arrays
x1 = x1.reshape(61,1)
x2 = x2.reshape(61,1)
x3 = x3.reshape(61,1)
y = y.reshape(61,1)

#Create column of 1's
x4 = np.ones([61,1])

#Create X's by catenating with column of 1's
X1 = np.concatenate((x4,x1),axis = 1)
X2 = np.concatenate((x4,x2),axis = 1)
X3 = np.concatenate((x4,x3),axis = 1)

#REGRESSION 1: lnRGDPpc and lnctfp
#Find Coefficient
B1 = np.dot(np.linalg.inv(np.dot(X1.T,X1)),np.dot(X1.T,y))
#Find Predicted Values
predicted_values1 = np.dot(X1, B1)

#Plot Predicted Values with Earlier Scatterplot
plt.plot(x1, predicted_values1, label='Predicted')
plt.scatter(x1, y, label = 'Data')
plt.title('Relationship between Real GDP per-capita and Total Factor Productivity, with Predicted Values')
plt.xlabel('Total Factor Productivity')
plt.ylabel('Real GDP per-capita')
plt.legend(loc='lower right')
plt.show()



#REGRESSION 2: lnRGDPpc and lncnpc
#Find Coefficient
B2 = np.dot(np.linalg.inv(np.dot(X2.T,X2)),np.dot(X2.T,y))
#Find Predicted Values
predicted_values2 = np.dot(X2, B2)

plt.plot(x2, predicted_values2, label='Predicted')
plt.scatter(x2, y, label = 'Data')
plt.title('Relationship between Real GDP per-capita and Capital Stock per-capita, with Predicted Values')
plt.xlabel('Capital Stock per-capita')
plt.ylabel('Real GDP per-capita')
plt.legend(loc='lower right')
plt.show()



#REGRESSION 3: lnRGDPpc and lnavh
#Find Coefficient
B3 = np.dot(np.linalg.inv(np.dot(X3.T,X3)),np.dot(X3.T,y))
#Find Predicted Values
predicted_values3 = np.dot(X3, B3)

plt.plot(x3, predicted_values3, label='Predicted')
plt.scatter(x3, y, label = 'Data')
plt.title('Relationship between Real GDP per-capita and Average Hours Worked, with Predicted Values')
plt.xlabel('Average Hours Worked')
plt.ylabel('Real GDP per-capita')
plt.legend(loc='lower right')
plt.show()


