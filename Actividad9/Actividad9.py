
# coding: utf-8

# In[67]:

import numpy as np
import math
from numpy import pi
import matplotlib.pyplot as plt
#Parametros

g = 9.806
l = 1.00
n = 500
e = 0.001


#Valores iniciales

th0 = np.linspace(e, pi-e, n)



I = [0 for i in range(n)]
I0 = [0 for i in range(n)]
T= [0 for i in range(n)]
sine = [0 for i in range(n)]
er = [0 for i in range(n)]



#Periodo para angulos pequeños
T0 = 2.0*(pi)*(np.sqrt(l/g))

#calcular la serie de potencias
L = 2 #Terminos de la serie 
for i in range(L):
    for j in range(0,n):
        
        fac1 = float(math.factorial(2*(i)))
        fac2 = float((2**(i)*math.factorial(i))**2)
        sine[j] = np.sin(th0[j]/2)**(2*(i))
        I[j] = ((fac1/fac2)**2)*sine[j]
        I0[j] = I0[j] + I[j]
        T[j] = 2.0*(pi)*(np.sqrt(l/g)*I0[j])
        er[j] = (T[j]/T0)
#Grafica
plt.plot(theta0, er, 'purple', label="T2")
plt.xlim(0,np.radians(180))
plt.xlabel("Angulo en radianes")
plt.ylabel("Error Relativo")
plt.title("Error relativo usando una serie de potencias ")
plt.legend(loc='best')
plt.show()
plt.grid()

