
# coding: utf-8

# In[2]:


#modelo de infeccion latente
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
plt.ion()
plt.rcParams['figure.figsize'] = 10, 8


P = 0        # Nacimientos 
D = 0.0001  # Muertes Naturales 
B = 0.0095  # Transmision   
G = 0.0001  # Removidos         
A = 0.005   # Destruidos        
o = 0.0001  #infectados
#sistema de eacuaciones
def f(y, t):
    Si = y[0]
    Zi = y[1]
    Ri = y[3]
    Ii = y[2]

    f0 = P - B*Si*Zi - d*Si
    f1 = B*Si*Zi + G*Ri - A*Si*Zi
    f3 = B*Si*Zi-o-d*Ii
    f2 = d*Si+d*Ii+ A*Si*Zi - G*Ri
    
    return [f0, f1, f2,f3]

# initial conditions
S0 = 500.                        # Poblacion Inicial
Z0 = 0                           # Zombie Inicial
R0 = 0                           # Muertos Inicial
y0 = [S0, Z0, R0]                # Condicion Inicial
t  = np.linspace(0, 14., 1000)   #Tiempo

# solucion
soln = odeint(f, y0, t)
S = soln[:, 0]
Z = soln[:, 1]
R = soln[:, 2]
I = soln[:, 3]
# plot 
plt.figure()
plt.ylim(0,600)
plt.grid(True)
plt.plot(t, S, label='Vivos')
plt.plot(t, Z, label='Zombies')
plt.xlabel('Tiempo')
plt.ylabel('Poblacion')
plt.title('Apocalipsis Zombie - Infeccion latente')
plt.legend(loc="best")



# In[ ]:

#modelo con cuerentena
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
plt.ion()
plt.rcParams['figure.figsize'] = 10, 8

P = 0         # Nacimientos 
D = 0.0001   # Muertes Naturales 
B = 0.0095   # Transmision    
G = 0.0001   # Removidos         
A = 0.0001   # Destruidos        
o = 0.05     # Infected          
K = 0.15     # Infectados 
S = 0.10     # Infected          
M = 0.001    # Infected          

#Sistema de Ecuaciones
def f(y, t):
    Si = y[0]
    Zi = y[1]
    Ri = y[2]
    Ii = y[3]
    Qi = y[4]
    # Modelo
    f0 = P - *Si*Zi - D*Si                        
    f1 = R*Ii + G*Ri - A*Si*Zi - S*Zi           
    f2 = Del*Si + Del*Ii + Alf*Si*Zi - G*Ri + M*Qi  
    f3 = B*Si*Zi -oIi - D*Ii - K*Ii            
    f4 = K*Ii + S*Zi - M*Qi                       
    return [f0, f1, f2, f3, f4]

S0 = 500.                        # Poblacion Inicial
Z0 = 0.                          # Zombie Inicial
R0 = 0.                          # Muertos Inicial
I0 = 1.                          # Infectados Inicial
Q0 = 0.                            # Cuarentena Inicial
y0 = [S0, Z0, R0, I0, Q0]        # Condiciones Iniciales
t  = np.linspace(0., 30., 1000)  # Tiempo

# Solucion 
soln = odeint(f, y0, t)
S = soln[:, 0]
Z = soln[:, 1]
R = soln[:, 2]
I = soln[:, 3]
Q = soln[:, 4]
# plot
plt.figure()
plt.ylim(0,600)
plt.grid(True)
plt.plot(t, S, label='Vivos')
plt.plot(t, Z, label='Zombies')
plt.xlabel('Tiempo')
plt.ylabel('Poblacion')
plt.title('Apocalipsis Zombie - Modelo Caurentena.')
plt.legend(loc="best")


# In[ ]:

#modelo con cura
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
plt.ion()
plt.rcParams['figure.figsize'] = 10, 8

  = 0        # Nacimientos 
D = 0.0001   # Muertes Naturales
B = 0.0095   # Transmision       
G = 0.0001   # Removidos    
A = 0.0001   # Destruidos        
o = 0.05     # Infected         
C  = 0.05     # Cura              

#Sistema de Ecuaciones 
def f(y, t):
    Si = y[0]
    Zi = y[1]
    Ri = y[2]
    Ii = y[3]
    # Modelo
    f0 = P - B*Si*Zi - D*Si +C*Zi             
    f1 = Ii + G*Ri - Alf*Si*Zi -C*Zi         
    f2 = D*Si + D*Ii + A*Si*Zi - G*Ri       
    f3 = B*Si*Zi -o*Ii - D*Ii                 
    
    return [f0, f1, f2, f3]

S0 = 500.                        # Poblacion Inicial
Z0 = 0.                          # Zombie Inicial
R0 = 0.                          # Muertos Inicial
I0 = 1.                          # Infectados Inicial
y0 = [S0, Z0, R0, I0]            # Condiciones Iniciales
t  = np.linspace(0., 30., 1000)  # Tiempo

# Solucion 
soln = odeint(f, y0, t)
S = soln[:, 0]
Z = soln[:, 1]
R = soln[:, 2]
I = soln[:, 3]
# plot
plt.figure()
plt.ylim(0,500)
plt.grid(True)
plt.plot(t, S, label='Vivos')
plt.plot(t, Z, label='Zombies')
plt.xlabel('Tiempo')
plt.ylabel('Poblacion')
plt.title('Apocalipsis Zombie - Modelo Tratamiento.')
plt.legend(loc="best")


# In[ ]:

#Erradicacion
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
plt.ion()
plt.rcParams['figure.figsize'] = 10, 8

P  = 0        # Nacimientos 
D = 0.0001   # Muertes Naturales 
B = 0.0055   # Transmision       
G = 0.0900   # Removidos         
A = 0.0075   # Destruidos      
k= 0.25
n=4

# solve the system dy/dt = f(y, t)
def f(y, t):
    Si = y[0]
    Zi = y[1]
    Ri = y[2]
    # Modelo
    f0 = P - B*Si*Zi - D*Si                 
    f1 = B*Si*Zi + G*Ri - Alf*Si*Zi          
    f2 = D*Si + A*Si*Zi - Z*Ri              
    f3 = -k*n*Zi                                 
    
    return [f0, f1, f2, f3]

# initial conditions
S0 = 500.                        # Poblacion Inicial
Z0 = 0.                          # Zombie Inicial
R0 = 0.                          # Muertos Inicial
DZ0 = 0.                          # Infectados Inicial
y0 = [S0, Z0, R0, DZ0]            # Condiciones Iniciales
t  = np.linspace(0., 130., 1000)  # Tiempo

# 
soln = odeint(f, y0, t)
S = soln[:, 0]
Z = soln[:, 1]
R = soln[:, 2]
I = soln[:, 3]
# plot results
plt.figure()
plt.ylim(0,500)
plt.grid(True)
plt.plot(t, S, label='Vivos')
plt.plot(t, Z, label='Zombies')
plt.xlabel('Tiempo')
plt.ylabel('Poblacion')
plt.title('Apocalipsis Zombie - Modelo Erradicacion.')
plt.legend(loc="best")


# In[ ]:

#modelo basico
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
plt.ion()
plt.rcParams['figure.figsize'] = 10, 8

P = 0        # Nacimientos 
D = 0.0001  # Muertes Naturales
B = 0.0095  # Transmision       
G = 0.0001  # Removidos         
A = 0.005   # Destruidos        

#Sistema de Ecuaciones
def f(y, t):
    Si = y[0]
    Zi = y[1]
    Ri = y[2]
    # Modelo 
    f0 = P - B*Si*Zi - D*Si              
    f1 = B*Si*Zi + G*Ri - A*Si*Zi       
    f2 = D*Si + A*Si*Zi - Z*Ri          
    return [f0, f1, f2]

S0 = 500.                        # Poblacion Inicial
Z0 = 0                           # Zombie Inicial
R0 = 0                           # Muertos Inicial
y0 = [S0, Z0, R0]                # Condicion Inicial
t  = np.linspace(0, 14., 1000)   #Tiempo

# Solucion 
soln = odeint(f, y0, t)
S = soln[:, 0]
Z = soln[:, 1]
R = soln[:, 2]
# plot
plt.figure()
plt.ylim(0,600)
plt.grid(True)
plt.plot(t, S, label='Vivos')
plt.plot(t, Z, label='Zombies')
plt.xlabel('Tiempo')
plt.ylabel('Poblacion')
plt.title('Apocalipsis Zombie - Modelo Basico')
plt.legend(loc="best")

