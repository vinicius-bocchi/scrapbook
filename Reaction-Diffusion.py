from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
from scipy.interpolate import griddata
import datetime

def test_function_1(x,y):
    return x**2+y**2

def test_function_2(x,y):
    return x+y

def test_function_3(x,y):
    return 10*(y*math.sin(x))

def test_function_4(x,y):
    return 100*math.sin(13*math.pi*x)*math.sin(5*math.pi*y)

def PrintMatrix(A,dim1,dim2):
    print(" ") 
    for i in range (0,dim1):
        for j in range (0,dim2):
            print (A[i,j], end=" ")
        print(" ")
    print(" ")

def PrintVector(v,dim):
    print(" ")
    for i in range(0,dim):
        print(v[i], end=" ")
    print(" ")
    
def Solve1(A,v,g,v_prior,dim,epsilon,h):
    begin_time = datetime.datetime.now()
    aval=8*epsilon/(h**2)+1
    for k in range(1,100000):
        tau=1/((aval+1)/2+((aval-1)/2)*math.cos((math.pi*(2*k-1)/4)))
        v=v_prior+tau*(g-np.matmul(A,v_prior))
        v_prior=v
        r=np.matmul(A,v)-g
        if math.sqrt(np.matmul(r,r)/len(v))<0.01:
            print("Number of iterations Method1 = ",k)
            print("Execution time Method1 = ",datetime.datetime.now() - begin_time)
            print("Error for Method1 = ",math.sqrt(np.matmul(r,r)/len(v)))
            break
    return v

def Solve2(A,v,g,v_prior,dim):
    begin_time = datetime.datetime.now()
    beta=1
    r=np.matmul(A,v_prior)-g
    tau=np.matmul(r,r)/np.matmul(np.matmul(A,r),r)
    v=v_prior-tau*r
    
    v_prior_prior=v_prior
    v_prior=v
    tau_prior=tau
    beta_prior=beta
    r_prior=r
    r=np.matmul(A,v)-g
    for k in range(1,1000):
        tau=np.matmul(r,r)/np.matmul(np.matmul(A,r),r)
        beta=1/(1-(tau/tau_prior)*np.matmul(r,r)/np.matmul(r_prior,r_prior)*(1/beta_prior))
        v=beta*v_prior-beta*tau*np.matmul(A,v_prior)+(1-beta)*v_prior_prior+beta*tau*g
        v_prior_prior=v_prior
        v_prior=v
        tau_prior=tau
        beta_prior=beta
        r_prior=r
        r=np.matmul(A,v)-g
        if math.sqrt(np.matmul(r,r)/len(v))<0.01:
            print("Number of iterations Method2 = ",k)
            print("Execution time Method2 = ",datetime.datetime.now() - begin_time)
            print("Error for Method2 = ",math.sqrt(np.matmul(r,r)/len(v)))
            break
    return v

#Data entry from user
N=20
f=test_function_4
epsilon=1
alpha=0

#Creating variables for the solver
h=1/N
dim=(N-1)*(N-1)
v=np.zeros(dim)
v_0=np.zeros(dim)
A=np.zeros((dim,dim))
func_vec=np.zeros(dim)
v_1=np.zeros(dim)
v_2=np.zeros(dim)

#Filling the coefficient matrix
for i in range(0,dim):
    A[i,i]=4*epsilon/(h**2)+1
for i in range(1,dim-1):
    A[i-1,i]=-epsilon/(h**2)
    A[i,i-1]=-epsilon/(h**2)
for i in range(1,N-1):
    for j in range(0,N-1):
        A[(N-1)*i+j,(N-1)*(i-1)+j]=-epsilon/(h**2)
for i in range(1,N-1):
    for j in range(0,N-1):
        A[(N-1)*(i-1)+j,(N-1)*i+j]=-epsilon/(h**2)
        
#Filling the inicial iteration value
for i in range(0,dim):
    v_0[i]=alpha
v_prior1=v_0
v_prior2=v_0
    
#Filling function values
for i in range(0,N-1):
    for j in range(0,N-1):
        func_vec[(N-1)*i+j]=f((i+1)*h,(j+1)*h)
for j in range(0,N-1):
    func_vec[j]+=epsilon*alpha/(h**2)
    func_vec[(N-2)*(N-1)+j]+=epsilon*alpha/(h**2)
    func_vec[(N-1)*j]+=epsilon*alpha/(h**2)
    func_vec[(N-2)+(N-1)*j]+=epsilon*alpha/(h**2)
    

#Checking matrix A
#PrintMatrix(A,dim,dim)
  
#Checking v_prior
#PrintVector(v_0,dim)

#Checking free coefficients vector
#PrintVector(func_vec,dim)

#Solution for first method
v_1=Solve1(A,v,func_vec,v_prior1,dim,epsilon,h)
#PrintVector(v_1,dim)

#Solution for second method
v_2=Solve2(A,v,func_vec,v_prior2,dim)
#PrintVector(v_2,dim)

#Data conversion for plotting
dim_ext=(N+1)*(N+1)
X=np.zeros(dim_ext)
Y=np.zeros(dim_ext)
Z=np.zeros(dim_ext)

for k in range(0,dim_ext):
    X[k]=(k%(N+1))*h
    Y[k]=(k//(N+1))*h
    
for i in range(0,N+1):
    for j in range(0,N+1):
        if (i==0 or i==N or j==0 or j==N):
            Z[(N+1)*i+j]=alpha
        else:
            Z[(N+1)*i+j]=v_1[(N-1)*(i-1)+j-1]
        
        
#PrintVector(X,dim_ext)
#PrintVector(Y,dim_ext)
#PrintVector(Z,dim_ext)

#Conversion to dataframe for plotting
xyz = {'x': X, 'y': Y, 'z': Z}
df = pd.DataFrame(xyz, index=range(len(xyz['x'])))
x1 = np.linspace(df['x'].min(), df['x'].max(), len(df['x'].unique()))
y1 = np.linspace(df['y'].min(), df['y'].max(), len(df['y'].unique()))
x2, y2 = np.meshgrid(x1, y1)
z2 = griddata((df['x'], df['y']), df['z'], (x2, y2), method='linear')

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x2, y2, z2, rstride=1, cstride=1, cmap=cm.coolwarm,
    linewidth=0, antialiased=False)
ax.set_zlim(df['z'].min(), df['z'].max())

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title('Numerical Solution')


plt.show()

