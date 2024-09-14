# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1. Import the required libraries.
2. Load the dataset.
3. Define X and Y array.
4. Define a function for costFunction,cost and gradient.
5. Define a function to plot the decision boundary. 6.Define a function to predict the 
   Regression value.
```
## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: karthikeyan m
RegisterNumber: 212223040088
*/
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("ex2data1.txt",delimiter=',')
X=data[:,[0,1]]
y=data[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
    return 1/(1+np.exp(-z))

plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    grad=np.dot(X.T,h-y)/X.shape[0]
    return J,grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

def cost (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    return J

def gradient (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    grad=np.dot(X.T,h-y)/X.shape[0]
    return grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    X_plot=np.c_[xx.ravel(),yy.ravel()]
    X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot=np.dot(X_plot,theta).reshape(xx.shape)
    
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()


plotDecisionBoundary(res.x,X,y)

prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
    X_train =np.hstack((np.ones((X.shape[0],1)),X))
    prob=sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)
np.mean(predict(res.x,X)==y)
```

## Output:
![image](https://github.com/Rahulv2005/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/152600335/89d5958a-fe3c-4625-822a-113646f95e66)
![image](https://github.com/Rahulv2005/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/152600335/c8516a0d-45e3-45a6-9bc7-a34c367fbb28)
![image](https://github.com/Rahulv2005/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/152600335/ea06c206-13d6-4d88-814a-ba2510687bfd)
![image](https://github.com/Rahulv2005/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/152600335/3b4886b6-6700-4a0c-8b7c-8228c3788a7d)
![image](https://github.com/Rahulv2005/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/152600335/291f254b-5238-4d1c-9f7a-8fdce029e3b0)
![image](https://github.com/Rahulv2005/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/152600335/8808f15c-bb61-44fa-ba5f-ff8cc8b2ae64)
![image](https://github.com/Rahulv2005/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/152600335/c6512723-3b63-4f6d-b8ff-1830279b3754)
![image](https://github.com/Rahulv2005/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/152600335/a3eb4154-7dd5-4edc-a634-29d56f9e4c18)
![image](https://github.com/Rahulv2005/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/152600335/77429d42-c607-40a9-a19e-e6ff99b8f3cd)
![image](https://github.com/Rahulv2005/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/152600335/6dc17a7f-5d2d-47f6-9abe-350093abece6)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

