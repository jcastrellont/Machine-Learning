
##LABORATORIO 1

#Generating some training data
Each indiviudal has two atributes (x0,x1)

-x0 is heigth and

-x1 is weight

There are three clases of individuals

-Babies,

-Teens and

-adults

We generate training examples

50 babies ~ Gaussian([60, 30], [[50,15],[15, 30]])

70 teens ~ Gaussian([120, 50], [[90,10 ],[10, 40]])

120 adults ~ Gaussian([150, 80], [[60, 10],[10, 40]])


```python
import numpy as np
import matplotlib.pyplot as plt

babies = np.random.multivariate_normal([60, 30], [[50,15],[15, 30]], 50)
teens = np.random.multivariate_normal([120, 50], [[90,10],[10, 40]], 70)
adults = np.random.multivariate_normal([150, 80], [[60, 10],[10, 40]], 120)
x = np.concatenate((babies,teens,adults),axis=0)

y1 = np.empty(len(babies))
y2= np.empty(len(teens))
y3 = np.empty(len(adults))
y1.fill(0)
y2.fill(1)
y3.fill(2)
y = np.concatenate((y1,y2,y3),axis=0)

color= ['red' if v == 1 else  'blue' if v == 0 else 'green' for v in y]
plt.scatter(x[:,0], x[:,1],   c=color, s=50)
plt.title('Data Set', size=16);
```


![png](output_2_0.png)



```python
fig, ax = plt.subplots()
color= ['red' if v == 1 else  'blue' if v == 0 else 'green' for v in y]
ax.scatter(x[:,0], x[:,1],  c=color, s=50);
ax.set_title('Naive Bayes Model', size=14)
xlim = (10, 180)
ylim = (10, 110)

xg = np.linspace(xlim[0], xlim[1], 60)
yg = np.linspace(ylim[0], ylim[1], 40)
xx, yy = np.meshgrid(xg, yg)
Xgrid = np.vstack([xx.ravel(), yy.ravel()]).T

for label, color in enumerate(['blue', 'red', 'green']):
    mask = (y == label)
    mu, std = x[mask].mean(0), x[mask].std(0)
    P = np.exp(-0.5 * (Xgrid - mu) ** 2 / std ** 2).prod(1)
    Pm = np.ma.masked_array(P, P < 0.03)
    ax.pcolorfast(xg, yg, Pm.reshape(xx.shape), alpha=0.5,
                  cmap=color.title() + 's')
    ax.contour(xx, yy, P.reshape(xx.shape),
               levels=[0.01, 0.1, 0.5, 0.9],
               colors=color, alpha=0.2)
    
ax.set(xlim=xlim, ylim=ylim);
```


![png](output_3_0.png)



```python
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(x, y);
```

# The desicion boundary in Gaussian naive Bayes is quadratic.


```python
x0 =  np.linspace(80,180,101)
x1 =  np.linspace(5,120,101)
x2 =  np.linspace(20,120,101)
g = np.meshgrid(x0,x1,x2)
xmesh = np.append(g[0].reshape(-1,1),g[1].reshape(-1,1),axis=1)
ymesh= model.predict(xmesh)

colormesh= ['red' if v == 1 else  'blue' if v == 0 else 'green' for v in ymesh]
plt.scatter(xmesh[:, 0], xmesh[:, 1], c=colormesh )
plt.title('Desicion Boundary', size=14);
```


![png](output_6_0.png)


# Testing the model with some random data


```python
rng = np.random.RandomState(0)
xnew = [80, 20] + [100, 100] * rng.rand(500, 2) 
ynew = model.predict(xnew)
```


```python
color= ['red' if v == 1 else  'blue' if v == 0 else 'green' for v in y]
plt.scatter(x[:,0].tolist(), x[:,1].tolist(),  c=color, s=50);
lim = plt.axis()

colornew= ['red' if v == 1 else  'blue' if v == 0 else 'green' for v in ynew]
plt.scatter(xnew[:, 0], xnew[:, 1], c=colornew, s=30,  alpha=0.3)
plt.axis(lim);
```


![png](output_9_0.png)


#Pobabilistic classification - posterior probabilities for the classes


```python
xnew[-20:].round(2)
```




    array([[ 102.01,   82.29],
           [  91.15,   65.9 ],
           [ 112.23,   51.65],
           [ 128.26,   92.98],
           [  86.92,  107.92],
           [ 153.48,   37.65],
           [ 173.92,   70.63],
           [ 179.98,   39.73],
           [ 133.49,   49.02],
           [ 110.42,   79.11],
           [ 172.17,  100.53],
           [ 152.39,   75.92],
           [ 172.23,   69.24],
           [ 167.38,  103.4 ],
           [ 101.38,   97.12],
           [  81.22,   52.28],
           [ 102.96,   70.69],
           [ 153.69,   29.77],
           [ 131.49,  113.84],
           [ 102.86,   87.71]])




```python
yprob = model.predict_proba(xnew[-20:])
yprob.round(2)
```




    array([[ 0.  ,  0.9 ,  0.1 ],
           [ 0.  ,  1.  ,  0.  ],
           [ 0.  ,  1.  ,  0.  ],
           [ 0.  ,  0.  ,  1.  ],
           [ 0.  ,  0.  ,  1.  ],
           [ 0.  ,  1.  ,  0.  ],
           [ 0.  ,  0.  ,  1.  ],
           [ 0.  ,  1.  ,  0.  ],
           [ 0.  ,  1.  ,  0.  ],
           [ 0.  ,  0.53,  0.47],
           [ 0.  ,  0.  ,  1.  ],
           [ 0.  ,  0.  ,  1.  ],
           [ 0.  ,  0.  ,  1.  ],
           [ 0.  ,  0.  ,  1.  ],
           [ 0.  ,  0.  ,  1.  ],
           [ 0.03,  0.97,  0.  ],
           [ 0.  ,  1.  ,  0.  ],
           [ 0.  ,  1.  ,  0.  ],
           [ 0.  ,  0.  ,  1.  ],
           [ 0.  ,  0.09,  0.91]])


