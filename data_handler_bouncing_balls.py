
"""
This script comes from the RTRBM code by Ilya Sutskever from 
http://www.cs.utoronto.ca/~ilya/code/2008/RTRBM.tar
"""

import sys
import numpy as np
import scipy
import os
import utils

shape_std=scipy.shape
def shape(A):
    if isinstance(A, scipy.ndarray):
        return shape_std(A)
    else:
        return A.shape()

size_std = scipy.size
def size(A):
    if isinstance(A, scipy.ndarray):
        return size_std(A)
    else:
        return A.size()

det = np.linalg.det

def new_speeds(m1, m2, v1, v2):
    new_v2 = (2*m1*v1 + v2*(m2-m1))/(m1+m2)
    new_v1 = new_v2 + (v2 - v1)
    return new_v1, new_v2
    

def norm(x): return scipy.sqrt((x**2).sum())
def sigmoid(x):        return 1./(1.+scipy.exp(-x))

SIZE=10
# size of bounding box: SIZE X SIZE.

def bounce_n(T=128, n=2, r=None, m=None):
    if r is None: r=scipy.array([1.2]*n)
    if m==None: m=scipy.array([1]*n)
    # r is to be rather small.
    X=scipy.zeros((T, n, 2), dtype='float')
    v = scipy.randn(n,2)
    v = v / norm(v)*.5
    good_config=False
    while not good_config:
        x = 2+scipy.rand(n,2)*8
        good_config=True
        for i in range(n):
            for z in range(2):
                if x[i][z]-r[i]<0:      good_config=False
                if x[i][z]+r[i]>SIZE:     good_config=False

        # that's the main part.
        for i in range(n):
            for j in range(i):
                if norm(x[i]-x[j])<r[i]+r[j]:
                    good_config=False
                    
    
    eps = .5
    for t in range(T):
        # for how long do we show small simulation

        for i in range(n):
            X[t,i]=x[i]
            
        for _ in range(int(1/eps)):

            for i in range(n):
                x[i]+=eps*v[i]

            for i in range(n):
                for z in range(2):
                    if x[i][z]-r[i]<0:  v[i][z]= abs(v[i][z]) # want positive
                    if x[i][z]+r[i]>SIZE: v[i][z]=-abs(v[i][z]) # want negative


            for i in range(n):
                for j in range(i):
                    if norm(x[i]-x[j])<r[i]+r[j]:
                        # the bouncing off part:
                        w    = x[i]-x[j]
                        w    = w / norm(w)

                        v_i  = scipy.dot(w.transpose(),v[i])
                        v_j  = scipy.dot(w.transpose(),v[j])

                        new_v_i, new_v_j = new_speeds(m[i], m[j], v_i, v_j)
                        
                        v[i]+= w*(new_v_i - v_i)
                        v[j]+= w*(new_v_j - v_j)

    return X

def ar(x,y,z):
    return z/2+scipy.arange(x,y,z,dtype='float')

def matricize(X,res,r=None):

    T, n= shape(X)[0:2]
    if r is None: r=scipy.array([1.2]*n)

    A=scipy.zeros((T,res,res), dtype='float')
    
    [I, J]=scipy.meshgrid(ar(0,1,1./res)*SIZE, ar(0,1,1./res)*SIZE)

    for t in range(T):
        for i in range(n):
            A[t]+= scipy.exp(-(  ((I-X[t,i,0])**2+(J-X[t,i,1])**2)/(r[i]**2)  )**4    )
            
        A[t][A[t]>1]=1
    return A

def bounce_vec(res, n=2, T=128, r =None, m =None):
    if r==None: r=scipy.array([1.2]*n)
    x = bounce_n(T,n,r,m);
    V = matricize(x,res,r)
    return V.reshape(T, res**2)  

if __name__ == "__main__":
    res=80
    T=20
    N=100
    M=200
    target_dir = os.path.join('data', 'bouncing_balls_diff0p0_bae_0p0')
    os.makedirs(target_dir, exist_ok=True)
    nrof_balls = 1
    for j in range(M):
        print('.', end='')
        sys.stdout.flush()
        dat=scipy.empty((N), dtype=object)
        for i in range(N):
            dat[i]=bounce_vec(res=res, n=nrof_balls, T=T)
        data = np.reshape(scipy.stack(dat), (N, T, res, res))
        utils.save_pickle(os.path.join(target_dir, 'train_%03d.pkl' % j), data)
    print('\nDone')
    
    N=100
    M=10
    dat=scipy.empty((N), dtype=object)
    for j in range(M):
        for i in range(N):
            dat[i]=bounce_vec(res=res, n=nrof_balls, T=T)
        data = np.reshape(scipy.stack(dat), (N, T, res, res))
        utils.save_pickle(os.path.join(target_dir, 'test_%03d.pkl' % j), data)

