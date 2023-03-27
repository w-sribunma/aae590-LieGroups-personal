import math
import numpy as np
import casadi as ca
from casadi.tools.graph import graph
import os
import pydot

eps = 1e-7 # to avoid divide by zero

'''
SO2Lie with applications of casadi
'''

class so2: #so2 Lie Algebra Function
    group_shape = (2, 2)
    group_params = 4
    algebra_params = 1

    def __init__(self, SO2=None):
        pass

    @classmethod
    def check_group_shape(cls, a):
        assert a.shape == cls.group_shape or a.shape == (cls.group_shape[0],)

    @classmethod #takes 6x1 lie algebra
    def ad_matrix(cls, v): #input vee operator [theta]
        ad_so2 = ca.SX(2, 2)
        ad_so2[0,1] = -v[0]
        ad_so2[1,0] = v[0] 
        return ad_so2

    @classmethod
    def check_group_shape(cls, a):
        assert a.shape == cls.group_shape or a.shape == (cls.group_shape[0],)

    @classmethod
    def vee(cls, X):
        '''
        This takes in an element of the SO2 Lie Group (Wedge Form) and returns the so2 Lie Algebra elements 
        '''
        # print(type(X))
        # print(X.shape)
        v = ca.SX.zeros(1) #[theta]
        v[0] = X[0] #theta
        return v

    @classmethod
    def wedge(cls, v): #input v = [x,y,theta]
        '''
        This takes in an element of the so2 Lie Algebra and returns the so2 Lie Algebra matrix
        '''
        X = ca.SX.zeros(2, 2)
        X[0, 1] = -v[0]
        X[1, 0] = v[0]
        return X        

    @classmethod
    def matmul(cls, a, b):
        cls.check_group_shape(a)
        cls.check_group_shape(b)
        return ca.mtimes(a, b)

    @classmethod
    def exp(cls, v): #accept input in wedge operator form
        v = cls.vee(v)
        theta = v[0]
        
        if type(v[0]) == 'int' and theta < eps:
            a = theta - theta ** 3 / 6 + theta ** 5 / 120
            b = 1 - theta **2 / 2 + theta **4 / 24 - theta **6 / 720
        else:
            a = ca.sin(theta)
            b = ca.cos(theta)
 
        R = ca.SX(2,2) #Exp(wedge(theta))
        R[0,0] = b
        R[0,1] = -a
        R[1,0] = a
        R[1,1] = b
        
        return R


class SO2:
    group_shape = (2, 2)
    group_params = 4
    algebra_params = 1

    def __init__(self, theta):
        # cls.theta = ca.SX((theta),())
        self.theta = ca.SX([theta])
        # pass

    @classmethod
    def check_group_shape(cls, a):
        assert a.shape == cls.group_shape or a.shape == (cls.group_shape[0],)
    
    @classmethod
    def one(cls):
        return ca.SX.zeros(1, 1)
    
    @classmethod
    def product(cls, a, b):
        cls.check_group_shape(a)
        cls.check_group_shape(b)
        return ca.mtimes(a, b)

    @classmethod
    def inv(cls, a): #input a matrix of ca.SX form
        cls.check_group_shape(a)
        # theta = a[0]
        # a_inv = ca.SX.zeros(2,2)
        a_inv = a.T #check transpose
        return a_inv
    
    @classmethod
    def neutral(cls, a): #neutral element
        return ca.SX.eye(2)
    
    @classmethod
    def log(self, a):
        cos_th = a[0,0] # 'b' term for cos(theta)
        theta = ca.acos(cos_th)
        return so2.wedge(theta)


def dot_plot_draw(u, **kwargs):
    F = ca.sparsify(u)

    output_dir = '/home/wsribunm/Documents/GitHub/aae590-LieGroups-personal/fig' #change directory if needed
    os.makedirs(output_dir, exist_ok=True)
    g = graph.dotgraph(F)
    g.set('dpi', 180)
    g.write_png(os.path.join(output_dir, 'result_test.png'))