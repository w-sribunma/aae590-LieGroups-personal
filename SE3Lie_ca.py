import casadi as ca
from pyecca.lie import so3, r3, se3
# from pyecca.lie.util import DirectProduct
from pyecca.lie.so3 import Quat, Dcm, Euler, Mrp
# from pyecca.lie.r3 import R3
# from pyecca.lie.se3 import SE3
# from pyecca.test import test_lie
import numpy as np
from casadi.tools.graph import graph, dotdraw
import os
import pydot
import math
from sympy import *


eps = 1e-7 # to avoid divide by zero

def so3_vee(X):
    v = ca.SX(3, 1)
    v[0, 0] = X[2, 1]
    v[1, 0] = X[0, 2]
    v[2, 0] = X[1, 0]
    return v


def so3_wedge(v):
    X = ca.SX(3, 3)
    X[0, 1] = -v[2]
    X[0, 2] = v[1]
    X[1, 0] = v[2]
    X[1, 2] = -v[0]
    X[2, 0] = -v[1]
    X[2, 1] = v[0]
    return X

'''
SE3Lie extension with applications of casadi
'''

class se3: #se3 Lie Algebra Functions
    group_shape = (6, 6)
    group_params = 12
    algebra_params = 6

    # coefficients 
    x = ca.SX.sym('x')
    C1 = ca.Function('a', [x], [ca.if_else(ca.fabs(x) < eps, 1 - x ** 2 / 6 + x ** 4 / 120, ca.sin(x)/x)])
    C2 = ca.Function('b', [x], [ca.if_else(ca.fabs(x) < eps, 0.5 - x ** 2 / 24 + x ** 4 / 720, (1 - ca.cos(x)) / x ** 2)])
    C3 = ca.Function('d', [x], [ca.if_else(ca.fabs(x) < eps, 0.5 + x**2/12 + 7*x**4/720, x/(2*ca.sin(x)))])
    C4 = ca.Function('f', [x], [ca.if_else(ca.fabs(x) < eps, (1/6) - x**2/120 + x**4/5040, (1-C1(x))/(x**2))])
    del x

    def __init__(self, SO3=None):
        if SO3 == None:
            self.SO3 = so3.Dcm()

    @classmethod
    def check_group_shape(cls, a):
        assert a.shape == cls.group_shape or a.shape == (cls.group_shape[0],)

    @classmethod #takes 6x1 lie algebra
    def ad_matrix(cls, v): #input vee operator [x,y,z,theta1,theta2,theta3]
        ad_se3 = ca.SX(6, 6)
        ad_se3[0,1] = -v[5]
        ad_se3[0,2] = v[4]
        ad_se3[0,4] = -v[2]
        ad_se3[0,5] = v[1]
        ad_se3[1,0] = v[5]
        ad_se3[1,2] = -v[3]
        ad_se3[1,3] = v[2]
        ad_se3[1,5] = -v[0]  
        ad_se3[2,0] = -v[4]
        ad_se3[2,1] = v[3]
        ad_se3[2,3] = -v[1]
        ad_se3[2,4] = v[0]
        ad_se3[3,4] = -v[5]
        ad_se3[3,5] = v[4]
        ad_se3[4,3] = v[5]
        ad_se3[4,5] = -v[3]
        ad_se3[5,3] = -v[4]
        ad_se3[5,4] = v[3]
        return ad_se3

    @classmethod
    def check_group_shape(cls, a):
        assert a.shape == cls.group_shape or a.shape == (cls.group_shape[0],)

    @classmethod
    def vee(cls, X):
        '''
        This takes in an element of the SE3 Lie Group (Wedge Form) and returns the se3 Lie Algebra elements 
        '''
        v = ca.SX(6, 1) #CORRECTED to [x,y,z,theta1,theta2,theta3]
        v[0,0] = X[0, 3]
        v[1,0] = X[1, 3]
        v[2,0] = X[2, 3]
        v[3,0] = X[2,1]
        v[4,0] = X[0,2]
        v[5,0] = X[1,0]
        return v

    @classmethod
    def wedge(cls, v):
        '''
        This takes in an element of the se3 Lie Algebra and returns the se3 Lie Algebra matrix
        '''
        X = ca.SX.zeros(4, 4) ##Corrected to form [x,y,z,theta1,theta2,theta3]
        X[0, 1] = -v[5]
        X[0, 2] = v[4]
        X[1, 0] = v[5]
        X[1, 2] = -v[3]
        X[2, 0] = -v[4]
        X[2, 1] = v[3]
        X[0, 3] = v[0]
        X[1, 3] = v[1]
        X[2, 3] = v[2]
        return X        

    @classmethod
    def matmul(cls, a, b):
        cls.check_group_shape(a)
        cls.check_group_shape(b)
        return ca.mtimes(a, b)

    @classmethod
    def exp(cls, v): #accept input in wedge operator form in se(3) lie algebra
        v = cls.vee(v)
        v_so3 = v[3:6] #grab only rotation terms for so3 uses ##corrected to v_so3 = v[3:6]
        X_so3 = so3.wedge(v_so3) #wedge operator for so3
        theta = ca.norm_2(so3.vee(X_so3)) #theta term using norm for sqrt(theta1**2+theta2**2+theta3**2)
        
        # translational components u
        u = ca.SX(3, 1)
        u[0, 0] = v[0]
        u[1, 0] = v[1]
        u[2, 0] = v[2]

        # R = exp(so3_wedge(v_so3))
        R = so3.Dcm.exp(v_so3) #'Dcm' for direction cosine matrix representation of so3 LieGroup Rotational
        V = ca.SX.eye(3) + cls.C2(theta)*X_so3 + cls.C4(theta)*ca.mtimes(X_so3, X_so3)
        horz = ca.horzcat(R, ca.mtimes(V,u))
        
        lastRow = ca.transpose((ca.SX([0,0,0,1])))
        
        return ca.vertcat(horz, lastRow)
    
    # @classmethod
    # def exp_Ad(cls, v): #accepts adjoint matrix and perform Exponential map on adjoint matrix of se(3) lie algebra
    #     R = v[:3,:3]
    #     t = v[:3,3:]
    #     theta = ca.norm_2(so3.vee(R)) #theta term using norm for sqrt(theta1**2+theta2**2+theta3**2)
        
    #     # translational components u
    #     u = ca.SX(3, 1)
    #     u[0] = t[2,1]
    #     u[1] = t[0,2]
    #     u[2] = t[1,0]

    #     R_exp = so3.Dcm.exp(so3_vee(R)) #'Dcm' for direction cosine matrix representation of so3 LieGroup Rotational
    #     # V = ca.SX.eye(3) + cls.C2(theta)*R + cls.C4(theta)*ca.mtimes(R, R)
    #     a = R/theta
    #     J = ca.sin(theta)/theta*ca.SX_eye(3) + (1- (ca.sin(theta)/theta))*a@ca.transpose(a)+ ((1-ca.cos(theta))/theta)*so3_wedge(a)

    #     horz = ca.horzcat(R_exp, so3_wedge(J@u)@R_exp)#so3_wedge(so3.Dcm.exp(u))@R_exp)

    #     zeros3 = ca.SX.zeros(3,3)
    #     lastRow = ca.horzcat(zeros3,R_exp)
        
    #     return ca.vertcat(horz, lastRow)
    @classmethod
    def exp_Ad(cls, ad): #accepts adjoint matrix se(3) lie algebra
        R = ad[:3,:3]
        t = ad[:3,3:]
        theta = ca.norm_2(so3.vee(R)) #theta term using norm for sqrt(theta1**2+theta2**2+theta3**2)
        
        ca_sin = ca.sin(theta)
        ca_cos = ca.cos(theta)
        term1 = np.identity(6) + ((3*ca_sin-theta*ca_cos)/(2*theta))*ad
        term2 = ((4-theta*ca_sin-4*ca_cos)/(2*(theta**2)))*ad@ad
        term3 = ((ca_sin-theta*ca_cos)/(2*(theta**3)))*ad@ad@ad
        term4 = ((2-theta*ca_sin-2*ca_cos)/(2*(theta**4)))*ad@ad@ad@ad
        return term1 + term2 + term3 + term4



class SE3:
    group_shape = (6, 6)
    group_params = 12
    algebra_params = 6

    # coefficients 
    x = ca.SX.sym('x')
    C1 = ca.Function('a', [x], [ca.if_else(ca.fabs(x) < eps, 1 - x ** 2 / 6 + x ** 4 / 120, ca.sin(x)/x)])
    C2 = ca.Function('b', [x], [ca.if_else(ca.fabs(x) < eps, 0.5 - x ** 2 / 24 + x ** 4 / 720, (1 - ca.cos(x)) / x ** 2)])
    del x

    def __init__(self, SO3=None):
        if SO3 == None:
            self.SO3 = so3.Dcm()

    @classmethod
    def check_group_shape(cls, a):
        assert a.shape == cls.group_shape or a.shape == (cls.group_shape[0],)
    
    @classmethod
    def one(cls):
        return ca.SX.zeros(6, 1)
    
    @classmethod
    def product(cls, a, b):
        cls.check_group_shape(a)
        cls.check_group_shape(b)
        return ca.mtimes(a, b)

    @classmethod
    def identity(self):
        return ca.SX.eye(4)

    @classmethod
    def inv(cls, a): #input a matrix of SX form from casadi
        cls.check_group_shape(a)
        a_inv = ca.solve(a,ca.SX.eye(6)) #Google Group post mentioned ca.inv() could take too long, and should explore solve function
        return ca.transpose(a)
    
    @classmethod
    def log(cls, G):
        R = G[:3,:3]
        theta = ca.arccos((ca.trace(R) - 1) / 2)
        wSkew = so3.wedge(so3.Dcm.log(R))

        x = ca.SX.sym('x')
        C1 = ca.Function('a', [x], [ca.if_else(ca.fabs(x) < eps, 1 - x ** 2 / 6 + x ** 4 / 120, ca.sin(x)/x)])
        C2 = ca.Function('b', [x], [ca.if_else(ca.fabs(x) < eps, 0.5 - x ** 2 / 24 + x ** 4 / 720, (1 - ca.cos(x)) / x ** 2)])

        V_inv = (
            ca.SX.eye(3)
            - wSkew / 2
            + (1 / theta**2) * (1 - C1(theta) / (2 * C2(theta))) * wSkew @ wSkew
        )        
        # t is translational component vector
        t = ca.SX(3, 1)     
        t[0] = G[0, 3]
        t[1] = G[1, 3]
        t[2] = G[2, 3]
        
        uInv = ca.mtimes(V_inv, t) 
        horz2 = ca.horzcat(wSkew, uInv)
        lastRow2 = ca.transpose((ca.SX([0,0,0,0])))
        return ca.vertcat(horz2, lastRow2)
    
    @classmethod
    def SE3Elems(cls, G): #accepts elements of SE(3) Lie Group in wedge
        '''
        Representation Matrix for SE(3) Lie Group 3D Rigid transformation
        '''
        # theta = G[:3,:3]
        # theta_norm = ca.norm_2(so3.vee(theta))
        # a = theta/theta_norm
        # R = ca.cos(theta_norm)@ca.SX_eye(3) + (1 - ca.cos(theta_norm))@a@ca.transpose(a) + ca.sin(theta_norm)@se3.wedge(a)
        # J = ca.sin(theta_norm)/theta_norm*ca.SX_eye(3) + (1- (ca.sin(theta_norm)/theta_norm))*a@ca.transpose(a)+ ((1-ca.cos(theta_norm))/theta_norm)*se3.wedge(a)
        
        R = G[:3,:3]
        
        # t is translational component vector
        t = ca.SX(3, 1)     
        t[0] = G[0,3]
        t[1] = G[1,3]
        t[2] = G[2,3]
        
        # t_J = J@t
        horz2 = ca.horzcat(R, t)
        lastRow2 = ca.transpose((ca.SX([0,0,0,1])))
        return ca.vertcat(horz2, lastRow2)
        
    @classmethod
    def Ad(cls, G): #accepts elements of SE(3) Lie Group, big Ad
        R = G[:3,:3]
        t = so3_wedge(G[:3,3]) #t_x
        txR = ca.mtimes(t,R) 
        first = ca.horzcat(R,txR)
        zeros3 = ca.SX.zeros(3,3)
        second = ca.horzcat(zeros3,R)
        return ca.vertcat(first, second)
    # @classmethod
    # def Ad(cls, G): #accepts adjoint matrix and perform Exponential map on adjoint matrix of se(3) lie algebra
    #     R = G[:3,:3]
    #     t = G[:3,3]
    #     theta = ca.norm_2(so3.vee(R)) #theta term using norm for sqrt(theta1**2+theta2**2+theta3**2)
        
    #     # translational components u
    #     u = ca.SX(3, 1)
    #     u[0] = t[0]
    #     u[1] = t[1]
    #     u[2] = t[2]

    #     R_exp = so3.Dcm.exp(so3_vee(R)) #'Dcm' for direction cosine matrix representation of so3 LieGroup Rotational
    #     a = R/theta
    #     J = ca.sin(theta)/theta*ca.SX_eye(3) + (1- (ca.sin(theta)/theta))*a@ca.transpose(a)+ ((1-ca.cos(theta))/theta)*so3_wedge(a)

    #     horz = ca.horzcat(R_exp, so3_wedge(J@u)@R_exp)#so3_wedge(so3.Dcm.exp(u))@R_exp)

    #     zeros3 = ca.SX.zeros(3,3)
    #     lastRow = ca.horzcat(zeros3,R_exp)
        
    #     return ca.vertcat(horz, lastRow)


def dot_plot_draw(u, **kwargs):
    F = ca.sparsify(u)

    output_dir = '/home/wsribunm/Documents/GitHub/aae590-LieGroups-personal/fig' #change directory if needed
    os.makedirs(output_dir, exist_ok=True)
    g = graph.dotgraph(F)
    g.set('dpi', 180)
    g.write_png(os.path.join(output_dir, 'result_fromSE3Lie.png'))
