# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 22:02:41 2022

@author: Alain
"""

import numpy as np
from scipy.special import comb

def get_bezier_parameters(X, Y, degree=2):
    """ Least square qbezier fit using penrose pseudoinverse.

    Parameters:

    X: array of x data.
    Y: array of y data. Y[0] is the y point for X[0].
    degree: degree of the Bézier curve. 2 for quadratic, 3 for cubic.

    Based on https://stackoverflow.com/questions/12643079/b%C3%A9zier-curve-fitting-with-scipy
    and probably on the 1998 thesis by Tim Andrew Pastva, "Bézier Curve Fitting".
    """
    if degree < 1:
        raise ValueError('degree must be 1 or greater.')

    if len(X) != len(Y):
        raise ValueError('X and Y must be of the same length.')

    if len(X) < degree + 1:
        raise ValueError(f'There must be at least {degree + 1} points to '
                         f'determine the parameters of a degree {degree} curve. '
                         f'Got only {len(X)} points.')

    def bpoly(n, t, k):
        """ Bernstein polynomial when a = 0 and b = 1. """
        return t ** k * (1 - t) ** (n - k) * comb(n, k)

    def bmatrix(T):
        """ Bernstein matrix for Bézier curves. """
        return np.matrix([[bpoly(degree, t, k) for k in range(degree + 1)] for t in T])

    def least_square_fit(points, M):
        M_ = np.linalg.pinv(M)
        return M_ * points

    T = np.linspace(0, 1, len(X))
    M = bmatrix(T)
    points = np.array(list(zip(X, Y)))
    return least_square_fit(points, M).tolist()


def bernstein_poly(i, n, T):
    """
     The Bernstein polynomial of n, i as a function of t
    """
   
    return comb(n, i) * ( T**(n-i) ) * (1 - T)**i



def bezier_curve(ControlPoints, nTimes=50):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(ControlPoints)
    
    __ControlPoints = np.array(ControlPoints)
    __ControlPoints = np.transpose(__ControlPoints)
    
    xCtrlPoints = __ControlPoints[0]
    yCtrlPoints = __ControlPoints[1]

    T = np.linspace(0.0, 1.0, nTimes)
    poly = lambda i : bernstein_poly(i, nPoints-1, T)

    polynomial_array = np.array([ poly(i) for i in range(nPoints)   ])
        
    xvals = np.dot(xCtrlPoints, polynomial_array)
    yvals = np.dot(yCtrlPoints, polynomial_array)

    return xCtrlPoints, yCtrlPoints, xvals, yvals, 


# =============================================================================
# #
xpoints = [19.21270, 19.21269, 19.21268, 19.21266, 19.21264, 19.21263, 19.21261, 19.21261, 19.21264, 19.21268,19.21274, 19.21282, 19.21290, 19.21299, 19.21307, 19.21316, 19.21324, 19.21333, 19.21342]
ypoints = [-100.14895, -100.14885, -100.14875, -100.14865, -100.14855, -100.14847, -100.14840, -100.14832, -100.14827, -100.14823, -100.14818, -100.14818, -100.14818, -100.14818, -100.14819, -100.14819, -100.14819, -100.14820, -100.14820]
# 
# # 1
# points = []
# for i in range(len(xpoints)):
#    points.append([xpoints[i],ypoints[i]])
# 
# # 2   
# points=[[xpoints[i],ypoints[i]] for i in range(len(xpoints)) ]
# 
# # 3
# points=list(zip(xpoints, ypoints))
# =============================================================================

                                             
import matplotlib.pyplot as plt
# Plot the original points
plt.plot(xpoints, ypoints, "ro",label='Original Points')
# Get the Bezier parameters based on a degree.
data = get_bezier_parameters(xpoints, ypoints, degree=5)

# Plot the resulting Bezier curve
x_ctrls, y_ctrls, xvals, yvals = bezier_curve(data, nTimes=1000)
plt.plot(xvals, yvals, 'b-', label='B Curve')

# Plot the control points
plt.plot(x_ctrls,y_ctrls,'k--o', label='Control Points')

plt.legend()
plt.show()