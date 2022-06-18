# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 22:02:41 2022

@author: Alain Van Hout
"""

import numpy as np
from scipy.special import comb

import matplotlib.pyplot as plt
import matplotlib.animation as animation


__parameters_array_cache = {} 

def get_bezier_parameters(x_base, y_base, degree=2):
    """ Least square qbezier fit using penrose pseudoinverse.

    Based on https://stackoverflow.com/questions/12643079/b%C3%A9zier-curve-fitting-with-scipy
    and thesis by Tim Andrew Pastva, "Bézier Curve Fitting".
    
    Written and optimised by A. Van Hout
    
    """

    one_parameters_array = __parameters_array_cache.setdefault(degree,None)
    if one_parameters_array is not None:   
        return (x_base,y_base) @ one_parameters_array
    
    
    T  = np.linspace(0, 1, len(x_base), dtype=np.float16)  
    
    def built_poly_array():
        
        d_size = degree+1
            
        R = np.empty((d_size,T.size), dtype=np.float32) 
        RT = np.empty_like(R)
        
        R[0]    = 1 # all line at one
        RT[-1]  = 1 # idem T0 = np.ones(T.size) 
        Re      = 1     
        
        for n in np.arange(1,d_size):
            Re =  Re * T             
            R[n] = comb(degree, n) * Re
            RT[degree-n] = np.flipud(Re) # or Re[::-1]
    
        return R * RT
    
    one_parameters_array = np.linalg.pinv(built_poly_array())
    __parameters_array_cache[degree]=one_parameters_array
                    
    return (x_base,y_base) @ one_parameters_array
    
    
    
# pour la mise en cache des matrices de fonction polynomiale
# definies par le nombre de points de contrôle
__poly_array_cache = {} 

def bezier_curve(x_ctrls:np.ndarray, y_ctrls:np.ndarray, nb_step = 200):
    """
        With a set of control points (x,y), this function return the
        bezier curve defined by this one and Tp_size for the number of step.

        See http://processingjs.nihongoresources.com/bezierinfo/

        Written by A. Van Hout
    """

    one_poly_array = __poly_array_cache.setdefault(x_ctrls.size,None)
    if one_poly_array is not None:   
        return (x_ctrls,y_ctrls) @ one_poly_array

    Tp   = np.linspace(0, 1, nb_step, dtype=np.float16)  
    n_points = x_ctrls.size-1    
     
    def built_poly_array():
        
        R   = np.empty((x_ctrls.size,Tp.size), dtype=np.float32) 
        RT  = np.empty_like(R)  

        R[0]    = 1 #all line at one               
        RT[-1]  = 1 #idem np.ones(T.size)
        Re      = 1

        for n in np.arange(1,x_ctrls.size):
            Re =  Re * Tp             
            R[n] = comb(n_points, n) * Re
            RT[n_points-n] = np.flipud(Re) # or Re[::-1]

        return R * RT
    
    one_poly_array = built_poly_array()
    __poly_array_cache[x_ctrls.size] = one_poly_array  
    
    return (x_ctrls,y_ctrls) @  one_poly_array




NB_STEP = 120
DEGREE  = 8
PERIOD = 2*np.pi

fig = plt.figure("From Function To Bezier Curve") # initialise la figure
fig.set_dpi(100)
fig.set_size_inches(8, 3)

plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.00, #space between all grids
                    hspace=0.1)

#Fond grisé des grilles
plt.style.use("ggplot")


#============================================
# 1 : Real Sine 
ax1 = plt.subplot(131)

plt.xlim(0, PERIOD)
plt.ylim(-3, 3)
plt.title('Original Function')

line_fct,   = ax1.plot([], [], "r-.", lw=1, label='Original Points')

ax1.legend()

#============================================
# 3 : Ctrls + Bezier 
ax3 = plt.subplot(133)

plt.xlim(0, PERIOD)
plt.ylim(-3, 3)
plt.title('Bezier result')

ax3.set_yticklabels([])

line_ctrl,  = ax3.plot([], [], marker = 'o', linestyle = '-.', markersize = 3, lw=.5)
line_bez,   = ax3.plot([], [], 'b--', lw=1, label='B Curve')

ax3.legend()


#============================================
# 2: Histo control
ax2 = plt.subplot(132) 

plt.xlim(-NB_STEP, 0)
plt.ylim(-3, 3)
plt.title('Histo control')

ax2.set_yticklabels([])

lines = []  # set of lines to trace in this graph
colors = [] # set of colors used by line

y3steps = np.zeros((DEGREE, NB_STEP))
x3steps = np.arange(-NB_STEP,0)

for j in np.arange(DEGREE):
    line,  = ax2.plot(x3steps, y3steps[0], '', lw=1, label=f'ctrl {j}')
    lines.append(line)
    colors.append(line.get_color())
    
for j in np.arange(DEGREE):
    
    #NB_STEP-1 pour mieux voir la boule
    lines.append(ax2.plot([NB_STEP-1,], [0,], marker = 'o', linestyle = 'none', markersize = 5, c=colors[j], lw=1)[0])


 
#================================================

#xpoints = np.linspace(0, PERIOD, NB_STEP)

def animate(num): 
    
    global y3steps
    global x3steps   
    global timestep
    
    # # function
    phase_shift = (2*np.pi/NB_STEP) * (4*num % NB_STEP) 
    # ypoints = np.sin(xpoints+phase_shift)*xpoints*.10
    
  # #2) sinusoïd(good result with degee= 5)
    xpoints = np.linspace(0, 2*np.pi, 20) 
    ypoints = np.sin(xpoints+phase_shift)

    # #3) circle (good result with degree = 8)
    # #x = a + R cos w ; y = b + R sin w

    
    line_fct.set_data(xpoints, ypoints)    
   
    # Get and trace the Bezier parameters based on a degree.
    xctrls, yctrls = get_bezier_parameters(xpoints, ypoints, degree=DEGREE)  
    line_ctrl.set_data(xctrls,yctrls)   
    
    # Get and trace the Bezier curve points
    line_bez.set_data(*bezier_curve(xctrls, yctrls) )
    
    #shift right to left (style oscilloscope)    
    x3steps      = np.roll(x3steps, -1) 
    x3steps[-1]  = num 
    plt.xlim(x3steps[0], num)
    
    y3steps      = np.roll(y3steps, -1)    
     
    for j in np.arange(DEGREE):
        y3steps[j][-1] = yctrls[j]

        lines[j].set_data(x3steps , y3steps[j])
        
        #boule de mise en évidence du moment des paramètres de controle
        lines[j+DEGREE].set_data([num,],yctrls[j])
 

ani = animation.FuncAnimation(fig, animate, interval=20, repeat=True)
plt.show()

