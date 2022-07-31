# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 22:02:41 2022

@author: Alain
"""
import multiprocessing as mp
import numpy as np
from scipy.special import comb

import matplotlib.pyplot as plt
import matplotlib.animation as animation

def get_bezier_parameters(x_base, y_base, degree=2):
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

    if len(x_base) != len(y_base):
        raise ValueError('X and Y must be of the same length.')

    if len(x_base) < degree + 1:
        raise ValueError(f'There must be at least {degree + 1} points to '
                         f'determine the parameters of a degree {degree} curve. '
                         f'Got only {len(x_base)} points.')
    
        
    """ Bernstein polynomial when a = 0 and b = 1. 
    Bernstein matrix for Bézier curves. """  
    
    T  = np.linspace(0, 1, len(x_base))  
    Tp = 1 - T
    
    d_size = degree+1
    poly_array = np.empty((d_size, T.size))
    
    TT = 1
 
    for n in range(d_size): 
       #poly_array[n] = comb(degree, n) * (T ** n) * ( Tp**(degree - n) )
       #simplify by :
       poly_array[n] = comb(degree, n) * TT * Tp**(degree - n) 
       TT = TT * T
      
    poly_array = np.linalg.pinv(poly_array) 
            
    return (x_base,y_base) @ poly_array 
    

def bezier_curve(x_ctrls, y_ctrls, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    n_points = x_ctrls.size-1
    
    T   = np.linspace(0, 1, nTimes)
    Tp  = 1 - T
       
    poly_array = np.empty((x_ctrls.size,T.size))
    
    TP = 1
    
    for n in range(x_ctrls.size):
        # poly_array[n] = comb(n_points, n) * ( T**(n_points-n) ) * ( Tp**n )
        # simplify as :        
        poly_array[n] = comb(n_points, n) * ( T**(n_points-n) ) *  TP
        TP = TP * Tp
        
    return (x_ctrls,y_ctrls) @ poly_array


# =============================================================================
# 
#1) path
#xpoints = [19.212722, 19.21269, 19.21268, 19.21266, 19.21264, 19.21263, 19.21261, 19.21261, 19.21264, 19.21268,19.21274, 19.21282, 19.21290, 19.21299, 19.21307, 19.21316, 19.21324, 19.21333, 19.21342]
#ypoints = [-100.14895, -100.14885, -100.14875, -100.14865, -100.14855, -100.14847, -100.14840, -100.14832, -100.14827, -100.14823, -100.14818, -100.14818, -100.14818, -100.14818, -100.14819, -100.14819, -100.14819, -100.14820, -100.14820]

# #2) sinusoïd(good result with degee= 5)
#xpoints = np.linspace(0, 2*np.pi, 20) 
#ypoints = np.sin(xpoints+dephasage)

# #3) circle (good result with degree = 8)
# #x = a + R cos w ; y = b + R sin w
# xpoints = [5+2*np.cos(w) for w in np.linspace(0, np.pi, 50)] 
# ypoints = [5+2*np.sin(w) for w in np.linspace(0, np.pi, 50)] 


# 
# =============================================================================


NB_STEP = 50
DEGREE  = 6

PERIOD = 2*np.pi

fig = plt.figure("emul_sine") # initialise la figure
fig.set_dpi(200)
fig.set_size_inches(10, 10)

#Fond grisé des grilles
plt.style.use("ggplot")


#============================================
# 1 : Real Sine 
ax1 = plt.subplot(231)

plt.xlim(0, PERIOD)
plt.ylim(-3, 3)
plt.title('Sinusoid (shift phase)')


xpoints = np.linspace(0, PERIOD, NB_STEP)

line_sin,   = ax1.plot([], [], "r-.", lw=1, label='Original Points')

ax1.legend()


#============================================
# 3 : Ctrls + Bezier 
ax3 = plt.subplot(233)

plt.xlim(0, PERIOD)
plt.ylim(-3, 3)
plt.title('Bezier result')

xpoints = np.linspace(0, PERIOD, NB_STEP)

line_ctrl,  = ax3.plot([], [], marker = 'o', linestyle = 'none', markersize = 5)
line_bez,   = ax3.plot([], [], 'b--', lw=1, label='B Curve')
ax3.legend()

#============================================
# 6: Error
ax6=plt.subplot(236)
plt.xlim(0, PERIOD)
plt.ylim(-1, 1)
plt.title('Error')

line_err, = ax6.plot([], [], lw=1, label='taux')
ax6.legend()

#============================================
# 2: Histo control
ax2 = plt.subplot(232) # 3 em position dans grille de 2/2
#ax2.xticks([], [])

plt.xlim(-NB_STEP, 0)
plt.ylim(-3, 3)
plt.title('Histo control')

#x3steps = np.zeros(NB_STEP)
y3steps = np.zeros((DEGREE, NB_STEP))

lines = []  # ensemble des lignes à tracer dans ce graphe
colors = []

for j in range(DEGREE):
    line,  = ax2.plot([], [], '', lw=1, label=f'ctrl {j}')
    lines.append(line)
    colors.append(line.get_color())

for j in range(DEGREE):
    
    #NB_STEP-1 pour mieux voir la boule
    lines.append(ax2.plot([NB_STEP-2,], [0,], marker = 'o', linestyle = 'none', markersize = 5, c=colors[j], lw=1)[0])


 
#================================================
   
def init():
    
    return line_sin, line_ctrl, line_bez, line_err, lines


timestep = -NB_STEP

def animate(num): 
    
    #trace sinusoïd 
    phase_shift = PERIOD/NB_STEP*num
    ypoints = np.sin(xpoints+phase_shift)
    line_sin.set_data(xpoints, ypoints)    
   
    # Get and trace the Bezier parameters based on a degree.
    xctrls, yctrls = get_bezier_parameters(xpoints, ypoints, degree=DEGREE)  
    line_ctrl.set_data(xctrls,yctrls)   
    
    # Get and trace the Bezier curve points
    xbvals, ybvals = bezier_curve(xctrls, yctrls, nTimes=NB_STEP)    
    line_bez.set_data(xbvals, ybvals)
    
    
    global timestep
    timestep+=1
    
  
    
    #plt.xlim(timestep, NB_STEP+timestep)
    (xmin,xmax) = ax2.xaxis.get_view_interval()
    plt.xlim(xmin+1, xmax+1)
    (xmin,xmax) = ax2.xaxis.get_view_interval()
    
    for j in range(DEGREE):
        # #shift right to left (style oscilloscope)
        e = np.empty_like(y3steps[j])
        e[-1:] = yctrls[j]
        e[:-1] = y3steps[j][1:]
        y3steps[j]=e

        lines[j].set_data(range(timestep,timestep+NB_STEP) , y3steps[j])
        
        #boule de mise en évidence du moment des paramètres de controle
        lines[j+DEGREE].set_data([xmax,],[yctrls[j],])
    
    # shift axis x  
    
   


    return line_sin, line_ctrl, line_bez, line_err, lines
     

ani = animation.FuncAnimation(fig, animate, init_func=init, frames=NB_STEP, interval=10, repeat=True)


plt.show()

