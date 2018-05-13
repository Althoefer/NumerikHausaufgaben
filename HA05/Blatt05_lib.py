# -*- coding: utf-8 -*-
"""
Gleichungssystem für Blatt 5, Aufgabe 5
"""
from pylab import ones, linspace, meshgrid, sqrt, figure, cm, colorbar
from mpl_toolkits.mplot3d import Axes3D

from scipy.sparse import spdiags


def system(m):
    n = m*m
    
    e = ones(n)
    l = ones(n)
    l[m-1::m] = 0.0
    r = ones(n)
    r[::m] = 0.0
    
    A = spdiags([-e, -l, 4.0*e, -r, -e], [-m, -1, 0, 1, m], n, n, format='csr')
    
    b = -e / float(n)
    
    return A, b


def plotxk(xk):
    n = len(xk)
    m = int(sqrt(n))
    
    h = linspace(0, 1, m)
    yy,xx = meshgrid(h,h)
    
    fig = figure('xk, m = {0}'.format(m))
    ax  = fig.gca(projection='3d')
    
    surf = ax.plot_surface(xx, yy, xk.reshape(m,m), cmap = cm.jet, rstride = 5, cstride = 5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Hoehe")
    colorbar(surf)
    


# -*- coding: utf-8 -*-
"""
Hilfsfunktionen für Blatt 6, Aufgabe 2, 4
"""
from pylab import ones, linspace, meshgrid, arange, sqrt, cos, pi, c_, figure, cm, colorbar
from mpl_toolkits.mplot3d import Axes3D

from scipy.sparse import spdiags


def Ablock(m):
    # Blockmatrix fuerr 2d-Laplace
    n = m*m
    
    e = ones(n)
    l = ones(n)
    l[m-1::m] = 0.0
    r = ones(n)
    r[::m] = 0.0
    
    A = spdiags([-e, -l, 4.0*e, -r, -e], [-m, -1, 0, 1, m], n, n, format='csr')
    
    return A.toarray()
    

def ew_exakt(m):
    # exakte Eigenwerte fuer 2d-Laplace Blockmatrix, aufsteigend sortiert
    ew1d = 2.0 * (1.0 - cos( (arange(m) + 1.0) * pi / (m + 1.0) ))

    ew = (c_[ew1d] + ew1d).flatten()
    ew.sort()
    
    return ew


def plotev(xk):
    # Eigenvektoren fuer 2d-Laplace Blockmatrix graphisch darstellen
    n = len(xk)
    m = int(sqrt(n))
    
    h = linspace(0, 1, m)
    yy,xx = meshgrid(h,h)
    
    fig = figure()
    ax  = fig.gca(projection='3d')
    
    surf = ax.plot_surface(xx, yy, xk.reshape(m,m), cmap = cm.jet, rstride = 1, cstride = 1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Hoehe")
    colorbar(surf)



import matplotlib.animation as animation  
from mpl_toolkits.mplot3d import axes3d  

def animev(xk):
    # Eigenvektoren fuer 2d-Laplace Blockmatrix animieren

    n = len(xk)
    m = int(sqrt(n))
    
    h = linspace(0, 1, m)
    yy,xx = meshgrid(h,h)
    zz    = xk.reshape(m,m)
    
    zmax = 1.1 * abs(zz).max()
    
    def a(nf = 100, inter = 100, rep = False):
        fig = figure()
        #ax  = fig.gca(projection='3d')
        ax = axes3d.Axes3D(fig)
        ax.set_axis_off()
        ax.grid(False)
        
        surf = ax.plot_surface(xx, yy, zz, cmap = cm.jet, rstride = 1, cstride = 1)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("Hoehe")
        #colorbar(surf)
        
        ax.set_zlim(-zmax, zmax)
        
        def update(i, ax, fig):
            ax.cla()
            phi = i * 180.0 / 3.14159 / nf
            zzi = cos(phi) * zz
            wframe = ax.plot_surface(xx, yy, zzi, cmap = cm.jet, rstride = 1, cstride = 1)
            ax.set_zlim(-zmax, zmax)
            ax.set_axis_off()
            ax.grid(False)
            return wframe,
        
        return animation.FuncAnimation(fig, update, 
                frames = range(nf), 
                fargs  = (ax, fig), interval = inter, repeat = rep)
        
    return a