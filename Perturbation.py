import numpy as np
import sympy as sp
sp.init_printing()

# Configuration des états considérés
ConsideredStates = [(1,0,0), (2, 0, 0), (2, 1, 0), (2, 1, -1), (2,1,1)]


# Symboles
pi, e, x, y, z, phi, theta, r = sp.symbols('pi e x y z phi theta r', real=True)
Ex, Ey, Ez = sp.symbols('Ex Ey Ez', real=True)
a0 = sp.symbols('a0', positive=True)
n, l, m = sp.symbols('n l m', integer=True)
f= sp.symbols('f', cls=sp.Function)
Eh = sp.symbols('Eh')


# Partie radiale
def R(n, l, r):
    R=sp.sqrt((2/(n*a0))**3*sp.factorial(n-l-1)/(2*n*sp.factorial(n+l)))*sp.exp(-r/(n*a0))*(2*r/(n*a0))**l*sp.assoc_laguerre(n-l-1, 2*l+1, 2*r/(n*a0))
    return R


# Partie angulaire
def Y(l, m, theta, phi):
    return sp.Ynm(l, m, theta, phi).expand(func=True)

def Psi(n, l, m, r, theta, phi):
    return R(n, l, r) * Y(l, m, theta, phi)


# Perturbation stark selon z
def f(r, theta, phi):
    return -e * Ez * r * sp.cos(theta) - e * Ex * r * sp.sin(theta) * sp.cos(phi) - e * Ey * r * sp.sin(theta) * sp.sin(phi)


# Calcul des élements de matrice
def matrix_element(n1, l1, m1, n2, l2, m2):
    psi1 = Psi(n1, l1, m1, r, theta, phi)
    psi2 = Psi(n2, l2, m2, r, theta, phi)
    f_expr = f(r, theta, phi)
    integrand = sp.conjugate(psi1) * f_expr * psi2 * r**2 * sp.sin(theta)
    integral = sp.integrate(integrand, (r, 0, sp.oo), (theta, 0, pi), (phi, 0, 2*pi))
    integral = integral.subs(pi, sp.pi)
    return integral
file = open('Matrix_elements.txt', 'w')
for state1 in ConsideredStates:
    for state2 in ConsideredStates:
        if matrix_element(*state1, *state2) != 0:
            file.write('<{}|f|{}> = {}\n'.format(state1, state2, matrix_element(*state1, *state2)))
file.close()
