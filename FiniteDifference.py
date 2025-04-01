from numpy import size, zeros, abs, imag, real, sqrt, meshgrid, linspace, array, ones, transpose, linalg
import jax.numpy as jnp
from jax import jacfwd as jax_jacfwd
import numpy as np

#----------------Header-----------------------
# Author: Brigham Ostergaard
# Date: 2-6-2025
# Title: FiniteDifference
# Description: Package containing methods to evaluate derivates. Method include:
#   1. Forward Difference
#   2. Backward Differnce
#   3. Central Difference
#   4. Complex Step
#   5. Algorithmic Differentiation

# inputs:
#   - Point about which to compute gradient
#   - Vector of functions to evaluate jacobian
#   - Step size

# Note: 
#   - You have to have at least one design varible and one function, 
#       but not necessarily equal numbers of either.

# Output:
#   - Jacobian of f with respect to x

# Note, can I force my functions to take in imaginary numbers?

# In running the formulas, we only perturb 1 value of x (not entire range) at a time. (new adition, 2/17)
def Forward_difference(f,x,h=1e-8):
    nx = size(x)    # Determine number of design variables
    nf = size(f)    # Number of functions?
    if(nf == 1):
        J = zeros(nx)
    else:
        J = np.squeeze(zeros([nf,nx]))
    f_0 = f(x)
    for j in range(nx):
        dx = h*(1+abs(x[j]))       
        x[j] = x[j]+dx                # faster than making a copy
        f_pos = f(x)            # Current function evaluation
        J[j] = (f_pos - f_0)/dx
        x[j] = x[j]-dx

    return J        # Returns Jacobian. Can handle mulitdimensional?
def Forward_difference_constraint(f, x, h=1e-8):
    # f(x) returns the stress of each member. x is the area osf each member
    nx = size(x)    # Determine number of design variables
    nf = size(f(x))    # Number of functions?
    J = zeros([nf, nx])
    f_0 = f(x)
    for j in range(nx):
        dx = h * (1 + abs(x[j]))  # Perturbation step size
        x[j] += dx  # Perturb the j-th variable
        f_pos = f(x)  # Evaluate function at perturbed point
        x[j] -= dx  # Restore the original value of x[j]
        J[:,j] = (f_pos - f_0) / dx  # Compute the finite difference, store column vector in J

    # return J
    return J # Returns Jacobian (nf x nx)

def Backward_difference(f,x,h=1e-6):        # Adjust to look like Forward_difference
    nx = size(x)    # Determine number of design variables
    nf = size(f)    # Number of functions?
    J = np.squeeze(zeros([nf,nx]))
    f_0 = f(x)
    for j in range(nx):
        dx = h*(1+abs(x[j]))
        x[j] = x[j]-dx
        f_neg = f(x)            # Current function evaluation
        J[j] = (f_0 - f_neg)/dx
        x[j] = x[j]+dx

    return J
def Backward_difference_constraint(f,x,h=1e-6):
    # f(x) returns the stress of each member. x is the area osf each member
    nx = size(x)    # Determine number of design variables
    nf = size(f(x))    # Number of functions?
    J = zeros([nf, nx])
    f_0 = f(x)
    for j in range(nx):
        dx = h * (1 + abs(x[j]))  # Perturbation step size
        x[j] -= dx  # Perturb the j-th variable
        f_neg = f(x)  # Evaluate function at perturbed point
        x[j] += dx  # Restore the original value of x[j]
        J[:,j] = (f_0 - f_neg) / dx  # Compute the finite difference, store column vector in J

    # return J
    return np.squeeze(J) # Returns Jacobian (nf x nx)


def Central_difference(f,x,h=1e-4):
    # f(x) returns the stress of each member. x is the area osf each member
    nx = size(x)    # Determine number of design variables
    nf = size(f(x))    # Number of functions?
    J = zeros([nf, nx])
    for j in range(nx):
        dx = h * (1 + abs(x[j]))  # Perturbation step size
        x[j] -= dx  # Perturb the j-th variable
        f_neg = f(x)  # Evaluate function at perturbed point
        x[j] += 2*dx  # Restore the original value of x[j]
        f_pos = f(x)
        x[j] -= dx
        J[:,j] = (f_pos - f_neg) / (2*dx)  # Compute the finite difference, store column vector in J

    # return J
    return J # Returns Jacobian (nf x nx)
def Central_difference_constraint(f,x,h=1e-4):
        # f(x) returns the stress of each member. x is the area osf each member
    nx = size(x)    # Determine number of design variables
    nf = size(f(x))    # Number of functions?
    J = zeros([nf, nx])
    for j in range(nx):
        dx = h * (1 + abs(x[j]))  # Perturbation step size
        x[j] -= dx  # Perturb the j-th variable
        f_neg = f(x)  # Evaluate function at perturbed point
        x[j] += 2*dx  # Restore the original value of x[j]
        f_pos = f(x)
        x[j] -= dx
        J[:,j] = (f_pos - f_neg) / (2*dx)  # Compute the finite difference, store column vector in J

    # return J
    return J # Returns Jacobian (nf x nx)

def Complex_Step(f,x,h=1e-20):  #Notice the smaller h
    x = array(x,dtype = complex)    # Convert x to complex
    nx = size(x)
    J = zeros(nx)
    F = f
    for j in range(nx):
        x[j] = x[j] + complex(0,h)     #x + (0 + hi), add imaginary step
        f_pos = F(x)                # Requires the function to be able to handle imaginary numbers
        x[j] = x[j] - complex(0,h)     #x - (0 + hi), subtract imaginary step
        J[j] = imag(f_pos)/h

    return J
def Complex_Step_constraint(f,x,h=1e-20):  #Notice the smaller h
    x = array(x,dtype = complex)    # Convert x to complex
    nx = size(x)
    nf = size(f(x))
    J = zeros([nf,nx])
    F = f
    for j in range(nx):
        x[j] = x[j] + complex(0,h)     #x + (0 + hi), add imaginary step
        f_pos = F(x)                # Requires the function to be able to handle imaginary numbers
        x[j] = x[j] - complex(0,h)     #x - (0 + hi), subtract imaginary step
        J[:,j] = imag(f_pos)/h

    return J
    # This is probably wrong. Lets see what we can do here.

def Forward_AD(f,x):
    # We can use jax.grad(f)(x) to evaluate the gradient.
    # However, the same is done if we do jax_jacfwd(f)(x), but this can handle the full jacobian too.

    J = jax_jacfwd(f)(x)
    # J = np.array(J,dtype=float)                 # Convert from jax array to numpy array
    # print("Returning Jacobian: ", J," \nShape = ", J.shape)

    return J

# def Forward_AD(f,x):
#     def wrapped_jacobian(x):
#         J = jax_jacfwd(f)(x)  # Compute the Jacobian at x
#         return np.array(J, dtype=float)  # Convert JAX array to NumPy
#     return wrapped_jacobian  # Return function, not evaluated array

# def Forward_AD_constraint(f,x):
#     def wrapped_jacobian(x):
#         J = jax_jacfwd(f)(x)
#         return np.array(J, dtype=float)
#     return wrapped_jacobian
#-----------------Examples on how to use the package-------------------------

# def f1(x):
#     return x[0]*x[1] + 2*x[0]*x[1]**2

# def f2(x):
#     return x[0]*x[1]**3 - x[1]

# def f3(x):
#     return x[0]-x[1]/3

# def f4(x):
#     return x[0]**(0.5) + x[1]**2

# x_min = -10
# x_max = 10
# x = linspace(x_min,x_max,100)
# Y,X = meshgrid(x,x)

# x0 = [1,3]                        # Can be any size
# x1 = [-1,3]
# func_array = array([f1,f2,f3])      # Can be any size
# func_array2 = array([f1])

# J = Forward_difference(x0,func_array)
# print("\nForward Difference: \n", J)

# J = Backward_difference(x0,func_array)
# print("\nBackward Difference: \n", J)

# J = Central_difference(x0, func_array)
# print("\nCentral Difference: \n", J)

# J_2d1 = Forward_difference(x0,f1)
# print("\nForward Differnce(1 function): \n", J_2d1)

# J_2d2 = Complex_Step(x0, f1)
# print("\nComplex Step (1 Function): \n", J_2d2)

# J_3d2 = Complex_Step(x1,f4)
# print("\nComplex Step (1 Function,possible imaginary): \n", J_3d2)

# alpha = 0.05

# # What are p1 and p2? Well they are the gradients we get from previous homework.
# # Could be Steepest Ascent, Conjugate Gradient, Newton, Quasi-Newton, etc.
# p1 = -J_2d1/linalg.norm(J_2d1)
# p2 = -J_2d2/linalg.norm(J_2d2)
# p3 = -J_3d2/max(J_3d2)

# x_new1 = x0 + transpose(alpha*p1)     # step along gradient direction
# x_new2 = x0 + transpose(alpha*p2)
# x_new3 = x1 + transpose(alpha*p3)

# print("\nForward Difference (alpha =", alpha, ")\nx0=", x0, " ---> x1 =", x_new1)
# print("\nComplex Step (alpha =",alpha, ")\nx0=", x0, " ---> x1 =", x_new2)
# print("\nComplex Step (alpha =",alpha, ")\nx0=", x1, " ---> x1 =", x_new3)


# print("")