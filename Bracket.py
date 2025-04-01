import scipy as sci
import numpy as np
from Pinpoint import pinpoint       # Our self-made function.
# ------------------------------ Header ------------------------------------------
# Author: Brigham Ostergaard
# Title: Bracketing
# Date: 1/28/2025
# Description: From a given search direction, we can iterate along path to find a suitable
#    minimum (satifying our Strong Wolfe Conditions). Returns new point from a given input
#    as well as the step size. Implements pinpoint to determine the step size.

# In this function we exectue 6 function calls, but some of them multiple times.
#---------------------------------------------------------------------------------
def Bracket_minimize(x0, p, f, df,alpha_star,tolerance=1e-6):
    # Here we bracket our search and calculate the step size. The actual changes occur in bracket_alpha.
    difference = 1
    n_steps = 0
    alpha_star_track = 0
    # prev_df = np.array([None for _ in range(np.size(x0))],dtype=float)
    prev_p = np.array([None for _ in range(np.size(x0))],dtype=float)

    while difference > tolerance and np.abs(alpha_star) > tolerance and n_steps < 50:
        n_steps = n_steps + 1                                   # increment counter
        phi_0 = f(x0)                                           # Function call, evaluate phi at current point
        grad_cur_f = df(f,x0)                                     # Funciton call, evaluate gradient at current point
        dphi_0 = np.matmul(grad_cur_f,p)                   # Gradient at x0 along p (original gradient to compare against)
        if(n_steps == 1):
            alpha_prev = 1
        else:
            alpha_prev = alpha_star
        alpha_star = bracket_alpha(x0, p, phi_0, dphi_0, f, df,alpha_prev) # Calculate new step size
        x_star = x0 + alpha_star*p              # Take a step along p_hat (thats what we have here)
        grad_x_star = df(f,x_star)

        difference = np.abs(np.linalg.norm(grad_cur_f) - np.linalg.norm(grad_x_star))      # Check the tolerance (change in value)       # 2 function calls
        x0 = x_star                             # Set current position to new minimum
        alpha_star_track = alpha_star_track + alpha_star    #Store step distance from initial x0 we go
        # prev_df = grad_x_star    # calculate gradient at new x0                          
        if(n_steps > 1):
            prev_p = np.vstack([prev_p,p])          # store previous direction
        else:
            prev_p = p

    return x0, alpha_star_track

def bracket_alpha(x, p, phi_0, dphi_0, f, df,alpha_init):
    # Input parameters
    # p      => directional derivative
    # phi_0  => initial function value
    # dphi_0 => gradient at intial point
    # f      => function used to evaluate for given alpha values
    # df     => function used to evalute derivative for given alpha values

    # Return:
    # alpha_star => step size in direction p (not the actual point, just the magnitude)

    # I want to adjust the step size due to the gradient - it doesn't work yet
    # if(n_func > 3):
    #     alpha_init = alpha_prev*((np.transpose(prev_df)*prev_p)/(np.transpose(cur_df)*p))
    # else:
    #     alpha_init = 1 

    # keep this for now. I want to switch it to the previous lines if I can figure it out
    mu1 = 0.1                   # sufficient decrease factor
    mu2 = 0.3                   # sufficienct curvature factor

    #0 < mu1 < mu2 < 1

    # 0 < mu1 < mu2 < 1
    sigma = 2                   # step size increase factor
    backtrack_factor = 0.9      # step size decrease

    alpha1 = 0                  #Step size to get to point 1 (i.e. starting point)
    alpha2 = alpha_init         #Step size to get to the end of the bracket (intial step size)      # Is this right?
    phi1 = phi_0 #f(x+(alpha1*p))
    dphi1 = dphi_0 #np.dot(df(x+(alpha1*p)),p)
    first = True      # This forces our function to use pinpoint on the first loop even if we don't meet sufficient decrease
    while True:
        phi2 = f(x+(alpha2*p))                                                      # Function call
        phi_low = min(phi1, phi2)
        if((phi2 > phi_0 + (mu1*alpha2*dphi1)) or ((first == False) and (phi2 > phi1))):
            alpha_star = pinpoint(x, alpha1, alpha2,f,df,mu1,mu2,phi_0,dphi_0,phi_low,p) # Can overload a tolerance parmaeter too
            return alpha_star
        dphi2 = np.dot(df(f,x+alpha2*p),p)                                            # Function call
        if(phi2 < phi1 + (mu1*alpha2*dphi1)):       #  Strong Wolfe condition 1    
            if(dphi2 > -mu2*dphi1):                 # if we are increasing at alpha2, we hit a minimum between alpha1 and alpha2, so find it with pinpoint     
                alpha_star = pinpoint(x, alpha1, alpha2, f, df, mu1, mu2, phi_0, dphi_0, phi_low,p) # Can overload a tolerance parmaeter too
                return alpha_star
            elif(np.abs(dphi2) <= (mu2*np.abs(dphi1))):      # Strong Wolfe condition 2      (dphi1 or dphi_0?)          
                alpha_star = alpha2                          # our second point is our final point (if our slope is really small)
                return alpha_star
            else:                                                                       # slope is negative
                alpha1 = alpha2
                alpha2 = sigma*alpha2                                                   # increase step
                phi1 = phi2
        else:
            alpha2 = alpha1 + backtrack_factor*alpha2   


        first = False