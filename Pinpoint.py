import numpy as np
# -------------------------------- Header -----------------------------------------
# Author: Brigham Ostergaard

# In this function we exectue 8 function calls, but some of them multiple times, but not all of them execute each time.

# We aren't doing this function right. We are only leaving if we hit the max iteration, but without any improvment.

def pinpoint(x,alpha1, alpha2, f, df, mu1, mu2,phi_0, dphi_0,phi_low,p,tolerance=1e-6):
    k = 0
    max_iteration = 100              # I only have this for troubleshooting. Theoretically it shouldn't be necessary.
    while True and k < max_iteration:     # We loop until we hit the return statement or just until we've iterated enough
        # interpolate using 4.3.3 to get a value for alpha_p
        phi_1 = f(x+(alpha1*p))     # Function value along p direction after a step size alpha1         # Function Call
        phi_2 = f(x+(alpha2*p))     # Function value along p direction after a step size alpha2         # Function Call
        dphi_1 = np.dot((df(f,x+(alpha1*p))),p)   # Inital gradient along p direction at step alpha1      # Function Call
        phi_high = max(phi_1, phi_2)
        phi_low = min(phi_1,phi_2)

        # Set our brackets
        # if(phi_high == phi_1):
        #     alpha_high = alpha1
        #     alpha_low = alpha2
        # else:
        #     alpha_high = alpha2
        #     alpha_low = alpha1
        alpha_high = np.max([alpha1, alpha2])
        alpha_low = np.min([alpha1,alpha2])

#--------------------- PROBLEM: alpha_p goes negative sometimes as dphi_1 goes positive ---------------------------------------------

        # Find minimum between our bracketted points along the projected line (eq. 4.35)
        # Fit a quadratic curve to the space between alpha1 and alpha2 and take the minimum
        interp_order = 2               # quadratic(2) or cubic(3)
        if(np.abs(phi_2 - phi_1) > tolerance):                  # Tolerance solves problems when we become singular
            if interp_order == 2:
                num = 2*alpha1*(phi_2 - phi_1) + dphi_1*(alpha1**2 - alpha2**2)     
                denom = 2*(phi_2 - phi_1 + dphi_1*(alpha1 - alpha2))
                alpha_p = num/denom                                     # basically alpha*
            else:
                dphi_2 = np.dot((df(x+(alpha2*p))),p)   # Gradient at step alpha along p. Only needed for quartic
                Beta_1 = dphi_1 + dphi_2 - 3*((phi_1 - phi_2)/(alpha1-alpha2))     
                Beta_2 = np.sign(alpha2 - alpha1)*np.sqrt(Beta_1**2 - (dphi_1*dphi_2)) 
                num = dphi_2 + Beta_2 - Beta_1
                denom = dphi_2 - dphi_1 + (2*Beta_2)
                alpha_p = alpha2 - ((alpha2 - alpha1)*(num/denom))      
        else:   
            alpha_p = (alpha2 + alpha1)/2        # This occurs if alpha2 and alpha1 are really close. In which case, don't interpolate

        # This is our "good enough" approximation fo the step size along our line search
        # It represents the step size to the minimum along the quadratic fit

        # use calculated step toward minimum to calculate the new minimum, phi_p
        phi_p = f(x+(alpha_p*p))                                                        # Function Call

        # Check that our point satisfies sufficient decrease
        if((phi_p > (phi_0 + (mu1*alpha_p*dphi_0)))):      # Doesn't satisfy
            alpha_high = alpha_p
            phi_high = phi_p
            if(interp_order == 3):
                dphi_2 = np.dot(df(f,x+(alpha_high*p)),p)                       # only need for higher order interpolation, Function Call
        else:
            dphi_p = np.dot(df(f,alpha_p*p),p)                                  # Function Call
            if(np.abs(dphi_p) <= -mu2*dphi_0):      # We satisfy sufficient decrease and stop searching
                alpha_star = alpha_p
                return alpha_p
            elif (np.dot(df(f,(alpha_high - alpha_low)*p),p) >= 0):             # Function Call  
                alpha_high = alpha_low
    
            alpha_low = alpha_p
        # Alright, we are letting alpha1 and alpha2 get really close before moving on
        alpha1 = alpha_low              # reset alpha values here to use new alpha_high and alpha_low
        alpha2 = alpha_high
        if(np.abs(alpha1 - alpha2) < tolerance):
            alpha_p = (alpha1 + alpha2)/2
            return alpha_p
        k = k+1
    
    # If we are taking a lot of iterations to converge to a point in pinpoint, I print this statment
    return alpha_p      # return if we hit max iterations