import numpy as np
import Bracket as linesearch
import matplotlib.pyplot as plt

# Basically We use newton to get p, and then do a linesearch
# See section 4.4.4

def quasi_Newton(f, df, x, alpha_init,return_points,tol=1e-6,plot_inner_steps=False):
    x_keep = np.array(x)

    converged = False
    k = 0
    cur_grad = df(f,x)
    alpha_init = 1
    reset = False
    n = np.size(x)
    x_prev = np.ones(n)     # hold previous iterations x value. Only needed when k>0
    I = np.diag(np.ones(n))
    reset_tol = 1e-3
    reset = False

    while(converged == False):
        if((k == 0) or (reset == True)):
            V_cur = (1/np.linalg.norm(cur_grad))*I
            reset = False
        else:
            s = x - x_prev
            y = cur_grad - grad_prev
            sigma = 1/(np.matmul(s,y))
            st = np.transpose(s)
            yt = np.transpose(y)
            V_cur = (I - (sigma*s*yt))*V_prev*(I - (sigma*y*st)) + (sigma*s*st)

        p = -np.matmul(V_cur,cur_grad)             # determine quasi-newton search direction
        p = p/np.linalg.norm(p)                    # normalize p
        x_new, alpha = linesearch.Bracket_minimize(x, p, f, df,alpha_init,tolerance=1e-6)        
        # Perform linesearch along search direction. Linesearch return updated x as well
        x_keep = np.vstack([x_keep,x_new])      # Store our input variables
        grad_prev = cur_grad            # set current gradient to past gradient for next iteration
        cur_grad = df(f,x_new)            # Find gradient with new x point

        if(np.max(np.abs(cur_grad)) <= tol):    # infinity norm
            converged = True
    
        V_prev = V_cur
        x_prev = x
        x = x_new

        k=k+1

        if(np.dot(np.transpose(cur_grad),p) > reset_tol):
            reset = True
        # Can add a condition here where we updat "reset" to be true if np.transpose(grad)*p > tolerance (unknown exactly what value)

    if(return_points):
        return x_keep
    else:
        return x