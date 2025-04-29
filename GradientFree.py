import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize as opt
from numpy import cos, sum, mean, size, delete
import random           # for gauss distribution
import os
from scipy.stats import qmc


global n_fun_local
n_fun_local = 0
# Header
# Author: Brigham Ostergaard
# Title: Gradient Free Algorithms
# Date: 3/14/2025
# Description: A package to implemnt a user written algorithms 
# for gradient-free optimization: 
#       1. Nelder-Mead 
#       2. Geneatic Algorithm
#       3. Particle Swarm Optimization (not necessary but I want to try it)

# Current Progress:
# 1. Nelder-Mead is implemented and working
# 2. Genetic Algorithm is implemented and working           
# 3. Particle Swarm Optimization is not implemented yet.

def test_f(x):
    global n_fun_local
    n_fun_local = n_fun_local+1
    return x[0]**2 - x[1]
def rosenbrock(x):
    global n_fun_local
    n_fun_local = n_fun_local+1
    return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2
def bean(x):
    global n_fun_local
    n_fun_local = n_fun_local+1
    return (1-x[0])**2 + (1-x[1])**2 + (1/2)*(2*x[1] - x[0]**2)**2
def egg_carton(x):
    global n_fun_local
    n_fun_local = n_fun_local+1
    return 0.1*x[0]**2 + 0.1*x[1]**2 - cos(3*x[0]) - cos(3*x[1])

# Nelder-Mead
def Nelder_Mead(f, x0, bounds=((-2,2.5),(-1,3)), max_iter=50, tol=1e-6, plot_flag=False,L=10,save_fig=False):
    # note, for each algorithm in the book, their n+1 is my n
    tol_x = tol
    tol_f = tol
    n = size(x0)+1                 # number of points in simplex (n_dimensions + 1)
    l = L                        # length of simplex (starting side length)
    xs = np.zeros([n, n-1])           # simplex
    # create simplex
    for j in range(n):
        for i in range(size(x0)):    # for each dimension of x
            si = (l/(n*np.sqrt(2)))*(np.sqrt(n+1)-1)
            if(i == j):
                si = si+(l/np.sqrt(2)) 
            xs[j,i] = x0[i] + si
    
    # evaluate function at each point in simplex
    fs = np.zeros(n)
    for i in range(n):
        fs[i] = f(xs[i])

    # sort simplex by function value
    idx = np.argsort(fs)           # we want to the best (smallest) value first
    xs = xs[idx]    # sort simplex
    fs = fs[idx]    # sort function values
    
    # x_store = np.vstack([x_store, xs[0,:]])     # store best point

    Delta_x = 0
    for i in range(n-1): 
        Delta_x = Delta_x + (np.linalg.norm(xs[i] - xs[-1]))   # eq. 7.6

    f_bar = mean(fs)
    Delta_f = 0
    for i in range(n-1):
        Delta_f = Delta_f + (fs[i] - f_bar)**2     # eq. 7.7
    Delta_f = np.sqrt(Delta_f/(n))

    gen = 0

    # Start looping
    while(Delta_x > tol_x and Delta_f > tol_f and gen < max_iter):
        if(plot_flag):  # only works with 2D x values
            plot_generation_NM(f,xs, gen, bounds,save_fig)
        gen = gen+1
        # sort list
        idx = np.argsort(fs)           # we want to the best (smallest) value first
        xs = xs[idx]    # sort simplex
        fs = fs[idx]    # sort function values
        if(gen == 1):
            x_store = xs[0,:]
        else:
            x_store = np.vstack([x_store, xs[0,:]])     # store best point
        x_c = np.sum(xs[:-1], axis=0)/(n-1)     # centroid of simplex (excluding worst point)
        # relfect
        x_r = x_c + (x_c - xs[-1])      # alpha = 1
        f_r = f(x_r)

        # check if reflected point is better than the best popint
        if(f_r < fs[0]):
            x_e = x_c + 2*(x_r - x_c)   # expand the point
            f_e = f(x_e)
            if(f_e < fs[0]):    #are we better than the best point? (should I do 2nd worst though)
                xs[-1] = x_e    # replace worst point with expanded point
                fs[-1] = f_e    # replace worst function value with expanded function value
            else:
                xs[-1] = x_r    # accept the relfected point, not the expansion
                fs[-1] = f_r
        elif(f_r <= fs[-2]): # compare to 2nd worst point (not worst becuase we only accept it if we made improvment)
            xs[-1] = x_r    
            fs[-1] = f_r
        else:
            # check if the reflected point is worst than the worst point
            if(f_r > fs[-1]):
                x_ic = x_c + 0.5*(x_c - xs[-1]) # contract
                if(f(x_ic) < fs[-1]):   # if we are better, keep the contracted point
                    xs[-1] = x_ic
                    fs[-1] = f(x_ic)
                else:
                    for j in range(n-1):
                        xs[j+1] = xs[0] + 0.5*(xs[j+1] - xs[0])
                        fs[j+1] = f(xs[j+1])
            else:
                x_oc = x_c + 0.5*(x_c - xs[-1]) # outside contract
                if(f(x_oc) < f_r):  # is contraction better than reflection
                    xs[-1] = x_oc
                    fs[-1] = f(x_oc)
                else:
                    for j in range(n-1):
                        xs[j+1] = xs[0] + 0.5*(xs[j+1] - xs[0])
                        fs[j+1] = f(xs[j+1])

        # update deltas
        Delta_x = 0
        for i in range(n-1): Delta_x = Delta_x + (np.linalg.norm(xs[i] - xs[-1]))

        f_bar = mean(fs)
        Delta_f = 0
        for i in range(n-1): Delta_f = Delta_f + (fs[i] - f_bar)**2
        Delta_f = np.sqrt(Delta_f/(n))

    x_store = np.vstack([x_store, xs[0,:]])

    if(plot_flag):  # only works with 2D x values
        plot_generation_NM(f,xs, gen, bounds)
    
    return xs[0], fs[0], xs, gen, x_store       # any of the points honestly. Should be close

def particle_swarm(f,bounds, pop_size, generations,dims,plot_flag=False,save_fig=False):
    # Generate initial population
    dims = len(bounds)
    l_bounds = np.zeros([dims])
    u_bounds = np.zeros([dims])
    for i in range(dims):
        l_bounds[i] = bounds[i][0]
        u_bounds[i] = bounds[i][1]

    # generate initial population with Latin-Hypercube Sampling
    sampler = qmc.LatinHypercube(d=dims)
    sample = sampler.random(n=pop_size)
    pop = qmc.scale(sample, l_bounds, u_bounds)
    x_best = np.zeros([len(pop),dims])      # individual best
    x_star = np.zeros([1,dims])      # global best

    # Define weight limits
    beta_max = 2            # 0-2
    gamma_max = 2           # 0-2
    dx = np.ones([pop_size,dims])
    dx_max = 1
    dx_max_red = 0.9        # between (0,1) that reduces the value of dx_max so the particles slow to a stop
    # 10% reduction seems to do the trick

    for gen in range(generations):
        print('Generation: ',gen+1)
        # define weights for the current generation (changes each generation, but applied uniformily to the population)
        alpha = np.random.uniform(0.8,1.2)
        beta = np.random.uniform(0,beta_max)
        gamma = np.random.uniform(0,gamma_max)

        # determine fitness
        f_vals = np.zeros(pop_size)
        for i in range(pop_size): 
            f_vals[i] = f(pop[i,:])
        idx = np.argsort(-f_vals)      # sort in descending order, since we want to minimize the function      
        pop = pop[idx]
        f_vals = f_vals[idx]        
        x_best = np.zeros([pop_size,dims])
        f_star = f_vals[0]                 # best f value, set as infinity so we always have a value.
    
        # iterate thoough population and update the velocity and position of each particle
        for i in range(pop_size):
            new_pop = pop
            if(gen == 0):
                x_min = pop[np.argmin(f_vals)]      # find the best and worst particle
                x_max = pop[np.argmax(f_vals)]
                dx[i,:] = 0.1*(np.random.uniform(x_min, x_max,dims))  # start with a random speed, a scale between the bounds of xmin and xmax
            else:
                dx[i,:] = alpha*dx[i,:] + beta*(x_best[i,:] - pop[i,:]) + gamma*(x_star - pop[i,:])  # eq. 7.29, update particle velocity
                if(np.linalg.norm(dx[i,:]) > dx_max):
                    # I want to scale the vector so its norm is dx_max
                    dx[i,:] = dx_max*(dx[i,:]/np.linalg.norm(dx[i,:]))

            new_pop[i,:] = pop[i,:] + dx[i,:]    #eq. 7.30, update particle position

        # update best x for each particle and overall best.
        for i in range(pop_size):
            if(f_star > f(pop[i,:])):
                x_best[i,:] = pop[i,:]  
                f_star = f(pop[i,:])                            
       
        if(plot_flag):
            plot_generation_PS(pop,dx,gen,bounds,X,Y,Z,save_fig)     # I diferentiate pop/new_pop so that I can plot pop with its vectors.
        pop = new_pop
        dx_max = dx_max * dx_max_red

    # sort at the end, so I can get the best point (only need to sort at the end)
    f_vals = []
    for i in range(pop_size): 
        if(np.size(f_vals) == 0):
            f_vals = np.array(f(pop[i,:]))
        else:
            f_vals = np.vstack([f_vals,f(pop[i,:])])
    idx = np.argsort(-f_vals)      # sort in descending order       
    pop = pop[idx,:]
    f_vals = f_vals[idx]

    return pop[0],f_vals[0], pop, generations

def genetic_algorithm(f,fitness,bounds, pop_size, generations, selection = "Roullette",plot_flag=False, save_fig=False, mutation_rate = 0.05):
    # define values for plotting
    dims = len(bounds)
    l_bounds = np.zeros([dims])
    u_bounds = np.zeros([dims])
    for i in range(dims):
        l_bounds[i] = bounds[i][0]
        u_bounds[i] = bounds[i][1]

    # generate initial population with Latin-Hypercube Sampling
    sampler = qmc.LatinHypercube(d=dims)
    sample = sampler.random(n=pop_size)
    pop = qmc.scale(sample, l_bounds, u_bounds)

    
    if(plot_flag):
        plot_generation_GA(f,pop, 0, bounds,save_fig)
    # Evaluate fitness
    fit = fitness(pop,f,pop_size)   # values bewteen 0 and 1. 1 is the best.
    # Sort population by fitness
    idx = np.argsort(-fit)      # sort in descending order       
    pop = pop[idx]
    fit = fit[idx]
    # Loop through generations
    for i in range(generations):
        # Select parents
        # roullette or tournament selection
        if(selection == "Roullette"):
            parent_pairs = roullette_selection(fit,pop,pop_size,dims)     # returns [n/2,4] array of parent pairs
        else:
            parent_pairs = tournament_selection(fit,pop,pop_size,dims)

        # Crossover
        children = np.zeros([int(pop_size),dims])    # 20 children, same as # of parents
        for j in range(int(pop_size/2)):
            # linear crossover
            children[2*j] = 0.5*(parent_pairs[j,0:dims] + parent_pairs[j,dims-1:-1])        # child 1: average of 2 parents
            children[(2*j)+1] = 2*(parent_pairs[j,0:dims])-parent_pairs[j,dims-1:-1]            # child 2: linear fit of 2 parents (weighted toward better parent)

        # Mutation
        p = mutation_rate # mutation rate (0.005 to 0.05 generally)
        n_muations = 0    # internal check to see how many mutation occur
        for j in range(pop_size):
            mutate_prop = np.random.random()
            if(mutate_prop < p):
                # mutate the child
                n_muations+=1
                for k in range(2):
                    # mutate the child by add/sub a random number from a normal distribution
                    sigma = random.gauss(0.5, 0.25)     # (mean, std)
                    sign = np.random.random()
                    if(sign < 0.5):
                        children[j,k] = children[j,k] -  sigma
                    else:
                        children[j,k] = children[j,k] +  sigma 

        # sort the children, like the parents, by fitness.              <------- This was my error. I hadn't sorted the children.
        fit_c = fitness(children,f,pop_size)
        idx = np.argsort(-fit_c)      # sort in descending order
        children = children[idx]
        fit_c = fit_c[idx]

        # I want to set some of the parents as the children
        # New generation consists of the top 20% of the parents and the top 80% of the children
        pop_top = pop[0:int(pop_size/5)]    # top 20% of the parents
        children_top = children[:-int(pop_size/5)]  # top 80% of the children
        new_gen = np.vstack([pop_top,children_top])

        # Set new population as children
        pop = new_gen          #children become the new population
        # sort the population by fitness
        fit = fitness(pop,f,pop_size)
        idx = np.argsort(-fit)      # sort in descending order
        pop = pop[idx]
        fit = fit[idx]

        if(plot_flag):
            plot_generation_GA(f,pop, i+1, bounds,save_fig)
        a=0


    return pop[0],f(pop[0]), pop, generations #x_star, f_star, xs, gen

def roullette_selection(fit,pop,pop_size,dims): #Pair parents together roullete style (this is the way to date)
    Parent_pairs = np.zeros([int(pop_size/2),dims*2])
    # we recieve the fitness and their respective poputlation (may not need the population)
    dF = 1.1*max(fit) - 0.1*min(fit)  # scale the fitness values
    # Convert objective function values to a fitness value (F)
    F = np.zeros_like(fit)
    for i in range(len(fit)):
        F[i] = (-fit[i]+dF)/(max(1,dF-min(fit)))  # scale the fitness values, eq.7.19
    
    # create sections of the roullette wheel 
    S = np.zeros_like(F)
    for i in range(len(F)):
        S[i] = np.sum(F[0:i])/np.sum(F)  # eq. 7.20, sum of previous fitness values divided by the sum of all fitness values

    # Now, since each fitness has a probability of being selected, we can randomly select a number between 0 and 1 and pair groups together
    # Some parents can be selected more than once, but that is okay.
    for i in range(int(len(F)/2)):
        # Select parent 1
        parent = np.zeros(dims*2)
        for k in range(2):
            r = np.random.rand()
            for j in range(int(len(S))):
                if(r > S[-1]):          # if our random number is greater than the last probability, select the last parent
                    parent[2*k:(2*k)+2] = pop[-1]
                    # parent[2*k:(2*k)+2] = (np.size(pop)/2)-1        #index istead of values
                    break
                elif(r < S[j]):
                    parent[2*k:(2*k)+2] = pop[j]        #I could also just return the index, not the population value.
                    if(j > 0 and j < len(F)-1):
                        pop = np.vstack((pop[0:j],pop[j+1:]))      # remove the parent from the population so it doens't get selected again?
                        S = np.hstack((S[0:j],S[j+1:]))            # remove the probablity from the stack
                    else:
                        if(j == 0):
                            pop = pop[1:]       # remove the first element
                            S = S[1:]
                        else:
                            pop = pop[:-1]      # remove the last element
                            S = S[:-1]
                    
                    break

        Parent_pairs[i] = parent
    return Parent_pairs

def tournament_selection(fit,pop,pop_size,dims): #Pair parents together tournament style
    Parent_pairs = np.zeros([int(pop_size/2),dims*2])
    for i in range(int(pop_size/2)):
        # Select parent 1
        parent = np.zeros(dims*2)
        for k in range(2):
            # Select 2 random parents
            idx = np.random.randint(0,pop_size,2)
            # select the better of the 2 parents to give 1/2 of the genes to the child
            if(fit[idx[0]] > fit[idx[1]]):
                parent[2*k:(2*k)+2] = pop[idx[0]]
            else:
                parent[2*k:(2*k)+2] = pop[idx[1]]
        Parent_pairs[i] = parent

    return Parent_pairs
   
def fit_func(x,f,pop_size):
    func = np.zeros(pop_size)
    fit = np.zeros(pop_size)
    best_f = np.inf
    # determine the best value of f (the smallest)
    for i in range(pop_size):
        profit = f(x[i,:])
        if(profit == 0):
            func[i] = np.inf
        else:
            func[i] = 1/(1-profit)

        if(func[i] < best_f):
            best_f = func[i]
    if(np.max(best_f) == np.inf and np.min(best_f) == np.inf):                  # this occurs if all the values come out to be 0.
        best_f = 0
    # calculate the fitness of each individual as fit[i] = 1/(1+(f(x[i])-best_f)^2)
    # try new fitness: fit[i] =  -f(x[i])  
    for i in range(pop_size):
        fit[i] = 1/(1+(func[i]-best_f)**2)      # 1/(1+x) where x is some value that approaches 0 when f(x) == f*(x) of the current generation. i.e. when we are the best, we get 1, when we are the worst, we get near 0.
        # fit[i] = -f(x[i])                       # This makes the biggest number the most negative.

    # fitness is between 0 and 1. 1 means we are the best.

    return fit


def single_point_crossover(parent1, parent2):       # not what we want, I don't think
    crossover_point = np.random.randint(len(parent1))
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2

def plot_generation_NM(f,xs,gen,bounds,save_fig=False,save_directiory="Figures/Nelder Mead/"):
    x_lim = np.linspace(bounds[0][0], bounds[0][1], 100)
    y_lim = np.linspace(bounds[1][0], bounds[1][1], 100)
    X,Y = np.meshgrid(x_lim,y_lim)
    xs = np.vstack((xs,xs[0,:]))
    Z = f([X,Y])

    plt.ion()  # Turn on interactive mode
    plt.figure(1)
    plt.clf()  # Clear the current figure
    plt.contourf(X, Y, Z, cmap='viridis')
    plt.colorbar()
    plt.contour(X, Y, Z, cmap='viridis')
    plt.plot(xs[:, 0], xs[:, 1], 'ro-')

    plt.xlabel('X')
    plt.ylabel('f(x)')
    plt.title(f'Generation {gen}')
    plt.xlim(bounds[0])
    plt.ylim(bounds[1])
    plt.draw()  # Update the figure

    if(save_fig):
        if(gen%5 == 0):
            filename = f"Generation {gen}"
            filepath = os.path.join(save_directiory, filename)
            plt.savefig(filepath)

    plt.pause(0.001)  # Pause to allow the plot to update
    input("Press Enter to continue...")

def plot_generation_GA(f,xs,gen,bounds,save_fig=False,save_directiory="Figures/Genetic Algorithm/"):
    x_lim = np.linspace(bounds[0][0], bounds[0][1], 100)
    y_lim = np.linspace(bounds[1][0], bounds[1][1], 100)
    X,Y = np.meshgrid(x_lim,y_lim)
    xs = np.vstack((xs,xs[0,:]))
    Z = f([X,Y])

    plt.ion()  # Turn on interactive mode
    plt.figure(1)
    plt.clf()  # Clear the current figure
    plt.contourf(X, Y, Z, cmap='viridis')
    plt.colorbar()
    plt.contour(X, Y, Z, cmap='viridis')
    # scatter plot the population
    plt.scatter(xs[:,0], xs[:,1],15,'r')

    plt.xlabel('X')
    plt.ylabel('f(x)')
    plt.title(f'Generation {gen}')
    plt.xlim(bounds[0])
    plt.ylim(bounds[1])
    plt.draw()  # Update the figure

    if(save_fig):
        if(gen%5 == 0):
            filename = f"Generation {gen}"
            filepath = os.path.join(save_directiory, filename)
            plt.savefig(filepath)

    plt.pause(0.001)  # Pause to allow the plot to update
    input("Press Enter to continue...")

def plot_generation_PS(xs,vs,gen,bounds,X,Y,Z,save_fig=False,save_directiory="Figures/Particle Swarm/"):
    plt.ion()  # Turn on interactive mode
    plt.figure(1)
    plt.clf()  # Clear the current figure
    plt.contourf(X, Y, Z, cmap='viridis')
    plt.colorbar()
    plt.contour(X, Y, Z, cmap='viridis') 
    # scatter plot the population
    plt.scatter(xs[:,0], xs[:,1],15,'r')
    if(gen > 0):
        plt.quiver(xs[:,0], xs[:,1], vs[:,0], vs[:,1], color='r', scale=1, scale_units='xy')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title(f'Generation: ({gen})')
    plt.xlim(bounds[0])
    plt.ylim(bounds[1])
    plt.draw()  # Update the figure
    plt.pause(0.001)  # Pause to allow the plot to update
    input("Press Enter to close figure and continue...")
    if(save_fig):
        if(gen%5 == 0 or gen == 1):
            filename = f"Generation {gen}"
            filepath = os.path.join(save_directiory, filename)
            plt.savefig(filepath)
    plt.close()

def plot_final_point(f,xs,gen,bounds):
    x_lim = np.linspace(bounds[0][0], bounds[0][1], 100)
    y_lim = np.linspace(bounds[1][0], bounds[1][1], 100)
    X,Y = np.meshgrid(x_lim,y_lim)
    Z = f([X,Y])

    plt.ion()  # Turn on interactive mode
    plt.figure(2)
    plt.contourf(X, Y, Z, cmap='viridis')
    plt.colorbar()
    plt.contour(X, Y, Z, cmap='viridis')
    plt.plot(xs[0], xs[1], 'r*',label='Optimal')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title(f'Final Generation ({gen})')
    plt.xlim(bounds[0])
    plt.ylim(bounds[1])
    plt.draw()  # Update the figure
    plt.pause(0.001)  # Pause to allow the plot to update
    input("Press Enter to continue...")

def plot_function(f,bounds,save_plot=False,save_directory="Figures/"):
    x_lim = np.linspace(bounds[0][0], bounds[0][1], 100)
    y_lim = np.linspace(bounds[1][0], bounds[1][1], 100)
    X,Y = np.meshgrid(x_lim,y_lim)
    Z = f([X,Y])


    # I want to create a 3D plot in the same figure, as a subplot.
    # This way I can better visualize the function.
    plt.ion()
    fig = plt.figure(figsize=(12, 6))

    # 2D Contour Plot - left side figure
    ax1 = fig.add_subplot(1, 2, 1)
    contour = ax1.contourf(X, Y, Z, cmap='viridis')
    plt.colorbar(contour, ax=ax1)
    ax1.contour(X, Y, Z, cmap='viridis')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_xlim(bounds[0])
    ax1.set_ylim(bounds[1])
    ax1.set_title('2D Contour Plot')

    # 3D Surface Plot - Right side figure
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('f(X, Y)')
    ax2.set_title('3D Surface Plot')

    plt.tight_layout()
    input("Press Enter to close figure and continue...")
    if(save_plot):
        filepath = os.path.join(save_directory, "Initial_design_space.jpg")
        plt.savefig(filepath)
    plt.close()

def convergence_NM(F, xs, save_plot=False, save_directory = "Figures/"):
    npoints = np.shape(xs)[0]
    f = np.zeros(npoints)
    for i in range(npoints):
        f[i] = F(xs[i,:])
    
    f_star = f[-1]
    dF = np.abs(f - f_star)
    iterations = np.linspace(1,npoints,npoints)
    
    fig,ax = plt.subplots()
    plt.ion()
    plt.plot(iterations, dF)
    plt.xlabel('Iteration')
    plt.ylabel('|f - f*|')
    ax.set_yscale('log')
    

    if(save_plot):
        filepath = os.path.join(save_directory, "Nelder Mead/NM_Convergence.jpg")
        plt.savefig(filepath)

    plt.pause(0.5)
    plt.close()


# ------------------------------------- Main body of Package ------------------------------------
def main():
    fun_num = 1
    method_list = [Nelder_Mead,genetic_algorithm,particle_swarm]
    method = method_list[1]
    plot_flag = False
    save_plot = False
    pop_size = 30   # size of population in each generations (Genetic/Particle Swarm)
    n_gen = 30      # number of generations (Genetic/Particle Swarm) or max generations (Nelder-Mead)
    mutation_rate = 0.005    # Genetic algorithm only. Should be on range (0.005, 0.05)

    if(fun_num == 1):
        f = egg_carton
        x0 = np.array([-0.5,-0.5])
        # Define other intial points we can use to test Nelder-Mead
        x0 = np.array([1,-2])
        x0 = np.array([-3,-3])
        x0 = np.array([1.1,3.5])
        x0 = np.array([0,2.4])
        the_bounds = ((-4,4),(-4,4))
    else:
        f = rosenbrock
        x0 = np.array([0,0])
        the_bounds = ((-4,4),(-4,4))

    # plot the initial function
    if(plot_flag):
        plot_function(f, the_bounds, save_plot)


    if(method == genetic_algorithm):
        #f,fitness, bounds, pop_size, generations,plot_flag
        # as a note: the population size should be divisible by 5 and 2. This is because we are pairing parents together and take the top 20% (1/5) of the parents and the top 80% (4/5) of the children.
        selection_methods = ["Roullette","Tournament"]
        selection = selection_methods[0]
        x_star, f_star, xs, gen = method(f,fit_func,bounds=the_bounds, pop_size=pop_size, generations=n_gen, selection=selection, plot_flag=plot_flag, save_fig = save_plot, mutation_rate=mutation_rate)
    elif(method == Nelder_Mead):
        x_star, f_star, xs, gen, x_store = method(f, x0, bounds=the_bounds, max_iter=n_gen, tol=1e-12, plot_flag=plot_flag, L=2,save_fig=save_plot) # You need a pretty low tolerance to get close to the optimizer
        res_NM = opt(f,x0,method="Nelder-Mead",tol=1e-12)
        if(plot_flag):
            convergence_NM(f,x_store,save_plot)
    elif(method == particle_swarm):
        x_star, f_star, xs, gen = method(f, bounds=the_bounds, pop_size=pop_size, generations=n_gen, plot_flag=plot_flag,save_fig=save_plot)

    print(f"x*_user = ", x_star, "f(x*) = ", f_star, "nfun = ", n_fun_local)
    if(plot_flag):  
        plot_final_point(f,x_star, gen, the_bounds)

    if(method == Nelder_Mead):
        print(f"x*_scipy = ", res_NM.x, "f(x*) = ", res_NM.fun, "nfun = ", res_NM.nfev)

    # x_star_opt = opt(f, x0, bounds=the_bounds)
    # print(f"x* = ", x_star_opt.x, "f(x*) = ", x_star_opt.fun)