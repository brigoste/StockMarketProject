import scipy as sci
from scipy.optimize import minimize as opt
import numpy as np
import random 
from scipy.stats import qmc
import quasi_Newton_method as qNm
import FiniteDifference as FD
import matplotlib.pyplot as plt

def genetic_algorithm(f, fitness, bounds, pop_size, generations, dims, selection = "Roullette", plot_flag=False, save_fig=True, mutation_rate = 0.05):
    
    # with how many variables we have, I don't know how I want to plot yet.
    # # define values for plotting
    # nx = 10
    # x_range = np.linspace(bounds[0][0], bounds[0][1], nx)
    # y_range = np.linspace(bounds[1][0], bounds[1][1], nx)
    # X, Y = np.meshgrid(x_range, y_range)
    # Z = np.zeros_like(X)
    # for i in range(X.shape[0]):
    #     for j in range(X.shape[1]):
    #         Z[i, j] = f([X[i, j], Y[i, j]])

    # Generate initial population using latin hypercube sampling
    dims = dims                          # the dimension of the problem. How many design variables.

    # seperate lower and upper bounds to instantiate the Latin Hypercube sample space.
    l_bounds = np.zeros(dims)
    u_bounds = np.zeros(dims)
    for i in range(dims):
        l_bounds[i] = bounds[i][0]   
        u_bounds[i] = bounds[i][1]

    sampler = qmc.LatinHypercube(d=dims)                       # change for number of dimensions (I think we have 5)
    sample = sampler.random(n=pop_size)
    pop = qmc.scale(sample, l_bounds, u_bounds)

    # if(plot_flag):
    #     plot_generation_GA(f,pop, 0, bounds,save_fig)

    # Evaluate fitness    
    fit = fitness(pop,f,pop_size)   # values bewteen 0 and 1. 1 is the best.
    # Sort population by fitness
    idx = np.argsort(-fit)      # sort in descending order       
    pop = pop[idx]
    fit = fit[idx]
    # Loop through generations
    for i in range(generations):
        # print(f"Generation {i+1}")
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
            children[2*j] = 0.5*(parent_pairs[j,0:dims] + parent_pairs[j,dims:2*dims])        # child 1: average of 2 parents
            children[(2*j)+1] = 2*(parent_pairs[j,0:dims])-parent_pairs[j,dims:2*dims]            # child 2: linear fit of 2 parents (weighted toward better parent)

        # Mutation
        p = mutation_rate # mutation rate (0.005 to 0.05 generally)
        n_muations = 0    # internal check to see how many mutation occur
        for j in range(pop_size):
            mutate_prop = np.random.random()
            if(mutate_prop < p):
                # mutate the child
                n_muations+=1
                for k in range(dims):
# mutate the child by add/sub a random number from a normal distribution
                    sigma = random.gauss(0, 0.5)     # (mean, std)                              # Do I scale this for our problem?
                    children[j,k] = children[j,k] + sigma

        # sort the children, like the parents, by fitness. 
        fit_c = fitness(children,f,pop_size)
        idx = np.argsort(-fit_c)      # sort in descending order
        children = children[idx]
        fit_c = fit_c[idx]

        # I want to set some of the parents as the children
        # New generation consists of the top 20% of the parents and the top 80% of the children
        pop_top = pop[0:int(pop_size/5)]    # top 20% of the parents
        children_top = children[:-int(pop_size/5)]  # top 80% of the children
        new_gen = np.vstack([pop_top,children_top])

        # apply boundary conditions
        for j in range(pop_size):
            for k in range(dims):
                if(new_gen[j,k] < bounds[k][0]):
                    new_gen[j,k] = bounds[k][0]
                elif(new_gen[j,k] > bounds[k][1]):
                    new_gen[j,k] = bounds[k][1]

        # Set new population as children
        pop = new_gen          #children become the new population
        # sort the population by fitness
        fit = fitness(pop,f,pop_size)
        idx = np.argsort(-fit)      # sort in descending order
        pop = pop[idx]
        fit = fit[idx]

        # if(plot_flag):
        #     plot_generation_GA(f,pop, i+1, bounds,save_fig)
    # [rr,sr, br, si, bi, dca_i, b,a] = pop[0]
    return pop[0],f(pop[0]), pop, generations #x_star, f_star, xs, gen

def roullette_selection(fit,pop,pop_size,dims): #Pair parents together roullete style (this is the way to date)
    Parent_pairs = np.zeros([int(pop_size/2),2*dims])
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
                    parent[k*dims:((k+1))*dims] = pop[-1]
                    # parent[2*k:(2*k)+2] = (np.size(pop)/2)-1        #index istead of values
                    break
                elif(r < S[j]):
                    parent[k*dims:((k+1))*dims] = pop[j]        
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
    Parent_pairs = np.zeros([int(pop_size/2), 2*dims])
    for i in range(int(pop_size/2)):
        # Select parent 1
        parent = np.zeros(4)
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
