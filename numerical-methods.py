import numpy as np

def Euler(fprime, time_max,time_step)->tuple[np.ndarray, np.ndarray]:
    """Generates array of `f(t)` values by Euler's Method.

    Args:
        fprime (Callable): Function `f'(t)` for unknown function `f(t)`. Callable functions, returns a 'float'.
        time_max (float): Final time in iteration
        time_step (float): Time difference `Δt` between evaluated points

    Returns:
        time (ArrayLike):  Array of time values from `t=0` to `t=time_max` evenly spaced by `Δt`. 
        f_array  (ArrayLike): Array of approximated `f(x)` values.
    """

    time = np.arange(0,time_max,time_step)
    f = []
    t = 0
    while t*time_step < time_max:
        f_t1 = f[t] + fprime * time_step
        f.append(f_t1)
        t+=1
    f_array = np.array(f)
    return time, f_array

def Armijo(x,f,g,p,c,maxa,r):
    """Produces optimal step length `α` by Armijo Backtracking line search

    Args:
        x (ArrayLike): Initi
        f (Callable): _description_
        g (Callable): _description_
        p (ArrayLike): _description_
        c (float): _description_
        maxa (float): _description_
        r (float): _description_

    Returns:
        a (float): Optimal step length, satisfying Weak Wolfe conditions
    """
    v=f(x)
    ai = maxa
    d = ((g.T)@p).item()
    v1 = f(x+(maxa*p))
    while v1 > v + c*(ai*d):
        ai = ai * r
        v1 = f(x+ai*p)
    a = ai
    return a