import numpy as np

def Euler(fprime, time_max,time_step):
    time = np.arange(0,time_max,time_step)
    f = []
    t = 0
    while t*time_step < time_max:
        f_t1 = f[t] + fprime * time_step
        f.append(f_t1)
        t+=1
    f_array = np.array(f)
    return time, f_array