import numpy as np

def Zipf_dist(N, alpha):
    Z = sum(1 / (i ** alpha) for i in range(1, N + 1))
    prob = [(1 / (i ** alpha)) / Z for i in range(1, N + 1)]
    # shuffle the list
    np.random.shuffle(prob)
    return prob

def Lyapunov_Function(zu, zi, n):
    ans = zu
    for i in range(n):
        ans += (zi[i] ** 2)
    return ans

def Lyapunov_Drift(L, L_past):
    return L - L_past