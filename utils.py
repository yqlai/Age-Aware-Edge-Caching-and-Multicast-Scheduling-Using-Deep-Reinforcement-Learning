import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

def plot_AAoI(arrs_aoi, T, window=1, labels=None):

    # Let L be the minimum length of all the arrays
    L = min([len(arr) for arr in arrs_aoi])

    # Compute the moving average of aoi_arr
    for i in range(len(arrs_aoi)):
        data = pd.Series(arrs_aoi[i][:L])
        moving_avg = data.rolling(window=window).mean()
        if labels:
            plt.plot(range(L-window+1), moving_avg[:(L-window+1)], label=labels[i])
        else:
            plt.plot(range(L-window+1), moving_avg[:(L-window+1)])

    # plt.ylim(0, 30)

    plt.xlabel('Time')
    plt.ylabel('Average Age of Information')
    plt.title('Average Age of Information vs Time')
    
    plt.legend(title='Method')
    
    plt.show()