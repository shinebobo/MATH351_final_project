import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, lagrange
from time import time

def lagrange_approximate(x_list, f):
    y = f(x_list)
    approx = lagrange(x_list, y)
    return approx

def cubic_spline(x_list, f):
    y = f(x_list)
    approx = CubicSpline(x_list, y)
    return approx


def linear_regression_1st(x_list, f):
    n = len(x_list)
    X = np.zeros((n, 2))
    w = np.zeros((2, 1))
    b = np.zeros((n, 1))
    
    for i, x in enumerate(x_list):
        X[i][0] = 1
        X[i][1] = x
        b[i] = f(x)
    
    w = np.matmul(np.linalg.inv(np.matmul(np.transpose(X), X)), np.matmul(np.transpose(X), b))
    
    return np.poly1d(np.flip(w.flatten()))

def linear_regression_2nd(x_list, f):
    n = len(x_list)
    X = np.zeros((n, 3))
    w = np.zeros((3, 1))
    b = np.zeros((n, 1))
    
    for i, x in enumerate(x_list):
        X[i][0] = 1
        X[i][1] = x
        X[i][2] = x**2
        b[i] = f(x)
    
    w = np.matmul(np.linalg.inv(np.matmul(np.transpose(X), X)), np.matmul(np.transpose(X), b))
    
    return np.poly1d(np.flip(w.flatten()))

def linear_regression_3rd(x_list, f):
    n = len(x_list)
    X = np.zeros((n, 4))
    w = np.zeros((4, 1))
    b = np.zeros((n, 1))
    
    for i, x in enumerate(x_list):
        X[i][0] = 1
        X[i][1] = x
        X[i][2] = x**2
        X[i][3] = x**3
        b[i] = f(x)
    
    w = np.matmul(np.linalg.inv(np.matmul(np.transpose(X), X)), np.matmul(np.transpose(X), b))
    
    return np.poly1d(np.flip(w.flatten()))


def f_1(input):
    return input**4 + 3*input**3 + 2*input**2 + 3*input + 5

def f_2(input):
    return np.exp(input)

def f_3(input):
    return np.sin(input)

def random_choose(dataset, point_num):
    dataset = np.random.choice(dataset, size=point_num-1, replace = False)
    return np.sort(dataset)


def validation(approx_funct, x_list, y_list):
    mse = 0
    for x, y in zip(x_list, y_list):
        mse = mse + ((y - approx_funct(x))**2)/2
    return mse

def result_plot(mse_list, time_list, name_list, label):
    plt.subplot(1,2,1)
    plt.plot(name_list, mse_list, label=label)
    plt.title("MSE")
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(name_list, time_list, label=label)
    plt.title("TIME")
    plt.legend()
    
def graph_plot(approx_funct, real_funct):
    x = np.linspace(-5, 5, 10, endpoint= False)
    y_exact = real_funct(x[:80])
    y_approx = approx_funct(x[:80])
    plt.plot(x[:80], y_exact, label = 'exact')
    plt.plot(x[:80], y_approx, label = 'approximate')
    plt.legend()

def graph_all_plot(approx_funct_list, real_funct, name_list, given_points):
    x = np.linspace(-5, 5, 10, endpoint= False)
    y_exact = real_funct(x)
    y_given = real_funct(given_points)
    n = len(name_list)
    
    for i in range(n):
        y_approx = approx_funct_list[i](x)
        plt.subplot(1,n,i+1)
        plt.plot(x, y_exact, label = 'exact')
        plt.plot(given_points, y_given, '-o')
        plt.plot(x, y_approx, label = 'approximate')
        plt.title(name_list[i])
        plt.legend()
        
        

EXP_LIST = [lagrange_approximate, cubic_spline, linear_regression_1st, linear_regression_2nd, linear_regression_3rd]
NAME_LIST = ["Lagrange", "Cubic", "Reg_1st", "Reg_2nd", "Reg_3rd"]
MSE_LIST = []
TIME_LIST = []
APPROX_LIST = []
EXACT_F = f_1
NUM_DATA = 2

x_dataset = np.linspace(-5, 5, 10, endpoint=False)
y_dataset = EXACT_F(x_dataset)
x_set = random_choose(x_dataset, num_data)
print(x_set)
fig = plt.figure(figsize =(16, 8))

for exper in EXP_LIST:
    compute_time = 0
    mse = 0
    start = time()
    approx = exper(x_set, EXACT_F)
    end = time()
    mse = validation(approx, x_dataset, y_dataset)
    compute_time = (end - start)
    APPROX_LIST.append(approx)
    MSE_LIST.append(mse)
    TIME_LIST.append(compute_time)
result_plot(MSE_LIST, TIME_LIST, NAME_LIST, str(NUM_DATA))

print(MSE_LIST)
print(TIME_LIST)

fig = plt.figure(figsize =(16, 8))
graph_all_plot(APPROX_LIST,f_1, NAME_LIST, x_set)
        
         