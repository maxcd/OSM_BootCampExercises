'''
3 period lived OG model
'''
# import statments
import numpy as np
import scipy.optimize as opt

# define arameters
## households parameters
### discount factor
beta = 0.442
### risk aversion coefficient
sigma = 3.0
### depreciation rate per period
delta = 0.6415

## aggregate labor suppöy
n_1 = 1.0
n_2 = 1.0
n_3 = 0.2
nvec = np.array([n_1, n_2, n_3])

# firms parameters
## TFP
A = 1.0
## capital share of income
alpha = 0.35

# write down the functions
## HH
## marginal utility of consumption
def get_mu(c_s, sigma):
    return c_s ** (-1 * sigma)

## budget constraint in period s = 1
def get_c_1(n_1, w, b_2):
    return n_1 * w - b_2

### in period s = 2
def get_c_2(n_2, w, r, b_2, b_3):
    return n_2 * w + (1 + r) * b_2 - b_3
### in period s = 3
def get_c_3(n_3, w, r, b_3):
    return n_3 * w + (1 + r) * b_3

## Firms
def get_r(alpha, A, L, K, delta):
    return alpha * A * (L / K) ** (1 - alpha) - delta

def get_w(alpha, A, L, K):
    return (1 - alpha) * A * (K / L) ** alpha

# market clearing
## exogenous aggregate labor supply
def get_L(nvec):
    return nvec.sum()

## caital market clearing
def get_K(bvec):
    return bvec.sum()
# euler equations dependent on b_s2,, b_s3
def euler_errs(bvec, *args):
    beta, A, alpha, sigma, nvec, delta = args
    b_2, b_3 = bvec[0], bvec[1]
    K = get_K(bvec)
    L = get_L(nvec)
    w = get_w(alpha, A, L, K)
    r = get_r(alpha, A, L, K, delta)

    c1 = get_c_1(nvec[0], w, b_2)
    c2 = get_c_2(nvec[1], w, r, b_2, b_3)
    c3 = get_c_3(nvec[2], w, r, b_3)

    # first euler equation
    lhs_1 = get_mu(c1, sigma)
    rhs_1 = beta * (1 + r) * get_mu(c2, sigma)
    err_1 = lhs_1 - rhs_1

    # second euler equations
    lhs_2 = get_mu(c2, sigma)
    rhs_2 = beta * (1 + r) * get_mu(c3, sigma)
    err_2 = lhs_2 - rhs_2

    errors = np.array([err_1, err_2])
    return errors

# assign arbitrary initial values
b2_init = 0.05
b3_init = 0.05
b_init = np.array([b2_init, b3_init])

# putthe parameters in a tuple
params = (beta, A, alpha, sigma, nvec, delta)
errs_init = euler_errs(b_init, beta, A, alpha, sigma, nvec, delta)
# compute steady state
ss_results = opt.root(euler_errs, b_init, args=(params))
print(ss_results)
print(ss_results.x)
