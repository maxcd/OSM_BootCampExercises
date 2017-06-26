'''
3 period lived OG model
# Excersise 5.1
'''
# import statments
import numpy as np
import scipy.optimize as opt

# define arameters
## households parameters
### discount factor
beta_pa = 0.96
beta = beta_pa ** 20
### risk aversion coefficient
sigma = 3.0
### depreciation rate per period
delta_pa = 0.05
delta = 1 - (1 - 0.05) ** 20

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
    return c_s ** (-1.0 * sigma)

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

# put the parameters in a tuple
params = (beta, A, alpha, sigma, nvec, delta)
errs_init = euler_errs(b_init, beta, A, alpha, sigma, nvec, delta)

# compute steady state
ss_results = opt.root(euler_errs, b_init, args=(params))
print(ss_results)

# calculate the other steady state variables
b_ss = ss_results.x
b2_ss, b3_ss = b_ss

def get_SS(bvec):
    b2_ss, b3_ss = bvec
    K_ss = get_K(b_ss)
    L_ss = get_L(nvec)
    r_ss = get_r(alpha, A, L_ss, K_ss, delta)
    w_ss = get_w(alpha, A, L_ss, K_ss)

    c1_ss = get_c_1(n_1, w_ss, b2_ss)
    c2_ss = get_c_2(n_2, w_ss, r_ss, b2_ss, b3_ss)
    c3_ss = get_c_3(n_3, w_ss, r_ss, b3_ss)

    return [K_ss, L_ss, r_ss, w_ss, c1_ss, c2_ss, c3_ss]

def print_SS(var_list):
    _, _, r_ss, w_ss, c1_ss, c2_ss, c3_ss = var_list
    print('''\nSteady State\n\nr_bar:\t{}\nw_bar:\t{}\nc1_bar:\t{}\nc2_bar:\t{}\nc3_bar:\t{}
      '''.format(r_ss.round(3), w_ss.round(3), c1_ss.round(3),
                 c2_ss.round(3), c3_ss.round(3)))

ss_list = get_SS(b_ss)
print_SS(ss_list)

'''
# Excefrcise 5.2
What happens if agents get more patient i.e. beta increases?

When beta increases, first the interest rate will decrease because agents
propensity to borrow decreases. In turn they actually save more. This leads
to a higher capital stock in the Steady State. Since, wage equals marginal
product of labor which is an increasing function in the capital stock, the
is higher as well. Since frims produce more, due to the higher capital stock,
consumption is higher in every age. This result probabliy relies on the
exogenous labor supply to some extent
'''

'''
# Excersise 5.3: TPI
'''

# getting the savings of the middle aged cohoret when they are old
# check again for algebra errors
# def get_b32(b_2_1, wvec, rvec, nvec, beta, sigma):
#     n1, n2, n3 = nvec
#     w_1, w_2 = wvec[0], wvec[1]
#     r_1, r_2 = rvec[0], rvec[1]
#
#     top = n2 * w_1 + (1+r_1) * b_2_1 - n3 * (beta * (1+r_2)) ** (-1.0/sigma)
#     bottom = 1 + (beta * (1+r_2)) ** (-1.0/sigma) * (1+r_2)
#     b_3_2 = top / bottom
#
#     return b_3_2

def b23_errs(b32_init, *args):
    b_2_1, wvec, rvec, nvec, beta, sigma = args
    n1, n2, n3 = nvec
    w_1, w_2 = wvec[0], wvec[1]
    r_1, r_2 = rvec[1], rvec[2]

    c2 = get_c_2(n2, w_1, r_1, b_2_1, b32_init)
    c3 = get_c_3(n3, w_2, r_2, b32_init)

    lhs = get_mu(c2, sigma)
    rhs = beta * (1 + r_2) * get_mu(c3, sigma)

    err = lhs - rhs
    return err

# start actual TPI algorithm
def euler_errs_tpi(b_coh, *args):
    beta, A, alpha, sigma, nvec, delta, w_coh, r_coh = args
    b_2_i0, b_3_i1 = b_coh[0], b_coh[1]
    w_i0, w_i1, w_i2 = w_coh
    r_i2, r_i3 = r_coh

    c1 = get_c_1(nvec[0], w_i0, b_2_i0)
    c2 = get_c_2(nvec[1], w_i1, r_i2, b_2_i0, b_3_i1)
    c3 = get_c_3(nvec[2], w_i2, r_i3, b_3_i1)

    # first euler equation
    lhs_1 = get_mu(c1, sigma)
    rhs_1 = beta * (1 + r_i2) * get_mu(c2, sigma)
    err_1 = lhs_1 - rhs_1

    # second euler equations
    lhs_2 = get_mu(c2, sigma)
    rhs_2 = beta * (1 + r_i3) * get_mu(c3, sigma)
    err_2 = lhs_2 - rhs_2

    return np.array([err_1, err_2])

# guess T_max
T_max = 30

r_ss_5 = np.repeat(ss_list[2], 5)
w_ss_5 = np.repeat(ss_list[3], 5)

def tpi(b_ss, K_ss, T_max, max_iters, eps_tol, xi, *args):
    beta, A, alpha, sigma, nvec, delta = args

    eps_list = []
    repeat = True
    iters = 1
    m = 5
    periods = T_max + m
    while repeat == True:

        if iters == 1:
            # initiate Kvec for the first time
            b_2_1 = 0.8 * b_ss[0]
            b_3_1 = 1.1 * b_ss[1]
            K_init = b_2_1 + b_3_1
            Kvec = np.linspace(K_init, K_ss, periods)
            L = get_L(nvec)
        # execute this always
        rvec = get_r(alpha, A, L, Kvec, delta)
        rvec = np.concatenate((rvec, r_ss_5))

        wvec = get_w(alpha, A, L, Kvec)
        wvec = np.concatenate((wvec, w_ss_5))
        b23args = b_2_1, wvec, rvec, nvec, beta, sigma
        b_3_2 = opt.root(b23_errs, b_3_1, args=(b23args))

        # initiate bvec
        bvec = np.zeros([periods, 2])

        bvec[0,:] = b_2_1, b_3_1
        bvec[1,1] = b_3_2.x

        try:
            for i in range(periods-2):
                b_coh = np.array([bvec[i, 0], bvec[i+1, 1]])
                w_coh = wvec[i:i+3]
                r_coh = rvec[i+1:i+3]

                #errs = euler_errs_tpi(b_coh, beta, A, alpha, sigma, nvec, delta, w_coh, r_coh)
                coh_args = beta, A, alpha, sigma, nvec, delta, w_coh, r_coh
                res = opt.root(euler_errs_tpi, b_coh, args=(coh_args))

                b_cohi1 = res.x
                bvec[i+1,0], bvec[i+2, 1] = b_cohi1[0], b_cohi1[1]

        except ValueError as e:
            print('stopped at iteration {}'.format(i))

        K_i_prime = bvec.sum(axis=1)[:30]
        K_i = Kvec[:30]

        deviation = K_i_prime - K_i
        eps = np.linalg.norm(deviation)
        eps_list.append(eps)
        if eps <= eps_tol:
            print('Convergence achieved after {} iterations'.format({iters}))
            repeat = False
        else:
            Kvec = xi * K_i_prime + (1 - xi) * K_i

        iters += 1
        if iters == max_iters:
            print('maximum itterations reached')
            break

    return bvec

max_iters = 100
K_ss = ss_list[0]

bpath = tpi(b_ss, K_ss, T_max, max_iters, 1e-9,
            0.5, beta, A, alpha, sigma, nvec, delta)

# Excersise 5.4
## plot convergence of the capital stock
Kpath = bpath.sum(axis=1)
dev = Kpath - K_ss
T = np.where(dev <= 1e-4)
T = T[0][0]

print('''\nIt takes only {} period to get within 0.0001 of the Steady State.
      ...if my time path iteration is correct.'''.format(T))


import matplotlib.pyplot as plt
fig, ax = plt.subplots()
t = np.arange(1, 15)
ax.plot(t, Kpath[:14], label='capital stock')
ax.plot(t, np.repeat(K_ss, 14), '--', color='k',
        label='steady state', alpha=0.3)
plt.xlabel('Period')
plt.title('Transition to Steady State')
plt.legend()
plt.show()
