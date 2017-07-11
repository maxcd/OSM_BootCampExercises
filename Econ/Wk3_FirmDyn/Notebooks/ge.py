'''
write a programm to solve for the generall equilibrium of
the model with heterogenuos firms and adjustment costs
'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from numba import jit
from scipy.optimize import fminbound, root
# imports for Ada Cooper
from scipy.stats import norm
import scipy.integrate as integrate

'''
    Define he functinos to be numbafied first
'''
@jit
def get_VF_outside(V_init, Vmat, shocks, e, sizek, betafirm):

    for k in range(len(shocks)): # looop over z'
        z = shocks[k]
        for i in range(sizek):  # loop over k
            for j in range(sizek):  # loop over k'
                Vmat[i, j, k] = e[i, j, k] + betafirm * V_init[j, k]
    return Vmat

@jit
def get_Gamma_outside(Gamma, HGamma, sizez, sizek, Pol_Fun, Pi):
    for i in range(sizez):  # z
        for j in range(sizek):  # k
            for m in range(sizez):  # z'
                HGamma[m, Pol_Fun[i, j]] = \
                    HGamma[m, Pol_Fun[i, j]] + Pi[i, m] * Gamma[i, j]
    return HGamma

class firms_model(object):

    '''
    parameters

    alpha_k : prod function parameters for capital
    alpha_l : prod fun parameter for labor
    delta : depreciation rate
    psi : coeficient on adjustment cost
    r : riskless rate
    sigma_eps : std dev of disturbances to productivity
    N : number of shocks to approximate
    mu : mean of shocks to disturbances
    '''

    def __init__(self, w=0.7, h=6.616, alpha_k=0.29715, alpha_l=0.65 ,delta=0.154,
                 psi=1.08, r=0.04, sigma_eps=0.213, rho=0.7605, N=9, dens=1,
                 mu=0.0, num_draws=100000):

        self.w = w # am imitial guess for the wage  rate

        self.h = h # labor disutillity coefficient on households

        self.alpha_k = alpha_k
        self.alpha_l = alpha_l
        self.delta = delta
        self.psi = psi
        self.r = r
        self.sigma_eps = sigma_eps

        self.rho = rho
        self.sigma_z = self.sigma_eps / ((1 - rho ** 2) ** (1 / 2))
        self.betafirm = (1 / (1 + self.r))
        self.mu = mu
        self.N = N
        self.draws = num_draws
        self.eps = np.random.normal(self.mu,self.sigma_z, size=self.draws)

        self.dens = dens # density of the grid for k

        self.V = None
        self.Pol_Fun = None

        # setup some matrices for the VFI loop
        # self.Vmat = np.zeros((self.sizek, self.sizek, self.N))
        # self.V = np.zeros((self.sizek, self.N))

    def update_wage(self, new_wage):
        self.w = new_wage

    def make_z(self):
        #N, mu, sigma_z, rho, sigma_eps = self.N, self.mu, self.sigma_z, self.rho, self.sigma_eps
        z = np.empty(self.draws)
        z[0] = 0.0 + self.eps[0]
        for i in range(1, self.draws):
            z[i] = self.rho * z[i - 1] + (1 - self.rho) * self.mu + self.eps[i]

        z_cutoffs = (self.sigma_z * norm.ppf(np.arange(self.N + 1) / self.N)) + self.mu
        z = ((self.N * self.sigma_z * (norm.pdf((z_cutoffs[:-1] - self.mu) / self.sigma_z)
                              - norm.pdf((z_cutoffs[1:] - self.mu) / self.sigma_z))) + self.mu)


        # define function that we will integrate
        def integrand(x, sigma_z, sigma_eps, rho, mu, z_j, z_jp1):
            val = (np.exp((-1 * ((x - mu) ** 2)) / (2 * (sigma_z ** 2)))
                * (norm.cdf((z_jp1 - (mu * (1 - rho)) - (rho * x)) / sigma_eps)
                   - norm.cdf((z_j - (mu * (1 - rho)) - (rho * x)) / sigma_eps)))
            return val

        pi = np.empty((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):

                results = integrate.quad(integrand, z_cutoffs[i], z_cutoffs[i + 1],
                                         args = (self.sigma_z, self.sigma_eps, self.rho, self.mu,
                                                 z_cutoffs[j],
                                                 z_cutoffs[j + 1]))
                pi[i,j] = (self.N / np.sqrt(2 * np.pi * self.sigma_z ** 2)) * results[0]

        self.z_grid = np.exp(z)
        self.pi = pi


    def make_k(self):
        dens, betafirm= self.dens, self.betafirm
        delta, alpha_l, alpha_k = self.delta, self.alpha_l, self.alpha_k
        # put in bounds here for the capital stock space
        kstar = ((((1 / betafirm - 1 + delta) * ((self.w / alpha_l) **
                                                 (alpha_l / (1 - alpha_l)))) /
                 (alpha_k * (1 ** (1 / (1 - alpha_l))))) **
                 ((1 - alpha_l) / (alpha_k + alpha_l - 1)))
        kbar = 2*kstar
        lb_k = 0.001
        ub_k = kbar
        krat = np.log(lb_k / ub_k)
        numb = np.ceil(krat / np.log(1 - delta))

        K = np.zeros(int(numb * dens))
        # we'll create in a way where we pin down the upper bound - since
        # the distance will be small near the lower bound, we'll miss that by little
        for j in range(len(K)):
            K[j] = ub_k * (1 - delta) ** (j / dens)
        kvec = K[::-1]
        sizek = kvec.shape[0]

        self.kvec= kvec
        self.sizek = sizek

    def op_ret(self, z): #alpha_l, alpha_k, w, z, kvec):
        opp = ((1 - self.alpha_l) * (self.alpha_l / self.w) ** (self.alpha_l / (1 - self.alpha_l))*
               (self.kvec ** (self.alpha_k / (1 - self.alpha_l))) * z ** (1 / (1 -self.alpha_l)))
        return opp

    def cost_fun(self, k, k_prime):
        adj_cost = (self.psi / 2) * ((k_prime - ((1 - self.delta) * k)) ** 2) / k
        net_invest_cost = k_prime - (1 - self.delta) * k

        total_cost = adj_cost + net_invest_cost
        return total_cost

    def make_flows(self):
        #e = get_flows_outside(self.z_grid, self.sizek, self.sizek, self.N)
        e = np.zeros((self.sizek, self.sizek, self.N))
        for k in range(self.N):
            z = self.z_grid[k]
            op = self.op_ret(z)
            for i in range(self.sizek):
                for j in range(self.sizek):
                    e[i, j, k] = op[i] - self.cost_fun(k=self.kvec[i], k_prime=self.kvec[j])

        self.e = e


    def plot_val_funs(self):
        fig, ax = plt.subplots(figsize=(9, 6))
        #ax.set_ylim(-40, 10)
        ax.set_xlim(np.min(self.kvec), np.max(self.kvec))
        for k in range(self.N):
            lb = 'z={}'.format(np.around(self.z_grid[k], 2))
            ax.plot(self.kvec, self.V[:,k], color=plt.cm.jet(k / self.N), lw=2, alpha=0.6, label=lb)

        plt.legend(loc="upper left") #bbox_to_anchor=(1.05,1)
        plt.ylabel('Value Function')
        plt.xlabel('Size of Capital Stock')
        plt.title('Value Function')
        plt.show()


    def get_VF(self, V_init, Vmat):
        #e, N, sizek = self.e, self.N, self.sizek
        #Vmat = np.zeros((self.sizek, self.sizek, self.N))
        Vmat = get_VF_outside(V_init, Vmat, self.z_grid, self.e, self.sizek, self.betafirm)

        return Vmat

    def VFI(self, VFmaxiter=3000, VFtol=1e-6, plot_result=True, message=True):
        Pol_Fun = np.ones((self.N, self.sizek), dtype=int)
        total_start_time = time.clock()
        #for k in range(len(z_shocks)):
        V = np.zeros((self.sizek, self.N))  # initial guess at value function
        Vmat = np.zeros((self.sizek, self.sizek, self.N))  # initialize Vmat matrix
        VFiter = 1
        VFdist = 3

        start_time = time.clock()
        while VFdist > VFtol and VFiter < VFmaxiter:

            TV = np.copy(V)
            TV_e = TV @ self.pi.T

            Vmat = self.get_VF(TV_e, Vmat)

            V = Vmat.max(axis=1)                    # apply max operator to Vmat (to get V(k))
            #Vstore[:, VFiter, :] = V     # store value function at each iteration for graphing later
            VFdist = (np.absolute(V - TV)).max()
            VFiter += 1

            time_elapsed = time.clock() - total_start_time

        if message:
            if VFiter < VFmaxiter:
                print('\nValue function converged after', VFiter, 'iterations')
                print('\ntotal time elapsed:', time_elapsed, 'seconds')
            else:
                print('Value function did not converge')

        Pol_Fun = np.argmax(Vmat, axis=1)

        self.V = V
        self.Pol_Fun = Pol_Fun

        if plot_result:
            self.plot_val_funs()

    def plot_policy(self):
        fig, ax = plt.subplots()
        for k in range(self.N):
            optK = self.kvec[self.Pol_Fun[:,k]]
            optI = optK - (1 - self.delta) * self.kvec
            lb = 'z={}'.format(np.around(self.z_grid[k], 2))
            ax.plot(self.kvec, optK, '--', color=plt.cm.jet(k / self.N), lw=2, alpha=0.8, label=lb)
        ax.plot(self.kvec, self.kvec, 'k:', label='45 degree line')
        ax.plot(self.kvec, self.kvec * (1-self.delta),  'k*', label='$(1-\delta)$', alpha=0.8)
        # Now add the legend with some customizations.
        legend = ax.legend(bbox_to_anchor=(1.0, 1.05))
        # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
        frame = legend.get_frame()
        frame.set_facecolor('0.90')
        # Set the fontsize
        for label in legend.get_texts():
            label.set_fontsize('large')
        for label in legend.get_lines():
            label.set_linewidth(1.5)  # the legend line width
        plt.xlabel('Size of Capital Stock')
        plt.ylabel('Optimal Choice of Capital Next Period')
        plt.title('Policy Function for initial states of z')
        plt.show();

    def solve_and_plot(self):
        self.make_z()
        self.make_k()
        self.make_flows()
        self.VFI()
        self.plot_policy()

    def make_opt_k(self):
        k_opt = np.zeros((self.N, self.sizek))
        for k in range(self.N):
            k_opt[k,:] = self.kvec[self.Pol_Fun[:,k]]

        self.k_opt= k_opt

    def get_funs(self):
        self.make_z()
        self.make_k()
        self.make_flows()
        self.VFI(plot_result=False, message=False)
        self.make_opt_k()

    def make_labor_d(self):
        alpha_l, alpha_k, = self.alpha_l, self.alpha_k
        one_alph = (1 / (1 - alpha_l))
        ka = self.kvec ** (alpha_k * one_alph)

        l = np.zeros((self.N, self.sizek))
        for i,z in enumerate(self.z_grid):
            l[i,:] = (alpha_l / self.w) ** one_alph * z ** one_alph * ka

        self.labor_d = l
        return l

    def make_inv_d(self):
        inv_d = self.k_opt - (1 - self.delta) * self.kvec.T

        self.inv_d = inv_d

    def get_Gamma(self, Gamma, HGamma):
        Pi, Pol_Fun, sizek = self.pi, self.Pol_Fun, self.sizek
        sizez, z, kvec = self.N, self.z_grid, self.kvec

        HGamma = get_Gamma_outside(Gamma, HGamma, sizek, sizez, Pol_Fun, Pi)
        return HGamma

    def solve_dist(self, Dtol = 1e-12, Dmaxiter=1000, message=True):
        Pi, Pol_Fun, sizek = self.pi, self.Pol_Fun.T, self.sizek
        sizez, z, kvec = self.N, self.z_grid, self.kvec

        ' initiaize  Gamma'
        Gamma = np.ones((sizez, sizek)) * (1 / (sizek * sizez))
        Diters = 0
        Ddist = 7.0    # some value to get the loop going

        '''
        ------------------------------------------------------------------------
        Compute the stationary distribution of firms over (k, z)
        ------------------------------------------------------------------------
        Dtol     = tolerance required for convergence of SD
        dist    = distance between last two distributions
        iters    = current iteration
        Dmaxiter = maximium iterations allowed to find stationary distribution
        Gamma     = stationary distribution
        HGamma    = operated on stationary distribution
        ------------------------------------------------------------------------
        '''

        while Ddist > Dtol and Dmaxiter > Diters:
            HGamma = np.zeros((sizez, sizek))
            HGamma = get_Gamma_outside(Gamma, HGamma, sizez, sizek, Pol_Fun, Pi)
            # for i in range(sizez):  # z
            #     for j in range(sizek):  # k
            #         for m in range(sizez):  # z'
            #             HGamma[m, Pol_Fun[i, j]] = \
            #                 HGamma[m, Pol_Fun[i, j]] + Pi[i, m] * Gamma[i, j]
            Ddist = (np.absolute(HGamma - Gamma)).max()
            Gamma = HGamma
            Diters += 1

        if message:
            if Diters < Dmaxiter:
                print('\nfirms distribution converged after', Diters,'iterations: ')
            else:
                print('\nfirms distribution did not converge')

        ''' firms_dist[i, k] where i is the productivity shocks
            and k is the size of the capital stock
        '''

        self.firms_dist = Gamma

    def aggregate(self):
        firms_capital = self.firms_dist.sum(axis=0)

        L_d_bar =  self.labor_d * self.firms_dist

        I_bar = (self.k_opt - (1 - self.delta) * self.kvec) * self.firms_dist

        invest = (self.k_opt - (1 - self.delta) * self.kvec) / self.kvec
        Psi_bar = (self.psi / 2) * invest ** 2 * self.kvec * self.firms_dist

        cap_prod = self.z_grid.reshape(9,1) * self.kvec.T ** self.alpha_k
        labor_prod = self.labor_d ** self.alpha_l
        Y_bar = cap_prod * labor_prod * self.firms_dist

        self.L_d_bar = L_d_bar.sum()
        self.I_bar = I_bar.sum()
        self.Psi_bar = Psi_bar.sum()
        self.Y_bar = Y_bar.sum()

        C_bar = self.Y_bar - self.I_bar - self.Psi_bar
        self.C_bar = C_bar

        L_s = self.w / (self.h * self.C_bar)

        self.L_s_bar = L_s

        return self.L_d_bar, self.L_s_bar, self.I_bar, self.Psi_bar, self.Y_bar, self.C_bar

    def make_labor_s(self):

        L_s = self.w / (self.h * self.C_bar)

        self.L_s_bar = L_s
        return L_s

    def check_L_clearing(self, w=None):
        if w is None:
            w = self.w

        self.update_wage(w)
        #self.make_k()
        self.make_flows()
        self.VFI(plot_result=False, message=False)
        self.make_opt_k()
        self.make_labor_d()
        self.make_inv_d
        self.solve_dist(message=False)
        self.aggregate()

        diff = self.L_s_bar - self.L_d_bar
        return diff

    def get_SS(self):
        result = root(self.check_L_clearing, self.w, tol=1e-5)
        w = result.x[0]
        self.update_wage(w)
        return result

    def plot_dist(self):
        fig, ax = plt.subplots( figsize=(12,4))
        ax.plot(self.kvec, self.firms_dist.sum(axis=0))
        ax.set_xlabel('Size of Capital Stock')
        ax.set_ylabel('Density')
        #axes[0].suptitle('Stationary Distribution over Capital')

        #axes[1].plot(np.log(self.z_grid), self.firms_dist.sum(axis=1))
        #axes[1].set_xlabel('Log Productivity')
        #axes[1].set_ylabel('Density')

        plt.suptitle('Stationary Distribution of Firms')
        plt.show()

# model = firms_model()
# model.make_z()
# model.make_k()
# model.make_flows()
# model.check_L_clearing()
# results = model.get_SS()
# model.check_L_clearing()
# model.plot_dist()
#1.05969465
#result = fminbound(model.check_L_clearing, 0, 3, xtol=1e-6, maxfun=30, full_output=True, disp=3)
