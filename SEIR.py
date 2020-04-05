import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
import datetime


class SEIR:

    def __init__(self):

        # Initial stat for SEIR model
        self.s = 0.
        self.e = 0.
        self.i = 0.
        self.r = 0.

        self.N = 0

        self.max_days = 0

        # Define default varlues
        self.T_inc = 5.2  # average incubation period
        self.T_inf = 2.9  # average infectious period
        self.R_0 = 3.954  # reproduction number

        self.init_SEIR()

    def set_params(self, T_inc, T_inf, R_0):
        self.T_inc = T_inc
        self.T_inf = T_inf
        self.R_0 = R_0

    def read_data(self):

        train = pd.read_csv('./input/covid19-global-forecasting-week-2/train.csv')
        train['Date_datetime'] = train['Date'].apply(lambda x: (datetime.datetime.strptime(x, '%Y-%m-%d')))

        pop_info = pd.read_csv('./input/covid19-population-data/population_data.csv')

        Country = 'Hubei'
        self.N = pop_info[pop_info['Name'] == Country]['Population'].tolist()[0]  # Hubei Population

        # Load dataset of Hubei
        self.train_loc = train[train['Country_Region'] == Country].query('ConfirmedCases > 0')
        if len(self.train_loc) == 0:
            self.train_loc = train[train['Province_State'] == Country].query('ConfirmedCases > 0')

    def init_SEIR(self):

        self.read_data()

        n_infected = self.train_loc['ConfirmedCases'].iloc[0]  # start from first comfirmedcase on dataset first date
        self.max_days = len(self.train_loc)  # how many days want to predict

        self.s = (self.N - n_infected) / self.N
        self.i = n_infected / self.N

    def dS_dt(self, S, I, R_t, T_inf):
        return -(R_t / T_inf) * I * S

    def dE_dt(self, S, E, I, R_t, T_inf, T_inc):
        return (R_t / T_inf) * I * S - (T_inc ** -1) * E

    def dI_dt(self, I, E, T_inc, T_inf):
        return (T_inc ** -1) * E - (T_inf ** -1) * I

    def dR_dt(self, I, T_inf):
        return (T_inf ** -1) * I

    def SEIR_model(self, t, y, R_t, T_inf, T_inc):
        if callable(R_t):
            reproduction = R_t(t)
        else:
            reproduction = R_t

        S, E, I, R = y

        S_out = self.dS_dt(S, I, reproduction, T_inf)
        E_out = self.dE_dt(S, E, I, reproduction, T_inf, T_inc)
        I_out = self.dI_dt(I, E, T_inc, T_inf)
        R_out = self.dR_dt(I, T_inf)

        return [S_out, E_out, I_out, R_out]

    def solve(self):
        sol = solve_ivp(self.SEIR_model, [0, self.max_days], [self.s, self.e, self.i, self.r],
                        args=(self.R_0, self.T_inf, self.T_inc),
                        t_eval=np.arange(self.max_days))

        return sol
