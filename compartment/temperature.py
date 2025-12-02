import jax.numpy as np
from jax import lax

#Rewriting temperature_seasonality function in jax 
# 0 north, 1 south
def temperature_seasonality_jax(Tmax, Tmin, Tmean, numeric_day, hemisphere):
    '''
    Calculates temperature using a sin curve to represent seasonality for each day of simulation.
    Compatible with JAX transformations (e.g., odeint, grad).
    '''
    
    def temp_north(_):
        temp = ((Tmax-Tmin)/2)*np.cos((2*np.pi/365)*(numeric_day+(365/2))) + Tmean
        return np.clip(temp, Tmin, Tmax)

    def temp_south(_):
        temp = ((Tmax-Tmin)/2)*np.cos((2*np.pi/365)*numeric_day) + Tmean
        return np.clip(temp, Tmin, Tmax)

    return lax.cond(hemisphere == 0, temp_north, temp_south, operand=None)

def briere_jax(temp, c, Tmin, Tmax):
    '''
    Fits Briere function to calculate thermal responses.
    JAX-compatible version using lax.cond.
    '''
    def in_range(_):
        return c * temp * (temp - Tmin) * np.sqrt(Tmax - temp)

    def out_of_range(_):
        return 0.0

    # Check if temp is within [Tmin, Tmax]
    return lax.cond(
        (temp < Tmin) | (temp > Tmax),
        out_of_range,
        in_range,
        operand=None
    )
    
def quadratic_jax(temp, c, Tmin, Tmax):
    ''' Fits quadratic function to calculate thermal responses '''
    def in_range(_):
        return c*(temp-Tmin)*(temp-Tmax)

    def out_of_range(_):
        return 0.0

    # Check if temp is within [Tmin, Tmax]
    return lax.cond(
        (temp < Tmin) | (temp > Tmax),
        out_of_range,
        in_range,
        operand=None
    )

def calculate_death_rate_jax(temp, c, Tmin, Tmax):
    ''' Fits quadratic function to calculate the inverse of mosquito lifespan as death rate (see pg. 7 of Huber) '''
    def in_range(_):
        return 1/(c*(temp-Tmin)*(temp-Tmax))

    def out_of_range(_):
        return 24.0

    # Check if temp is within [Tmin, Tmax]
    return lax.cond(
        (temp <= Tmin) | (temp >= Tmax),
        out_of_range,
        in_range,
        operand=None
    )

def calculate_surviving_offspring(T0=29):
    l_V_T0 = briere_jax(T0, 8.56e-3, 14.58, 34.61) #number of eggs laid per female
    s_V_T0 = quadratic_jax(T0, -5.99e-3, 13.56, 38.29)  # probability of egg-to-adult survival
    d_V_T0 = briere_jax(T0, 7.86e-5, 11.36, 39.17) #mosquitio egg-to-adult development rate
    mu_V_T0 = calculate_death_rate_jax(T0, -1.48e-1, 9.16, 37.73)

    return (l_V_T0 * s_V_T0 * d_V_T0) / mu_V_T0, mu_V_T0

def get_carrying_capacity(temperature, vector_surviving_offspring, mu_V_T0, N_v_m, E_a, N, T0, k):
    return ((vector_surviving_offspring - mu_V_T0) / vector_surviving_offspring) * N_v_m * N * np.exp((-E_a) * ((temperature - T0)**2) / (k * (temperature + 273.15) * (T0 + 273.15)))

def calculate_thermal_responses(temperature):
    l_V_T = briere_jax(temperature, 8.56e-3, 14.58, 34.61)          # number of eggs laid per female
    s_V_T= quadratic_jax(temperature, -5.99e-3, 13.56, 38.29)       # probability of egg-to-adult survival
    d_V_T = briere_jax(temperature, 7.86e-5, 11.36, 39.17)          # mosquitio egg-to-adult development rate
    epsilon_V_T = briere_jax(temperature, 6.65e-5, 10.68, 45.90)    # Virus extrinsic incubation rate
    delta_V_T = briere_jax(temperature, 4.91e-4, 12.22, 37.46)      # Probability of infection per bite on infectious host TODO - double check that this is correct
    gamma_T = briere_jax(temperature, 8.49e-4, 17.05, 35.83)        # Probability of mosquito infectiousness
    b_V_T = briere_jax(temperature, 2.02e-4, 13.35, 40.08)          # Biting rate
    mu_V_T = calculate_death_rate_jax(temperature, -1.48e-1, 9.16, 37.73)

    return l_V_T, s_V_T, d_V_T, epsilon_V_T, delta_V_T, gamma_T, b_V_T, mu_V_T