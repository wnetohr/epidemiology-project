# Imports
from scipy.integrate import odeint
import numpy as np
import plotly.graph_objects as go
from ipywidgets import interact
import os
import plotly.io as pio
from ipywidgets import IntSlider, FloatSlider, Layout, Dropdown, interact_manual, VBox, HBox
import ipywidgets as widgets
from ipywidgets import IntSlider, FloatSlider, Layout

def ode_model(z, t, beta, sigma, gamma, mu):
    S, E, I, R, D = z
    N = S + E + I + R + D
    dSdt = -beta*S*I/N
    dEdt = beta*S*I/N - sigma*E
    dIdt = sigma*E - gamma*I - mu*I
    dRdt = gamma*I
    dDdt = mu*I
    return [dSdt, dEdt, dIdt, dRdt, dDdt]

def ode_solver(t, initial_conditions, params):
    initE, initI, initR, initN, initD = initial_conditions
    beta, sigma, gamma, mu = params
    initS = initN - (initE + initI + initR + initD)
    res = odeint(ode_model, [initS, initE, initI, initR, initD], t, args=(beta, sigma, gamma, mu))
    return res

def add_pie_chart(S, E, I, R, D, days):
    # Valores finais
    final_values = [S[-1], E[-1], I[-1], R[-1], D[-1]]
    labels = ['Susceptible', 'Exposed', 'Infected', 'Recovered', 'Deaths']
    colors = ['lightblue', 'orange', 'red', 'green', 'black']
    
    fig_pie = go.Figure(data=[go.Pie(labels=labels, values=final_values, 
                                     marker_colors=colors)])
    fig_pie.update_layout(title=f'Population Distribution at Day {days}',
                         width=400, height=400)
    return fig_pie

def calculate_metrics(S, E, I, R, D, beta, gamma, mu, initN):
    # R0 (Número básico de reprodução)
    R0 = beta / (gamma + mu)
    
    # Pico de infectados
    peak_infected = max(I)
    peak_day = np.argmax(I)
    
    # Taxa de ataque final
    attack_rate = (initN - S[-1]) / initN * 100
    
    # Mortalidade final
    mortality_rate = D[-1] / initN * 100
    
    return {
        'R0': R0,
        'Peak Infected': int(peak_infected),
        'Peak Day': int(peak_day),
        'Attack Rate (%)': round(attack_rate, 2),
        'Mortality Rate (%)': round(mortality_rate, 4)
    }

def plot_daily_incidence(sol, tspan, initN):
    S, E, I, R, D = sol[:, 0], sol[:, 1], sol[:, 2], sol[:, 3], sol[:, 4]
    
    # Calcular novos casos diários
    new_infections = np.diff(np.concatenate([[0], initN - S]))
    new_deaths = np.diff(np.concatenate([[0], D]))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tspan[:-1], y=new_infections, 
                            mode='lines', name='New Infections'))
    fig.add_trace(go.Scatter(x=tspan[:-1], y=new_deaths, 
                            mode='lines', name='New Deaths'))
    
    fig.update_layout(title='Daily Incidence',
                     xaxis_title='Day',
                     yaxis_title='Daily Cases',
                     width=700, height=300)
    return fig

def create_main_plot(tspan, S, E, I, R, D, days):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tspan, y=S, mode='lines', name='Susceptible', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=tspan, y=E, mode='lines', name='Exposed', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=tspan, y=I, mode='lines', name='Infected', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=tspan, y=R, mode='lines', name='Recovered', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=tspan, y=D, mode='lines', name='Deaths', line=dict(color='black')))
    
    if days <= 30:
        step = 1
    elif days <= 90:
        step = 7
    else:
        step = 30
    
    fig.update_layout(
        title='SEIRD Model Simulation',
        xaxis_title='Day',
        yaxis_title='Population',
        height=500,
        hovermode='x unified'
    )
    fig.update_xaxes(tickmode='array', tickvals=np.arange(0, days + 1, step))
    return fig