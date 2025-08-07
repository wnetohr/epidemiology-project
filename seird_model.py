import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.integrate import odeint
import plotly.express as px
import utils

st.set_page_config(
    page_title="SEIRD Model Dashboard",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.title("🦠 SEIRD Epidemiological Model Dashboard")
st.markdown("Interactive simulation of disease spread using the SEIRD (Susceptible-Exposed-Infected-Recovered-Deaths) model")

disease_presets = {
    'Custom': {'beta': 0.5, 'sigma': 0.2, 'gamma': 0.1, 'mu': 0.01},
    'COVID-19 (Original)': {'beta': 0.40, 'sigma': 0.197, 'gamma': 0.1, 'mu': 0.0272},
    'COVID-19 (Delta)': {'beta': 0.78, 'sigma': 0.227, 'gamma': 0.1, 'mu': 0.03},
    'COVID-19 (Omicron)': {'beta': 0.24, 'sigma': 0.292, 'gamma': 0.143, 'mu': 0.0009}  
}

st.sidebar.header("📊 Model Parameters")

disease = st.sidebar.selectbox(
    "Select Disease Preset:",
    options=list(disease_presets.keys()),
    index=1
)

preset = disease_presets[disease]

st.sidebar.subheader("🔢 Initial Conditions")
initN = st.sidebar.number_input("Population (N)", min_value=1000, max_value=10000000, value=100000, step=1000)
initE = st.sidebar.number_input("Initial Exposed (E)", min_value=0, max_value=10000, value=10, step=1)
initI = st.sidebar.number_input("Initial Infected (I)", min_value=1, max_value=10000, value=5, step=1)
initR = st.sidebar.number_input("Initial Recovered (R)", min_value=0, max_value=10000, value=0, step=1)
initD = st.sidebar.number_input("Initial Deaths (D)", min_value=0, max_value=10000, value=0, step=1)

st.sidebar.subheader("🧬 Disease Parameters")
beta = st.sidebar.slider("Infection Rate (β)", min_value=0.0, max_value=4.0, value=preset['beta'], step=0.01)
sigma = st.sidebar.slider("Incubation Rate (σ)", min_value=0.0, max_value=2.0, value=preset['sigma'], step=0.001)
gamma = st.sidebar.slider("Recovery Rate (γ)", min_value=0.0, max_value=1.0, value=preset['gamma'], step=0.001)
mu = st.sidebar.slider("Mortality Rate (μ)", min_value=0.0, max_value=0.1, value=preset['mu'], step=0.001)

st.sidebar.subheader("⏱️ Simulation Time")
days = st.sidebar.slider("Simulation Days", min_value=1, max_value=600, value=180, step=7)

if st.sidebar.button("🚀 Run Simulation", type="primary"):

    initial_conditions = [initE, initI, initR, initN, initD]
    params = [beta, sigma, gamma, mu]
    tspan = np.arange(0, days, 1)
    sol = utils.ode_solver(tspan, initial_conditions, params)
    S, E, I, R, D = sol[:, 0], sol[:, 1], sol[:, 2], sol[:, 3], sol[:, 4]

    metrics = utils.calculate_metrics(S, E, I, R, D, beta, gamma, mu, initN)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_main = utils.create_main_plot(tspan, S, E, I, R, D, days)
        st.plotly_chart(fig_main, use_container_width=True)
    
    with col2:
        st.subheader("📈 Key Metrics")
        
        col_metric1, col_metric2 = st.columns(2)
        with col_metric1:
            st.metric("R₀", metrics['R0'])
            st.metric("Peak Day", metrics['Peak Day'])
        
        with col_metric2:
            st.metric("Peak Infected", f"{metrics['Peak Infected']:,}")
            st.metric("Attack Rate", f"{metrics['Attack Rate (%)']}%")
        
        st.metric("Mortality Rate", f"{metrics['Mortality Rate (%)']}%")
    
    col3, col4 = st.columns(2)
    
    with col3:
        fig_pie = utils.add_pie_chart(S, E, I, R, D, days)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col4:
        if days > 7:
            fig_incidence = utils.plot_daily_incidence(sol, tspan, initN)
            st.plotly_chart(fig_incidence, use_container_width=True)
    
    st.subheader("📋 Final Population Distribution")
    final_data = {
        'Category': ['Susceptible', 'Exposed', 'Infected', 'Recovered', 'Deaths'],
        'Count': [int(S[-1]), int(E[-1]), int(I[-1]), int(R[-1]), int(D[-1])],
        'Percentage': [
            round(S[-1]/initN*100, 2),
            round(E[-1]/initN*100, 2),
            round(I[-1]/initN*100, 2),
            round(R[-1]/initN*100, 2),
            round(D[-1]/initN*100, 2)
        ]
    }
    st.dataframe(final_data, use_container_width=True)

else:
    st.info("👈 Adjust parameters in the sidebar and click 'Run Simulation' to start!")
    
    st.subheader("📚 About the SEIRD Model")
    st.markdown("""
    The SEIRD model divides the population into five compartments:
    
    - **S (Susceptible)**: Individuals who can become infected
    - **E (Exposed)**: Individuals who have been exposed but are not yet infectious
    - **I (Infected)**: Individuals who are currently infectious
    - **R (Recovered)**: Individuals who have recovered and gained immunity
    - **D (Deaths)**: Individuals who have died from the disease
    
    ### Parameters:
    - **β (Beta)**: Infection rate - how quickly the disease spreads
    - **σ (Sigma)**: Incubation rate - rate at which exposed become infectious
    - **γ (Gamma)**: Recovery rate - rate at which infected recover
    - **μ (Mu)**: Mortality rate - rate at which infected die
    """)

st.markdown("---")
st.markdown("Created by Washington Rocha using Streamlit | SEIRD Epidemiological Model")