import streamlit as st
import numpy as np
import pandas as pd
import os
import sklearn
from joblib import dump, load

# text elements
st.title('Previsão de falhas em máquinas')
st.write('##### App que utiliza machine learning para prever os tipos de falhas.')

# input widgets
airtemperature = st.number_input('Temperatura do ar [°C]:', min_value = 0.00, max_value = 45.00, step = 0.05)

processtemperature = st.number_input('Temperatura do processo [°C]:', min_value = 0.00, max_value = 45.00, step = 0.05)

rotationalspeed = st.number_input('Velocidade de rotação [rpm]:', min_value = 0, max_value = 3000, step = 1)

torque = st.number_input('Torque [Nm]:', min_value = 0.0, max_value = 100.0, step = 0.1)

tollwear = st.number_input('Tempo para o início do desgate [min]:', min_value = 0, max_value = 300, step = 1)

lista1 = ['0', '1']
type1 = st.radio('Baixa qualidade do equipamento [0: não] [1: sim]:', options = lista1)

lista2 = ['0', '1']
type2 = st.radio('Média qualidade do equipamento [0: não] [1: sim]:', options = lista2)

lista3 = ['0', '1']
type3 = st.radio('Alta qualidade do equipamento [0: não] [1: sim]:', options = lista3)


if (os.path.exists('classifier_multiclass.pkl')):
    modelo = load('classifier_multiclass.pkl')
    botao = st.button('Previsão do tipo de falha [Multiclasse]')
    if(botao):
        listaValores = np.array([[airtemperature, processtemperature, rotationalspeed, torque, tollwear, type1, type2, type3]])
        resultado = modelo.predict(listaValores)

        if(resultado[0] == 'Random Failures'):
            st.success('##### Random Failures')
        
        elif(resultado[0] == 'Tool Wear Failure'):
            st.success('##### Tool Wear Failure')
    
        elif(resultado[0] == 'Overstrain Failure'):
            st.success('##### Overstrain Failure')

        elif(resultado[0] == 'Power Failure'):
            st.success('##### Power Failure')
    
        elif(resultado[0] == 'Heat Dissipation Failure'):
            st.success('##### Heat Dissipation Failure')
        else:
            st.success('##### No Failure')

else:
    st.error('Erro ao carregar o modelo preditivo. Contacte o administrador do sistema.')
