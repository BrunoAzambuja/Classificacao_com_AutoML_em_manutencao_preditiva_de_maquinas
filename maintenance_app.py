import streamlit as st
import numpy as np
import pandas as pd
import os
import sklearn
from joblib import dump, load

# text elements
st.title(' Previsão de falhas em equipamentos')
st.write('##### App que utiliza machine learning para prever os tipos de falhas.')
st.write(' Preencha os dados de medição do equipamento na barra lateral [>] antes de selecionar o botão de previsão abaixo.')

# input widgets

Air_temperature = st.sidebar.number_input('Temperatura do ar [°C]:', min_value = 0.00, max_value = 45.00, step = 0.05)

Process_temperature = st.sidebar.number_input('Temperatura do processo [°C]:', min_value = 0.00, max_value = 45.00, step = 0.05)

Rotational_speed = st.sidebar.number_input('Velocidade de rotação [rpm]:', min_value = 0, max_value = 3000, step = 1)

Torque = st.sidebar.number_input('Torque [Nm]:', min_value = 0.0, max_value = 100.0, step = 0.1)

Toll_wear = st.sidebar.number_input('Tempo para o início do desgate [min]:', min_value = 0, max_value = 300, step = 1)

lista = ['Baixa', 'Média', 'Alta']
Type = st.sidebar.radio('Qualidade do equipamento:', options = lista)

if Type == 'Baixa':
    Type_L = '1'
    Type_M = '0'
    Type_H = '0'
    
elif Type == 'Média':
    Type_L = '0'
    Type_M = '1'
    Type_H = '0'
    
else:
    Type_L = '0'
    Type_M = '0'
    Type_H = '1'

#lista1 = ['0', '1']
#Type_L = st.sidebar.radio('Baixa qualidade do equipamento [0: não] [1: sim]:', options = lista1)

#lista2 = ['0', '1']
#Type_M = st.sidebar.radio('Média qualidade do equipamento [0: não] [1: sim]:', options = lista2)

#lista3 = ['0', '1']
#Type_H = st.sidebar.radio('Alta qualidade do equipamento [0: não] [1: sim]:', options = lista3)

features = {'Air_temperature': Air_temperature, 'Process_temperature': Process_temperature,
            'Rotational_speed': Rotational_speed, 'Torque': Torque,
            'Toll_wear': Toll_wear, 'Type_L': Type_L,
            'Type_M': Type_M, 'Type_H': Type_H
            }

features_df  = pd.DataFrame([features], dtype=float)

#st.table(features_df)

if (os.path.exists('classifier_multiclass.pkl')):
    modelo = load('classifier_multiclass.pkl')
    botao = st.button('Previsão do tipo de falha')
    if(botao):
        
        resultado = modelo.predict(features_df)
        
        if resultado[0] == 'Random Failures':
            st.success('##### Random Failures')
        
        elif resultado[0] == 'Tool Wear Failure':
            st.success('##### Tool Wear Failure')
    
        elif resultado[0] == 'Overstrain Failure':
            st.success('##### Overstrain Failure')

        elif resultado[0] == 'Power Failure':
            st.success('##### Power Failure')
    
        elif resultado[0] == 'Heat Dissipation Failure':
            st.success('##### Heat Dissipation Failure')

        else:
            st.success('##### No Failure')

else:
    st.error('Erro ao carregar o modelo preditivo. Contacte o administrador do sistema.')



