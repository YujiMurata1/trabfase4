import streamlit as st
import ipeadatapy as ip
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import numpy as np
import warnings
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
from PIL import Image


with st.sidebar:
    topico1 = 'INTRODUÇÃO'
    topico2 = 'DESENVOLVIMENTO'
    topico3 = 'NARRATIVA'
    opcoes_topicos = [topico1, topico2, topico3]
    topico_selecionado = st.sidebar.selectbox('**Escolha um tópico**', opcoes_topicos)

if topico_selecionado == topico1:
    st.write(f'# O GRUPO')
    paragraphs = [
        "**Grupo 49**",
        "Aluna: Carolina Passos Morelli Cunha"
    ]
    for paragraph in paragraphs:
        st.write(paragraph)

    st.write(f'# A PROPOSTA')
    paragraphs = [
        "Você foi contratado(a) para uma consultoria, e seu trabalho envolve analisar os dados de preço do petróleo brent, que pode ser encontrado no site do ipea. Essa base de dados histórica envolve duas colunas: data e preço (em dólares).",
        "Um grande cliente do segmento pediu para que a consultoria desenvolvesse um dashboard interativo e que gere insights relevantes para tomada de decisão. Além disso, solicitaram que fosse desenvolvido um modelo de Machine Learning para fazer o forecasting do preço do petróleo.",
        "Seu objetivo é:",
        "*	Criar um dashboard interativo com ferramentas à sua escolha.",
        "*	Seu dashboard deve fazer parte de um storytelling que traga insights relevantes sobre a variação do preço do petróleo, como situações geopolíticas, crises econômicas, demanda global por energia e etc. Isso pode te ajudar com seu modelo. É obrigatório que você traga pelo menos 4 insights neste desafio.",
        "*	Criar um modelo de Machine Learning que faça a previsão do preço do petróleo diariamente (lembre-se de time series). Esse modelo deve estar contemplado em seu storytelling e deve conter o código que você trabalhou, analisando as performances do modelo.",
        "*	Criar um plano para fazer o deploy em produção do modelo, com as ferramentas que são necessárias.",
        "*	Faça um MVP do seu modelo em produção utilizando o Streamlit."
    ]    
    for paragraph in paragraphs:
        st.write(paragraph)

elif topico_selecionado == topico2:
    st.write(f'# FONTE DOS DADOS')
    paragraphs = [
        "**FONTE:** [IPEA](http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view)",
        " *Os dados são atualizados semanalmente no site.*"
    ]
    for paragraph in paragraphs:
        st.write(paragraph)

    #obtendo dos dados
    df_ipea = ip.timeseries("EIA366_PBRENT366")
    df_ipea = df_ipea.dropna() #removendo linhas sem valores
    df_ipea = df_ipea.reset_index('DATE')
    df_ipea['DATE'] = pd.to_datetime(df_ipea['DATE'], dayfirst=True) #realizando a conversão da data para formato datetime
    df_ipea = df_ipea.sort_values(by='DATE',ascending=True)
    df = df_ipea
    # #REMOVENDO as colunas desnecessárias
    df = df.drop(columns=['CODE'])
    df = df.drop(columns=['RAW DATE'])
    df = df.drop(columns=['DAY'])
    df = df.drop(columns=['MONTH'])
    df = df.drop(columns=['YEAR'])
    # Criando coluna com % de variação do fechamento
    df["Variacao_Fechamento"] = df["VALUE (US$)"].pct_change(periods=1)*100
    df['Variacao_Fechamento'] = df['Variacao_Fechamento'].fillna(0)


    if st.button("Atualizar"):
        #obtendo dos dados
        df_ipea = ip.timeseries("EIA366_PBRENT366")
        df_ipea = df_ipea.dropna() #removendo linhas sem valores
        df_ipea = df_ipea.reset_index('DATE')
        df_ipea['DATE'] = pd.to_datetime(df_ipea['DATE'], dayfirst=True) #realizando a conversão da data para formato datetime
        df_ipea = df_ipea.sort_values(by='DATE',ascending=True)
        df = df_ipea

        #REMOVENDO as colunas desnecessárias
        df = df.drop(columns=['CODE'])
        df = df.drop(columns=['RAW DATE'])
        df = df.drop(columns=['DAY'])
        df = df.drop(columns=['MONTH'])
        df = df.drop(columns=['YEAR'])
        # Criando coluna com % de variação do fechamento
        df["Variacao_Fechamento"] = df["VALUE (US$)"].pct_change(periods=1)*100
        df['Variacao_Fechamento'] = df['Variacao_Fechamento'].fillna(0)

    def convert_df(df):
        return df.to_csv()

    csv = convert_df(df)

    st.download_button(
        label="Download dados",
        data=csv,
        file_name="dados_ipea.csv",
        mime="text/csv"
    )

    
    st.write(f'# Variação do Preço do Petróleo')
    st.write(f'(EIA366_PBRENT366)')

    st.dataframe(df)

    x = df['DATE']
    y = df['VALUE (US$)']
    plt.figure(figsize=(15, 10))
    plt.plot(x, y, linestyle='-', color='c')
    plt.title('Preço do Petróleo Bruto - Brent (FOB) - USD')
    plt.xlabel('Data')
    plt.ylabel('Preço (USD)')
    plt.xticks(rotation=45)
    plt.grid(True)
    st.pyplot(plt)

    x = df['DATE']
    y_variacao = df['Variacao_Fechamento']
    plt.figure(figsize=(15, 10))
    plt.plot(x, y_variacao, linestyle='-', color='c')
    plt.title('Variação do Preço do Petróleo Bruto - Brent (FOB) - %')
    plt.xlabel('Data')
    plt.ylabel('% Variação')
    plt.xticks(rotation=45)
    plt.grid(True)
    st.pyplot(plt)

    # Calcular e plotar desvio padrão móvel
    st.write(f'## Desvio Padrão')    
    window = 30  # Janela de 30 dias
    df['DesvioPadrao'] = df['VALUE (US$)'].rolling(window=window).std()
    y_desviopadrao = df['DesvioPadrao']
    plt.figure(figsize=(15, 10))
    df['VALUE (US$)'].plot(x="Major", y="DATE", color='c')
    df['DesvioPadrao'].plot(x="Major", y="", color='b')
    plt.xlabel('Data')
    plt.ylabel('Preço (USD)')
    plt.xticks(rotation=45)
    plt.title('Desvio Padrão Móvel e Preço (USD)')
    plt.legend()
    st.pyplot(plt)

    ##INICIO Modelo Preditivo
    st.write(f'# Modelo Preditivo do Petróleo Brent') 

    #carrega modelo
    with open('modelo_brent.pkl', 'rb') as file2:
        modelo_brent = joblib.load(file2)

    lags = 7
    for lag in range(1, lags + 1):
        df[f'Preço_lag_{lag}'] = df['VALUE (US$)'].shift(lag)

    # Removemos quaisquer linhas com valores NaN que foram criados ao fazer o shift
    df = df.dropna()

    # Preparando os dados para treinamento
    X = df[['Preço_lag_1', 'Preço_lag_2', 'Preço_lag_3', 'Preço_lag_4', 'Preço_lag_5', 'Preço_lag_6', 'Preço_lag_7']].values  # Inputs são os preços atrasados
    y = df['VALUE (US$)'].values  # Output é o preço atual

    # Dividir os dados em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Fazer previsões
    predictions = modelo_brent.predict(X_test)

    # Avaliar o modelo
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    st.write(f'### O erro quadrado médio é de: {mse:.2f}') 
    st.write(f'### O erro absoluto médio é de: {mae:.2f}') 


    # Fazer previsões para a próxima semana usando os últimos dados conhecidos
    last_known_data = X[-1].reshape(1, -1)
    next_week_predictions = []
    for _ in range(7):  # para cada dia da próxima semana
        next_day_pred = modelo_brent.predict(last_known_data)[0]
        next_week_predictions.append(next_day_pred)
        last_known_data = np.roll(last_known_data, -1)
        last_known_data[0, -1] = next_day_pred

    # As datas correspondentes à próxima semana
    next_week_dates = pd.date_range(df['DATE'].iloc[-1], periods=8)[1:]

    # Selecionar os dados da semana atual (últimos 7 dias do dataset)
    current_week_dates = df['DATE'].iloc[-7:]
    current_week_prices = df['VALUE (US$)'].iloc[-7:]

    for week, pred in zip(next_week_dates, next_week_predictions):
        print(f'{week}: {pred:.2f}')

    # Plotar os preços reais da semana atual e as previsões para a próxima semana
    st.write(f'# Previsões para a Próxima Semana')    
    plt.figure(figsize=(10, 5))
    plt.plot(current_week_dates, current_week_prices, 'bo-', label='Preços Atuais')
    plt.plot(next_week_dates, next_week_predictions, 'r--o', label='Previsões para a Próxima Semana')

    # Formatar o eixo x para exibir datas
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.gcf().autofmt_xdate()  # Ajustar formato das datas para evitar sobreposição

    plt.xlabel('Data')
    plt.ylabel('Preço')
    plt.title('Preços Reais e Previsões para as Últimas Duas Semanas')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    #carrega modelo
    with open('modelo_brent.pkl', 'rb') as file2:
        modelo_brent = joblib.load(file2)

    #Avaliar o modelo
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    print("Mean Squared Error:", mse)
    print("Mean Absolute Error:", mae)
    # Fazer previsões para a próxima semana usando os últimos dados conhecidos
    last_known_data = X[-1].reshape(1, -1)
    next_week_predictions = []
    for _ in range(7):  # para cada dia da próxima semana
        next_day_pred = modelo_brent.predict(last_known_data)[0]
        next_week_predictions.append(next_day_pred)
        last_known_data = np.roll(last_known_data, -1)
        last_known_data[0, -1] = next_day_pred

    # As datas correspondentes à próxima semana
    next_week_dates = pd.date_range(df['DATE'].iloc[-1], periods=8)[1:]

    # Selecionar os dados da semana atual (últimos 7 dias do dataset)
    current_week_dates = df['DATE'].iloc[-7:]
    current_week_prices = df['VALUE (US$)'].iloc[-7:]

    for week, pred in zip(next_week_dates, next_week_predictions):
        print(f'{week}: {pred:.2f}')

    # Plotar os preços reais da semana atual e as previsões para a próxima semana
    plt.figure(figsize=(10, 5))
    plt.plot(current_week_dates, current_week_prices, 'bo-', label='Preços Atuais')
    plt.plot(next_week_dates, next_week_predictions, 'r--o', label='Previsões para a Próxima Semana')

    # Formatar o eixo x para exibir datas
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.gcf().autofmt_xdate()  # Ajustar formato das datas para evitar sobreposição

    plt.xlabel('Data')
    plt.ylabel('Preço')
    plt.title('Preços Reais e Previsões para a Próxima Semana')
    plt.legend()
    plt.grid(True)
    plt.show()

    st.pyplot(plt)

    ##IM


elif topico_selecionado == topico3:
    st.write('# NARRATIVA')
    paragraphs = [
        "O petróleo é um dos principais fatores que influenciam a economia mundial.",
        "**O que é o Petróleo Brent?**",
        "O Petróleo Bruto Brent é um tipo de petróleo cru extraído do Mar do Norte e é usado como uma referência para precificar muitos tipos de petróleo ao redor do mundo. O preço do petróleo bruto Brent é influenciado por uma série de fatores, incluindo a oferta e demanda global, eventos geopolíticos, políticas dos países produtores de petróleo, condições econômicas e muito mais."
        "",
        "**O que significa o termo FOB?**",
        "FOB - Free On Board, que, em português, pode ser traduzido para “livre a bordo”. Essa sigla serve para descrever um tipo de frete em que o comprador assume todos os riscos e custos com o transporte de alguma mercadoria assim que ela é colocada a bordo de um meio de transporte",
        "",
        "**Como é definido o preço do petróleo no mercado internacional?**",
        "O principal meio de precificação é através da lei da oferta e da demanda."
        "Como o petróleo é uma commodity — ou seja, um bem pouco industrializado com pouca variação de características, ele é negociado a preços padronizados em todo o planeta.",
        "O barril de petróleo padrão no mercado global corresponde a 42 galões americanos, o equivalente a aproximadamente 159 litros.",
        "O valor do Brent também sofre forte influência dos interesses da Organização dos Países Exportadores de Petróleo (Opep), que controla boa parte de sua produção.",
        "Como a commodity é precificada em dólares, a oscilação da moeda americana também afeta o preço do petróleo no mercado internacional.",
        "",
        "**O que é a OPEP?**",
        "Organização dos Países Exportadores de Petróleo (Opep) foi fundada em 1960, com o objetivo de coordenar a produção de petróleo entre nações para exercer maior influência sobre o preço do petróleo global.",
        "Sob a liderança efetiva da Arábia Saudita, a Opep atua orientando seus membros a cortar ou elevar a produção para controlar a oferta e manter os preços estáveis.",
        "Em 2016, um grupo formado por dez outros países relevantes no setor de petróleo colabora extraoficialmente com as estratégias da Opep, embora não sejam membros da organização.",
        "Este bloco, chamado informalmente de Opep+, é liderado pela Rússia e inclui México, Sudão, Sudão do Sul, Azerbaijão, Malásia, Omã, Brunei, Cazaquistão e Bahrein."
        "**Quem são os membros da OPEP?**",
        "A Opep (ou OPEC, na sigla em inglês) é um grupo composto por 13 dos principais países exportadores de petróleo em todo o mundo:",
        " * Arábia Saudita", "* Emirados Árabes Unidos,", "* Irã", "* Iraque", "* Líbia", "* Argélia", "* Venezuela", "* Angola", "* Kuwait", "* Congo", "* Nigéria", "* Gabão", "* Guiné Equatorial", 
        "",
        "**Quem são os maiores produtores de petróleo?**",         
    ]
    for paragraph in paragraphs:
        st.write(paragraph)

    url_produtor = "/mount/src/techchallenge4/imagens/produtores_2022.png"
    imagem2 = Image.open(url_produtor)
    st.image(imagem2, use_column_width=True, caption='Consumidores')
    
    paragraphs = [
        "",
        "**Quem são os maiores consumidores de petróleo?**",
    ]

    for paragraph in paragraphs:
        st.write(paragraph)

    url_consumidor = "/mount/src/techchallenge4/imagens/consumidores_2022.png"
    imagem2 = Image.open(url_consumidor)
    st.image(imagem2, use_column_width=True, caption='Consumidores')


    st.write('## Predição')
    paragraphs = [
            "Desenvolvido um modelo de Machine Learning em Python para análise de séries temporais, utilizando a base de dados de preços do petróleo Brent - US$.",
            "Período utilizado:",
            "Data de Início: 20/05/1987",
            "Data de Término: 13/05/2024",
        ]
    
    for paragraph in paragraphs:
        st.write(paragraph)


    paragraphs = [
        "",
        "",
        "",
        "*Fontes:*",
        "* https://warren.com.br/magazine/preco-do-petroleo/",
        "* Energy Institute (https://www.energyinst.org/statistical-review/resources-and-data-downloads)"  
    ]
    for paragraph in paragraphs:
        st.write(paragraph)



