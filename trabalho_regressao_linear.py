"""
Grupo: Gabriel Fassina Fereguetti, Marllon Monteiro Cardoso e Thiago Piffer Lauro
 
Relação escolhida para regressão linear simples: linhas de código (X) x complexidade ciclomática (Y)

Fontes: 
-Dataset: https://www.kaggle.com/datasets/aczy156/software-defect-prediction-nasa
-Exemplo com explicação do dataset: https://github.com/MostafaMohamedEr/JM1-SoftwareDefectPrediction-
-Exemplo de uso: https://github.com/messuuu/jm1-software-defect-prediction-
 
Dependências para rodar o código: pip install pandas numpy matplotlib scipy kagglehub
"""


#Imports necessários
import kagglehub
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


#Definindo variáveis globais
DATASET_LINK = "aczy156/software-defect-prediction-nasa"#Guarda o link do dataset usado
 
COL_X = "loc"#Guarda o nome da coluna referente à quantidade de linhas de código
COL_Y = "v(g)"#Guarda o nome da coluna referente à complexidade ciclomática (número de caminhos lógicos)
 
PREVISOES_LOC = [10, 50, 100, 200, 300, 500, 1000, 2000, 3000]#Lista com alguns valores de linhas de código para teste de previsão
 

#Funções para carregar os dados do dataset
def localizar_csv():#Função que localiza e retorna o arquivo ".csv" com os dados usados
 
    pasta = kagglehub.dataset_download(DATASET_LINK)#Baixa o dataset indicado pelo link e guarda o caminho
 
    if os.path.isfile(pasta):#Se o caminho informado já for um arquivo válido, usa ele diretamente
        return pasta
 
    candidatos = []#Lista dos arquivos ".csv" que têm as colunas procuradas
 
    #Percorre todos os ".csv" indicados pelo link
    for arq in sorted(glob.glob(os.path.join(pasta, "**", "*.csv"), recursive=True)):#Percorre todos os arquivos ".csv" da pasta
        colunas = pd.read_csv(arq, nrows=0).columns#Recebe as colunas
        if COL_X in colunas and COL_Y in colunas:#Verifica se as procuradas estão inclusas
            candidatos.append(arq)
 
    #Retorna erro caso nenhum ".csv" tenha as colunas procuradas
    if not candidatos:
        raise FileNotFoundError(
            f"Erro: Nenhum CSV com as colunas {COL_X} e {COL_Y} em {pasta}. "
        )
 
    return candidatos[0]#Retona o primeiro ".csv" encontrado
 
def carregar_dados():#Função que filtra e retorna os dados
 
    arq = localizar_csv()#Obtem o caminho do arquivo ".csv"
 
    df = pd.read_csv(arq, usecols=[COL_X, COL_Y])#Carrega apenas as duas colunas usadas na análise
 
    #Converte os valores das duas colunas para números
    df[COL_X] = pd.to_numeric(df[COL_X], errors="coerce")#Valores inválidos viram NaN
    df[COL_Y] = pd.to_numeric(df[COL_Y], errors="coerce")#Valores inválidos viram NaN
 
    df = df.dropna(subset=[COL_X, COL_Y])#Apaga as linhas com dados vazios
 
    df = df[(df[COL_X] > 0) & (df[COL_Y] > 0)]#Mantém só módulos com pelo menos 1 linha de código e complexidade maior que 0
 
    return df.reset_index(drop=True)#Enumera as linhas a partir de 0 e retorna a tabela
 
 
#Funções para análise e exibição de dados
def classificar_correlacao(r):#Função que classifica o coeficiente de correlação linear
 
    rA = round(r, 3)#Arredonda o coeficiente para 3 casas decimais
 
    #Verifica os 3 casos extremos e já gera retorno
    if  rA == 1:
        return "perfeita e positiva"
    if  rA == -1:
        return "perfeita e negativa"
    if  rA == 0:
        return "inexistente (sem relação linear)"
 
    forca = abs(r)#Pega o módulo do coeficiente de correlação e verifica em que intervalo ele se encaixa
    if forca < 0.3:
        txt = "muito fraca"
    elif forca < 0.6:
        txt = "relativamente fraca"
    else:
        txt = "significativa"
 
    sentido = "positiva" if r > 0 else "negativa"#Verifica se é positivo ou negativo
 
    return f"{txt} e {sentido}"#Retorna a classificação
 
def main():#Função main do programa
 
    df = carregar_dados()#Recebe a tabela com os dados tratados
    x = df[COL_X].to_numpy()#Converte a coluna das linhas de código em array
    y = df[COL_Y].to_numpy()#Converte a coluna da complexidade ciclomática em array
    n = len(x)#Variável que guarda o número de registros
 
    #1-Pesquisa sobre os dados
    print("\n" + "=" * 30)#Somente a exibição dos dados pelo programa
    print("DESCRIÇÃO DOS DADOS")
    print("=" * 30)
    print(f"Variável X: linhas de código ({COL_X}).")
    print(f"Variável Y: complexidade ciclomática ({COL_Y}).")
    print(f"Quantidade de registros: n = {n}.")
 
    resumo = df[[COL_X, COL_Y]].describe().round(2)#Faz a descrição dos dados das colunas
    resumo.columns = ["Linhas de código (X)", "Complexidade ciclomática (Y)"]#Determina os nomes de cada coluna
    print(resumo)#Exibe o resultado
 
    #2-Diagrama de dispersão
    fig1, ax1 = plt.subplots(figsize=(9, 6))#Cria a imagem e o gráfico contido nela
    ax1.scatter(x, y, alpha=0.4, s=25, color="#1f4e9c", edgecolor="none")#Desenha os pontos X e Y em cada eixo
 
    ax1.set_title("Diagrama de dispersão: linhas de código x complexidade ciclomática")#Define o título do gráfico
    ax1.set_xlabel("Linhas de código (X)")#Define o rótulo do eixo X
    ax1.set_ylabel("Complexidade ciclomática (Y)")#Define o rótulo do eixo Y
 
    ax1.grid(alpha=0.3)#Deixa a grade do gráfico mais escondida para facilitar visualização
    fig1.tight_layout()#Ajuste de margens
    fig1.savefig("diagrama_de_dispersao.png", dpi=150)#Salva a imagem do gráfico na pasta atual
 
    #3-Cálculo e análise da correlação
    x_med, y_med = x.mean(), y.mean()#Calcula as média de X e de Y
 
    sxy = np.sum(x * y) - n * x_med * y_med#Calcula Σxy − n·x̄·ȳ
    sxx = np.sum(x**2) - n * x_med**2#Calcula Σx² − n·x̄²
    syy = np.sum(y**2) - n * y_med**2#Calcula Σy² − n·ȳ²
 
    r = sxy / np.sqrt(sxx * syy)#Calcula o coeficiente de correlação
 
    print("\n" + "=" * 30)#Somente a exibição dos dados pelo programa
    print("CORRELAÇÃO LINEAR")
    print("=" * 30)
    print(f"x̄ = {x_med:.4f} linhas | ȳ = {y_med:.4f} (complexidade).")
    print(f"Σxy - n·x̄·ȳ = {sxy:,.2f}.")
    print(f"Σx² - n·x̄²  = {sxx:,.2f}.")
    print(f"Σy² - n·ȳ²  = {syy:,.2f}.")
    print(f"r = {r:.4f}.")
    print(f"r² = {r**2:.4f}.")
    print(f"Classificação: correlação linear {classificar_correlacao(r)}.")
    print(f"Cerca de {r**2 * 100:.1f}% da variação da complexidade ciclomática é explicada "
          "linearmente pelas linhas de código.")
 
    #4-Equação da regressão
    a = sxy / sxx#Calcula o coeficiente A
    b = y_med - a * x_med#Calcula o coeficiente B
 
    res = stats.linregress(x, y)#Calcula a regressão linear simples
    assert np.isclose(a, res.slope) and np.isclose(b, res.intercept)#Calcula se o cálculo manual e automatico deram no mesmo resultado
 
    print("\n" + "=" * 30)#Somente a exibição dos dados pelo programa
    print("REGRESSÃO LINEAR SIMPLES")
    print("=" * 30)
    print(f"a = Sxy / Sxx = {a:.4f}.")
    print(f"b = ȳ - a·x̄   = {b:.4f}.")
    print(f"Equação:  Y = {a:.4f}X + {b:.4f}.")
 
    #5-Diagrama de dispersão com a reta no mesmo gráfico
    xs = np.linspace(x.min(), x.max(), 200)#Cria 200 valores de X igualmente espaçados, seguindo o intervalo do menor ao maior X
    fig2, ax2 = plt.subplots(figsize=(9, 6))#Cria a imagem e o gráfico contido nela
    ax2.scatter(x, y, alpha=0.4, s=25, color="#1f4e9c", edgecolor="none", label="Observações")#Cria cada um dos pontos e define a legenda "Observações"
 
    #Cria a reta de regressão, tendo a equação e o r na legenda
    ax2.plot(xs, a * xs + b, color="#d62728", linewidth=2.5,
             label=f"Reta: y = {a:.4f}·x {b:+.2f}; (r = {r:.2f}, R² = {r**2:.2f})")
 
    ax2.set_title("Regressão linear: linhas de código x complexidade ciclomática")#Define o título do gráfico
    ax2.set_xlabel("Linhas de código (X)")#Define o rótulo do eixo X
    ax2.set_ylabel("Complexidade ciclomática (Y)")#Define o rótulo do eixo Y
 
    ax2.grid(alpha=0.3)#Deixa a grade do gráfico mais escondida para facilitar visualização
    ax2.legend()#Exibe a caixa de legenda
    fig2.tight_layout()#Ajuste de margens
    fig2.savefig("diagrama_regressao_reta.png", dpi=150)#Salva a imagem do gráfico na pasta atual
 
    #6-Previsões
    print("\n" + "=" * 30)#Somente a exibição dos dados pelo programa
    print("PREVISÕES SEGUNDO A EQUAÇÃO")
    print("=" * 30)
 
    for loc in PREVISOES_LOC:#Percorre a lista com linhas de código determinadas e calcula o Y pela equação da reta, exibindo o resultado
        prev = a * loc + b
 
        aviso = "  (fora do intervalo observado)." \
            if loc < x.min() or loc > x.max() else ""#Exibe aviso caso o valor seja superior ou inferior aos observados
 
        print(f"{loc:>5} linhas de código -> v(g) previsto = {prev:>7.2f}{aviso}")
 
    plt.show()#Abre janelas exibindo os gráficos
 
if __name__ == "__main__":#Roda o programa chamando a função "main()"
    main()