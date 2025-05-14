# Previsão de Demanda com Dados Heterogêneos: Uma Pipeline de Machine Learning Aplicado ao Varejo

Autor: Gustavo Gabriel Ribeiro
Data: 11 de maio de 2025

---

## Abstract

With the increasing complexity of market scenarios, the need for robust and intelligent demand forecasting solutions is growing, especially in the retail sector. This work presents a study based on data from the Kaggle competition "Store Sales - Time Series Forecasting," which provides over four years of sales data from a national retail company in Ecuador. The proposed approach involves the integration of multiple data sources, including holidays, oil prices, transaction volume, and regional events, followed by preprocessing, feature engineering, evaluation of predictive models, hyperparameter selection and tuning, and the application of ensemble techniques. The results demonstrate that the ensemble outperforms individual models, highlighting the potential of machine learning in demand forecasting in complex and dynamic environments.

## Resumo

Com a crescente complexidade dos cenários de mercado, aumenta a necessidade por soluções robustas e inteligentes para previsão de demanda, especialmente no setor varejista. Este trabalho apresenta um estudo baseado nos dados da competição do Kaggle "Store Sales - Time Series Forecasting", que disponibiliza mais de quatro anos de vendas de uma empresa nacional de varejo no Equador. A abordagem proposta envolve a integração de múltiplas fontes de dados, incluindo feriados, preços de petróleo, volume de transações e eventos regionais, seguida de pré-processamento, feature engineering, avaliação de modelos preditivos, seleção e tunagem de hyperparametros, além da aplicação de técnicas de ensemble. Os resultados demonstram que o ensemble supera modelos individuais, evidenciando o potencial do aprendizado de máquina na previsão de demanda em ambientes complexos e dinâmicos.

---

## Seção 1 - Introdução

### 1.1 Motivação

A previsão precisa de demanda desempenha um papel fundamental no setor de varejo, especialmente para lojas e mercados que lidam com bens perecíveis e uma demanda altamente variável. Antecipar corretamente as necessidades dos consumidores é essencial para manter uma gestão de estoque eficiente, reduzir o desperdício de produtos e evitar a escassez de itens populares, o que poderia acarretar perdas financeiras e insatisfação dos clientes.

Com o avanço das técnicas de aprendizado de máquina (AM), tornou-se possível desenvolver modelos preditivos mais precisos, robustos, automatizados e orientados por dados. Tais modelos oferecem grande potencial para lidar com a complexidade e dinamicidade dos ambientes de varejo, possibilitando tomadas de decisão mais embasadas.

### 1.2 Apresentação da Competição

O objetivo central deste projeto é desenvolver e avaliar uma pipeline completa de aprendizado de máquina para prever as vendas unitárias de milhares de itens em diversas lojas da Corporación Favorita, uma grande varejista equatoriana. Esta tarefa é baseada nos dados da competição [Store Sales - Time Series Forecasting hospedada na plataforma Kaggle](https://www.kaggle.com/competitions/store-sales-time-series-forecasting). Especificamente, o modelo deve prever as vendas para um horizonte de 15 dias subsequentes à última data presente no conjunto de dados de treinamento.

Os dados fornecidos abrangem informações sobre as lojas, incluindo metadados como cidade, estado, tipo e cluster que a loja pertence; dados históricos de vendas unitárias para uma vasta gama de produtos; preços diários do petróleo, um fator economicamente relevante para o Equador; informações sobre promoções aplicadas aos itens; o número de transações diárias por loja; e um calendário detalhado de feriados nacionais, regionais e locais, além de eventos especiais que podem impactar o volume de vendas.

---

### Seção 2 - Conjuntos de Dados e Pré-processamento

#### Conjuntos de Dados

A competição disponibiliza ao todo 6 conjuntos de dados fundamentais para a tarefa de previsão de demanda. Exemplo dos datasets são incluídos nas tabelas subsequentes.

Os arquivos principais `train.csv` e `test.csv` abrangem, respectivamente, os períodos de 1º de janeiro de 2013 a 15 de agosto de 2017 (1687 dias) e de 16 a 31 de agosto de 2017 (15 dias), totalizando 1702 dias. Estes arquivos detalham as vendas (`sales`) e os itens em promoção (`onpromotion`) para cada data, loja (`store_nbr`) e família de produtos (`family`). A granularidade, com 54 lojas e 33 famílias, resulta em 1.782 séries temporais distintas. O desafio consiste em, portanto, prever para essas 1.782 séries temporais únicas a demanda diária dos 15 dias presentes no período de teste.

| id  | date       | store_nbr | family     | sales | onpromotion |
| --- | ---------- | --------- | ---------- | ----- | ----------- |
| 0   | 2013-01-01 | 1         | AUTOMOTIVE | 0.0   | 0           |
| 1   | 2013-01-01 | 1         | BABY CARE  | 0.0   | 0           |
| 2   | 2013-01-01 | 1         | BEAUTY     | 0.0   | 0           |
| 3   | 2013-01-01 | 1         | BEVERAGES  | 0.0   | 0           |
| 4   | 2013-01-01 | 1         | BOOKS      | 0.0   | 0           |
Descrição: `train.csv` e `test.csv`, 3000888 rows × 6 columns. Informações de promoção e vendas (target variable) para 1.782 séries temporais distintas, especificadas por `store_nbr`e `family` em um período de 1º de janeiro de 2013 a 15 de agosto de 2017 (1687 dias de treino) e de 16 a 31 de agosto de 2017 (15 dias), totalizando 1702 dias.

Informações contextuais sobre as lojas são fornecidas pelo dataset `stores.csv`, que detalha a localização (cidade, estado) e categoriza as lojas por tipo (type) e cluster (cluster). Embora os critérios desses agrupamentos não sejam explícitos nas especificações da competição, eles habilitam, através de técnincas de ensemble, a utilização de modelos especializados ajustadas a cada perfil de loja. Essa técninca será explorada na seção 4.3.

| store_nbr | city  | state     | type | cluster |
| --------- | ----- | --------- | ---- | ------- |
| 1         | Quito | Pichincha | A    | 13      |
| 2         | Quito | Pichincha | B    | 13      |
| 3         | Quito | Pichincha | C    | 8       |
| 4         | Quito | Pichincha | D    | 9       |
| 5         | Quito | Pichincha | E    | 13      |
Descrição: Dataset `stores.csv`, 54 rows × 5 columns. Informações das 54 lojas presentes na competição.

O conjunto `oil.csv` fornece os preços diários do barril de petróleo ao longo de todo o intervalo de treino e teste. Considerando que o Equador tem sua economia fortemente baseada nas exportações de petróleo  (Figura X), oscilações nesse preço podem afetar diretamente o poder de compra da população e, consequentemente, o comportamento de consumo. Essa variável, portanto, é de alta relevância e deve ser incorporada à modelagem.

![Quadro de Exportações do Ecuador](Ecuador_Product_Exports_(2019).svg.png)

| date       | dcoilwtico |
| ---------- | ---------- |
| 2013-01-01 | NaN        |
| 2013-01-02 | 93.14      |
| 2013-01-03 | 92.97      |
| 2013-01-04 | 93.12      |
| 2013-01-07 | 93.2       |
Descrição: Dataset `oil.csv`, 1218 rows × 2 columns. Preço do barril de petróleo por data.

O arquivo `holidays_events.csv` fornece informações sobre feriados e eventos ocorridos no Equador. Este conjunto de dados detalha o tipo de evento: nacional, regional, local, sua descrição, e se houve transferência da data de comemoração. É importante notar a presença tanto de feriados recorrentes com potencial de forte impacto nos padrões de consumo, como Natal, Ano Novo e o Dia dos Finados, quanto de eventos pontuais, como competições esportivas.

| date       | type    | locale   | locale_name | description                   | transferred |
| ---------- | ------- | -------- | ----------- | ----------------------------- | ----------- |
| 2012-03-02 | Holiday | Local    | Manta       | Fundacion de Manta            | False       |
| 2012-04-01 | Holiday | Regional | Cotopaxi    | Provincializacion de Cotopaxi | False       |
| 2012-04-12 | Holiday | Local    | Cuenca      | Fundacion de Cuenca           | False       |
| 2012-04-14 | Holiday | Local    | Libertad    | Cantonizacion de Libertad     | False       |
| 2012-04-21 | Holiday | Local    | Riobamba    | Cantonizacion de Riobamba     | False       |

Descrição: Dataset `holidays_events.csv`, 350 rows × 6 columns. Informações de feriados e eventos comemorativos.

O dataset `transactions.csv` oferece contagens de transações diárias por loja. O interessante é que possuimos esses dados apenas para o período de treino e sem granularidade por família de produto, apenas por loja, o que exige um processamento adequado para que seja possível seu uso para os dados de teste.

| date       | store_nbr | transactions |
| ---------- | --------- | ------------ |
| 2013-01-01 | 25        | 770          |
| 2013-01-02 | 1         | 2111         |
| 2013-01-02 | 2         | 2358         |
| 2013-01-02 | 3         | 3487         |
| 2013-01-02 | 4         | 1922         |

Descrição: Dataset `transactions.csv`, 83488 rows × 3 columns. Transações diárias por loja.

Por fim, informações sobre alguns eventos contextuais são fornecidos pela competição, como os pagamentos quinzenais no setor público, realizados nos dias 15 e no último dia de cada mês, que podem influenciar o consumo, e o terremoto de magnitude 7.8 ocorrido em 16 de abril de 2016. Este último teve impacto significativo nas vendas de itens essenciais por várias semanas, afetando o consumo. A modelagem correta desses eventos pode garantir previsões mais precisas e robustas.

#### Pré-processamento

A etapa de pré-processamento de dados foi focada no tratamento de valores ausentes e na análise de outliers. Em relação aos outliers, não se observaram ocorrências semanticamente inconsistentes, como vendas negativas, que necessitassem de remoção ou substituição.

Os valores faltantes, por outro lado, foram tratados de maneira específica para cada conjunto de dados. Nos dados de preços do petróleo, por exemplo, as ausências concentravam-se majoritariamente nos finais de semana, período em que, presume-se, não há negociação de mercado. Para esses casos, aplicou-se a técnica de forward fill, onde os valores do final de semana são preenchidos com o valor registrado na sexta-feira anterior. Já para os dados de transações diárias, a interpolação linear foi utilizada para estimar os valores ausentes durante o período de teste.

Após a conclusão dessas etapas de tratamento, os diversos conjuntos de dados processados foram consolidados em um único dataset completo, que serviu de base para a próxima etapa de feature engineering.

---

### Seção 3 - Feature Engineering

No processo de feature engineering para a previsão de demanda, foram implementadas diversas técnicas para extrair informações relevantes. A partir da data, foram criadas features de calendário, incluindo ano, mês, dia do mês, dia da semana (numérico e nominal), nome do mês, semana do ano, trimestre, dia do ano, e indicadores binários para fins de semana e início/fim de mês, trimestre e ano. O propósito dessas features é fornecer ao modelo sinais explícitos sobre sazonalidades, ciclicidades e eventos calendáricos que influenciam o comportamento das vendas.

Features de Lag e rolling statistics foram incorporadas para capturar a dinâmica e a dependência temporal dos dados. As features de lag consistem nos valores de variáveis em instantes anteriores (e.g., Xt−1​,Xt−2​,...,Xt−k​), permitindo que o modelo aprenda com o passado recente e capture a inércia ou efeitos retardados. Foram criadas lags de até 7 dias para as variáveis de preço do petróleo, promoções, transações e para a própria varíavel alvo de vendas. Para as mesmas variáveis foram criada as rolling statistics, que calculam médias e desvios padrão sobre janelas deslizantes de 7, 14 e 28 dias. Elas resumem o comportamento local da série, suavizando ruídos, destacando tendências de curto prazo e quantificando a volatilidade recente.

Para tratar eventos específicos, como os dias de pagamento (mencionados na competição), foram criadas features binárias de proximidade (e.g., janela de +/- 15 dias) e contadores de dias relativos ao ciclo de pagamento. Analogamente, para o terremoto de 16 de abril de 2016, foram desenvolvidas features para indicar a janela do evento e a contagem de dias antes e depois do sismo.

A fim de eliminar os tipos categóricos de algumas variáveis, foi utilizado a técnina de target encoding. Diferentemente de métodos como one-hot encoding, que podem expandir significativamente o espaço de features com variáveis de alta cardinalidade, o target encoding atribui a cada categoria um único valor numérico. No caso desse estudo, a média da variável alvo observada para os registros que contêm aquela categoria específica. Dessa forma, é embutida a relação entre a categoria e o target na representação da feature.

### Seção 4 - Treinamento dos Modelos

#### Metodologia

A pipeline de modelagem utilizada incluiu as seguintes etapas:
    1. Treinamento e avaliação de modelos de regressão comuns, culminando na identificação e seleção do LightGBM como algoritmo central, com base em sua superioridade em performance e eficiência.
    2. Otimização de hiperparâmetros do modelo LightGBM selecionado e subsequente reavaliação de seu desempenho.
    3. Aplicação de técnicas de ensemble para o aprimoramento final dos resultados.

As métricas de avaliação utilizadas foram MAE, RMSE, R² e RMSLE, sendo esta última a principal, conforme critério da competição. A Raiz do Erro Quadrático Médio Logarítmico (RMSLE) é calculada como:
RMSLE=n1​i=1∑n​(ln(pi​+1)−ln(ai​+1))2​
(com pi​ sendo a predição e ai​ o valor real). O uso do logaritmo (com adição de 1 para tratar zeros) significa que a RMSLE mede a razão entre os valores previstos e reais, tornando-a sensível a erros percentuais e menos afetada por outliers de grande escala. Isso é particularmente útil para dados de vendas, que podem apresentar grande variação.

#### Série de Modelos

A etapa preliminar de modelagem compreendeu a seleção e avaliação comparativa de um grupo de algoritmos de regressão. Foram considerados modelos de referência (baselines), como o Dummy Regressor (predição baseada na média dos dados de treinamento) e a Regressão Linear, progredindo para algoritmos de maior complexidade. Estes incluíram a Árvore de Decisão, ensembles de árvores como o HistGradientBoosting, e o algoritmo de gradient boosting avançado LightGBM.

Todos os modelos foram inicialmente treinados utilizando suas configurações de hiperparâmetros padrão. Para a avaliação de desempenho, empregou-se de validação cruzada específica para séries temporais, configurada com 4 partições (folds). As métricas gerais de desempenho reportadas para cada modelo representam a média dos resultados obtidos de seus folds. Os resultados detalhados desta etapa são apresentados nas tabelas subsequentes.

DummyRegressor
| Fold | MAE      | RMSE      | RMSLE  | R²      | Time (s) |
| ---- | -------- | --------- | ------ | ------- | -------- |
| 1    | 397.0867 | 966.0150  | 3.8712 | -0.0082 | 0.0175   |
| 2    | 456.3634 | 1080.6752 | 3.7774 | -0.0077 | 0.0068   |
| 3    | 530.0777 | 1274.8923 | 3.4555 | -0.0123 | 0.0070   |
| 4    | 583.0717 | 1374.3693 | 3.4317 | -0.0112 | 0.0083   |

LinearRegression (Linear)
| Fold | MAE      | RMSE     | RMSLE  | R²     | Time (s) |
| ---- | -------- | -------- | ------ | ------ | -------- |
| 1    | 147.7540 | 598.4125 | 1.4660 | 0.6131 | 44.3978  |
| 2    | 130.2411 | 610.0230 | 1.2187 | 0.6789 | 103.8828 |
| 3    | 114.4227 | 528.9300 | 0.9045 | 0.8258 | 174.2278 |
| 4    | 108.7634 | 450.2158 | 0.7852 | 0.8915 | 203.1671 |

DecisionTree
| Fold | MAE      | RMSE     | RMSLE  | R²     | Time (s)  |
| ---- | -------- | -------- | ------ | ------ | --------- |
| 1    | 127.4308 | 500.0793 | 1.4526 | 0.7298 | 287.0430  |
| 2    | 97.9187  | 454.6134 | 1.1981 | 0.8217 | 1045.9238 |
| 3    | 76.4641  | 405.9469 | 0.6564 | 0.8974 | 1491.5473 |
| 4    | 70.2314  | 382.1742 | 0.5889 | 0.9212 | 2834.2071 |

HistGradientBoosting
| Fold | MAE      | RMSE     | RMSLE  | R²     | Time (s) |
| ---- | -------- | -------- | ------ | ------ | -------- |
| 1    | 112.0526 | 425.7947 | 1.9667 | 0.8041 | 4.6875   |
| 2    | 85.1386  | 361.9302 | 1.2179 | 0.8870 | 8.2149   |
| 3    | 78.5190  | 387.0799 | 0.9382 | 0.9067 | 13.1811  |
| 4    | 78.8274  | 308.6743 | 0.9091 | 0.9490 | 15.3855  |

LightGBM
| Fold | MAE      | RMSE     | RMSLE  | R²     | Time (s) |
| ---- | -------- | -------- | ------ | ------ | -------- |
| 1    | 109.9611 | 407.6579 | 2.2111 | 0.8204 | 3.5430   |
| 2    | 84.8564  | 359.7692 | 1.2187 | 0.8883 | 5.4818   |
| 3    | 78.4244  | 388.3210 | 0.8983 | 0.9061 | 8.5012   |
| 4    | 78.6972  | 304.3005 | 0.9059 | 0.9504 | 9.4887   |

Comparativo de Desempenho Médio dos Modelos na Validação Cruzada
| Modelo               | RMSLE      | RMSE         | MAE         | R²         | Tempo Médio (s) |
| -------------------- | ---------- | ------------ | ----------- | ---------- | --------------- |
| DummyRegressor       | 3.6340     | 1173.9880    | 491.6499    | -0.0099    | 0.01            |
| LinearRegression     | 2.2518     | 448.1833     | 149.9042    | 0.8411     | 9.51            |
| DecisionTree         | 1.0936     | 546.8953     | 125.2953    | 0.7523     | 131.42          |
| RandomForest         | **0.9740** | 435.7035     | 93.0113     | 0.8425     | 1414.68         |
| HistGradientBoosting | 1.2580     | 370.8698     | 88.6344     | 0.8867     | 10.37           |
| LightGBM             | 1.3085     | **365.0122** | **87.9848** | **0.8913** | **6.75**        |

A análise dos resultados médios da validação cruzada (Tabela X) evidencia a superioridade dos modelos de ensemble e gradient boosting em relação aos baselines (DummyRegressor, LinearRegression) e à DecisionTree simples. O RandomForest se destaca ao obter o menor RMSLE (0.9740), métrica princiapl da competição, mas ao custo de um tempo de treinamento significativamente alto (1414.68s). HistGradientBoosting apresenta uma performance boa e rápida, e o LightGBM demonstra o melhor desempenho global em termos de RMSE (365.0122), MAE (87.9848) e R² (0.8913), e se consagra como o mais eficiente em tempo de treinamento, com incriveis 6.75s totais de processamento. Com base nos dados de avaliação, o modelo LightGBM é selecionado para as fases subsequentes do estudo, primariamente devido à sua performance e ao seu tempo de processamento reduzido. A próxima etapa da análise foca em investigar o grau de melhoria no desempenho do LightGBM após a otimização de seus hiperparâmetros.

A Figura X detalha a importância relativa das 50% de features consideradas mais preditivas pelo modelo LightGBM. Esta análise da contribuição das features revela que diversas categorias de variáveis são determinantes para a acurácia da predição de demanda. Características categóricas intrínsecas, como family (categoria do produto) e store_nbr (identificador da loja), posicionam-se entre as mais significativas, assim como dados transacionais diretos (transactions) e as médias móveis derivadas destes. Variáveis de natureza temporal, incluindo tanto componentes de calendário (day_of_year, day, day_of_week) quanto diversas formas de vendas defasadas e suas respectivas estatísticas móveis (e.g., sales_lag21, sales_rolling_mean_lag16_window_size28) também demonstraram elevada relevância, o que corrobora a forte dependência temporal inerente às séries de vendas. Adicionalmente, fatores promocionais (onpromotion) e variáveis exógenas como o preço do petróleo (dcoilwtico) e indicadores de eventos (por exemplo, features relacionadas ao terremoto) também integram o conjunto de features significativas.

![Feature Importance](image-1.png)

#### Tunagem de Hyperparametros

A etapa de otimização de hiperparâmetros do LightGBM foi conduzida utilizando o framework [Optuna](https://optuna.org/). A escolha desta ferramenta, em detrimento de abordagens como a exploração manual, Grid Search ou Random Search (oferecidas por bibliotecas como Scikit-learn), ou outros frameworks (Hyperopt, Ax), baseia-se na crescente proeminência do Optuna na comunidade científica e em sua capacidade de alcançar resultados notáveis através de algoritmos de busca eficientes.

Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. In Proceedings of the 25th ACM SIGKDD International 1 Conference on Knowledge Discovery & Data Mining 2 (pp. 2623-2631).

Durante a otimização, um conjunto de hiperparâmetros chave foi explorado, incluindo o número de árvores (n_estimators), a taxa de aprendizado (learning_rate), a complexidade das árvores (e.g., num_leaves), os coeficientes de regularização L1 e L2, e as frações de amostragem de dados e atributos (bagging_fraction, feature_fraction). O processo envolveu a execução de 25 trials. Em cada trial, o Optuna propõe uma configuração de hiperparâmetros, treina o modelo e avalia seu desempenho, utilizando essa informação para guiar a exploração subsequente do espaço de busca de forma adaptativa e eficiente.

| Modelo             | RMSLE  | RMSE     | MAE     | R²     | Tempo de Treinamento (s) |
| ------------------ | ------ | -------- | ------- | ------ | ------------------------ |
| LightGBM Básico    | 0.8919 | 307.8006 | 78.8475 | 0.9493 | 13.8394                  |
| LightGBM Otimizado | 0.4693 | 285.6477 | 66.1276 | 0.9563 | 170.8635                 |
Descrição: Resultados comparativos para a otimização de hyperparametros para o modelo LightGBM. 

Os resultados, apresentados na Tabela X, demonstram que a otimização de hiperparâmetros produziu um modelo LightGBM significativamente mais eficiente. O modelo otimizado apresentou reduções nos valores de RMSLE (0.4693), RMSE (285.6477) e MAE (66.1276), além de um aumento no coeficiente R² (0.9563), em relação à configuração básica. Esses ganhos refletem uma melhora substancial na precisão e na capacidade preditiva do modelo, evidenciando a importância da etapa de otimização em aplicações de aprendizado de máquina. Observa-se também um acréscimo considerável no tempo de treinamento, que aumentou de 13,84s para 170,86s. Tal crescimento é atribuído à maior complexidade do modelo, que passou de árvores com profundidade máxima de 5 para 11 e de 100 para até 1500 estimadores. No contexto deste estudo, o ganho de desempenho justifica o aumento no custo computacional. Contudo, em outras aplicações, esse trade-off deve ser cuidadosamente avaliado.

#### Ensemble

A considerável variabilidade inerente às séries temporais de vendas motiva a busca por soluções que superam um modelo monolítico em performance, o qual pode não ser capaz de aprender todos os padrões dos dados. Para contornar esse desafio, utiliza-se a criação de múltiplos modelos especializados, cada um treinado em segmentos específicos do conjunto de dados original. A união das previsões desses modelos visa a um resultado final mais robusto e inteligente. 

Nesse viês, nessa aplicação é desenvolvida três abordagens de ensemble:

 1. Por Família de Produtos: Criam-se 33 modelos distintos, cada um especializado em uma das 33 famílias de produtos.
 2. Por Tipo de Loja: Desenvolvem-se 5 modelos, um para cada um dos cinco tipos de loja (A,B,C,D,E).
 3. Por Cluster de Loja: Geram-se 17 modelos, correspondendo a cada um dos clusters de loja existentes.

As performances individuais dos modelos especializados em cada abordagem são avaliadas. Subsequentemente, considera-se a formação de um ensemble final pela média das previsões dos modelos especializados.

Os resultados obtidos são verificados na tabela X

| Ensemble  | Nº de Modelos | RMSLE (média) | RMSE (média) | MAE (média) | R² (média) |
| --------- | ------------- | ------------- | ------------ | ----------- | ---------- |
| `family`  | 33            | **0.3145**    | **49.9787**  | **27.6311** | 0.8957     |
| `type`    | 5             | 0.5511        | 139.6999     | 39.6427     | 0.9799     |
| `cluster` | 17            | 0.5219        | 104.1676     | 32.5279     | **0.9895** |
| `all`     | 45            | 0.4625        | 97.9487      | 33.2672     | 0.9550     |

Os resultados da avaliação das estratégias de ensemble (Tabela Y) indicam um desempenho diferenciado conforme a forma de especialização dos modelos. A abordagem que especializa modelos por família de produtos (33 modelos) demonstrou ser a mais eficaz na minimização dos erros, alcançando os menores RMSLE (0.3145), RMSE (49.9787) e MAE (27.6311). Por outro lado, o ensemble focado em clusters de loja (17 modelos) destacou-se com o maior R² (0.9895) e um MAE competitivo (32.5279). A estratégia por tipo de loja (5 modelos), embora tenha apresentado um R² (0.9799) superior ao do ensemble por família, registrou os piores indicadores para RMSLE (0.5511), RMSE (139.6999) e MAE (39.6427). O ensemble all (45 modelos), cujas métricas representam a média dos três ensembles especializados, teve um desempenho misto: foi inferior ao ensemble family na maioria das métricas de erro (RMSLE, RMSE, MAE) e não atingiu o R² dos ensembles cluster ou type. Estes resultados sugerem que a especialização criteriosa dos modelos pode ser mais benéfica do que uma simples agregação geral, e que um número maior de modelos no ensemble all não garantiu, por si só, a melhor performance em todas as métricas. Conclui-se, portanto, que os modelos especializados por família de produtos apresentaram o melhor desempenho entre as abordagens avaliadas. Isso sugere que a segmentação por família é eficaz, pois os produtos dentro de uma mesma família tendem a compartilhar características comportamentais e padrões de demanda semelhantes. Essa homogeneidade interna facilita o aprendizado de padrões relevantes pelos modelos especializados, resultando em uma capacidade de generalização superior quando comparada a outras formas de agrupamento.

Aqui está a seção de conclusão revisada, sem o último parágrafo sobre limitações e trabalhos futuros, e com um tom mais direto:

#### Seção 5 - Conclusão

Este estudo desenvolveu e avaliou uma pipeline de aprendizado de máquina para previsão de demanda no varejo, baseada em dados heterogêneos da competição Kaggle "Store Sales - Time Series Forecasting". A metodologia incluiu integração e pré-processamento de dados, robusta engenharia de atributos, seleção do LightGBM como modelo base, otimização de hiperparâmetros e a aplicação de estratégias de ensemble.

Os resultados confirmaram a eficácia da pipeline. A otimização de hiperparâmetros melhorou significantemente o desempenho do LightGBM individual, e as técnicas de ensemble especializadas demonstraram superioridade. Especificamente, o ensemble por família de produtos alcançou a melhor performance na minimização de erros (RMSLE 0.3145, RMSE 49.9787, MAE 27.6311), o que é consistente com a alta importância da feature family identificada na análise do modelo.

Este estudo evidencia o alto potencial do aprendizado de máquina, particularmente modelos de gradient boosting e ensembles customizados, para a complexa tarefa de previsão de demanda no varejo com dados diversos. A engenharia de atributos e a otimização de hiperparâmetros provaram-se cruciais, mas a especialização de modelos por família de produtos demonstrou ser a estratégia de ensemble mais vantajosa, resultando em ganhos de performance substanciais sobre abordagens genéricas.