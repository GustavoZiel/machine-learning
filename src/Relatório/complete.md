# Previsão de Demanda com Dados Heterogêneos: Uma Pipeline de Machine Learning Aplicado ao Varejo

Autor: Gustavo Gabriel Ribeiro
Data: 11 de maio de 2025

---

## Resumo

Com a crescente complexidade dos cenários de mercado, aumenta a necessidade por soluções robustas e inteligentes para previsão de demanda, especialmente no setor varejista. Este trabalho apresenta um estudo baseado nos dados da competição do Kaggle "Store Sales - Time Series Forecasting", que disponibiliza mais de quatro anos de vendas de uma empresa nacional de varejo no Equador. A abordagem proposta envolve a integração de múltiplas fontes de dados — incluindo feriados, preços de petróleo, volume de transações e eventos regionais — seguida de pré-processamento, feature engineering, avaliação de modelos preditivos, seleção e tunagem com Optuna, além da aplicação de técnicas de ensemble. Os resultados demonstram que o ensemble supera modelos individuais, evidenciando o potencial do aprendizado de máquina na previsão de demanda em ambientes complexos e dinâmicos.

---

## Seção 1 - Introdução

### 1.1 Motivação

A previsão precisa de demanda desempenha um papel fundamental no setor de varejo, especialmente para lojas e mercados que lidam com bens perecíveis e uma demanda altamente variável. Antecipar corretamente as necessidades dos consumidores é essencial para manter uma gestão de estoque eficiente, reduzir o desperdício de produtos e evitar a escassez de itens populares, o que poderia acarretar perdas financeiras e insatisfação dos clientes.

Com o avanço das técnicas de aprendizado de máquina (AM), tornou-se possível desenvolver modelos preditivos mais precisos, robustos, automatizados e orientados por dados. Tais modelos oferecem grande potencial para lidar com a complexidade e dinamicidade dos ambientes de varejo, possibilitando tomadas de decisão mais embasadas.

### 1.2 Apresentação da Competição

O objetivo central deste projeto é desenvolver e avaliar uma pipeline completa de aprendizado de máquina para prever as vendas unitárias de milhares de itens em diversas lojas da Corporación Favorita, uma grande varejista equatoriana. Esta tarefa é baseada nos dados da competição Store Sales - Time Series Forecasting hospedada na plataforma Kaggle (TODO CITE). Especificamente, o modelo deve prever as vendas para um horizonte de 15 dias subsequentes à última data presente no conjunto de dados de treinamento.

Os dados fornecidos abrangem informações sobre as lojas, incluindo metadados como cidade, estado, tipo e cluster que a loja pertence; dados históricos de vendas unitárias para uma vasta gama de produtos; preços diários do petróleo, um fator economicamente relevante para o Equador; informações sobre promoções aplicadas aos itens; o número de transações diárias por loja; e um calendário detalhado de feriados nacionais, regionais e locais, além de eventos especiais que podem impactar o volume de vendas.

---

### Seção 2 - Conjuntos de Dados e Pré-processamento

#### Conjuntos de Dados

A competição disponibiliza ao todo 6 conjuntos de dados fundamentais para a tarefa de previsão de demanda. Exemplo dos datasets são incluídos nas tabelas subsequentes.

Os arquivos principais `train.csv` e `test.csv` abrangem, respectivamente, os períodos de 1º de janeiro de 2013 a 15 de agosto de 2017 (1687 dias) e de 16 a 31 de agosto de 2017 (15 dias), totalizando 1702 dias. Estes arquivos detalham as vendas (`sales`) e os itens em promoção (`onpromotion`) para cada data, loja (`store_nbr`) e família de produtos (`family`). A granularidade, com 54 lojas e 33 famílias, resulta em 1.782 séries temporais distintas. O desafio consiste, portanto em prever para as 1.782 séries temporais únicas a demanda, e portanto vendas, dos 15 dias do período de teste.

| id  | date       | store_nbr | family     | sales | onpromotion |
| --- | ---------- | --------- | ---------- | ----- | ----------- |
| 0   | 2013-01-01 | 1         | AUTOMOTIVE | 0.0   | 0           |
| 1   | 2013-01-01 | 1         | BABY CARE  | 0.0   | 0           |
| 2   | 2013-01-01 | 1         | BEAUTY     | 0.0   | 0           |
| 3   | 2013-01-01 | 1         | BEVERAGES  | 0.0   | 0           |
| 4   | 2013-01-01 | 1         | BOOKS      | 0.0   | 0           |

Informações contextuais sobre as lojas são fornecidas pelo `stores.csv`, que detalha a localização (cidade, estado) e categoriza as lojas por tipo (type) e cluster (cluster). Embora os critérios desses agrupamentos não sejam explícitos nas especificações, eles habilitam a exploração de modelos especializados, como técnicas de ensemble ajustadas a cada perfil de loja.

| store_nbr | city  | state     | type | cluster |
| --------- | ----- | --------- | ---- | ------- |
| 1         | Quito | Pichincha | A    | 13      |
| 2         | Quito | Pichincha | B    | 13      |
| 3         | Quito | Pichincha | C    | 8       |
| 4         | Quito | Pichincha | D    | 9       |
| 5         | Quito | Pichincha | E    | 13      |

O conjunto `oil.csv` fornece os preços diários do barril de petróleo ao longo de todo o intervalo de treino e teste. Considerando que o Equador tem sua economia fortemente baseada nas exportações de petróleo  (Figura X), oscilações nesse preço podem afetar diretamente o poder de compra da população e, consequentemente, o comportamento de consumo. Essa variável, portanto, é de alta relevância e deve ser incorporada à modelagem.

![Quadro de Exportações do Ecuador](Ecuador_Product_Exports_(2019).svg.png)

| date       | dcoilwtico |
| ---------- | ---------- |
| 2013-01-01 | NaN        |
| 2013-01-02 | 93.14      |
| 2013-01-03 | 92.97      |
| 2013-01-04 | 93.12      |
| 2013-01-07 | 93.2       |

O arquivo `holidays_events.csv` traz informações sobre feriados e eventos no Equador, incluindo tipo (nacional, regional, adicional etc.), possíveis transferências de data, e informações adicionais. Há feriados como Natal, Ano Novo, Dia dos Mortos que são recorrentes e podem afetar significativamente os padrões de consumo, e outros pontuais, como eventos esportivos, que podem gerar ruído ou infomrações úteis sobre os dados que também devem ser levados em consideração.

| date       | type    | locale   | locale_name | description                   | transferred |
| ---------- | ------- | -------- | ----------- | ----------------------------- | ----------- |
| 2012-03-02 | Holiday | Local    | Manta       | Fundacion de Manta            | False       |
| 2012-04-01 | Holiday | Regional | Cotopaxi    | Provincializacion de Cotopaxi | False       |
| 2012-04-12 | Holiday | Local    | Cuenca      | Fundacion de Cuenca           | False       |
| 2012-04-14 | Holiday | Local    | Libertad    | Cantonizacion de Libertad     | False       |
| 2012-04-21 | Holiday | Local    | Riobamba    | Cantonizacion de Riobamba     | False       |

O dataset `transactions.csv` oferece contagens de transações diárias por loja. O interessante é que possuimos esses dados apenas para o período de treino e sem granularidade por família de produto, apenas por loja, o que exige um processamento adequado para que seja possível seu uso para os dados de teste.

| date       | store_nbr | transactions |
| ---------- | --------- | ------------ |
| 2013-01-01 | 25        | 770          |
| 2013-01-02 | 1         | 2111         |
| 2013-01-02 | 2         | 2358         |
| 2013-01-02 | 3         | 3487         |
| 2013-01-02 | 4         | 1922         |

Por fim, informações sobre alguns eventos contextuais são fornecidos pela competição, como os pagamentos quinzenais no setor público, realizados nos dias 15 e no último dia de cada mês, que podem influenciar o consumo, e o terremoto de magnitude 7.8 ocorrido em 16 de abril de 2016. Este último teve impacto significativo nas vendas de itens essenciais por várias semanas, afetando o consumo. A modelagem correta desses eventos pode garantir previsões mais precisas e robustas.

#### Pré-processamento

O pré-processamento dos dados concentrou-se primordialmente no tratamento de valores ausentes e na análise de outliers. Com relação a outliers, não foram identificadas ocorrências significativas que demandassem remoção ou substituição, como valores semanticamente inconsistentes (e.g., vendas negativas). Os valores faltantes, por sua vez, foram tratados caso a caso. Nos dados de preços do petróleo, por exemplo, foram observadas ausências principalmente nos fins de semana, possivelmente devido à interrupção das negociações de mercado nesse período, e preenchidas utilizando-se de forward fill, i.e os valores faltantes no fim de semana serão iguais aos valores de sexta-feira. Sobre os dados de transações diárias, a fim de lidar com os dados faltantes no período de teste, foi empregada a interpolação linear, permitindo sua utilização pelo modelo. Após os processamentos realizados, os dados dos diferentes modelos foram unidos em apenas um grande dataset que é utilizado na etapa de feature engineering.

---

### Seção 3 - Feature Engineering

No processo de feature engineering para a previsão de demanda, foram implementadas diversas técnicas para extrair informações relevantes. A partir da data, foram criadas features de calendário, incluindo ano, mês, dia do mês, dia da semana (numérico e nominal), nome do mês, semana do ano, trimestre, dia do ano, e indicadores binários para fins de semana e início/fim de mês, trimestre e ano. O propósito dessas features é fornecer ao modelo sinais explícitos sobre sazonalidades, ciclicidades e eventos calendáricos que influenciam o comportamento das vendas.

Lag features e rolling statistics foram incorporadas para capturar a dinâmica e a dependência temporal dos dados. As lag features consistem nos valores de variáveis em instantes anteriores (e.g., Xt−1​,Xt−2​,...,Xt−k​), permitindo que o modelo aprenda com o passado recente e capture a inércia ou efeitos retardados. Foram criadas lags de até 7 dias para as variáveis de preço do petróleo, promoções, transações e para a própria varíavel alvo de vendas. Para as mesmas variáveis foram criada as rolling statistics, que calculam médias e desvios padrão calculados sobre janelas deslizantes de 7, 14 e 28 dias. Elas resumem o comportamento local da série, suavizando ruídos, destacando tendências de curto prazo e quantificando a volatilidade recente.

Para tratar eventos específicos, como os dias de pagamento (mencionados na competição), foram criadas features binárias de proximidade (e.g., janela de +/- 15 dias) e contadores de dias relativos ao ciclo de pagamento. Analogamente, para o terremoto de 16 de abril de 2016, foram desenvolvidas features para indicar a janela do evento e a contagem de dias antes e depois do sismo, buscando modelar seu impacto particular.

Por fim, a fim de eliminar os tipos categóricos de algumas variáveis, foi utilizado a técnina de target encoding. Diferentemente de métodos como one-hot encoding, que podem expandir significativamente o espaço de features com variáveis de alta cardinalidade, o target encoding atribui a cada categoria um único valor numérico. No caso desse estudo, a média da variável alvo observada para os registros que contêm aquela categoria específica. Dessa forma, é embutida a relação entre a categoria e o target na representação da feature.

<!-- No contexto de previsão de demanda com base em séries temporais para, a incorporação de informações relacionadas à estrutura temporal dos dados é essencial para capturar padrões sazonais, tendências e efeitos recorrentes. A partir da variável de data presente no conjunto de dados, foram extraídos diversos componentes temporais, tais como: ano, mês, dia do mês, dia da semana (em formatos numérico e nominal), nome do mês, semana do ano, trimestre e dia do ano. Adicionalmente, foram gerados indicadores binários que identificam se uma data corresponde a um fim de semana, bem como marcadores para início e término de meses, trimestres e anos. Essas variáveis enriquecem a capacidade do modelo de identificar comportamentos sazonais associados, por exemplo, ao dia da semana, ao calendário comercial ou a eventos periódicos. 

Complementando as variáveis derivadas de calendário, o processo de feature engineering incorporou técnicas para capturar dinâmicas temporais complexas. Foram introduzidas lag features (defasagens), construídas a partir de valores passados de séries relevantes, com o objetivo de permitir ao modelo aprender padrões de dependência de curto prazo e a inércia característica de fenômenos temporais. Adicionalmente, foram computadas rolling statistics, incluindo média, desvio padrão, valor máximo e mínimo, utilizando janelas deslizantes de 7, 14 e 28 dias. Estas janelas visam capturar comportamentos sazonais de curto a médio prazo, como ciclos semanais e quinzenais comuns no varejo, onde as médias móveis suavizam flutuações e destacam tendências locais, enquanto outras métricas avaliam volatilidade e extremos. Ambas as técnicas – lags e rolling statistics – foram aplicadas às features de promoção, transações das lojas e preço do petróleo.

Para endereçar eventos específicos mencionados pela competição, como os dias de pagamento no setor público, foram criadas features binárias indicando a proximidade a essas datas (e.g., ocorrência em uma janela de +/- N dias em torno do pagamento) e features numéricas para a contagem de dias até o próximo ciclo de pagamento. De modo análogo, para o terremoto de 16 de abril de 2016 no Equador, foram desenvolvidas variáveis para sinalizar a ocorrência do evento dentro de uma janela temporal específica e para quantificar a contagem de dias antes e após o sismo, permitindo modelar seu impacto disruptivo. -->

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

Na fase inicial, um conjunto de modelos de regressão foi selecionado para treinamento e avaliação. Estes incluíram: modelos de baseline (Dummy Regressor – previsão pela média, e Linear Regression), modelos lineares regularizados (Ridge, Lasso, ElasticNet), modelos não lineares (KNN, Decision Tree), ensembles de árvores (Random Forest, Gradient Boosting, HistGradientBoosting), algoritmos de gradient boosting avançados (LightGBM, XGBoost, CatBoost) e uma Rede Neural (MLP). Todos os modelos foram treinados com suas configurações padrões e submetidos a um processo de validação cruzada para séries temporais com 4 partições (folds). Os resultados das métricas dos modelos culminou na média geral da performance de seus folds. Os resultados obtidos são mostrados nas tabelas a seguir:

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

RandomForest

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


A análise dos resultados médios da validação cruzada (Tabela X) evidencia a superioridade dos modelos de ensemble e gradient boosting em relação aos baselines (DummyRegressor, LinearRegression) e à DecisionTree simples. O RandomForest se destaca ao obter o menor RMSLE (0.9740), métrica princiapl da competição, mas ao custo de um tempo de treinamento significativamente alto (1414.68s). HistGradientBoosting apresenta uma performance boa e rápida, e o LightGBM demonstra o melhor desempenho global em termos de RMSE (365.0122), MAE (87.9848) e R² (0.8913), e se consagra como o mais eficiente em tempo de treinamento, com incriveis 6.75s totais. Com base nos dados de avaliação, o modelo LightGBM é selecionado para as fases subsequentes do estudo, primariamente devido à sua performance e ao seu incrível tempo de processamento. A próxima etapa da análise foca em investigar o grau de melhoria no desempenho do LightGBM após a otimização de seus hiperparâmetros.

#### Tunagem de Hyperparametros

A etapa de otimização de hiperparâmetros do LightGBM foi conduzida utilizando o framework Optuna. A escolha desta ferramenta, em detrimento de abordagens como a exploração manual, Grid Search ou Random Search (oferecidas por bibliotecas como Scikit-learn), ou outros frameworks (Hyperopt, Ax), baseia-se na crescente proeminência do Optuna na comunidade científica e em sua capacidade de alcançar resultados notáveis através de algoritmos de busca eficientes.

Durante a otimização, um conjunto de hiperparâmetros chave foi explorado, incluindo o número de árvores (n_estimators), a taxa de aprendizado (learning_rate), a complexidade das árvores (e.g., num_leaves), os coeficientes de regularização L1 e L2, e as frações de amostragem de dados e atributos (bagging_fraction, feature_fraction). O processo envolveu a execução de 25 trials. Em cada um, o Optuna propõe uma configuração de hiperparâmetros, treina o modelo e avalia seu desempenho, utilizando essa informação para guiar a exploração subsequente do espaço de busca de forma adaptativa e eficiente.

| Modelo             | RMSLE  | RMSE     | MAE     | R²     | Tempo de Treinamento (s) |
| ------------------ | ------ | -------- | ------- | ------ | ------------------------ |
| LightGBM Básico    | 0.8919 | 307.8006 | 78.8475 | 0.9493 | 13.8394                  |
| LightGBM Otimizado | 0.4693 | 285.6477 | 66.1276 | 0.9563 | 170.8635                 |


Os resultados, apresentados na Tabela X, demonstram que a otimização de hiperparâmetros produziu um modelo LightGBM significativamente mais eficiente. O modelo otimizado apresentou reduções nos valores de RMSLE (0.4693), RMSE (285.6477) e MAE (66.1276), além de um aumento no coeficiente R² (0.9563), em relação à configuração básica. Esses ganhos refletem uma melhora substancial na precisão e na capacidade preditiva do modelo, evidenciando a importância da etapa de otimização em aplicações de aprendizado de máquina. Observa-se também um acréscimo considerável no tempo de treinamento, que aumentou de 13,84s para 170,86s. Tal crescimento é atribuído à maior complexidade do modelo, que passou de árvores com profundidade máxima de 5 para 11 e de 100 para até 1500 estimadores. No contexto deste estudo, o ganho de desempenho justifica o aumento no custo computacional. Contudo, em outras aplicações, esse trade-off deve ser cuidadosamente avaliado.

![Learning Curve](image.png)

![Feature Importance](image-1.png)


#### Ensemble

A considerável variabilidade inerente às séries temporais de vendas motiva a busca por soluções que superam um modelo monolítico em performance, o qual pode não ser capaz de aprender todos os padrões dos dados. Para contornar esse desafio, utiliza-se a criação de múltiplos modelos especializados, cada um treinado em segmentos específicos do conjunto de dados original. A união das previsões desses modelos visa a um resultado final mais robusto e inteligente. Nesse viês, nessa aplicação é desenvolvida três abordagens de ensemble:
    1. Por Família de Produtos: Criam-se 33 modelos distintos, cada um especializado em uma das 33 famílias de produtos.
    2. Por Tipo de Loja: Desenvolvem-se 5 modelos, um para cada um dos cinco tipos de loja (A,B,C,D,E).
    3. Por Cluster de Loja: Geram-se 17 modelos, correspondendo a cada um dos clusters de loja existentes. 
    
As performances individuais dos modelos especializados em cada abordagem são avaliadas. Subsequentemente, considera-se a formação de um ensemble final pela média das previsões dos modelos especializados.

Os resultados obtidos são verificados na tabela X

| Ensemble  | Nº de Modelos | RMSLE (média) | RMSE (média) | MAE (média) | R² (média) |
| --------- | ------------- | ------------- | ------------ | ----------- | ---------- |
| `family`  | 33            | **0.4287**    | **3.9264**   | **2.6135**  | **0.8129** |
| `type`    | 5             | **0.5032**    | **7.1221**   | **4.8914**  | **0.7324** |
| `cluster` | 7             | **0.4018**    | **2.9312**   | **1.9886**  | **0.8593** |
| `all`     | 45            | **0.4352**    | **4.1273**   | **2.7890**  | **0.8181** |

Os resultados da avaliação das estratégias de ensemble (Tabela Y) indicam um desempenho variável conforme a forma de especialização dos modelos. O ensemble que especializa modelos por cluster de loja (7 modelos) obteve consistentemente os melhores resultados, com RMSLE de 0.4018, RMSE de 2.9312, MAE de 1.9886 e R² de 0.8593. A abordagem baseada em família de produtos (33 modelos) apresentou o segundo melhor desempenho em RMSLE (0.4287) e nas outras métricas de erro, superando a estratégia por tipo de loja, que registrou os piores indicadores (RMSLE de 0.5032). Curiosamente, o ensemble all, que combina o maior número de modelos (45), não superou a performance da estratégia por cluster nem a por família na maioria das métricas, sugerindo que a simples adição de mais modelos não garante um melhor desempenho e que a heterogeneidade ou a qualidade variável dos modelos componentes podem diluir a precisão final.

#### Seção 5 - Conclusão

