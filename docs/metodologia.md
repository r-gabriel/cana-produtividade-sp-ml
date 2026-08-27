# Metodologia Inicial

## Unidade de análise

A unidade de análise do projeto será definida por:

Município + Ano

Dependendo da disponibilidade de dados climáticos, poderão ser criadas variáveis agregadas por mês, safra ou ano agrícola.

## Recorte geográfico

O foco analítico será o município de Guariba-SP, código IBGE 3518609.

Para treinamento e comparação, serão utilizados dados de outros municípios do estado de São Paulo, pois a análise de apenas um município gera poucas observações históricas para modelos de Machine Learning.

## Variável-alvo

A variável-alvo inicial será a produtividade da cana-de-açúcar, calculada por:

Produtividade = Quantidade produzida / Área colhida

A unidade esperada será tonelada por hectare.

## Variáveis explicativas previstas

- Área plantada;
- Área colhida;
- Quantidade produzida;
- Valor da produção;
- Temperatura média;
- Temperatura máxima;
- Temperatura mínima;
- Precipitação acumulada;
- Umidade relativa;
- Ano;
- Município;
- Indicadores de uso e cobertura do solo, quando disponíveis.

## Modelos previstos

- Regressão Linear como modelo de referência;
- Random Forest Regressor;
- XGBoost Regressor.

## Métricas

- MAE;
- RMSE;
- R².

## Considerações éticas

O projeto utilizará exclusivamente dados públicos e agregados por município, sem dados pessoais ou sensíveis.
Os resultados devem ser interpretados como estimativas analíticas, sem pretensão de substituir avaliações agronômicas especializadas.

## Produção Agrícola Municipal — PAM / SIDRA

INSTITUTO BRASILEIRO DE GEOGRAFIA E ESTATÍSTICA (IBGE).
Produção Agrícola Municipal (PAM), consulta realizada no Sistema IBGE de
Recuperação Automática (SIDRA).

Produto: Cana-de-açúcar.
Município: Guariba (SP).
Código municipal: 3518602.
Variáveis: área plantada, área colhida, quantidade produzida,
rendimento médio e valor da produção.
Período: 2000 a 2024.

Os dados foram obtidos por exportação da consulta configurada na interface
do SIDRA e armazenados em formato CSV na pasta data/raw/agricola.

## Validação dos dados agrícolas de Guariba-SP

A disponibilidade dos dados agrícolas foi validada diretamente na interface
oficial do SIDRA/IBGE, por meio da tabela 5457 da Pesquisa de Produção
Agrícola Municipal.

A consulta confirmou valores numéricos para o município de Guariba-SP,
código 3518602, relacionados à cultura de cana-de-açúcar.

Durante a fase inicial de automação, uma consulta ampla à API, utilizando
todas as variáveis e todos os períodos disponíveis, retornou marcadores
de indisponibilidade ("..."). Para reduzir ambiguidades e garantir
consistência, a estratégia de extração foi ajustada para realizar consultas
específicas por variável, produto e período.

A validação pela interface oficial demonstrou, por exemplo, área plantada
ou destinada à colheita de cana-de-açúcar em Guariba-SP de 17.000 hectares
no ano de 2003.

## Produto agrícola selecionado

A cultura analisada é a cana-de-açúcar, identificada na tabela 5457
da PAM/SIDRA pelo código 40106, dentro da classificação 782.

Uma tentativa inicial utilizou o código 401, que retornou valores
indisponíveis e não apresentou a identificação do produto. Após validação
na interface oficial e em consultas específicas à API, foi confirmado que
o código 40106 corresponde a Cana-de-açúcar.

Como validação, a consulta para Guariba-SP no ano de 2003 retornou
17.000 hectares para a variável Área plantada ou destinada à colheita.

## Tratamento e validação da base agrícola

Os dados brutos da Produção Agrícola Municipal (PAM), obtidos por meio da
API SIDRA/IBGE, foram transformados de formato longo para formato tabular
anual. A unidade de análise adotada foi Município + Ano.

Foram selecionadas as variáveis relacionadas à cultura de cana-de-açúcar:
área plantada ou destinada à colheita, área colhida, quantidade produzida,
rendimento médio da produção e valor da produção.

A produtividade agrícola foi calculada pela divisão entre a quantidade
produzida, em toneladas, e a área colhida, em hectares. Paralelamente,
o rendimento médio divulgado pelo IBGE, originalmente em quilogramas por
hectare, foi convertido para toneladas por hectare.

A validação comparou a produtividade calculada com o rendimento médio
oficial convertido. A diferença máxima observada foi de aproximadamente
0,000507 tonelada por hectare, atribuída a arredondamentos dos dados
divulgados. Dessa forma, a consistência da base analítica foi considerada
adequada para as etapas seguintes de análise exploratória e integração
com variáveis climáticas.

## Análise exploratória dos dados agrícolas

A análise exploratória utilizou dados anuais da Produção Agrícola Municipal
(PAM/IBGE) para a cultura de cana-de-açúcar no município de Guariba-SP,
considerando o período de 2000 a 2024.

No período analisado, a área plantada apresentou média de 19.098,80 hectares,
com variação entre 17.000 e 21.215 hectares. A área colhida apresentou
comportamento semelhante, indicando baixa diferença entre a área destinada à
colheita e a área efetivamente colhida.

A produção média anual foi de aproximadamente 1,63 milhão de toneladas. O
maior volume foi registrado em 2020, com 1.908.000 toneladas, enquanto o menor
foi observado em 2000, com 1.275.000 toneladas.

A produtividade média foi de 86,11 toneladas por hectare. O maior rendimento
foi observado em 2005, com 92,41 toneladas por hectare, e o menor em 2000,
com 75 toneladas por hectare. Entre os indicadores analisados, a produtividade
apresentou menor variabilidade relativa, com coeficiente de variação de 4,67%.

O valor da produção apresentou maior oscilação ao longo dos anos. Entretanto,
como os valores monetários são nominais, a comparação histórica deve considerar
a ausência de correção monetária pela inflação.

Os resultados obtidos serão posteriormente integrados a dados climáticos para
avaliar associações entre precipitação, temperatura, umidade e produtividade
da cana-de-açúcar.

## Dados climáticos

Os dados climáticos serão obtidos da plataforma NASA POWER, utilizando
informações diárias para um ponto geográfico representativo do município
de Guariba-SP.

Serão utilizadas variáveis de precipitação, temperatura média, temperatura
máxima, temperatura mínima, umidade relativa e radiação solar. Os dados
diários serão agregados anualmente para compatibilização com a granularidade
da Produção Agrícola Municipal, cuja unidade de análise é Município + Ano.

A utilização de coordenadas representativas do município constitui uma
simplificação metodológica. Portanto, os indicadores climáticos devem ser
interpretados como estimativas das condições médias locais, e não como
medições específicas de cada propriedade agrícola.

## Integração dos dados agrícolas e climáticos

Os dados agrícolas anuais da PAM/IBGE e os dados climáticos anuais obtidos
da NASA POWER foram integrados utilizando a chave temporal do ano de
referência. A base consolidada contém informações para o município de
Guariba-SP no período de 2000 a 2024.

Foram incluídas variáveis de precipitação anual, temperatura média, máxima
e mínima anuais, umidade relativa média e radiação solar média. Também foram
calculados indicadores derivados, como precipitação média diária e amplitude
térmica média.

A análise inicial utiliza correlação de Pearson e gráficos de dispersão para
investigar associações entre condições climáticas e indicadores agrícolas.
Os resultados devem ser interpretados como exploratórios, pois a série possui
25 observações anuais e os dados climáticos correspondem a anos-calendário,
enquanto o ciclo produtivo da cana pode atravessar mais de um período anual.

## Modelagem Preditiva da Produtividade

A etapa de modelagem preditiva teve como objetivo construir um modelo de
Machine Learning capaz de prever a produtividade da cana-de-açúcar em
Guariba-SP. A variável-alvo definida foi a `produtividade_ton_ha_ibge`.

Para a construção do modelo, foi utilizado o algoritmo Random Forest Regressor,
conhecido por sua robustez e capacidade de lidar com relações complexas entre
as variáveis. O conjunto de dados, enriquecido com atributos climáticos
sazonais e defasados, foi dividido em conjuntos de treino e teste.

O modelo foi avaliado utilizando métricas como RMSE (Root Mean Squared Error),
MAE (Mean Absolute Error) e R² (coeficiente de determinação). Adicionalmente,
foi realizada uma análise da importância dos atributos para identificar quais
variáveis contribuem mais significativamente para as previsões do modelo.

## Expansão e Refinamento do Dataset

Para superar as limitações de dados observadas na modelagem inicial (Etapa 10),
o dataset foi expandido para incluir múltiplos municípios produtores de
cana-de-açúcar no estado de São Paulo.

O processo de expansão envolveu:
1.  **Extração de dados agrícolas (PAM/IBGE):** Coleta de dados de produção,
    área e valor para 579 municípios paulistas no período de 2000 a 2024.
2.  **Obtenção de coordenadas geográficas:** Utilização de um dataset local
    (IBGE) para obter latitude e longitude de cada município.
3.  **Extração de dados climáticos (NASA POWER):** Coleta de dados diários
    de precipitação, temperatura, umidade e radiação solar para cada um dos
    579 municípios, agregados anualmente.
4.  **Integração das bases:** Consolidação dos dados agrícolas e climáticos
    em uma única base anual por município.
5.  **Engenharia de atributos climáticos:** Criação de atributos sazonais
    (por períodos de brotação, crescimento e maturação) e defasados
    (do ano anterior) para todos os municípios.

Adicionalmente, o modelo foi refinado removendo-se variáveis que poderiam
causar vazamento de dados (`producao_ton` e `valor_producao_mil_reais`),
garantindo que as previsões se baseiem apenas em fatores antecedentes.