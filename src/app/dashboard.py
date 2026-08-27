import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ============================================================
# CONFIGURAÇÕES E CARREGAMENTO DE DADOS/MODELO
# ============================================================

# Caminhos dos arquivos
ARQUIVO_BASE_ATRIBUTOS = Path("data/processed/base_cana_clima_atributos_multi_anual.parquet")
ARQUIVO_MODELO = Path("models/random_forest_produtividade_multi.joblib")
ARQUIVO_IMPORTANCIA_ATRIBUTOS = Path("outputs/tables/importancia_atributos_modelo_multi.csv")

# Variável alvo e features (devem ser as mesmas usadas no treinamento)
TARGET = "produtividade_ton_ha_ibge"
FEATURES = [
    "area_plantada_ha",
    "area_colhida_ha",
    "precipitacao_anual_mm",
    "temperatura_media_anual_c",
    "temperatura_maxima_anual_c",
    "temperatura_minima_anual_c",
    "umidade_relativa_media_pct",
    "radiacao_solar_media_mj_m2_dia",
    "precipitacao_media_diaria_mm",
    "amplitude_termica_media_c",
    "precipitacao_acumulada_mm_p1_brotacao_crescimento_inicial",
    "temperatura_media_c_p1_brotacao_crescimento_inicial",
    "dias_chuvosos_p1_brotacao_crescimento_inicial",
    "dias_secos_p1_brotacao_crescimento_inicial",
    "precipitacao_acumulada_mm_p2_crescimento_acumulo_sacarose",
    "temperatura_media_c_p2_crescimento_acumulo_sacarose",
    "dias_chuvosos_p2_crescimento_acumulo_sacarose",
    "dias_secos_p2_crescimento_acumulo_sacarose",
    "precipitacao_acumulada_mm_p3_maturacao_colheita",
    "temperatura_media_c_p3_maturacao_colheita",
    "dias_chuvosos_p3_maturacao_colheita",
    "dias_secos_p3_maturacao_colheita",
    "produtividade_ton_ha_ibge_lag1",
    "precipitacao_anual_mm_lag1",
]

@st.cache_data
def carregar_dados():
    """Carrega a base de dados e o modelo treinado."""
    if not ARQUIVO_BASE_ATRIBUTOS.exists():
        st.error(f"Arquivo de dados não encontrado: {ARQUIVO_BASE_ATRIBUTOS.resolve()}")
        st.stop()
    if not ARQUIVO_MODELO.exists():
        st.error(f"Modelo não encontrado: {ARQUIVO_MODELO.resolve()}")
        st.stop()

    df = pd.read_parquet(ARQUIVO_BASE_ATRIBUTOS)
    model = joblib.load(ARQUIVO_MODELO)

    df_importancia = pd.DataFrame()
    if ARQUIVO_IMPORTANCIA_ATRIBUTOS.exists():
        df_importancia = pd.read_csv(ARQUIVO_IMPORTANCIA_ATRIBUTOS)
    else:
        st.warning(f"Arquivo de importância de atributos não encontrado: {ARQUIVO_IMPORTANCIA_ATRIBUTOS.resolve()}")

    return df, model, df_importancia

df_completo, model, df_importancia_global = carregar_dados()

# Preencher NaNs para previsão (se houver, para evitar erros no predict)
# Uma estratégia simples é preencher com a média das colunas para os valores ausentes
# antes de fazer a previsão. Isso é feito apenas para a parte de previsão do dashboard.
df_previsao = df_completo.dropna(subset=FEATURES + [TARGET]).copy()
df_previsao["previsao"] = model.predict(df_previsao[FEATURES])

# Obter lista de municípios para seleção
municipios = sorted(df_completo["municipio"].unique())

# ============================================================
# FUNÇÕES DE VISUALIZAÇÃO
# ============================================================

def plot_produtividade_historica(df_municipio):
    fig = px.line(
        df_municipio,
        x="ano",
        y=TARGET,
        title="Produtividade Histórica (t/ha)",
        labels={"ano": "Ano", TARGET: "Produtividade (toneladas por hectare)"},
        markers=True,
        color_discrete_sequence=[px.colors.qualitative.Plotly[0]]
    )
    fig.update_layout(hovermode="x unified")
    return fig

def plot_clima_historico(df_municipio, variavel_climatica):
    fig = px.line(
        df_municipio,
        x="ano",
        y=variavel_climatica,
        title=f"{variavel_climatica.replace('_', ' ').title()} Histórica",
        labels={"ano": "Ano", variavel_climatica: variavel_climatica.replace('_', ' ').title()},
        markers=True,
        color_discrete_sequence=[px.colors.qualitative.Plotly[1]]
    )
    fig.update_layout(hovermode="x unified")
    return fig

def plot_previsao_vs_real(df_municipio):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_municipio["ano"],
        y=df_municipio[TARGET],
        mode='lines+markers',
        name='Produtividade Real',
        line=dict(color=px.colors.qualitative.Plotly[0])
    ))
    fig.add_trace(go.Scatter(
        x=df_municipio["ano"],
        y=df_municipio["previsao"],
        mode='lines+markers',
        name='Previsão do Modelo',
        line=dict(color=px.colors.qualitative.Plotly[1], dash='dash')
    ))
    fig.update_layout(
        title="Produtividade Real vs. Previsão do Modelo",
        xaxis_title="Ano",
        yaxis_title="Produtividade (toneladas por hectare)",
        hovermode="x unified"
    )
    return fig

def plot_shap_summary(model, X_data):
    """Gera o plot SHAP summary (dot) para um subconjunto de dados."""
    explainer = shap.TreeExplainer(model)
    # Usar um subconjunto de X_data para evitar lentidão em datasets muito grandes
    X_sample = X_data.sample(min(1000, len(X_data)), random_state=42)
    shap_values = explainer.shap_values(X_sample)

    # Plotar o summary plot (impacto e direção)
    # O SHAP cria sua própria figura e eixos, então não passamos 'ax'
    fig = plt.figure(figsize=(10, 6)) # Criar uma figura explícita para o SHAP
    shap.summary_plot(shap_values, X_sample, show=False, plot_type="dot")
    plt.title("Impacto e Direção das Features (SHAP)")
    plt.tight_layout()
    st.pyplot(fig) # Exibir o plot no Streamlit
    plt.close(fig) # Fechar a figura para liberar memória

# ============================================================
# LAYOUT DO DASHBOARD
# ============================================================

st.set_page_config(layout="wide", page_title="Produtividade Cana-de-Açúcar SP")

st.title("🌱 Previsão de Produtividade de Cana-de-Açúcar em SP")
st.markdown(
    """
    Este dashboard interativo oferece uma visão aprofundada sobre a produtividade da cana-de-açúcar
    no estado de São Paulo. Explore dados históricos, entenda as previsões do nosso modelo de Machine Learning
    e descubra quais fatores mais influenciam a produção.
    """
)
st.markdown("---")

# Sidebar para seleção de município
st.sidebar.header("⚙️ Configurações")
municipio_selecionado = st.sidebar.selectbox(
    "Selecione o Município para Análise:",
    municipios,
    index=municipios.index("Guariba (SP)") if "Guariba (SP)" in municipios else 0,
    help="Escolha um município para visualizar seus dados históricos e previsões."
)

# Filtrar dados para o município selecionado
df_municipio = df_previsao[df_previsao["municipio"] == municipio_selecionado].sort_values("ano")

# ============================================================
# SEÇÃO: SOBRE A CULTURA DA CANA-DE-AÇÚCAR
# ============================================================
st.header("🌿 Sobre a Cultura da Cana-de-Açúcar")
st.markdown(
    """
    A cana-de-açúcar é uma cultura de grande importância econômica para o Brasil,
    utilizada principalmente na produção de açúcar e etanol. Seu ciclo de vida
    é influenciado por diversos fatores, com destaque para as condições climáticas
    e o manejo agrícola.

    **Fases de Desenvolvimento:**
    *   **Brotação e Crescimento Inicial (Setembro a Dezembro):** Período de germinação
        e estabelecimento da planta, sensível à disponibilidade de água e temperatura.
    *   **Crescimento e Acúmulo de Sacarose (Janeiro a Abril):** Fase de maior
        desenvolvimento vegetativo e início da formação de açúcar, demandando
        muita água e radiação solar.
    *   **Maturação e Colheita (Maio a Agosto):** A planta direciona energia para
        o acúmulo de sacarose, idealmente com menor umidade e temperaturas mais amenas.

    A produtividade é medida em **toneladas por hectare (t/ha)** e reflete a
    eficiência da lavoura.
    """
)
st.markdown("---")

# ============================================================
# SEÇÃO 1: VISÃO GERAL E MÉTRICAS
# ============================================================
st.header(f"📊 Visão Geral: {municipio_selecionado}")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Produtividade Média Histórica (t/ha)",
        value=f"{df_municipio[TARGET].mean():.2f}",
        help="Média da produtividade de cana-de-açúcar em toneladas por hectare para este município ao longo dos anos registrados."
    )
with col2:
    st.metric(
        label="Precipitação Média Anual (mm)",
        value=f"{df_municipio['precipitacao_anual_mm'].mean():.2f}",
        help="Média da precipitação total anual em milímetros para este município."
    )
with col3:
    st.metric(
        label="Temperatura Média Anual (°C)",
        value=f"{df_municipio['temperatura_media_anual_c'].mean():.2f}",
        help="Média da temperatura anual em graus Celsius para este município."
    )

st.markdown("---")

# ============================================================
# SEÇÃO 2: ANÁLISE HISTÓRICA E PREVISÕES
# ============================================================
st.header("📈 Análise Histórica e Previsões")

st.subheader("Produtividade Real vs. Previsão do Modelo")
st.write(
    "Este gráfico compara a produtividade real da cana-de-açúcar (em toneladas por hectare) "
    "com as previsões geradas pelo nosso modelo de Machine Learning para o município selecionado. "
    "A linha verde representa os dados históricos observados, enquanto a linha tracejada vermelha "
    "mostra o que o modelo previu para cada ano."
)
st.plotly_chart(plot_previsao_vs_real(df_municipio), use_container_width=True)

st.subheader("Fatores Climáticos Históricos")
st.write(
    "As condições climáticas são cruciais para o desenvolvimento da cana-de-açúcar. "
    "Explore a evolução de diferentes variáveis climáticas ao longo dos anos para "
    "entender como elas podem ter influenciado a produtividade."
)
# Seleção de variável climática para plotar
variaveis_climaticas = [
    "precipitacao_anual_mm",
    "temperatura_media_anual_c",
    "umidade_relativa_media_pct",
    "radiacao_solar_media_mj_m2_dia",
    "amplitude_termica_media_c"
]
variavel_climatica_selecionada = st.selectbox(
    "Selecione uma variável climática para visualizar:",
    variaveis_climaticas,
    format_func=lambda x: x.replace('_', ' ').title(),
    help="Escolha uma variável climática para ver sua tendência histórica."
)
st.plotly_chart(plot_clima_historico(df_municipio, variavel_climatica_selecionada), use_container_width=True)

st.markdown("---")

# ============================================================
# SEÇÃO 3: INTERPRETABILIDADE DO MODELO
# ============================================================
st.header("🧠 Interpretabilidade do Modelo")

st.subheader("Importância Global das Features (Feature Importance)")
st.write(
    "Este ranking mostra quais fatores (features) o modelo considera mais importantes "
    "para prever a produtividade da cana-de-açúcar, em média, para todos os municípios. "
    "Quanto maior a porcentagem, maior a influência do fator na previsão."
)
if not df_importancia_global.empty:
    fig_feat_imp = px.bar(
        df_importancia_global.head(10),
        x="importance",
        y="feature",
        orientation="h",
        title="Top 10 Fatores Mais Importantes (Global)",
        labels={"importance": "Importância Relativa", "feature": "Fator"},
    )
    fig_feat_imp.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_feat_imp, use_container_width=True)
else:
    st.warning("Não foi possível carregar a importância dos atributos. Verifique se o arquivo 'importancia_atributos_modelo_multi.csv' existe.")

st.subheader("Impacto Individual dos Fatores (SHAP Values)")
st.write(
    "A análise SHAP nos ajuda a entender como cada fator contribui para uma previsão específica do modelo. "
    "Cada ponto no gráfico representa uma observação (um ano para um município). "
    "A posição horizontal indica o impacto do fator na previsão: para a direita, aumenta a produtividade; "
    "para a esquerda, diminui. A cor do ponto (vermelho para valores altos do fator, azul para baixos) "
    "mostra se um valor alto ou baixo do fator teve esse impacto."
)

# Para o SHAP, é crucial usar os dados de teste ou um subconjunto representativo
# que o modelo realmente viu durante o treinamento/avaliação.
# Vamos usar um subconjunto do df_previsao para o SHAP, garantindo que as colunas sejam as FEATURES.
X_shap = df_previsao[FEATURES]
plot_shap_summary(model, X_shap)

st.markdown("---")

# ============================================================
# SEÇÃO 4: PREVISÃO PARA UM ANO FUTURO
# ============================================================
st.header("🔮 Previsão de Produtividade para um Ano Futuro")
st.write(
    "Utilize esta seção para simular a produtividade da cana-de-açúcar em um ano futuro, "
    "com base em suas estimativas para os principais fatores. "
    "O modelo usará o último ano disponível do município selecionado como base para preencher "
    "os valores que você não alterar."
)

# Obter o último ano disponível para o município selecionado
ultimo_ano_df = df_municipio.sort_values("ano", ascending=False).iloc[0]
ano_futuro = st.number_input(
    "Ano da Previsão:",
    min_value=int(df_completo["ano"].max()) + 1,
    max_value=2050,
    value=int(df_completo["ano"].max()) + 1,
    step=1,
    help="Insira o ano para o qual você deseja fazer a previsão."
)

st.subheader("Fatores para a Previsão")

# Criar um dicionário para o ponto de dados futuro, preenchendo com o último ano disponível
data_futura = ultimo_ano_df[FEATURES].to_dict()
data_futura["ano"] = ano_futuro # Atualizar o ano

# Inputs para os fatores mais importantes (baseado na feature importance global)
# Isso torna a interface mais amigável, focando no que realmente importa
st.markdown("**Fatores Agrícolas:**")
col_agri1, col_agri2 = st.columns(2)
with col_agri1:
    data_futura["area_plantada_ha"] = st.number_input(
        "Área Plantada (hectares):",
        value=float(data_futura.get("area_plantada_ha", 0)),
        min_value=0.0,
        help="Área total em hectares dedicada ao plantio de cana-de-açúcar."
    )
with col_agri2:
    data_futura["area_colhida_ha"] = st.number_input(
        "Área Colhida (hectares):",
        value=float(data_futura.get("area_colhida_ha", 0)),
        min_value=0.0,
        help="Área total em hectares de cana-de-açúcar que foi colhida."
    )

st.markdown("**Fatores Climáticos Anuais:**")
col_clim_fut1, col_clim_fut2, col_clim_fut3 = st.columns(3)
with col_clim_fut1:
    data_futura["precipitacao_anual_mm"] = st.number_input(
        "Precipitação Anual (mm):",
        value=float(data_futura.get("precipitacao_anual_mm", 0)),
        min_value=0.0,
        help="Total de chuva esperada para o ano em milímetros."
    )
with col_clim_fut2:
    data_futura["temperatura_media_anual_c"] = st.number_input(
        "Temperatura Média Anual (°C):",
        value=float(data_futura.get("temperatura_media_anual_c", 0)),
        min_value=0.0,
        help="Temperatura média esperada para o ano em graus Celsius."
    )
with col_clim_fut3:
    data_futura["radiacao_solar_media_mj_m2_dia"] = st.number_input(
        "Radiação Solar Média (MJ/m²/dia):",
        value=float(data_futura.get("radiacao_solar_media_mj_m2_dia", 0)),
        min_value=0.0,
        help="Média da radiação solar diária em Megajoules por metro quadrado."
    )

st.markdown("**Fatores Climáticos Sazonais (Período de Crescimento e Acúmulo de Sacarose - Jan a Abr):**")
data_futura["precipitacao_acumulada_mm_p2_crescimento_acumulo_sacarose"] = st.number_input(
    "Precipitação Acumulada P2 (mm):",
    value=float(data_futura.get("precipitacao_acumulada_mm_p2_crescimento_acumulo_sacarose", 0)),
    min_value=0.0,
    help="Total de chuva esperada de Janeiro a Abril, crucial para o crescimento da cana."
)

st.markdown("**Produtividade do Ano Anterior:**")
data_futura["produtividade_ton_ha_ibge_lag1"] = st.number_input(
    "Produtividade do Ano Anterior (t/ha):",
    value=float(data_futura.get("produtividade_ton_ha_ibge_lag1", 0)),
    min_value=0.0,
    help="Produtividade observada no ano imediatamente anterior ao ano da previsão. Este é o fator mais influente."
)

# Botão para fazer a previsão
if st.button("Gerar Previsão Futura"):
    # Criar DataFrame para a previsão
    df_futuro = pd.DataFrame([data_futura])

    # Garantir que todas as colunas de FEATURES estejam presentes e na ordem correta
    # e preencher quaisquer NaNs restantes com a média do dataset de treino
    for feature in FEATURES:
        if feature not in df_futuro.columns:
            df_futuro[feature] = df_completo[feature].mean() # Preenche com a média global

    df_futuro = df_futuro[FEATURES]
    df_futuro = df_futuro.fillna(df_completo[FEATURES].mean()) # Preenche NaNs remanescentes com a média

    previsao_futura = model.predict(df_futuro)[0]

    st.success(f"**Previsão de Produtividade para {municipio_selecionado} em {ano_futuro}:**")
    st.markdown(f"## **{previsao_futura:.2f} toneladas por hectare**")
    st.write(
        "Esta previsão é uma estimativa baseada nos fatores informados. "
        "Lembre-se que o modelo considera as relações aprendidas com dados históricos "
        "e a precisão pode variar com a incerteza dos dados futuros."
    )

st.markdown("---")
st.caption("Desenvolvido por Gabriel Rocha para UNIVESP - Projeto Integrador")
