from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import joblib
import shap


# ============================================================
# CONFIGURAÇÕES
# ============================================================

ARQUIVO_ENTRADA = Path(
    "data/processed/base_cana_clima_atributos_multi_anual.parquet"
)

PASTA_MODELOS = Path("models")
PASTA_GRAFICOS = Path("outputs/figures")
PASTA_TABELAS = Path("outputs/tables")

ARQUIVO_PREVISOES = PASTA_TABELAS / "previsoes_modelo_produtividade_multi.csv"
ARQUIVO_IMPORTANCIA_ATRIBUTOS = (
    PASTA_TABELAS / "importancia_atributos_modelo_multi.csv"
)
ARQUIVO_MODELO_SALVO = PASTA_MODELOS / "random_forest_produtividade_multi.joblib"

TARGET = "produtividade_ton_ha_ibge"

# Variáveis a serem usadas como features (preditoras)
FEATURES = [
    "area_plantada_ha",
    "area_colhida_ha",
    # "producao_ton",  # Removido para evitar data leakage
    # "valor_producao_mil_reais", # Removido para evitar data leakage
    "precipitacao_anual_mm",
    "temperatura_media_anual_c",
    "temperatura_maxima_anual_c",
    "temperatura_minima_anual_c",
    "umidade_relativa_media_pct",
    "radiacao_solar_media_mj_m2_dia",
    "precipitacao_media_diaria_mm",
    "amplitude_termica_media_c",
    # Atributos sazonais
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
    # Atributos defasados
    "produtividade_ton_ha_ibge_lag1",
    "precipitacao_anual_mm_lag1",
]

# Cores para os gráficos
COR_PRINCIPAL = "#2E8B57"  # Verde mar
COR_PREVISAO = "#FF6347"   # Tomate


# ============================================================
# FUNÇÕES
# ============================================================

def carregar_dados() -> pd.DataFrame:
    """Carrega a base de dados com atributos."""
    if not ARQUIVO_ENTRADA.exists():
        raise FileNotFoundError(
            f"Arquivo de entrada não encontrado:\n{ARQUIVO_ENTRADA.resolve()}"
        )
    return pd.read_parquet(ARQUIVO_ENTRADA)


def preparar_dados(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Prepara os dados para o treinamento do modelo."""

    # Remover linhas com NaN na variável alvo ou nas features
    df_limpo = df.dropna(subset=[TARGET] + FEATURES).copy()

    X = df_limpo[FEATURES]
    y = df_limpo[TARGET]

    # Dividir em treino e teste (usando stratify para manter proporção se necessário,
    # mas para regressão simples, um split aleatório é comum)
    # Para séries temporais, seria ideal um split baseado no tempo, mas aqui
    # estamos tratando cada município-ano como uma amostra independente.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\nDados de treino: {len(X_train)} amostras")
    print(f"Dados de teste: {len(X_test)} amostras")

    return X_train, X_test, y_train, y_test, df_limpo


def otimizar_hiperparametros(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    """
    Otimiza os hiperparâmetros do RandomForestRegressor usando RandomizedSearchCV.
    """
    print("\nIniciando otimização de hiperparâmetros com RandomizedSearchCV...")

    # Definir o espaço de busca para os hiperparâmetros
    param_dist = {
        "n_estimators": [100, 200, 300, 400, 500],  # Número de árvores na floresta
        "max_features": ["sqrt", "log2", None],  # Número de features a considerar em cada split (auto foi removido em versões recentes)
        "max_depth": [10, 20, 30, 40, 50, None],  # Profundidade máxima da árvore
        "min_samples_split": [2, 5, 10],  # Número mínimo de amostras para dividir um nó
        "min_samples_leaf": [1, 2, 4],  # Número mínimo de amostras em um nó folha
        "bootstrap": [True, False],  # Amostragem com reposição
    }

    # Criar o modelo base
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)

    # Configurar o RandomizedSearchCV
    # n_iter: número de combinações de parâmetros a serem testadas
    # cv: número de folds para validação cruzada
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=20,  # Testar 20 combinações aleatórias
        cv=3,       # Validação cruzada com 3 folds
        verbose=2,  # Exibir detalhes do processo
        random_state=42,
        n_jobs=-1,  # Usar todos os cores disponíveis
        scoring="r2", # Otimizar para R²
    )

    # Executar a busca
    random_search.fit(X_train, y_train)

    print("\nOtimização concluída.")
    print(f"Melhores hiperparâmetros encontrados: {random_search.best_params_}")
    print(f"Melhor R² na validação cruzada: {random_search.best_score_:.2f}")

    return random_search.best_estimator_


def avaliar_modelo(
    model: RandomForestRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Avalia o desempenho do modelo no conjunto de teste."""

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = mse**0.5  # Calcular RMSE a partir do MSE
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n--- Métricas de Avaliação ---")
    print(f"RMSE (Root Mean Squared Error): {rmse:.2f}")
    print(f"MAE (Mean Absolute Error): {mae:.2f}")
    print(f"R² (Coeficiente de Determinação): {r2:.2f}")

    return {"rmse": rmse, "mae": mae, "r2": r2, "y_pred": y_pred}


def analisar_importancia_atributos(
    model: RandomForestRegressor, features: list
) -> pd.DataFrame:
    """Calcula e exibe a importância dos atributos para o modelo."""

    importances = model.feature_importances_
    feature_importances = pd.DataFrame(
        {"feature": features, "importance": importances}
    )
    feature_importances = feature_importances.sort_values(
        "importance", ascending=False
    ).reset_index(drop=True)

    print("\n--- Importância dos Atributos ---")
    print(feature_importances.head(10).to_string())

    return feature_importances


def plotar_previsoes(
    df_limpo: pd.DataFrame,
    model: RandomForestRegressor,
    X: pd.DataFrame,
    y: pd.Series,
) -> None:
    """
    Plota as previsões do modelo vs. valores reais para uma amostra de municípios.
    """
    df_plot = df_limpo.copy()
    df_plot["previsao"] = model.predict(X)

    # Selecionar alguns municípios para plotar (ex: os 6 primeiros)
    municipios_amostra = df_plot["codigo_ibge"].unique()[:6]

    plt.figure(figsize=(15, 15)) # Aumentar o tamanho da figura para 3x2 subplots
    for i, cod_ibge in enumerate(municipios_amostra):
        df_municipio = df_plot[df_plot["codigo_ibge"] == cod_ibge].sort_values("ano")

        plt.subplot(3, 2, i + 1) # Grid de 3 linhas e 2 colunas
        plt.plot(
            df_municipio["ano"],
            df_municipio[TARGET],
            marker="o",
            label="Real",
            color=COR_PRINCIPAL,
            linewidth=2,
        )
        plt.plot(
            df_municipio["ano"],
            df_municipio["previsao"],
            marker="x",
            label="Previsão",
            color=COR_PREVISAO,
            linestyle="--",
            linewidth=2,
        )
        plt.title(f"Produtividade em {df_municipio['municipio'].iloc[0]}")
        plt.xlabel("Ano")
        plt.ylabel("Produtividade (t/ha)")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    PASTA_GRAFICOS.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        PASTA_GRAFICOS / "12_produtividade_real_vs_previsao_multi.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print(f"Gráfico salvo: {PASTA_GRAFICOS / '12_produtividade_real_vs_previsao_multi.png'}")


def analisar_shap(model: RandomForestRegressor, X: pd.DataFrame, features: list) -> None:
    """
    Calcula e plota os SHAP values para interpretar o modelo.
    """
    print("\nCalculando SHAP values para interpretabilidade do modelo...")

    # Para TreeExplainer, shap_values pode ser uma lista de arrays se for multi-output
    # ou um único array para single-output. Para RandomForestRegressor, é um único array.
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Plotar o summary plot (importância global das features)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title("Importância Global das Features (SHAP)")
    plt.tight_layout()
    PASTA_GRAFICOS.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        PASTA_GRAFICOS / "12_shap_summary_bar.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print(f"Gráfico SHAP Summary (Bar) salvo: {PASTA_GRAFICOS / '12_shap_summary_bar.png'}")

    # Plotar o summary plot (impacto e direção)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False)
    plt.title("Impacto e Direção das Features (SHAP)")
    plt.tight_layout()
    PASTA_GRAFICOS.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        PASTA_GRAFICOS / "12_shap_summary_dot.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print(f"Gráfico SHAP Summary (Dot) salvo: {PASTA_GRAFICOS / '12_shap_summary_dot.png'}")

    print("Análise SHAP concluída.")


def salvar_modelo(model: RandomForestRegressor, filename: Path) -> None:
    """Salva o modelo treinado em disco."""
    PASTA_MODELOS.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filename)
    print(f"\nModelo salvo em: {filename.resolve()}")


def main():
    print("=" * 60)
    print("ETAPA 12.2 — OTIMIZAÇÃO E INTERPRETABILIDADE DO MODELO")
    print("=" * 60)

    df = carregar_dados()
    X_train, X_test, y_train, y_test, df_limpo = preparar_dados(df)

    # Otimizar hiperparâmetros e obter o melhor modelo
    model = otimizar_hiperparametros(X_train, y_train)

    # Avaliar o modelo otimizado no conjunto de teste
    metricas = avaliar_modelo(model, X_test, y_test)
    importancia_atributos = analisar_importancia_atributos(model, FEATURES)

    # Salvar previsões e importância dos atributos
    df_previsoes = df_limpo[["codigo_ibge", "municipio", "ano", TARGET]].copy()
    df_previsoes["previsao"] = model.predict(df_limpo[FEATURES])
    df_previsoes.to_csv(ARQUIVO_PREVISOES, index=False, encoding="utf-8-sig")
    importancia_atributos.to_csv(
        ARQUIVO_IMPORTANCIA_ATRIBUTOS, index=False, encoding="utf-8-sig"
    )

    plotar_previsoes(df_limpo, model, df_limpo[FEATURES], df_limpo[TARGET])
    analisar_shap(model, X_test, FEATURES) # Usar X_test para a análise SHAP

    salvar_modelo(model, ARQUIVO_MODELO_SALVO)

    print("\nOtimização e Interpretabilidade do Modelo concluídas com sucesso.")


if __name__ == "__main__":
    main()