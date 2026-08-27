from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURAÇÕES
# ============================================================

ARQUIVO_ENTRADA = Path(
    "data/processed/base_cana_clima_atributos_anual.parquet"
)

PASTA_MODELOS = Path("models")
PASTA_GRAFICOS = Path("outputs/figures")
PASTA_TABELAS = Path("outputs/tables")

ARQUIVO_PREVISOES = PASTA_TABELAS / "previsoes_modelo_produtividade.csv"
ARQUIVO_IMPORTANCIA_ATRIBUTOS = (
    PASTA_TABELAS / "importancia_atributos_modelo.csv"
)

TARGET = "produtividade_ton_ha_ibge"

# Variáveis a serem usadas como features (preditoras)
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

# Parâmetros do modelo Random Forest (ajustados para um dataset pequeno)
MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "min_samples_leaf": 2,
    "random_state": 42,
}

sns.set_theme(style="whitegrid", context="notebook")
COR_PRINCIPAL = "#2E7D32"
COR_PREVISAO = "#1565C0"


# ============================================================
# FUNÇÕES
# ============================================================

def carregar_dados() -> pd.DataFrame:
    """Carrega a base com atributos climáticos e agrícolas."""

    if not ARQUIVO_ENTRADA.exists():
        raise FileNotFoundError(
            f"Base de dados não encontrada:\n{ARQUIVO_ENTRADA.resolve()}"
        )

    df = pd.read_parquet(ARQUIVO_ENTRADA)
    return df


def preparar_dados(df: pd.DataFrame) -> tuple:
    """
    Prepara os dados para o treinamento do modelo,
    removendo NaNs e dividindo em treino/teste.
    """

    # Remove linhas com NaN, que são principalmente do primeiro ano
    # devido às variáveis defasadas.
    df_limpo = df.dropna(subset=FEATURES + [TARGET]).copy()

    X = df_limpo[FEATURES]
    y = df_limpo[TARGET]

    # Para um dataset pequeno (25 linhas), vamos usar uma divisão simples
    # O ideal seria validação cruzada, mas para a primeira versão,
    # uma divisão de 80/20 é um bom começo.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\nDados de treino: {len(X_train)} amostras")
    print(f"Dados de teste: {len(X_test)} amostras")

    return X_train, X_test, y_train, y_test, df_limpo


def treinar_modelo(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    """Treina o modelo Random Forest Regressor."""

    print("\nTreinando o modelo Random Forest...")
    model = RandomForestRegressor(**MODEL_PARAMS)
    model.fit(X_train, y_train)
    print("Modelo treinado com sucesso.")
    return model


def avaliar_modelo(
    model: RandomForestRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Avalia o desempenho do modelo no conjunto de teste."""

    y_pred = model.predict(X_test)

    # Calcula o MSE e depois a raiz quadrada para obter o RMSE
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse**0.5  # Ou np.sqrt(mse) se numpy for importado

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
    """Plota as previsões do modelo vs. valores reais."""

    df_plot = df_limpo.copy()
    df_plot["previsao"] = model.predict(X)

    plt.figure(figsize=(12, 7))
    plt.plot(
        df_plot["ano"],
        df_plot[TARGET],
        marker="o",
        label="Real",
        color=COR_PRINCIPAL,
        linewidth=2,
    )
    plt.plot(
        df_plot["ano"],
        df_plot["previsao"],
        marker="x",
        label="Previsão",
        color=COR_PREVISAO,
        linestyle="--",
        linewidth=2,
    )

    plt.title(
        "Produtividade Real vs. Previsão do Modelo (Guariba-SP, 2000-2024)"
    )
    plt.xlabel("Ano")
    plt.ylabel("Produtividade (t/ha)")
    plt.legend()
    plt.grid(True)

    PASTA_GRAFICOS.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        PASTA_GRAFICOS / "10_produtividade_real_vs_previsao.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print(f"Gráfico salvo: {PASTA_GRAFICOS / '10_produtividade_real_vs_previsao.png'}")


def main():
    print("=" * 60)
    print("ETAPA 10 — MODELAGEM PREDITIVA DA PRODUTIVIDADE")
    print("=" * 60)

    df = carregar_dados()
    X_train, X_test, y_train, y_test, df_limpo = preparar_dados(df)

    model = treinar_modelo(X_train, y_train)
    metricas = avaliar_modelo(model, X_test, y_test)
    importancia_atributos = analisar_importancia_atributos(model, FEATURES)

    # Salvar previsões e importância dos atributos
    df_previsoes = df_limpo[["ano", TARGET]].copy()
    df_previsoes["previsao"] = model.predict(df_limpo[FEATURES])
    df_previsoes.to_csv(ARQUIVO_PREVISOES, index=False, encoding="utf-8-sig")
    importancia_atributos.to_csv(
        ARQUIVO_IMPORTANCIA_ATRIBUTOS, index=False, encoding="utf-8-sig"
    )

    plotar_previsoes(df_limpo, model, df_limpo[FEATURES], df_limpo[TARGET])

    print("\nModelagem preditiva concluída com sucesso.")


if __name__ == "__main__":
    main()