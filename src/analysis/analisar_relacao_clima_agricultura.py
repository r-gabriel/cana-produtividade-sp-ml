from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# ============================================================
# CONFIGURAÇÕES
# ============================================================

ARQUIVO_ENTRADA = Path(
    "data/processed/base_cana_clima_guariba_anual.parquet"
)

PASTA_GRAFICOS = Path("outputs/figures")
PASTA_TABELAS = Path("outputs/tables")

ARQUIVO_CORRELACOES = (
    PASTA_TABELAS / "correlacoes_clima_produtividade_guariba.csv"
)

ARQUIVO_BASE_ANALISE = (
    PASTA_TABELAS / "base_analise_clima_produtividade_guariba.csv"
)

sns.set_theme(style="whitegrid", context="notebook")

COR_PRINCIPAL = "#2E7D32"
COR_REGRESSAO = "#C62828"

VARIAVEIS_CLIMATICAS = [
    "precipitacao_anual_mm",
    "temperatura_media_anual_c",
    "temperatura_maxima_anual_c",
    "temperatura_minima_anual_c",
    "umidade_relativa_media_pct",
    "radiacao_solar_media_mj_m2_dia",
    "amplitude_termica_media_c",
]

VARIAVEIS_AGRICOLAS = [
    "producao_ton",
    "produtividade_ton_ha_ibge",
    "area_plantada_ha",
    "area_colhida_ha",
]


# ============================================================
# FUNÇÕES
# ============================================================

def carregar_dados() -> pd.DataFrame:
    """Carrega a base anual integrada de agricultura e clima."""

    if not ARQUIVO_ENTRADA.exists():
        raise FileNotFoundError(
            f"Base integrada não encontrada:\n"
            f"{ARQUIVO_ENTRADA.resolve()}"
        )

    return pd.read_parquet(ARQUIVO_ENTRADA)


def salvar_grafico(nome: str) -> None:
    """Salva o gráfico atual em alta resolução."""

    PASTA_GRAFICOS.mkdir(parents=True, exist_ok=True)

    caminho = PASTA_GRAFICOS / nome

    plt.tight_layout()
    plt.savefig(caminho, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Gráfico salvo: {caminho}")


def gerar_tabela_correlacoes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula correlações de Pearson entre as variáveis climáticas
    e os indicadores agrícolas selecionados.
    """

    resultados = []

    for variavel_climatica in VARIAVEIS_CLIMATICAS:
        for variavel_agricola in VARIAVEIS_AGRICOLAS:
            correlacao = df[
                [variavel_climatica, variavel_agricola]
            ].corr().iloc[0, 1]

            resultados.append({
                "variavel_climatica": variavel_climatica,
                "variavel_agricola": variavel_agricola,
                "correlacao_pearson": correlacao,
                "correlacao_absoluta": abs(correlacao),
            })

    df_correlacoes = pd.DataFrame(resultados)

    return (
        df_correlacoes
        .sort_values(
            "correlacao_absoluta",
            ascending=False
        )
        .reset_index(drop=True)
    )


def grafico_matriz_correlacao(df: pd.DataFrame) -> None:
    """Gera heatmap das variáveis agrícolas e climáticas."""

    colunas = VARIAVEIS_AGRICOLAS + VARIAVEIS_CLIMATICAS

    matriz = df[colunas].corr()

    plt.figure(figsize=(14, 10))

    sns.heatmap(
        matriz,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        square=False,
    )

    plt.title(
        "Correlação entre Indicadores Agrícolas e Climáticos\n"
        "Guariba-SP, 2000–2024"
    )

    salvar_grafico("06_matriz_correlacao_clima_agricultura.png")


def grafico_dispersao(
    df: pd.DataFrame,
    x: str,
    y: str,
    titulo: str,
    nome_arquivo: str,
    rotulo_x: str,
    rotulo_y: str,
) -> None:
    """Gera gráfico de dispersão com regressão linear e identificação anual."""

    plt.figure(figsize=(10, 7))

    sns.regplot(
        data=df,
        x=x,
        y=y,
        scatter_kws={
            "s": 65,
            "color": COR_PRINCIPAL,
            "alpha": 0.85,
        },
        line_kws={
            "color": COR_REGRESSAO,
            "linewidth": 2,
        },
    )

    for _, linha in df.iterrows():
        plt.annotate(
            str(int(linha["ano"])),
            (linha[x], linha[y]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            alpha=0.75,
        )

    coeficiente = df[[x, y]].corr().iloc[0, 1]

    plt.title(f"{titulo}\nCorrelação de Pearson: {coeficiente:.2f}")
    plt.xlabel(rotulo_x)
    plt.ylabel(rotulo_y)

    salvar_grafico(nome_arquivo)


def main():
    print("=" * 60)
    print("ETAPA 8 — ANÁLISE CLIMA E PRODUTIVIDADE")
    print("=" * 60)

    PASTA_GRAFICOS.mkdir(parents=True, exist_ok=True)
    PASTA_TABELAS.mkdir(parents=True, exist_ok=True)

    df = carregar_dados()

    colunas_analise = [
        "ano",
        "producao_ton",
        "produtividade_ton_ha_ibge",
        "area_plantada_ha",
        "area_colhida_ha",
        *VARIAVEIS_CLIMATICAS,
    ]

    df_analise = df[colunas_analise].copy()

    df_analise.to_csv(
        ARQUIVO_BASE_ANALISE,
        index=False,
        encoding="utf-8-sig",
    )

    correlacoes = gerar_tabela_correlacoes(df_analise)

    correlacoes.to_csv(
        ARQUIVO_CORRELACOES,
        index=False,
        encoding="utf-8-sig",
    )

    grafico_matriz_correlacao(df_analise)

    grafico_dispersao(
        df=df_analise,
        x="precipitacao_anual_mm",
        y="produtividade_ton_ha_ibge",
        titulo="Precipitação Anual e Produtividade da Cana-de-Açúcar",
        nome_arquivo="07_precipitacao_vs_produtividade.png",
        rotulo_x="Precipitação anual (mm)",
        rotulo_y="Produtividade (t/ha)",
    )

    grafico_dispersao(
        df=df_analise,
        x="temperatura_media_anual_c",
        y="produtividade_ton_ha_ibge",
        titulo="Temperatura Média Anual e Produtividade da Cana-de-Açúcar",
        nome_arquivo="08_temperatura_vs_produtividade.png",
        rotulo_x="Temperatura média anual (°C)",
        rotulo_y="Produtividade (t/ha)",
    )

    grafico_dispersao(
        df=df_analise,
        x="umidade_relativa_media_pct",
        y="produtividade_ton_ha_ibge",
        titulo="Umidade Relativa e Produtividade da Cana-de-Açúcar",
        nome_arquivo="09_umidade_vs_produtividade.png",
        rotulo_x="Umidade relativa média (%)",
        rotulo_y="Produtividade (t/ha)",
    )

    print("\nCORRELAÇÕES MAIS FORTES:")
    print(
        correlacoes.head(10).to_string(
            index=False,
            formatters={
                "correlacao_pearson": "{:.3f}".format,
                "correlacao_absoluta": "{:.3f}".format,
            },
        )
    )

    correlacao_chuva_produtividade = (
        df_analise[
            ["precipitacao_anual_mm", "produtividade_ton_ha_ibge"]
        ]
        .corr()
        .iloc[0, 1]
    )

    print("\nCORRELAÇÃO PRINCIPAL:")
    print(
        "Precipitação anual × produtividade: "
        f"{correlacao_chuva_produtividade:.3f}"
    )

    print("\nArquivos gerados:")
    print(f"- {ARQUIVO_CORRELACOES.resolve()}")
    print(f"- {ARQUIVO_BASE_ANALISE.resolve()}")
    print(f"- {PASTA_GRAFICOS.resolve()}")

    print("\nAnálise climática concluída com sucesso.")


if __name__ == "__main__":
    main()