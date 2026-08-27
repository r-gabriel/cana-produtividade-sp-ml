from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# ============================================================
# CONFIGURAÇÕES
# ============================================================

ARQUIVO_ENTRADA = Path(
    "data/processed/pam_cana_guariba_anual.parquet"
)

PASTA_GRAFICOS = Path("outputs/figures")
PASTA_TABELAS = Path("outputs/tables")

ARQUIVO_RESUMO = PASTA_TABELAS / "resumo_exploratorio_guariba.csv"
ARQUIVO_CORRELACAO = PASTA_TABELAS / "correlacao_variaveis_agricolas.csv"

sns.set_theme(style="whitegrid", context="notebook")

COR_VERDE = "#2E7D32"
COR_LARANJA = "#EF6C00"
COR_AZUL = "#1565C0"
COR_VERMELHO = "#C62828"
COR_ROXO = "#6A1B9A"


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def salvar_grafico(nome_arquivo: str) -> None:
    """Salva gráfico em PNG com qualidade adequada para relatório."""
    PASTA_GRAFICOS.mkdir(parents=True, exist_ok=True)

    caminho = PASTA_GRAFICOS / nome_arquivo

    plt.tight_layout()
    plt.savefig(caminho, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Gráfico salvo: {caminho}")


def carregar_dados() -> pd.DataFrame:
    """Carrega a base agrícola processada."""
    if not ARQUIVO_ENTRADA.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado:\n{ARQUIVO_ENTRADA.resolve()}"
        )

    return pd.read_parquet(ARQUIVO_ENTRADA)


def criar_variacoes(df: pd.DataFrame) -> pd.DataFrame:
    """Cria indicadores de variação percentual anual."""
    df = df.sort_values("ano").copy()

    colunas = [
        "area_plantada_ha",
        "area_colhida_ha",
        "producao_ton",
        "produtividade_ton_ha_ibge",
        "valor_producao_mil_reais",
    ]

    for coluna in colunas:
        df[f"variacao_anual_{coluna}"] = (
            df[coluna]
            .pct_change()
            .mul(100)
        )

    return df


def gerar_resumo(df: pd.DataFrame) -> pd.DataFrame:
    """Gera estatísticas descritivas relevantes para o projeto."""
    indicadores = [
        "area_plantada_ha",
        "area_colhida_ha",
        "producao_ton",
        "produtividade_ton_ha_ibge",
        "valor_producao_mil_reais",
    ]

    resumo = df[indicadores].describe().T

    resumo["amplitude"] = resumo["max"] - resumo["min"]
    resumo["coeficiente_variacao_pct"] = (
        resumo["std"] / resumo["mean"] * 100
    )

    resumo = resumo.rename(
        columns={
            "count": "quantidade_anos",
            "mean": "media",
            "std": "desvio_padrao",
            "min": "minimo",
            "25%": "percentil_25",
            "50%": "mediana",
            "75%": "percentil_75",
            "max": "maximo",
        }
    )

    return resumo.round(2)


# ============================================================
# GRÁFICOS
# ============================================================

def grafico_producao_e_produtividade(df: pd.DataFrame) -> None:
    """Cria gráfico com produção e produtividade em eixos distintos."""
    fig, eixo_esquerdo = plt.subplots(figsize=(12, 6))

    eixo_esquerdo.plot(
        df["ano"],
        df["producao_ton"] / 1_000_000,
        marker="o",
        color=COR_VERDE,
        label="Produção",
    )

    eixo_esquerdo.set_xlabel("Ano")
    eixo_esquerdo.set_ylabel("Produção (milhões de toneladas)", color=COR_VERDE)
    eixo_esquerdo.tick_params(axis="y", labelcolor=COR_VERDE)

    eixo_direito = eixo_esquerdo.twinx()

    eixo_direito.plot(
        df["ano"],
        df["produtividade_ton_ha_ibge"],
        marker="s",
        color=COR_LARANJA,
        label="Produtividade",
    )

    eixo_direito.set_ylabel("Produtividade (t/ha)", color=COR_LARANJA)
    eixo_direito.tick_params(axis="y", labelcolor=COR_LARANJA)

    plt.title(
        "Evolução da Produção e Produtividade da Cana-de-Açúcar\n"
        "Guariba-SP, 2000–2024"
    )

    salvar_grafico("01_producao_produtividade_guariba.png")


def grafico_areas(df: pd.DataFrame) -> None:
    """Cria gráfico de área plantada e área colhida."""
    plt.figure(figsize=(12, 6))

    plt.plot(
        df["ano"],
        df["area_plantada_ha"],
        marker="o",
        color=COR_AZUL,
        label="Área plantada/destinada à colheita",
    )

    plt.plot(
        df["ano"],
        df["area_colhida_ha"],
        marker="s",
        color=COR_VERDE,
        label="Área colhida",
    )

    plt.title(
        "Evolução da Área Plantada e Colhida de Cana-de-Açúcar\n"
        "Guariba-SP, 2000–2024"
    )

    plt.xlabel("Ano")
    plt.ylabel("Área (hectares)")
    plt.legend()

    salvar_grafico("02_areas_plantada_colhida_guariba.png")


def grafico_valor_producao(df: pd.DataFrame) -> None:
    """Cria gráfico do valor nominal da produção."""
    plt.figure(figsize=(12, 6))

    plt.bar(
        df["ano"],
        df["valor_producao_mil_reais"] / 1_000,
        color=COR_ROXO,
    )

    plt.title(
        "Valor Nominal da Produção de Cana-de-Açúcar\n"
        "Guariba-SP, 2000–2024"
    )

    plt.xlabel("Ano")
    plt.ylabel("Valor da produção (milhões de R$)")
    plt.xticks(df["ano"], rotation=45)

    salvar_grafico("03_valor_producao_guariba.png")


def grafico_variacao_produtividade(df: pd.DataFrame) -> None:
    """Cria gráfico da variação anual da produtividade."""
    coluna = "variacao_anual_produtividade_ton_ha_ibge"

    cores = [
        COR_VERDE if valor >= 0 else COR_VERMELHO
        for valor in df[coluna].fillna(0)
    ]

    plt.figure(figsize=(12, 6))

    plt.bar(
        df["ano"],
        df[coluna],
        color=cores,
    )

    plt.axhline(
        y=0,
        color="black",
        linewidth=0.8
    )

    plt.title(
        "Variação Anual da Produtividade de Cana-de-Açúcar\n"
        "Guariba-SP, 2000–2024"
    )

    plt.xlabel("Ano")
    plt.ylabel("Variação anual da produtividade (%)")
    plt.xticks(df["ano"], rotation=45)

    salvar_grafico("04_variacao_anual_produtividade_guariba.png")


def grafico_correlacao(df: pd.DataFrame) -> pd.DataFrame:
    """Cria matriz de correlação entre os principais indicadores."""
    colunas = [
        "area_plantada_ha",
        "area_colhida_ha",
        "producao_ton",
        "produtividade_ton_ha_ibge",
        "valor_producao_mil_reais",
    ]

    correlacao = df[colunas].corr()

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        correlacao,
        annot=True,
        cmap="RdYlGn",
        fmt=".2f",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
    )

    plt.title(
        "Correlação entre Indicadores Agrícolas\n"
        "Guariba-SP, 2000–2024"
    )

    salvar_grafico("05_correlacao_variaveis_agricolas_guariba.png")

    return correlacao


# ============================================================
# EXECUÇÃO
# ============================================================

def main():
    print("=" * 60)
    print("ETAPA 5 — ANÁLISE EXPLORATÓRIA DOS DADOS AGRÍCOLAS")
    print("=" * 60)

    PASTA_GRAFICOS.mkdir(parents=True, exist_ok=True)
    PASTA_TABELAS.mkdir(parents=True, exist_ok=True)

    df = carregar_dados()
    df = criar_variacoes(df)

    resumo = gerar_resumo(df)
    correlacao = grafico_correlacao(df)

    resumo.to_csv(
        ARQUIVO_RESUMO,
        encoding="utf-8-sig",
    )

    correlacao.to_csv(
        ARQUIVO_CORRELACAO,
        encoding="utf-8-sig",
    )

    grafico_producao_e_produtividade(df)
    grafico_areas(df)
    grafico_valor_producao(df)
    grafico_variacao_produtividade(df)

    print("\nRESUMO ESTATÍSTICO:")
    print(resumo.to_string())

    print("\nANOS DE DESTAQUE:")

    ano_maior_producao = df.loc[df["producao_ton"].idxmax()]
    ano_menor_producao = df.loc[df["producao_ton"].idxmin()]

    ano_maior_produtividade = df.loc[
        df["produtividade_ton_ha_ibge"].idxmax()
    ]
    ano_menor_produtividade = df.loc[
        df["produtividade_ton_ha_ibge"].idxmin()
    ]

    print(
        f"- Maior produção: {int(ano_maior_producao['ano'])} "
        f"({ano_maior_producao['producao_ton']:,.0f} toneladas)"
    )

    print(
        f"- Menor produção: {int(ano_menor_producao['ano'])} "
        f"({ano_menor_producao['producao_ton']:,.0f} toneladas)"
    )

    print(
        f"- Maior produtividade: {int(ano_maior_produtividade['ano'])} "
        f"({ano_maior_produtividade['produtividade_ton_ha_ibge']:.2f} t/ha)"
    )

    print(
        f"- Menor produtividade: {int(ano_menor_produtividade['ano'])} "
        f"({ano_menor_produtividade['produtividade_ton_ha_ibge']:.2f} t/ha)"
    )

    print("\nArquivos gerados:")
    print(f"- {ARQUIVO_RESUMO.resolve()}")
    print(f"- {ARQUIVO_CORRELACAO.resolve()}")
    print(f"- {PASTA_GRAFICOS.resolve()}")

    print("\nAnálise exploratória concluída com sucesso.")


if __name__ == "__main__":
    main()