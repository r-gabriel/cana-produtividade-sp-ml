from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURAÇÕES
# ============================================================

ARQUIVO_AGRICOLA = Path(
    "data/processed/pam_cana_multi_municipios_anual.parquet"
)

ARQUIVO_CLIMA = Path(
    "data/processed/clima_multi_municipios_anual.parquet"
)

ARQUIVO_SAIDA_PARQUET = Path(
    "data/processed/base_cana_clima_multi_municipios_anual.parquet"
)

ARQUIVO_SAIDA_CSV = Path(
    "data/processed/base_cana_clima_multi_municipios_anual.csv"
)


# ============================================================
# FUNÇÕES
# ============================================================

def carregar_bases() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega as bases agrícola e climática processadas para múltiplos municípios."""

    if not ARQUIVO_AGRICOLA.exists():
        raise FileNotFoundError(
            f"Base agrícola não encontrada:\n{ARQUIVO_AGRICOLA.resolve()}"
        )

    if not ARQUIVO_CLIMA.exists():
        raise FileNotFoundError(
            f"Base climática não encontrada:\n{ARQUIVO_CLIMA.resolve()}"
        )

    df_agricola = pd.read_parquet(ARQUIVO_AGRICOLA)
    df_clima = pd.read_parquet(ARQUIVO_CLIMA)

    return df_agricola, df_clima


def integrar_bases(
    df_agricola: pd.DataFrame,
    df_clima: pd.DataFrame
) -> pd.DataFrame:
    """Integra as bases agrícola e climática por código IBGE e ano."""

    # Renomear colunas para evitar conflitos e padronizar para o merge
    df_clima = df_clima.rename(
        columns={
            "codigo_ibge_municipio": "codigo_ibge",
            "nome_municipio": "municipio",
        }
    )

    # Realizar o merge
    df_integrado = pd.merge(
        df_agricola,
        df_clima,
        on=["codigo_ibge", "ano"],
        how="inner",  # Usar inner para garantir que temos dados de ambos
        suffixes=("_agricola", "_clima"),
    )

    # Remover colunas duplicadas ou redundantes após o merge
    if "municipio_clima" in df_integrado.columns:
        df_integrado = df_integrado.drop(columns=["municipio_clima"])
    if "municipio_agricola" in df_integrado.columns:
        df_integrado = df_integrado.rename(
            columns={"municipio_agricola": "municipio"}
        )

    return df_integrado


def criar_atributos(df: pd.DataFrame) -> pd.DataFrame:
    """Cria atributos derivados na base integrada."""

    # Variação anual da produtividade
    df["variacao_anual_produtividade_pct"] = df.groupby("codigo_ibge")[
        "produtividade_ton_ha_ibge"
    ].pct_change() * 100

    # Variação anual da produção
    df["variacao_anual_producao_pct"] = df.groupby("codigo_ibge")[
        "producao_ton"
    ].pct_change() * 100

    # Precipitação média diária
    df["precipitacao_media_diaria_mm"] = (
        df["precipitacao_anual_mm"] / df["quantidade_dias"]
    )

    # Amplitude térmica média
    df["amplitude_termica_media_c"] = (
        df["temperatura_maxima_anual_c"] - df["temperatura_minima_anual_c"]
    )

    # Classificação da precipitação (exemplo: quartis)
    # df["faixa_precipitacao"] = pd.qcut(
    #     df["precipitacao_anual_mm"],
    #     q=4,
    #     labels=["muito_baixa", "baixa", "media", "alta"],
    #     duplicates="drop",
    # )

    return df


def validar_base(df: pd.DataFrame) -> None:
    """Executa validações básicas na base integrada."""

    print("\nVALIDAÇÃO DA BASE INTEGRADA")

    print(f"\nQuantidade de linhas: {len(df)}")
    print(f"Quantidade de municípios: {df['codigo_ibge'].nunique()}")
    print(f"Período: {df['ano'].min()} a {df['ano'].max()}")

    print("\nValores ausentes por coluna (top 10):")
    print(df.isna().sum().sort_values(ascending=False).head(10).to_string())

    print("\nAmostra da base integrada:")
    print(
        df[
            [
                "codigo_ibge",
                "municipio",
                "ano",
                "produtividade_ton_ha_ibge",
                "precipitacao_anual_mm",
                "temperatura_media_anual_c",
                "variacao_anual_produtividade_pct",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print("\nResumo da precipitação anual:")
    print(df["precipitacao_anual_mm"].describe().to_string())


def salvar_base(df: pd.DataFrame) -> None:
    """Salva a base integrada em Parquet e CSV."""

    ARQUIVO_SAIDA_PARQUET.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(
        ARQUIVO_SAIDA_PARQUET,
        index=False
    )

    df.to_csv(
        ARQUIVO_SAIDA_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nArquivos gerados:")
    print(f"- {ARQUIVO_SAIDA_PARQUET.resolve()}")
    print(f"- {ARQUIVO_SAIDA_CSV.resolve()}")


def main():
    print("=" * 60)
    print("ETAPA 11.8 — INTEGRAÇÃO ENTRE AGRICULTURA E CLIMA (MÚLTIPLOS MUNICÍPIOS)")
    print("=" * 60)

    df_agricola, df_clima = carregar_bases()

    print(f"\nRegistros agrícolas: {len(df_agricola)}")
    print(f"Registros climáticos: {len(df_clima)}")

    df_integrado = integrar_bases(df_agricola, df_clima)
    df_integrado = criar_atributos(df_integrado)

    validar_base(df_integrado)
    salvar_base(df_integrado)

    print("\nIntegração concluída com sucesso.")


if __name__ == "__main__":
    main()