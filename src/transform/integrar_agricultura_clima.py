from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURAÇÕES
# ============================================================

ARQUIVO_AGRICOLA = Path(
    "data/processed/pam_cana_guariba_anual.parquet"
)

ARQUIVO_CLIMA = Path(
    "data/processed/clima_guariba_anual.parquet"
)

ARQUIVO_SAIDA_PARQUET = Path(
    "data/processed/base_cana_clima_guariba_anual.parquet"
)

ARQUIVO_SAIDA_CSV = Path(
    "data/processed/base_cana_clima_guariba_anual.csv"
)


# ============================================================
# FUNÇÕES
# ============================================================

def carregar_bases() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega as bases agrícola e climática processadas."""

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
    """
    Une as bases usando município e ano como chaves.

    A validação 'one_to_one' garante que cada ano apareça uma única
    vez em cada base antes da integração.
    """

    colunas_clima = [
        "codigo_ibge",
        "ano",
        "precipitacao_anual_mm",
        "temperatura_media_anual_c",
        "temperatura_maxima_anual_c",
        "temperatura_minima_anual_c",
        "umidade_relativa_media_pct",
        "radiacao_solar_media_mj_m2_dia",
        "quantidade_dias",
    ]

    df_clima = df_clima[colunas_clima].copy()

    df_final = pd.merge(
        df_agricola,
        df_clima,
        on=["codigo_ibge", "ano"],
        how="inner",
        validate="one_to_one",
    )

    return df_final.sort_values("ano").reset_index(drop=True)


def criar_atributos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria indicadores derivados para a análise e para futuras etapas
    de Machine Learning.
    """

    df = df.copy()

    # Variações anuais dos principais indicadores agrícolas.
    df["variacao_anual_producao_pct"] = (
        df["producao_ton"].pct_change() * 100
    )

    df["variacao_anual_produtividade_pct"] = (
        df["produtividade_ton_ha_ibge"].pct_change() * 100
    )

    # Precipitação por dia: útil para comparação entre anos comuns e bissextos.
    df["precipitacao_media_diaria_mm"] = (
        df["precipitacao_anual_mm"] / df["quantidade_dias"]
    )

    # Amplitude térmica anual média.
    df["amplitude_termica_media_c"] = (
        df["temperatura_maxima_anual_c"]
        - df["temperatura_minima_anual_c"]
    )

    # Classificação simples de anos conforme a precipitação histórica.
    q25 = df["precipitacao_anual_mm"].quantile(0.25)
    q75 = df["precipitacao_anual_mm"].quantile(0.75)

    def classificar_chuva(valor):
        if valor <= q25:
            return "Baixa precipitação"
        if valor >= q75:
            return "Alta precipitação"
        return "Precipitação intermediária"

    df["faixa_precipitacao"] = df["precipitacao_anual_mm"].apply(
        classificar_chuva
    )

    return df


def validar_base(df: pd.DataFrame) -> None:
    """Executa verificações de consistência após a integração."""

    print("\nVALIDAÇÃO DA BASE INTEGRADA")

    print(f"\nQuantidade de linhas: {len(df)}")
    print(f"Período: {df['ano'].min()} a {df['ano'].max()}")

    print("\nValores ausentes por coluna:")
    print(
        df.isna()
        .sum()
        .sort_values(ascending=False)
        .to_string()
    )

    print("\nAmostra da base integrada:")
    print(
        df[
            [
                "ano",
                "producao_ton",
                "produtividade_ton_ha_ibge",
                "precipitacao_anual_mm",
                "temperatura_media_anual_c",
                "umidade_relativa_media_pct",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print("\nResumo da precipitação anual:")
    print(
        df["precipitacao_anual_mm"]
        .describe()
        .round(2)
        .to_string()
    )


def salvar_base(df: pd.DataFrame) -> None:
    """Salva a base integrada em CSV e Parquet."""

    ARQUIVO_SAIDA_PARQUET.parent.mkdir(
        parents=True,
        exist_ok=True
    )

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
    print("ETAPA 7 — INTEGRAÇÃO ENTRE AGRICULTURA E CLIMA")
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