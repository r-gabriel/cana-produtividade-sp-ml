from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURAÇÕES
# ============================================================

ARQUIVO_CLIMA_DIARIO = Path(
    "data/raw/clima/nasa_power_guariba_diario_raw.csv"
)

ARQUIVO_BASE_INTEGRADA = Path(
    "data/processed/base_cana_clima_guariba_anual.parquet"
)

ARQUIVO_SAIDA_PARQUET = Path(
    "data/processed/base_cana_clima_atributos_anual.parquet"
)

ARQUIVO_SAIDA_CSV = Path(
    "data/processed/base_cana_clima_atributos_anual.csv"
)

# Limiar para considerar um dia como chuvoso/seco
LIMIAR_CHUVA_MM = 1.0


# ============================================================
# FUNÇÕES
# ============================================================

def carregar_dados() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carrega os dados climáticos diários e a base anual integrada.
    """

    if not ARQUIVO_CLIMA_DIARIO.exists():
        raise FileNotFoundError(
            f"Arquivo climático diário não encontrado:\n"
            f"{ARQUIVO_CLIMA_DIARIO.resolve()}"
        )

    if not ARQUIVO_BASE_INTEGRADA.exists():
        raise FileNotFoundError(
            f"Base integrada anual não encontrada:\n"
            f"{ARQUIVO_BASE_INTEGRADA.resolve()}"
        )

    df_clima_diario = pd.read_csv(
        ARQUIVO_CLIMA_DIARIO,
        parse_dates=["data"],
    )

    df_base_anual = pd.read_parquet(ARQUIVO_BASE_INTEGRADA)

    return df_clima_diario, df_base_anual


def criar_atributos_sazonais(df_clima_diario: pd.DataFrame) -> pd.DataFrame:
    """
    Cria atributos climáticos agregados por períodos sazonais
    relevantes para a cana-de-açúcar.
    """

    df = df_clima_diario.copy()

    df["ano"] = df["data"].dt.year
    df["mes"] = df["data"].dt.month

    # Definindo os períodos sazonais
    # Período 1: Brotação e Crescimento Inicial (Setembro a Dezembro)
    # Período 2: Crescimento e Acúmulo de Sacarose (Janeiro a Abril)
    # Período 3: Maturação e Colheita (Maio a Agosto)

    def get_periodo(mes):
        if mes >= 9 and mes <= 12:
            return "p1_brotacao_crescimento_inicial"
        elif mes >= 1 and mes <= 4:
            return "p2_crescimento_acumulo_sacarose"
        elif mes >= 5 and mes <= 8:
            return "p3_maturacao_colheita"
        return None

    df["periodo_agronomico"] = df["mes"].apply(get_periodo)

    # Remove dias que não se encaixam nos períodos definidos (se houver)
    df = df.dropna(subset=["periodo_agronomico"])

    # Calcula dias secos/chuvosos
    df["dia_chuvoso"] = (df["precipitacao_mm_dia"] >= LIMIAR_CHUVA_MM).astype(int)
    df["dia_seco"] = (df["precipitacao_mm_dia"] < LIMIAR_CHUVA_MM).astype(int)

    # Agregação por ano e período
    df_sazonal = (
        df.groupby(["ano", "periodo_agronomico"])
        .agg(
            precipitacao_acumulada_mm=("precipitacao_mm_dia", "sum"),
            temperatura_media_c=("temperatura_media_c", "mean"),
            dias_chuvosos=("dia_chuvoso", "sum"),
            dias_secos=("dia_seco", "sum"),
        )
        .reset_index()
    )

    # Transforma para formato largo (pivot)
    df_sazonal_wide = df_sazonal.pivot_table(
        index="ano",
        columns="periodo_agronomico",
        values=[
            "precipitacao_acumulada_mm",
            "temperatura_media_c",
            "dias_chuvosos",
            "dias_secos",
        ],
    )

    # Renomeia as colunas para um formato mais claro
    df_sazonal_wide.columns = [
        f"{col[0].lower()}_{col[1]}"
        for col in df_sazonal_wide.columns
    ]

    df_sazonal_wide = df_sazonal_wide.reset_index()

    return df_sazonal_wide


def criar_atributos_defasados(df_base_anual: pd.DataFrame) -> pd.DataFrame:
    """
    Cria atributos defasados (lagged features) para a produtividade
    e precipitação do ano anterior.
    """

    df = df_base_anual.copy()

    df["produtividade_ton_ha_ibge_lag1"] = df[
        "produtividade_ton_ha_ibge"
    ].shift(1)

    df["precipitacao_anual_mm_lag1"] = df["precipitacao_anual_mm"].shift(1)

    return df


def integrar_atributos(
    df_base_anual: pd.DataFrame,
    df_atributos_sazonais: pd.DataFrame,
) -> pd.DataFrame:
    """
    Integra os novos atributos sazonais e defasados à base anual.
    """

    df_final = pd.merge(
        df_base_anual,
        df_atributos_sazonais,
        on="ano",
        how="left",
        validate="one_to_one",
    )

    df_final = criar_atributos_defasados(df_final)

    return df_final


def validar_base(df: pd.DataFrame) -> None:
    """
    Executa verificações de consistência após a engenharia de atributos.
    """

    print("\nVALIDAÇÃO DA BASE COM ATRIBUTOS")

    print(f"\nQuantidade de linhas: {len(df)}")
    print(f"Período: {df['ano'].min()} a {df['ano'].max()}")

    print("\nValores ausentes por coluna (top 10):")
    print(
        df.isna()
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .to_string()
    )

    print("\nAmostra da base com atributos:")
    print(
        df[
            [
                "ano",
                "produtividade_ton_ha_ibge",
                "precipitacao_acumulada_mm_p1_brotacao_crescimento_inicial",
                "temperatura_media_c_p2_crescimento_acumulo_sacarose",
                "dias_chuvosos_p3_maturacao_colheita",
                "produtividade_ton_ha_ibge_lag1",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


def salvar_base(df: pd.DataFrame) -> None:
    """
    Salva a base com os novos atributos em Parquet e CSV.
    """

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
    print("ETAPA 9 — ENGENHARIA DE ATRIBUTOS CLIMÁTICOS")
    print("=" * 60)

    df_clima_diario, df_base_anual = carregar_dados()

    df_atributos_sazonais = criar_atributos_sazonais(df_clima_diario)
    df_final = integrar_atributos(df_base_anual, df_atributos_sazonais)

    validar_base(df_final)
    salvar_base(df_final)

    print("\nEngenharia de atributos concluída com sucesso.")


if __name__ == "__main__":
    main()