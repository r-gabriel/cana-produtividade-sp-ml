from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURAÇÕES
# ============================================================

ARQUIVO_ENTRADA = Path(
    "data/raw/agricola_multi/pam_cana_multi_municipios_raw.csv"
)

ARQUIVO_SAIDA_PARQUET = Path(
    "data/processed/pam_cana_multi_municipios_anual.parquet"
)

ARQUIVO_SAIDA_CSV = Path(
    "data/processed/pam_cana_multi_municipios_anual.csv"
)

MAPA_VARIAVEIS = {
    "8331": "area_plantada_ha",
    "216": "area_colhida_ha",
    "214": "producao_ton",
    "112": "rendimento_kg_ha",
    "215": "valor_producao_mil_reais",
}


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def converter_valor(valor):
    """
    Converte os valores retornados pelo SIDRA para formato numérico.
    """
    if pd.isna(valor):
        return pd.NA
    valor = str(valor).strip()
    if valor in {"...", "-", ""}:
        return pd.NA
    valor = valor.replace(".", "").replace(",", ".")
    return pd.to_numeric(valor, errors="coerce")


def carregar_dados() -> pd.DataFrame:
    """
    Carrega os dados brutos coletados da API SIDRA para múltiplos municípios.
    """
    if not ARQUIVO_ENTRADA.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado:\n{ARQUIVO_ENTRADA.resolve()}"
        )
    df = pd.read_csv(
        ARQUIVO_ENTRADA,
        dtype={
            "D1C": str,
            "D2C": str,
            "D3C": str,
        }
    )
    return df


def transformar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra, padroniza, reorganiza e calcula indicadores agrícolas para
    múltiplos municípios.
    """
    # Mantém apenas as variáveis relevantes ao projeto.
    df = df[df["D2C"].isin(MAPA_VARIAVEIS)].copy()

    # Padronização de nomes.
    df["ano"] = pd.to_numeric(df["D3C"], errors="coerce")
    df["valor_numerico"] = df["V"].apply(converter_valor)
    df["variavel_padronizada"] = df["D2C"].map(MAPA_VARIAVEIS)

    # Mantém colunas essenciais antes da transformação.
    df_base = df[
        [
            "D1C",
            "D1N",
            "ano",
            "variavel_padronizada",
            "valor_numerico",
        ]
    ].copy()

    # Converte o formato longo para largo.
    df_wide = (
        df_base.pivot_table(
            index=["D1C", "D1N", "ano"],
            columns="variavel_padronizada",
            values="valor_numerico",
            aggfunc="first",
        )
        .reset_index()
    )

    # Remove o nome técnico atribuído automaticamente às colunas.
    df_wide.columns.name = None

    # Renomeia identificadores de município.
    df_wide = df_wide.rename(
        columns={
            "D1C": "codigo_ibge",
            "D1N": "municipio",
        }
    )

    # Adiciona UF e produto para facilitar integrações futuras.
    df_wide["uf"] = "SP" # Assumimos que todos são de SP
    df_wide["produto"] = "Cana-de-açúcar"

    # Calcula produtividade em toneladas por hectare.
    df_wide["produtividade_ton_ha_calculada"] = (
        df_wide["producao_ton"]
        / df_wide["area_colhida_ha"]
    )

    # Converte o rendimento oficial do IBGE de kg/ha para t/ha.
    df_wide["produtividade_ton_ha_ibge"] = (
        df_wide["rendimento_kg_ha"] / 1000
    )

    # Diferença para validação entre cálculo próprio e rendimento do IBGE.
    df_wide["diferenca_produtividade_ton_ha"] = (
        df_wide["produtividade_ton_ha_calculada"]
        - df_wide["produtividade_ton_ha_ibge"]
    )

    # Organiza as colunas.
    colunas_ordenadas = [
        "codigo_ibge",
        "municipio",
        "uf",
        "produto",
        "ano",
        "area_plantada_ha",
        "area_colhida_ha",
        "producao_ton",
        "rendimento_kg_ha",
        "produtividade_ton_ha_ibge",
        "produtividade_ton_ha_calculada",
        "diferenca_produtividade_ton_ha",
        "valor_producao_mil_reais",
    ]

    df_wide = df_wide[colunas_ordenadas]

    return df_wide.sort_values(["codigo_ibge", "ano"]).reset_index(drop=True)


def validar_dados(df: pd.DataFrame) -> None:
    """
    Executa verificações básicas de qualidade da base processada.
    """
    print("\nVALIDAÇÃO DA BASE PROCESSADA")

    print(f"\nQuantidade de linhas: {len(df)}")
    print(f"Quantidade de municípios: {df['codigo_ibge'].nunique()}")
    print(f"Primeiro ano: {df['ano'].min()}")
    print(f"Último ano: {df['ano'].max()}")

    print("\nValores ausentes por coluna:")
    print(df.isna().sum().sort_values(ascending=False).to_string())

    print("\nDiferença máxima entre produtividade calculada e IBGE:")
    diferenca_maxima = df[
        "diferenca_produtividade_ton_ha"
    ].abs().max()

    print(f"{diferenca_maxima:.6f} t/ha")

    print("\nPrimeiras linhas da base analítica:")
    print(df.head().to_string(index=False))


def salvar_dados(df: pd.DataFrame) -> None:
    """
    Salva a base final em Parquet e CSV.
    """
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
    print("ETAPA 11.4 — TRANSFORMAÇÃO DOS DADOS PAM / MÚLTIPLOS MUNICÍPIOS")
    print("=" * 60)

    df_raw = carregar_dados()
    df_processado = transformar_dados(df_raw)

    validar_dados(df_processado)
    salvar_dados(df_processado)

    print("\nTransformação concluída com sucesso.")


if __name__ == "__main__":
    main()