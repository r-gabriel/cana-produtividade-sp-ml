from pathlib import Path
import json

import pandas as pd
import requests


# ============================================================
# CONFIGURAÇÕES
# ============================================================

LATITUDE_GUARIBA = -21.359
LONGITUDE_GUARIBA = -48.228

DATA_INICIAL = "20000101"
DATA_FINAL = "20241231"

PARAMETROS = [
    "PRECTOTCORR",
    "T2M",
    "T2M_MAX",
    "T2M_MIN",
    "RH2M",
    "ALLSKY_SFC_SW_DWN",
]

URL_API = "https://power.larc.nasa.gov/api/temporal/daily/point"

PASTA_RAW = Path("data/raw/clima")
PASTA_PROCESSED = Path("data/processed")

ARQUIVO_JSON = PASTA_RAW / "nasa_power_guariba_diario_raw.json"
ARQUIVO_DAILY = PASTA_RAW / "nasa_power_guariba_diario_raw.csv"
ARQUIVO_ANNUAL_PARQUET = (
    PASTA_PROCESSED / "clima_guariba_anual.parquet"
)
ARQUIVO_ANNUAL_CSV = (
    PASTA_PROCESSED / "clima_guariba_anual.csv"
)


# ============================================================
# EXTRAÇÃO
# ============================================================

def consultar_nasa_power() -> dict:
    """Consulta dados climáticos diários da NASA POWER."""

    parametros_requisicao = {
        "parameters": ",".join(PARAMETROS),
        "community": "AG",
        "longitude": LONGITUDE_GUARIBA,
        "latitude": LATITUDE_GUARIBA,
        "start": DATA_INICIAL,
        "end": DATA_FINAL,
        "format": "JSON",
    }

    print("Consultando NASA POWER...")
    print(f"Latitude: {LATITUDE_GUARIBA}")
    print(f"Longitude: {LONGITUDE_GUARIBA}")
    print(f"Período: {DATA_INICIAL} a {DATA_FINAL}")

    resposta = requests.get(
        URL_API,
        params=parametros_requisicao,
        timeout=120,
    )

    print(f"Status HTTP: {resposta.status_code}")
    print(f"URL consultada: {resposta.url}")

    if resposta.status_code != 200:
        raise RuntimeError(
            f"Erro HTTP {resposta.status_code}\n"
            f"Resposta: {resposta.text[:2000]}"
        )

    dados = resposta.json()

    if "properties" not in dados:
        raise ValueError(
            "Resposta inesperada da NASA POWER: "
            "campo 'properties' não encontrado."
        )

    return dados


# ============================================================
# TRANSFORMAÇÃO
# ============================================================

def transformar_dados_diarios(dados: dict) -> pd.DataFrame:
    """
    Converte a estrutura JSON retornada pela NASA POWER para
    uma tabela diária.
    """

    parametros = dados["properties"]["parameter"]

    df = pd.DataFrame(parametros)

    df.index.name = "data"
    df = df.reset_index()

    df["data"] = pd.to_datetime(
        df["data"],
        format="%Y%m%d",
        errors="coerce",
    )

    df = df.rename(
        columns={
            "PRECTOTCORR": "precipitacao_mm_dia",
            "T2M": "temperatura_media_c",
            "T2M_MAX": "temperatura_maxima_c",
            "T2M_MIN": "temperatura_minima_c",
            "RH2M": "umidade_relativa_pct",
            "ALLSKY_SFC_SW_DWN": "radiacao_solar_mj_m2_dia",
        }
    )

    # Valores ausentes da NASA POWER são frequentemente codificados como -999.
    df = df.replace(-999, pd.NA)

    colunas_numericas = [
        "precipitacao_mm_dia",
        "temperatura_media_c",
        "temperatura_maxima_c",
        "temperatura_minima_c",
        "umidade_relativa_pct",
        "radiacao_solar_mj_m2_dia",
    ]

    for coluna in colunas_numericas:
        df[coluna] = pd.to_numeric(df[coluna], errors="coerce")

    return df


def consolidar_por_ano(df_diario: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega os dados diários para a granularidade Município + Ano,
    compatível com a base anual da PAM.
    """

    df = df_diario.copy()

    df["ano"] = df["data"].dt.year

    df_anual = (
        df.groupby("ano", as_index=False)
        .agg(
            precipitacao_anual_mm=(
                "precipitacao_mm_dia",
                "sum"
            ),
            temperatura_media_anual_c=(
                "temperatura_media_c",
                "mean"
            ),
            temperatura_maxima_anual_c=(
                "temperatura_maxima_c",
                "mean"
            ),
            temperatura_minima_anual_c=(
                "temperatura_minima_c",
                "mean"
            ),
            umidade_relativa_media_pct=(
                "umidade_relativa_pct",
                "mean"
            ),
            radiacao_solar_media_mj_m2_dia=(
                "radiacao_solar_mj_m2_dia",
                "mean"
            ),
            quantidade_dias=(
                "data",
                "count"
            ),
        )
    )

    df_anual["codigo_ibge"] = "3518602"
    df_anual["municipio"] = "Guariba (SP)"
    df_anual["uf"] = "SP"

    colunas = [
        "codigo_ibge",
        "municipio",
        "uf",
        "ano",
        "precipitacao_anual_mm",
        "temperatura_media_anual_c",
        "temperatura_maxima_anual_c",
        "temperatura_minima_anual_c",
        "umidade_relativa_media_pct",
        "radiacao_solar_media_mj_m2_dia",
        "quantidade_dias",
    ]

    return df_anual[colunas]


# ============================================================
# VALIDAÇÃO E SALVAMENTO
# ============================================================

def validar_dados(
    df_diario: pd.DataFrame,
    df_anual: pd.DataFrame
) -> None:
    """Exibe validações básicas da série climática."""

    print("\nVALIDAÇÃO DOS DADOS CLIMÁTICOS")

    print(f"\nTotal de registros diários: {len(df_diario):,}")
    print(
        f"Data inicial: "
        f"{df_diario['data'].min().strftime('%d/%m/%Y')}"
    )
    print(
        f"Data final: "
        f"{df_diario['data'].max().strftime('%d/%m/%Y')}"
    )

    print("\nQuantidade de registros anuais:")
    print(len(df_anual))

    print("\nDias por ano:")
    print(
        df_anual[
            ["ano", "quantidade_dias"]
        ].to_string(index=False)
    )

    print("\nValores ausentes por variável diária:")
    print(
        df_diario.isna()
        .sum()
        .sort_values(ascending=False)
        .to_string()
    )

    print("\nResumo climático anual:")
    print(
        df_anual[
            [
                "ano",
                "precipitacao_anual_mm",
                "temperatura_media_anual_c",
                "umidade_relativa_media_pct",
            ]
        ]
        .head()
        .to_string(index=False)
    )


def salvar_dados(
    dados_json: dict,
    df_diario: pd.DataFrame,
    df_anual: pd.DataFrame
) -> None:
    """Salva o JSON bruto e os dados diário e anual."""

    PASTA_RAW.mkdir(parents=True, exist_ok=True)
    PASTA_PROCESSED.mkdir(parents=True, exist_ok=True)

    with open(ARQUIVO_JSON, "w", encoding="utf-8") as arquivo:
        json.dump(
            dados_json,
            arquivo,
            ensure_ascii=False,
            indent=2
        )

    df_diario.to_csv(
        ARQUIVO_DAILY,
        index=False,
        encoding="utf-8-sig"
    )

    df_anual.to_csv(
        ARQUIVO_ANNUAL_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    df_anual.to_parquet(
        ARQUIVO_ANNUAL_PARQUET,
        index=False
    )

    print("\nArquivos gerados:")
    print(f"- {ARQUIVO_JSON.resolve()}")
    print(f"- {ARQUIVO_DAILY.resolve()}")
    print(f"- {ARQUIVO_ANNUAL_CSV.resolve()}")
    print(f"- {ARQUIVO_ANNUAL_PARQUET.resolve()}")


def main():
    print("=" * 60)
    print("ETAPA 6 — EXTRAÇÃO DE DADOS CLIMÁTICOS")
    print("NASA POWER / GUARIBA-SP")
    print("=" * 60)

    dados_json = consultar_nasa_power()
    df_diario = transformar_dados_diarios(dados_json)
    df_anual = consolidar_por_ano(df_diario)

    validar_dados(df_diario, df_anual)
    salvar_dados(dados_json, df_diario, df_anual)

    print("\nExtração climática concluída com sucesso.")


if __name__ == "__main__":
    main()