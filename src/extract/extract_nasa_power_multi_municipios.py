import json
import time
from pathlib import Path

import pandas as pd
import requests


# ============================================================
# CONFIGURAÇÕES GLOBAIS
# ============================================================

DATA_INICIAL = "20000101"
DATA_FINAL = "20241231"

PARAMETROS_NASA_POWER = [
    "PRECTOTCORR",  # Precipitação corrigida (mm/dia)
    "T2M",          # Temperatura média a 2 metros (C)
    "T2M_MAX",      # Temperatura máxima a 2 metros (C)
    "T2M_MIN",      # Temperatura mínima a 2 metros (C)
    "RH2M",         # Umidade relativa a 2 metros (%)
    "ALLSKY_SFC_SW_DWN", # Radiação solar de onda curta à superfície (MJ/m^2/dia)
]

URL_API_NASA_POWER = "https://power.larc.nasa.gov/api/temporal/daily/point"

PASTA_RAW_CLIMA = Path("data/raw/clima_multi")
PASTA_PROCESSED_CLIMA = Path("data/processed")

ARQUIVO_COORDENADAS_ENTRADA = PASTA_RAW_CLIMA / "municipios_sp_com_cana_coords.csv"
ARQUIVO_SAIDA_DIARIO_BRUTO = PASTA_RAW_CLIMA / "nasa_power_multi_municipios_diario_raw.csv"
ARQUIVO_SAIDA_ANUAL_PROCESSADO_PARQUET = PASTA_PROCESSED_CLIMA / "clima_multi_municipios_anual.parquet"
ARQUIVO_SAIDA_ANUAL_PROCESSADO_CSV = PASTA_PROCESSED_CLIMA / "clima_multi_municipios_anual.csv"

# Mapeamento dos nomes das variáveis da NASA POWER para nomes padronizados
MAPA_NOMES_VARIAVEIS = {
    "PRECTOTCORR": "precipitacao_mm_dia",
    "T2M": "temperatura_media_c",
    "T2M_MAX": "temperatura_maxima_c",
    "T2M_MIN": "temperatura_minima_c",
    "RH2M": "umidade_relativa_pct",
    "ALLSKY_SFC_SW_DWN": "radiacao_solar_mj_m2_dia",
}


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def consultar_nasa_power(
    latitude: float,
    longitude: float,
    data_inicial: str,
    data_final: str,
    parametros: list
) -> dict:
    """
    Consulta a API da NASA POWER para dados climáticos diários.
    """
    params = {
        "parameters": ",".join(parametros),
        "community": "AG",  # Agricultural community
        "longitude": longitude,
        "latitude": latitude,
        "start": data_inicial,
        "end": data_final,
        "format": "JSON",
    }

    try:
        response = requests.get(URL_API_NASA_POWER, params=params, timeout=120)
        response.raise_for_status()  # Levanta exceção para erros HTTP
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"  Erro ao consultar NASA POWER para lat={latitude}, lon={longitude}: {e}")
        return {}


def transformar_dados_diarios(dados_json: dict, codigo_ibge: str, nome_municipio: str) -> pd.DataFrame:
    """
    Transforma os dados JSON da NASA POWER em um DataFrame diário.
    """
    if not dados_json or "properties" not in dados_json:
        return pd.DataFrame()

    df_data = {}
    for param in PARAMETROS_NASA_POWER:
        if param in dados_json["properties"]["parameter"]:
            df_data[MAPA_NOMES_VARIAVEIS[param]] = dados_json["properties"]["parameter"][param]

    if not df_data:
        return pd.DataFrame()

    df = pd.DataFrame(df_data)
    df["data"] = pd.to_datetime(df.index, format="%Y%m%d")
    df["codigo_ibge"] = codigo_ibge
    df["municipio"] = nome_municipio

    # Converter colunas numéricas, tratando valores -999 (ausentes na NASA POWER)
    colunas_numericas = list(MAPA_NOMES_VARIAVEIS.values())
    for coluna in colunas_numericas:
        if coluna in df.columns:
            df[coluna] = pd.to_numeric(df[coluna], errors="coerce")
            df[coluna] = df[coluna].replace(-999.0, pd.NA) # Substituir -999 por NaN

    return df


def consolidar_por_ano(df_diario: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega os dados diários para a granularidade Município + Ano.
    """
    if df_diario.empty:
        return pd.DataFrame()

    df = df_diario.copy()
    df["ano"] = df["data"].dt.year

    df_anual = (
        df.groupby(["codigo_ibge", "municipio", "ano"], as_index=False)
        .agg(
            precipitacao_anual_mm=("precipitacao_mm_dia", "sum"),
            temperatura_media_anual_c=("temperatura_media_c", "mean"),
            temperatura_maxima_anual_c=("temperatura_maxima_c", "mean"),
            temperatura_minima_anual_c=("temperatura_minima_c", "mean"),
            umidade_relativa_media_pct=("umidade_relativa_pct", "mean"),
            radiacao_solar_media_mj_m2_dia=("radiacao_solar_mj_m2_dia", "mean"),
            quantidade_dias=("data", "count"),
        )
    )
    df_anual["uf"] = "SP"

    colunas_ordenadas = [
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
    return df_anual[colunas_ordenadas]


# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

def main():
    print("=" * 60)
    print("ETAPA 11.7 — EXTRAÇÃO CLIMÁTICA PARA MÚLTIPLOS MUNICÍPIOS")
    print("NASA POWER / SÃO PAULO")
    print("=" * 60)

    PASTA_RAW_CLIMA.mkdir(parents=True, exist_ok=True)
    PASTA_PROCESSED_CLIMA.mkdir(parents=True, exist_ok=True)

    if not ARQUIVO_COORDENADAS_ENTRADA.exists():
        raise FileNotFoundError(
            f"Arquivo de coordenadas não encontrado:\n"
            f"{ARQUIVO_COORDENADAS_ENTRADA.resolve()}"
        )

    df_coords = pd.read_csv(ARQUIVO_COORDENADAS_ENTRADA, dtype={"codigo_ibge": str})
    print(f"Total de municípios com coordenadas: {len(df_coords)}")

    todos_dados_diarios = []
    todos_dados_anuais = []

    for index, row in df_coords.iterrows():
        codigo_ibge = row["codigo_ibge"]
        nome_municipio = row["nome_municipio"]
        latitude = row["latitude"]
        longitude = row["longitude"]

        print(f"Coletando dados climáticos para {nome_municipio} ({codigo_ibge})...")
        dados_json = consultar_nasa_power(
            latitude, longitude, DATA_INICIAL, DATA_FINAL, PARAMETROS_NASA_POWER
        )

        df_diario_municipio = transformar_dados_diarios(dados_json, codigo_ibge, nome_municipio)
        if not df_diario_municipio.empty:
            todos_dados_diarios.append(df_diario_municipio)
            df_anual_municipio = consolidar_por_ano(df_diario_municipio)
            if not df_anual_municipio.empty:
                todos_dados_anuais.append(df_anual_municipio)

        time.sleep(0.5) # Delay para não sobrecarregar a API da NASA POWER

    if not todos_dados_diarios:
        print("Nenhum dado climático diário coletado.")
        return

    df_diario_final = pd.concat(todos_dados_diarios, ignore_index=True)
    df_diario_final.to_csv(
        ARQUIVO_SAIDA_DIARIO_BRUTO,
        index=False,
        encoding="utf-8-sig"
    )
    print(f"\nDados climáticos diários brutos salvos em: {ARQUIVO_SAIDA_DIARIO_BRUTO.resolve()}")
    print(f"Total de registros diários: {len(df_diario_final):,}")

    if not todos_dados_anuais:
        print("Nenhum dado climático anual consolidado.")
        return

    df_anual_final = pd.concat(todos_dados_anuais, ignore_index=True)
    df_anual_final.to_parquet(
        ARQUIVO_SAIDA_ANUAL_PROCESSADO_PARQUET,
        index=False
    )
    df_anual_final.to_csv(
        ARQUIVO_SAIDA_ANUAL_PROCESSADO_CSV,
        index=False,
        encoding="utf-8-sig"
    )
    print(f"Dados climáticos anuais processados salvos em: {ARQUIVO_SAIDA_ANUAL_PROCESSADO_PARQUET.resolve()}")
    print(f"Total de registros anuais: {len(df_anual_final):,}")

    print("\nExtração climática para múltiplos municípios concluída com sucesso.")


if __name__ == "__main__":
    main()