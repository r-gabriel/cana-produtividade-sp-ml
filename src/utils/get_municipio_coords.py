import time
from pathlib import Path

import pandas as pd
import requests


# ============================================================
# CONFIGURAÇÕES
# ============================================================

ARQUIVO_LISTA_MUNICIPIOS = Path(
    "data/raw/agricola_multi/municipios_sp_com_cana.csv"
)

ARQUIVO_COORDENADAS_SAIDA = Path(
    "data/raw/clima_multi/municipios_sp_com_cana_coords.csv"
)

# Usaremos a API do Nominatim (OpenStreetMap) para geocodificação
# É uma API gratuita, mas requer um User-Agent e tem limites de requisição.
# Seja gentil: adicione um delay entre as requisições.
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"


# ============================================================
# FUNÇÕES
# ============================================================

def obter_coordenadas(municipio: str, estado: str = "São Paulo") -> tuple[float, float]:
    """
    Obtém as coordenadas (latitude, longitude) de um município
    usando a API do Nominatim.
    """
    params = {
        "q": f"{municipio}, {estado}, Brasil",
        "format": "json",
        "limit": 1,
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.88 Safari/537.36"
    }

    try:
        response = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data:
            lat = float(data[0]["lat"])
            lon = float(data[0]["lon"])
            return lat, lon
        else:
            print(f"  Coordenadas não encontradas para {municipio}.")
            return pd.NA, pd.NA
    except requests.exceptions.RequestException as e:
        print(f"  Erro ao obter coordenadas para {municipio}: {e}")
        return pd.NA, pd.NA


def main():
    print("=" * 60)
    print("ETAPA 11.5.1 — OBTENÇÃO DE COORDENADAS DOS MUNICÍPIOS")
    print("=" * 60)

    PASTA_RAW = ARQUIVO_COORDENADAS_SAIDA.parent
    PASTA_RAW.mkdir(parents=True, exist_ok=True)

    if not ARQUIVO_LISTA_MUNICIPIOS.exists():
        raise FileNotFoundError(
            f"Arquivo de lista de municípios não encontrado:\n"
            f"{ARQUIVO_LISTA_MUNICIPIOS.resolve()}"
        )

    df_municipios = pd.read_csv(ARQUIVO_LISTA_MUNICIPIOS, dtype={"codigo_ibge": str})

    # Obter nomes dos municípios da API do IBGE para ter o nome completo
    url_ibge_municipios = "https://servicodados.ibge.gov.br/api/v1/localidades/municipios"
    response_ibge = requests.get(url_ibge_municipios, timeout=120)
    response_ibge.raise_for_status()
    municipios_ibge_json = response_ibge.json()

    mapa_codigo_nome = {
        str(m['id']): m['nome'] for m in municipios_ibge_json
    }

    df_municipios["nome_municipio"] = df_municipios["codigo_ibge"].map(mapa_codigo_nome)

    # Remover Guariba da lista para evitar duplicidade, já que já temos os dados dele
    # ou garantir que ele seja processado corretamente
    # df_municipios = df_municipios[df_municipios['codigo_ibge'] != '3518602'].copy()

    coordenadas = []
    for index, row in df_municipios.iterrows():
        codigo_ibge = row["codigo_ibge"]
        nome_municipio = row["nome_municipio"]

        print(f"Buscando coordenadas para {nome_municipio} ({codigo_ibge})...")
        lat, lon = obter_coordenadas(nome_municipio)
        coordenadas.append({
            "codigo_ibge": codigo_ibge,
            "nome_municipio": nome_municipio,
            "latitude": lat,
            "longitude": lon,
        })
        time.sleep(2)  # Delay de 1 segundo para não sobrecarregar a API do Nominatim

    df_coordenadas = pd.DataFrame(coordenadas)
    df_coordenadas = df_coordenadas.dropna(subset=["latitude", "longitude"])

    df_coordenadas.to_csv(
        ARQUIVO_COORDENADAS_SAIDA,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nCoordenadas salvas em: {ARQUIVO_COORDENADAS_SAIDA.resolve()}")
    print(f"Total de municípios com coordenadas: {len(df_coordenadas)}")
    print("\nObtenção de coordenadas concluída com sucesso.")


if __name__ == "__main__":
    main()