from pathlib import Path

import pandas as pd
import requests


# ============================================================
# CONFIGURAÇÕES GLOBAIS
# ============================================================

TABELA_PAM = "5457"
CLASSIFICACAO_PRODUTO = "782"
CODIGO_CANA_DE_ACUCAR = "40106"

ANOS = list(range(2000, 2025))

VARIAVEIS_PAM = {
    "8331": "area_plantada_ha",
    "216": "area_colhida_ha",
    "214": "producao_ton",
    "112": "rendimento_kg_ha",
    "215": "valor_producao_mil_reais",
}

PASTA_SAIDA_RAW = Path("data/raw/agricola_multi")
ARQUIVO_LISTA_MUNICIPIOS = PASTA_SAIDA_RAW / "municipios_sp_com_cana.csv"
ARQUIVO_DADOS_BRUTOS = PASTA_SAIDA_RAW / "pam_cana_multi_municipios_raw.csv"


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def converter_valor(valor):
    """
    Converte os valores retornados pela API SIDRA em valores numéricos.
    """
    if pd.isna(valor):
        return pd.NA
    valor = str(valor).strip()
    if valor in {"...", "-", ""}:
        return pd.NA
    return pd.to_numeric(valor, errors="coerce")


def obter_municipios_sp() -> pd.DataFrame:
    """
    Obtém a lista de municípios de São Paulo da API do IBGE.
    """
    url = "https://servicodados.ibge.gov.br/api/v1/localidades/estados/35/municipios"
    print(f"Obtendo lista de municípios de SP da API do IBGE: {url}")
    resposta = requests.get(url, timeout=120)
    resposta.raise_for_status()  # Levanta exceção para erros HTTP
    municipios_json = resposta.json()

    municipios_df = pd.DataFrame([
        {"codigo_ibge": str(m["id"]), "nome_municipio": m["nome"]}
        for m in municipios_json
    ])
    return municipios_df


def consultar_variavel_municipio(
    codigo_municipio: str,
    codigo_variavel: str
) -> pd.DataFrame:
    """
    Consulta uma variável agrícola de cana-de-açúcar para um município específico.
    """
    anos_consulta = ",".join(map(str, ANOS))

    url = (
        "https://apisidra.ibge.gov.br/values/"
        f"t/{TABELA_PAM}/"
        f"n6/{codigo_municipio}/"
        f"v/{codigo_variavel}/"
        f"p/{anos_consulta}/"
        f"c{CLASSIFICACAO_PRODUTO}/{CODIGO_CANA_DE_ACUCAR}"
    )

    # print(f"  Consultando {codigo_municipio} - {codigo_variavel}...")
    resposta = requests.get(url, timeout=120)

    if resposta.status_code != 200:
        print(
            f"  Erro HTTP {resposta.status_code} para {codigo_municipio} "
            f"- {codigo_variavel}. Resposta: {resposta.text[:100]}"
        )
        return pd.DataFrame()

    df = pd.DataFrame(resposta.json())

    # A primeira linha descreve os nomes das colunas.
    df = df[df["NC"] != "Nível Territorial (Código)"].copy()

    return df


# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

def main():
    print("=" * 60)
    print("ETAPA 11.3.2 — EXTRAÇÃO PAM PARA MÚLTIPLOS MUNICÍPIOS")
    print("=" * 60)

    PASTA_SAIDA_RAW.mkdir(parents=True, exist_ok=True)

    municipios_sp = obter_municipios_sp()
    print(f"Total de municípios de SP encontrados: {len(municipios_sp)}")

    todos_dados = []
    municipios_com_cana = set()

    # Iterar sobre cada município e cada variável
    for _, municipio_row in municipios_sp.iterrows():
        codigo_ibge = municipio_row["codigo_ibge"]
        nome_municipio = municipio_row["nome_municipio"]

        print(f"Processando município: {nome_municipio} ({codigo_ibge})")

        dados_municipio = []
        tem_cana_no_municipio = False

        for codigo_variavel, nome_padronizado in VARIAVEIS_PAM.items():
            df_variavel = consultar_variavel_municipio(
                codigo_ibge,
                codigo_variavel
            )

            if not df_variavel.empty:
                # Verifica se há valores numéricos válidos
                df_variavel["V_numerico"] = df_variavel["V"].apply(converter_valor)
                if df_variavel["V_numerico"].notna().any():
                    tem_cana_no_municipio = True

                df_variavel["variavel_padronizada"] = nome_padronizado
                dados_municipio.append(df_variavel)

        if tem_cana_no_municipio and dados_municipio:
            df_municipio_completo = pd.concat(dados_municipio, ignore_index=True)
            todos_dados.append(df_municipio_completo)
            municipios_com_cana.add(codigo_ibge)
            print(f"  -> Dados de cana encontrados para {nome_municipio}.")
        else:
            print(f"  -> Nenhum dado de cana com valores numéricos para {nome_municipio}.")


    if not todos_dados:
        print("Nenhum dado de cana encontrado para os municípios de SP.")
        return

    df_final = pd.concat(todos_dados, ignore_index=True)

    # Processamento final dos dados
    df_final["D1C"] = df_final["D1C"].astype(str)
    df_final["D3C"] = pd.to_numeric(df_final["D3C"], errors="coerce")
    df_final["V_numerico"] = df_final["V"].apply(converter_valor)

    print(f"\nTotal de registros brutos coletados: {len(df_final)}")
    print(f"Total de municípios com dados de cana: {len(municipios_com_cana)}")

    # Salvar a lista de municípios com cana
    pd.DataFrame(
        list(municipios_com_cana),
        columns=["codigo_ibge"]
    ).to_csv(
        ARQUIVO_LISTA_MUNICIPIOS,
        index=False,
        encoding="utf-8-sig"
    )
    print(f"\nLista de municípios com cana salva em: {ARQUIVO_LISTA_MUNICIPIOS.resolve()}")

    # Salvar os dados brutos coletados
    df_final.to_csv(
        ARQUIVO_DADOS_BRUTOS,
        index=False,
        encoding="utf-8-sig"
    )
    print(f"Dados brutos da PAM para múltiplos municípios salvos em: {ARQUIVO_DADOS_BRUTOS.resolve()}")

    print("\nExtração PAM para múltiplos municípios concluída com sucesso.")


if __name__ == "__main__":
    main()