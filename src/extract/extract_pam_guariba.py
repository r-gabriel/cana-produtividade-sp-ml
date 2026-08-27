from pathlib import Path

import pandas as pd
import requests


# ============================================================
# CONFIGURAÇÕES
# ============================================================

TABELA_PAM = "5457"
CODIGO_GUARIBA = "3518602"

CLASSIFICACAO_PRODUTO = "782"
CODIGO_CANA_DE_ACUCAR = "40106"

ANOS = list(range(1988, 2025))

VARIAVEIS_PAM = {
    "8331": "area_plantada_ha",
    "216": "area_colhida_ha",
    "214": "producao_ton",
    "112": "rendimento_kg_ha",
    "215": "valor_producao",
}

PASTA_SAIDA = Path("data/raw/agricola")
ARQUIVO_SAIDA = PASTA_SAIDA / "pam_cana_guariba_raw.csv"


def converter_valor(valor):
    """
    Converte valores retornados pela API SIDRA em valores numéricos.
    """

    if pd.isna(valor):
        return pd.NA

    valor = str(valor).strip()

    if valor in {"...", "-", ""}:
        return pd.NA

    return pd.to_numeric(valor, errors="coerce")


def consultar_variavel(codigo_variavel: str) -> pd.DataFrame:
    """
    Consulta uma variável agrícola de cana-de-açúcar para Guariba-SP.
    """

    anos_consulta = ",".join(map(str, ANOS))

    url = (
        "https://apisidra.ibge.gov.br/values/"
        f"t/{TABELA_PAM}/"
        f"n6/{CODIGO_GUARIBA}/"
        f"v/{codigo_variavel}/"
        f"p/{anos_consulta}/"
        f"c{CLASSIFICACAO_PRODUTO}/{CODIGO_CANA_DE_ACUCAR}"
    )

    print(f"\nConsultando variável {codigo_variavel}...")
    print(f"URL: {url}")

    resposta = requests.get(url, timeout=120)

    if resposta.status_code != 200:
        raise RuntimeError(
            f"Erro HTTP {resposta.status_code}\n"
            f"Resposta da API: {resposta.text[:1000]}"
        )

    df = pd.DataFrame(resposta.json())

    # A primeira linha descreve os nomes das colunas.
    df = df[
        df["NC"] != "Nível Territorial (Código)"
    ].copy()

    return df


def main():
    dados = []

    for codigo_variavel, nome_padronizado in VARIAVEIS_PAM.items():
        df_variavel = consultar_variavel(codigo_variavel)

        df_variavel["variavel_padronizada"] = nome_padronizado
        dados.append(df_variavel)

    df_final = pd.concat(dados, ignore_index=True)

    df_final["D1C"] = df_final["D1C"].astype(str)
    df_final["D3C"] = pd.to_numeric(df_final["D3C"], errors="coerce")
    df_final["V_numerico"] = df_final["V"].apply(converter_valor)

    resumo = (
        df_final.groupby(
            ["D2C", "D2N", "MN"],
            dropna=False
        )
        .agg(
            registros=("V", "count"),
            valores_disponiveis=(
                "V_numerico",
                lambda serie: serie.notna().sum()
            ),
            primeiro_ano=("D3C", "min"),
            ultimo_ano=("D3C", "max"),
        )
        .reset_index()
    )

    print("\nRESUMO DA EXTRAÇÃO")
    print(resumo.to_string(index=False))

    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

    df_final.to_csv(
        ARQUIVO_SAIDA,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nArquivo salvo em:\n{ARQUIVO_SAIDA.resolve()}")


if __name__ == "__main__":
    main()