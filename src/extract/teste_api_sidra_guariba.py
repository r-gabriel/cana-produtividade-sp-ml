from pathlib import Path

import pandas as pd
import requests


# ============================================================
# CONFIGURAÇÕES
# ============================================================

TABELA_PAM = "5457"
CODIGO_GUARIBA = "3518602"
CODIGO_CANA_DE_ACUCAR = "40106"

ANOS = list(range(2000, 2025))

VARIAVEIS_PAM = {
    "8331": "area_plantada_ha",
    "216": "area_colhida_ha",
    "214": "producao_ton",
    "112": "rendimento_kg_ha",
    "215": "valor_producao",
}

PASTA_SAIDA = Path("data/raw/agricola")
ARQUIVO_SAIDA = PASTA_SAIDA / "pam_cana_guariba_api_raw.csv"


# ============================================================
# FUNÇÕES
# ============================================================

def consultar_variavel(codigo_variavel: str) -> pd.DataFrame:
    """
    Consulta uma variável da PAM para Guariba-SP e cana-de-açúcar.
    """

    anos_consulta = ",".join(map(str, ANOS))

    url = (
        "https://apisidra.ibge.gov.br/values/"
        f"t/{TABELA_PAM}/"
        f"n6/{CODIGO_GUARIBA}/"
        f"v/{codigo_variavel}/"
        f"p/{anos_consulta}/"
        f"c782/{CODIGO_CANA_DE_ACUCAR}"
    )

    print(f"\nConsultando variável {codigo_variavel}...")
    print(url)

    resposta = requests.get(url, timeout=120)

    if resposta.status_code != 200:
        print(f"Erro HTTP {resposta.status_code}")
        print(resposta.text[:1000])
        return pd.DataFrame()

    dados = resposta.json()
    df = pd.DataFrame(dados)

    if df.empty:
        return pd.DataFrame()

    # Remove a primeira linha descritiva retornada pela API.
    df = df[df["NC"] != "Nível Territorial (Código)"].copy()

    return df


def converter_valor(valor):
    """
    Converte valores textuais do SIDRA em número.
    """

    if pd.isna(valor):
        return pd.NA

    valor = str(valor).strip()

    if valor in {"...", "-", ""}:
        return pd.NA

    valor = valor.replace(".", "").replace(",", ".")

    return pd.to_numeric(valor, errors="coerce")


# ============================================================
# EXECUÇÃO
# ============================================================

def main():
    dataframes = []

    for codigo_variavel, nome_padronizado in VARIAVEIS_PAM.items():
        df_variavel = consultar_variavel(codigo_variavel)

        if df_variavel.empty:
            print(
                f"Nenhum registro retornado para a variável: "
                f"{nome_padronizado}"
            )
            continue

        df_variavel["variavel_padronizada"] = nome_padronizado
        dataframes.append(df_variavel)

    if not dataframes:
        raise RuntimeError(
            "Nenhuma variável retornou dados. "
            "Verifique os parâmetros da consulta."
        )

    df_final = pd.concat(dataframes, ignore_index=True)

    df_final["D1C"] = df_final["D1C"].astype(str)
    df_final["D3C"] = pd.to_numeric(df_final["D3C"], errors="coerce")
    df_final["V_numerico"] = df_final["V"].apply(converter_valor)

    print("\nResumo dos dados retornados:")
    print(
        df_final.groupby(
            ["D2C", "D2N", "MN"],
            dropna=False
        )
        .agg(
            registros=("V", "count"),
            valores_numericos=(
                "V_numerico",
                lambda serie: serie.notna().sum()
            ),
            primeiro_ano=("D3C", "min"),
            ultimo_ano=("D3C", "max"),
        )
        .reset_index()
        .to_string(index=False)
    )

    print("\nAmostra dos dados:")
    print(
        df_final[
            [
                "D1C", "D1N",
                "D2C", "D2N",
                "D3C", "D3N",
                "MN", "V",
                "V_numerico"
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

    df_final.to_csv(
        ARQUIVO_SAIDA,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nArquivo salvo em:\n{ARQUIVO_SAIDA.resolve()}")


if __name__ == "__main__":
    main()