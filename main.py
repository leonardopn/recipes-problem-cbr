import pandas as pd
import asyncio
from watchfiles import awatch, run_process


DATASET_FILE = "./datasets/recipes.csv"


# 1. Carregar o dataset
# Certifique-se de que o nome do arquivo corresponde ao que você baixou.
# Se o arquivo estiver em outra pasta, coloque o caminho completo.
def load_data_set() -> pd.DataFrame:
    try:
        df = pd.read_csv(DATASET_FILE)
        print(df.head())
        return df
    except FileNotFoundError:
        print(
            "Erro: Arquivo CSV não encontrado. Verifique o nome e o caminho do arquivo."
        )
        exit()


# 2. Limpeza e preparação dos dados (uma etapa inicial)
# Vamos remover receitas que não têm ingredientes listados
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df.dropna(subset=["Cleaned_Ingredients"], inplace=True)
    # Para este exemplo, vamos trabalhar com uma amostra menor para ser mais rápido
    df_sample = df.sample(n=5000, random_state=42)
    return df_sample


# ===================================== APP ===================================== #


def main() -> None:
    print(f"{'=' * 80}Iniciando o processamento...{'=' * 80}\n")

    df = load_data_set()
    df = clean_data(df)

    print(f"\n{'=' * 80} Fim do processamento {'=' * 80}\n")


if __name__ == "__main__":
    run_process("./", target=main)
