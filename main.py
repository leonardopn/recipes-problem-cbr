import pandas as pd


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


# ===================================== APP ===================================== #


def main() -> None:
    load_data_set()


if __name__ == "__main__":
    main()
