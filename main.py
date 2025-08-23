import pandas as pd
from watchfiles import run_process
import cbrkit
import cbrkit.loaders
import ast

DATASET_FILE = "./datasets/recipes.csv"


# 1. Carregar o dataset
def load_data_set() -> pd.DataFrame:
    try:
        df = pd.read_csv(DATASET_FILE)
        return df
    except FileNotFoundError:
        print(
            "Erro: Arquivo CSV não encontrado. Verifique o nome e o caminho do arquivo."
        )
        exit()


# 2. Limpeza e preparação dos dados
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df.dropna(subset=["Cleaned_Ingredients"], inplace=True)
    # Convertendo a string de ingredientes em uma lista de verdade
    # Isso é importante para que a função de similaridade funcione corretamente
    df["Cleaned_Ingredients_List"] = df["Cleaned_Ingredients"].apply(
        lambda x: set(ast.literal_eval(x))
    )
    df_sample = df.sample(n=5000, random_state=42)
    return df_sample


# 3. Mapear o DataFrame para a estrutura do cbrkit
def map_dataframe_to_casebase(df: pd.DataFrame) -> cbrkit.loaders.pandas:
    # Mapeamos as colunas do DataFrame para os conceitos de "problema" e "solução"
    # O loader do cbrkit cuida da criação da base de casos
    casebase = cbrkit.loaders.pandas(df)
    return casebase


# ===================================== APP ===================================== #


def main() -> None:
    print(f"{'=' * 40} Iniciando o processamento... {'=' * 40}\n")

    # Passos 1 e 2: Carregar e limpar os dados
    df = load_data_set()
    df_cleaned = clean_data(df)
    print(f"Dataset carregado e limpo. Usando {len(df_cleaned)} receitas como amostra.")

    # Passo 3: Criar a base de casos usando o loader
    casebase = map_dataframe_to_casebase(df_cleaned)
    print("Base de casos criada com sucesso!")
    print(f"Número de casos na base: {len(casebase)}")

    # Exibir um caso de exemplo para verificação
    if casebase:
        first_case_key = next(iter(casebase.keys()))
        print("\nExemplo de um caso na base:")
        print(casebase[first_case_key])

    print(f"\n{'=' * 40} Fim do processamento {'=' * 40}\n")


if __name__ == "__main__":
    run_process("./", target=main)
