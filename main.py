from typing import TypedDict
from numpy import int64
import pandas as pd
from watchfiles import run_process
import cbrkit
import cbrkit.loaders
from constant.result_limit import RESULT_LIMIT
from evaluation.evaluate_with_leave_one_out import evaluate_with_leave_one_out
from helpers.clean_data import clean_data

# Importa a nova classe de similaridade TF-IDF
from sim_functions.tfidf_similarity import TFIDFSimilarity


# ===================================== CONSTANTES ===================================== #

DATASET_FILE = "./datasets/recipes.csv"

# ===================================== TYPES ===================================== #


class Recipe(TypedDict):
    Id: int64
    Title: str
    Ingredients: list[str]
    Instructions: str
    Image_Name: str
    Cleaned_Ingredients: list[str]
    Literal_Ingredients_List: set[str]
    Processed_Ingredients: set[str]  # Adicionado para clareza
    Ingredient_String: str  # Adicionado para clareza


class CaseResult:
    def __init__(self, case: Recipe, similarity: float):
        self.case = case
        self.similarity = similarity

    def to_string(self) -> str:
        return f"{self.case['Title']} (ID: {self.case['Id']}, Similaridade: {(self.similarity*100):.2f}%)"


# ===================================== OPERAÇÕES ===================================== #


# 1. Carregar o dataset
def load_data_set() -> pd.DataFrame:
    try:
        df = pd.read_csv(DATASET_FILE)  # Usar Id como índice
        return df
    except FileNotFoundError:
        print(
            "Erro: Arquivo CSV não encontrado. Verifique o nome e o caminho do arquivo."
        )
        exit()


# 3. Mapear o DataFrame para a estrutura do cbrkit
def map_dataframe_to_case_base(df: pd.DataFrame) -> cbrkit.loaders.pandas:
    # O loader do cbrkit usa o índice do DataFrame como chave do caso
    return cbrkit.loaders.pandas(df)


# 4. Executar a recuperação de casos
def perform_retrieval(case_base: cbrkit.loaders.pandas, similarity_func) -> None:
    retriever = cbrkit.retrieval.dropout(
        cbrkit.retrieval.build(
            similarity_func=similarity_func,
        ),
        limit=RESULT_LIMIT,
    )

    query_ingredients = {
        "cup of chopped tomatoes",
        "salt to taste",
        "cup of olive oil",
        "cloves of garlic",
        "teaspoon of sugar",
    }

    query = {
        "Literal_Ingredients_List": query_ingredients,
    }
    print(f"Buscando receitas com os ingredientes: {query_ingredients}\n")

    retrieved_cases = cbrkit.retrieval.apply_query(
        casebase=case_base, query=query, retrievers=retriever
    )

    print(f"Top {RESULT_LIMIT} receitas recomendadas:")
    matched_cases = retrieved_cases.casebase.items()
    position = 1

    for key, case in matched_cases:
        case_result = CaseResult(
            case=Recipe(**case), similarity=retrieved_cases.similarities[key]
        )

        print(position, case_result.to_string())
        position += 1



# ===================================== APP ===================================== #


def main() -> None:
    print(f"{'=' * 40} Iniciando o processamento... {'=' * 40}\n")

    # Passos 1 e 2: Carregar e limpar os dados
    df = load_data_set()
    df_cleaned = clean_data(df)
    print(f"Dataset carregado e limpo. Usando {len(df_cleaned)} receitas como amostra.")

    # Passo 3: Criar a base de casos usando o loader
    case_base = map_dataframe_to_case_base(df_cleaned)
    print("Base de casos criada com sucesso!")

    # Passo 4: Inicializar o modelo de similaridade TF-IDF
    # O modelo é treinado uma vez com todos os dados
    tfidf_similarity_calculator = TFIDFSimilarity(df_cleaned)

    # Passo 5: Executar a recuperação de casos
    perform_retrieval(case_base, tfidf_similarity_calculator)

    # Passo 6: Avaliar o sistema com Leave-One-Out
    # evaluate_with_leave_one_out(case_base, tfidf_similarity_calculator, sample_size=200)

    print(f"\n{'=' * 40} Fim do processamento {'=' * 40}\n")


if __name__ == "__main__":
    run_process("./", target=main)