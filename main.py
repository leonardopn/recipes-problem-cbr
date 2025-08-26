from typing import TypedDict
from numpy import int64
import pandas as pd
from watchfiles import run_process
import cbrkit
import cbrkit.loaders
import ast

from sim_functions.custom_ingredient_similarity import custom_ingredient_similarity

# ===================================== CONSTANTES ===================================== #

DATASET_FILE = "./datasets/recipes.csv"
RESULT_LIMIT = 10

# ===================================== TYPES ===================================== #


class Recipe(TypedDict):
    Id: int64
    Title: str
    Ingredients: list[str]
    Instructions: str
    Image_Name: str
    Cleaned_Ingredients: str
    Cleaned_Ingredients_List: set[str]


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
        df = pd.read_csv(DATASET_FILE)
        return df
    except FileNotFoundError:
        print(
            "Erro: Arquivo CSV não encontrado. Verifique o nome e o caminho do arquivo."
        )
        exit()


# 2. Limpeza e preparação dos dados
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Removendo linhas com valores nulos nas colunas essenciais
    df.dropna(subset=["Cleaned_Ingredients"], inplace=True)
    df.dropna(subset=["Title"], inplace=True)

    # Convertendo a string de ingredientes em uma lista de verdade
    # Isso é importante para que a função de similaridade funcione corretamente
    df["Cleaned_Ingredients_List"] = df["Cleaned_Ingredients"].apply(
        lambda x: set(ast.literal_eval(str(x.lower())))
    )

    return df


# 3. Mapear o DataFrame para a estrutura do cbrkit
def map_dataframe_to_case_base(df: pd.DataFrame) -> cbrkit.loaders.pandas:
    # Mapeamos as colunas do DataFrame para os conceitos de "problema" e "solução"
    # O loader do cbrkit cuida da criação da base de casos
    return cbrkit.loaders.pandas(df)


# 4. Executar a recuperação de casos
def perform_retrieval(case_base: cbrkit.loaders.pandas) -> None:
    similarity_func = cbrkit.sim.attribute_value(
        attributes={"Cleaned_Ingredients_List": custom_ingredient_similarity},
        aggregator=cbrkit.sim.aggregator(pooling="mean"),
    )

    retriever = cbrkit.retrieval.dropout(
        cbrkit.retrieval.build(
            similarity_func=similarity_func,
        ),
        limit=RESULT_LIMIT,
    )

    query_ingredients = {"bread", "cheese", "lettuce", "meat", "tomato"}

    query = {"Cleaned_Ingredients_List": query_ingredients}
    print(f"Buscando receitas com os ingredientes: {query_ingredients}\n")

    retrieved_cases = cbrkit.retrieval.apply_query(
        casebase=case_base, query=query, retrievers=retriever
    )

    print(f"Top {RESULT_LIMIT} receitas recomendadas:")
    result = retrieved_cases.final_step
    matched_cases = result.casebase.items()
    position = 1

    for case in matched_cases:
        case_result = CaseResult(
            case=Recipe(**case[1]), similarity=result.similarities[case[0]].value
        )

        print(position, case_result.to_string())
        position += 1


# 5. Avaliar o sistema com o método Leave-One-Out
def evaluate_with_leave_one_out(
    casebase: cbrkit.loaders.pandas, sample_size: int = 100
) -> None:
    print(
        f"\n{'=' * 40} Iniciando Avaliação Leave-One-Out (amostra de {sample_size} casos) {'=' * 40}\n"
    )

    similarity_func = cbrkit.sim.attribute_value(
        attributes={"Cleaned_Ingredients_List": custom_ingredient_similarity},
        aggregator=cbrkit.sim.aggregator(pooling="mean"),
    )

    correct_predictions = 0

    # Para não demorar muito, vamos testar com uma amostra da base de casos
    # Pega os primeiros `sample_size` IDs da base de casos para o teste
    case_ids_to_test = list(casebase.keys())[:sample_size]

    for i, case_id_to_hold_out in enumerate(case_ids_to_test):
        print(f"Testando caso {i+1}/{sample_size}...")

        # a. Separa o caso de teste (holdout)
        holdout_case = casebase[case_id_to_hold_out]
        print(holdout_case)
        query = holdout_case["problem"]
        correct_solution_title = holdout_case["solution"]["Title"]

        # b. Cria a base de casos para o teste (todos exceto o holdout)
        test_casebase = {
            key: value for key, value in casebase.items() if key != case_id_to_hold_out
        }

        # c. Constrói o recuperador para a base de teste, buscando apenas o caso mais similar
        retriever = cbrkit.retrieval.build(
            casebase=test_casebase,
            similarity_func=similarity_func,
            limit=1,
        )

        # d. Executa a recuperação
        retrieved_result = retriever(query)

        # e. Compara a solução
        if retrieved_result:
            retrieved_case_id = next(iter(retrieved_result))
            retrieved_solution_title = retrieved_result[retrieved_case_id]["solution"][
                "Title"
            ]

            if retrieved_solution_title == correct_solution_title:
                correct_predictions += 1

    # f. Calcula e exibe a acurácia final
    accuracy = (correct_predictions / sample_size) * 100
    print("\n-------------------- Resultado da Avaliação --------------------")
    print(f"Casos testados: {sample_size}")
    print(f"Previsões corretas: {correct_predictions}")
    print(f"Acurácia do sistema: {accuracy:.2f}%")
    print("----------------------------------------------------------------")


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
    print(f"Número de casos na base: {len(case_base)}")
    print(f"Um caso de exemplo seria: {list(case_base.items())[0][1]}")

    # Passo 4: Executar a recuperação de casos
    perform_retrieval(case_base)

    # Passo 5: Avaliar o sistema com Leave-One-Out
    evaluate_with_leave_one_out(case_base, sample_size=200)

    print(f"\n{'=' * 40} Fim do processamento {'=' * 40}\n")


if __name__ == "__main__":
    run_process("./", target=main)
