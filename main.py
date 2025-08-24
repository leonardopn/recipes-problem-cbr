from typing import TypedDict
from numpy import int64
import pandas as pd
from watchfiles import run_process
import cbrkit
import cbrkit.loaders
import ast

DATASET_FILE = "./datasets/recipes.csv"


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


# 4. Executar a recuperação de casos
def perform_retrieval(casebase: cbrkit.loaders.pandas) -> None:
    # 1. Definir a função de similaridade local para os ingredientes
    # Usaremos a similaridade de Jaccard, ideal para comparar conjuntos.
    ingredients_similarity = cbrkit.sim.collections.jaccard[set[str]]()

    # 2. Definir a similaridade global
    # Como nosso "problema" tem apenas um atributo ('Cleaned_Ingredients_List'),
    # a similaridade global será a mesma que a local.
    # O `attribute_value` nos ajuda a aplicar a função à chave correta.
    similarity_func = cbrkit.sim.attribute_value(
        attributes={"Cleaned_Ingredients_List": ingredients_similarity},
        aggregator=cbrkit.sim.aggregator(pooling="mean"),
    )

    # 3. Construir o recuperador (retriever)
    # Ele usará nossa função de similaridade e nos retornará os 5 casos mais similares.
    retriever = cbrkit.retrieval.dropout(
        cbrkit.retrieval.build(
            similarity_func=similarity_func,
        ),
        limit=5,
    )

    # 4. Criar uma consulta (query)
    # Vamos simular um usuário que tem frango, cebola, alho e tomate.
    query_ingredients = {"onion", "garlic", "chicken", "tomato"}
    query = {"Cleaned_Ingredients_List": query_ingredients}
    print(f"Buscando receitas com os ingredientes: {query_ingredients}\n")

    # 5. Executar a recuperação
    retrieved_cases = cbrkit.retrieval.apply_query(
        casebase=casebase, query=query, retrievers=retriever
    )

    # 6. Exibir os resultados (Reúso da Solução)
    print("Top 5 receitas recomendadas:")
    result = retrieved_cases.final_step

    for case in result.casebase.items():
        case_result = CaseResult(
            case=Recipe(**case[1]), similarity=result.similarities[case[0]].value
        )

        print(case_result.to_string())


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

    # Passo 4: Executar a recuperação de casos
    perform_retrieval(casebase)

    print(f"\n{'=' * 40} Fim do processamento {'=' * 40}\n")


if __name__ == "__main__":
    run_process("./", target=main)
