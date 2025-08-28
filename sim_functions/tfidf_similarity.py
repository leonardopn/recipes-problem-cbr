import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from helpers.clean_data import normalize_ingredients


class TFIDFSimilarity:
    def __init__(self, casebase_df: pd.DataFrame):
        """
        Inicializa o calculador de similaridade TF-IDF.
        O modelo é treinado uma vez com todos os casos para aprender os pesos IDF.
        """
        print("Treinando o modelo TF-IDF...")
        # 1. Prepara o corpus (a coleção de todos os "documentos" de ingredientes)
        corpus = casebase_df["Ingredient_String"].tolist()

        # 2. Cria e treina o Vectorizer
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

        # 3. Mantém um mapeamento do ID original do DataFrame para o índice na matriz
        self.case_id_to_matrix_idx = {
            case_id: i for i, case_id in enumerate(casebase_df.index)
        }
        print("Modelo TF-IDF treinado com sucesso!")

    def __call__(self, case: dict, query: dict) -> float:
        """
        Permite que a instância da classe seja chamada como uma função,
        facilitando a integração com cbrkit.
        """
        case_id = case["Id"]
        query_ingredients = query["Literal_Ingredients_List"]

        # Normaliza os ingredientes da query
        processed_query = normalize_ingredients(query_ingredients)
        query_string = " ".join(processed_query)

        # Transforma a query em um vetor TF-IDF usando o vocabulário já aprendido
        query_vector = self.vectorizer.transform([query_string])

        # Recupera o vetor pré-calculado do caso
        matrix_idx = self.case_id_to_matrix_idx.get(case_id)
        if matrix_idx is None:
            return 0.0
        case_vector = self.tfidf_matrix[matrix_idx]

        # Calcula a similaridade de cossenos
        similarity = cosine_similarity(query_vector, case_vector)

        # A similaridade é retornada dentro de uma matriz, então pegamos o primeiro valor
        return similarity[0][0]