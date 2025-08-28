import ast
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
import re


try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# --- Configuração inicial do NLTK (melhor fazer fora da função) ---
# Carrega as stopwords em inglês para um conjunto, que é mais rápido para buscas
ENGLISH_STOP_WORDS = set(stopwords.words("english"))
# Adiciona palavras comuns de receitas que não são stopwords padrão
CUSTOM_STOP_WORDS = {
    "cup",
    "cups",
    "teaspoon",
    "teaspoons",
    "tablespoon",
    "tablespoons",
    "oz",
    "ml",
    "g",
    "kg",
    "pinch",
    "taste",
}
FINAL_STOP_WORDS = ENGLISH_STOP_WORDS.union(CUSTOM_STOP_WORDS)

# Inicializa o Stemmer
stemmer = PorterStemmer()


def normalize_ingredients(ingredient_list: str | set[str]) -> set[str]:
    """
    Função auxiliar que processa uma única string de lista de ingredientes.
    Ex: '["1 cup of chopped tomatoes", "salt to taste"]'
    """
    processed_keywords = set()

    if isinstance(ingredient_list, str):
        # 1. Converte a string para uma lista de Python de forma segura
        try:
            ingredient_phrases = ast.literal_eval(str(ingredient_list).lower())
        except (ValueError, SyntaxError):
            # Se a string não for uma lista válida, retorna um conjunto vazio
            return processed_keywords
    else:
        ingredient_phrases = ingredient_list

    for phrase in ingredient_phrases:
        # 2. Limpa a frase: remove tudo que não for letra
        phrase = re.sub(r"[^a-z\s]", "", phrase)

        # 3. Tokeniza: quebra a frase em palavras
        tokens = phrase.split()

        for token in tokens:
            # 4. Remove stopwords e aplica o stemming
            if token not in FINAL_STOP_WORDS:
                # Reduz a palavra ao seu radical (ex: 'tomatoes' -> 'tomato')
                stemmed_token = stemmer.stem(token)
                if stemmed_token:  # Garante que não adiciona strings vazias
                    processed_keywords.add(stemmed_token)

    return processed_keywords


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Função de limpeza, com remoção de stopwords e stemming.
    """
    # Evita o uso de inplace=True para maior segurança e previsibilidade
    df_cleaned = df.replace("", pd.NA).dropna(how="any").copy()

    # Aplica a função de processamento robusta na coluna de ingredientes
    df_cleaned["Processed_Ingredients"] = df_cleaned[
        "Literal_Ingredients_List"
    ].apply(normalize_ingredients)

    # Cria uma nova coluna com os ingredientes como uma única string para o TF-IDF
    df_cleaned["Ingredient_String"] = df_cleaned["Processed_Ingredients"].apply(
        lambda s: " ".join(s)
    )

    return df_cleaned