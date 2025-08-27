from helpers.clean_data import normalize_ingredients


def custom_ingredient_similarity(case: set[str], ingredients: set[str]) -> float:
    """
    Calcula a similaridade entre dois conjuntos de ingredientes
    usando o Índice de Jaccard.
    """
    # Normaliza os dois conjuntos para garantir uma comparação justa
    norm_case = normalize_ingredients(case)
    norm_user = normalize_ingredients(ingredients)

    if not norm_case or not norm_user:
        return 0.0

    # Calcula a intersecção (ingredientes em comum)
    intersection = norm_case.intersection(norm_user)

    # Calcula a união (todos os ingredientes únicos somados)
    union = norm_case.union(norm_user)

    if len(union) == 0:
        return 0.0

    return len(intersection) / len(union)
