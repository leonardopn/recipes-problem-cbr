def custom_ingredient_similarity(case: set[str], ingredients: set[str]) -> float:
    if len(ingredients) == 0 or len(case) == 0:
        return 0

    points = 0
    faults = 0

    # Primeiro, verificar se algum dos ingredientes estão dentro da receita caso
    for ingredient in ingredients:
        joined_case_str = " ".join(case)
        if ingredient in joined_case_str:
            points += 1

    # Depois, se algum ingrediente do caso base não estiver na lista de ingredientes, adicionamos uma falta
    for ingredient in case:
        # Precisamos tokenizar pois os ingredientes nos casos estão da seguinte forma: "uma fatia de pão"
        # ou seja, se dentro da sentença, nenhuma das palavras estiver na lista de ingredientes, adicionamos uma falta
        # Fazemos isso, pois entendemos que se a receita precisar de mais itens, ela pode não funcionar corretamente
        tokens = ingredient.split()
        is_in = False
        for token in tokens:
            if token in ingredients:
                is_in = True
                break
        if not is_in:
            faults += 1

    return points / (len(ingredients) + faults)
