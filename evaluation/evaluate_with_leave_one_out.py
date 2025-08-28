import random
import cbrkit
from multiprocessing import Pool, cpu_count
from constant.result_limit import RESULT_LIMIT


def evaluate_single_case(args):
    """Função auxiliar para avaliar um único caso - será executada em paralelo"""
    case_id_to_hold_out, casebase, similarity_func = args

    # Separa o caso de teste (holdout)
    holdout_case = casebase[case_id_to_hold_out]
    query = {"Literal_Ingredients_List": holdout_case["Processed_Ingredients"]}

    # Cria a base de casos para o teste (todos exceto o holdout)
    test_casebase = {
        key: value for key, value in casebase.items() if key != case_id_to_hold_out
    }

    retriever = cbrkit.retrieval.dropout(
        cbrkit.retrieval.build(
            similarity_func=similarity_func,
        ),
        limit=RESULT_LIMIT,
    )

    # Constrói o recuperador para a base de teste
    retrieved_result = cbrkit.retrieval.apply_query(
        casebase=test_casebase, query=query, retrievers=retriever
    )

    # Compara a solução
    matched_cases = retrieved_result.casebase.items()
    sim_sum = 0

    for key, case in matched_cases:
        sim_sum += retrieved_result.similarities[key]

    case_avg_sim = sim_sum / len(matched_cases) if matched_cases else 0
    return case_avg_sim


# Avaliar o sistema com o método Leave-One-Out
def evaluate_with_leave_one_out(
    casebase: cbrkit.loaders.pandas, similarity_func, sample_size: int = 100
) -> None:
    print(
        f"\n{'=' * 40} Iniciando Avaliação Leave-One-Out (amostra de {sample_size} casos) {'=' * 40}\n"
    )

    # Para não demorar muito, vamos testar com uma amostra da base de casos
    all_case_ids = list(casebase.keys())
    # Garante que o sample_size não é maior que a base de casos
    actual_sample_size = min(sample_size, len(all_case_ids))
    case_ids_to_test = random.sample(all_case_ids, actual_sample_size)

    # Preparar argumentos para cada processo
    args_list = [(case_id, casebase, similarity_func) for case_id in case_ids_to_test]

    # Usar paralelização com multiprocessing
    num_processes = min(cpu_count(), len(case_ids_to_test))
    print(f"Usando {num_processes} processos paralelos...")

    with Pool(processes=num_processes) as pool:
        similarities = pool.map(evaluate_single_case, args_list)

    # Calcular a média das similaridades
    sim_avg = sum(similarities) / len(similarities) if similarities else 0

    similarity_percentage = sim_avg * 100
    print("\n-------------------- Resultado da Avaliação (TF-IDF) --------------------")
    print(f"Casos testados: {actual_sample_size}")
    print(f"Similaridade Média das Recomendações: {similarity_percentage:.2f}%")
    print("-------------------------------------------------------------------------")