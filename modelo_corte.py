# bb_cutting_with_model.py
# Branch-and-Bound manual para Corte Unidimensional (relaxações com GLOP)
# Usa apenas OR-Tools + bibliotecas padrão do Python.
#
# Execute: python bb_cutting_with_model.py
# Requisito: pip install ortools

from ortools.linear_solver import pywraplp
import math
import sys

# Tolerâncias
EPS = 1e-7
INT_TOL = 1e-6

class Node:
    def __init__(self, node_id, parent_id):
        self.id = node_id
        self.parent_id = parent_id
        self.lower_bounds = {}  # {var_index: lb}
        self.upper_bounds = {}  # {var_index: ub}
    def create_child(self, new_id):
        c = Node(new_id, self.id)
        c.lower_bounds = self.lower_bounds.copy()
        c.upper_bounds = self.upper_bounds.copy()
        return c

def generate_maximal_patterns(bar_length, sizes):
    """
    Gera padrões 'máximos': padrões onde não é possível adicionar mais nenhum item sem exceder bar_length.
    Retorna lista de padrões; cada padrão é lista de contagens por item (na ordem sizes).
    """
    m = len(sizes)
    patterns = []
    current = [0] * m

    def backtrack(i, current_len):
        if i == m:
            if current_len == 0:
                return
            # maximalidade: se existe algum item que possa ser adicionado, então não é maximal
            for k in range(m):
                if current_len + sizes[k] <= bar_length + EPS:
                    # existe espaço para adicionar mais um item k => não maximal
                    return
            patterns.append(current.copy())
            return

        max_ci = (bar_length - current_len) // sizes[i]
        for c in range(int(max_ci) + 1):
            current[i] = c
            new_len = current_len + c * sizes[i]
            if new_len <= bar_length + EPS:
                backtrack(i + 1, new_len)
        current[i] = 0

    backtrack(0, 0)
    return patterns

def solve_lp_relaxation(patterns, wastes, demands, node_bounds):
    """
    Monta e resolve o LP relaxado com GLOP.
    node_bounds: {'lb': {idx: lb}, 'ub': {idx: ub}}
    Retorna: status, obj_value, var_values
    """
    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver:
        raise RuntimeError("GLOP não disponível")

    infinity = solver.infinity()
    num_vars = len(patterns)
    vars = [solver.NumVar(0.0, infinity, f"x{p}") for p in range(num_vars)]

    # aplicar bounds
    for idx, lb in node_bounds.get('lb', {}).items():
        vars[idx].SetLb(float(lb))
    for idx, ub in node_bounds.get('ub', {}).items():
        vars[idx].SetUb(float(ub))

    # restrições: sum_p pattern[p][j] * x_p >= demand[j]
    for j, demand_j in enumerate(demands):
        ct = solver.Constraint(demand_j, infinity, f"dem_{j}")
        for p, pattern in enumerate(patterns):
            coeff = pattern[j]
            if coeff != 0:
                ct.SetCoefficient(vars[p], coeff)

    # objetivo: minimizar desperdício
    objective = solver.Objective()
    for p, w in enumerate(wastes):
        objective.SetCoefficient(vars[p], w)
    objective.SetMinimization()

    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        obj = objective.Value()
        vals = [v.solution_value() for v in vars]
    else:
        obj = None
        vals = None
    return status, obj, vals

def is_integer_solution(values):
    for v in values:
        if abs(v - round(v)) > INT_TOL:
            return False
    return True

def find_fractional_var(values):
    # retorna índice e valor da variável com maior parte fracionária (se houver)
    max_frac = 0.0
    idx = -1
    val = 0.0
    for i, v in enumerate(values):
        frac = abs(v - round(v))
        if frac > max_frac + 1e-12 and frac > INT_TOL:
            max_frac = frac
            idx = i
            val = v
    return idx, val

def write_model_to_log(log, patterns, wastes, demands):
    # matriz A: linhas = itens, colunas = padrões
    num_p = len(patterns)
    num_j = len(demands)
    log.write("----- MODELO (matriz A, b, c) -----\n")
    log.write(f"Itens (m): {num_j}, Padrões (n): {num_p}\n")
    log.write("Matriz A (cada linha = coeficientes do padrão para um item):\n")
    for j in range(num_j):
        row = [str(patterns[p][j]) for p in range(num_p)]
        log.write(" ".join(row) + "\n")
    log.write("b (demandas):\n")
    log.write(" ".join(str(int(d)) for d in demands) + "\n")
    log.write("c (custos/desperdícios por padrão):\n")
    log.write(" ".join(str(int(w)) for w in wastes) + "\n")
    log.write("-----------------------------------\n\n")
    log.flush()

def branch_and_bound(patterns, wastes, demands, log_path="logs.txt"):
    num_vars = len(patterns)
    best_obj = float('inf')
    best_solution = None
    nodes_processed = 0
    node_id_counter = 0

    log = open(log_path, "w", encoding="utf-8")
    log.write("ID | PAI | STATUS     | OBJETIVO     | VARS_RELAXADAS\n")
    log.flush()

    # grava modelo no log
    write_model_to_log(log, patterns, wastes, demands)

    # DFS stack
    stack = []
    root = Node(0, -1)
    stack.append(root)
    node_id_counter = 0

    while stack:
        node = stack.pop()
        nodes_processed += 1

        node_bounds = {'lb': node.lower_bounds, 'ub': node.upper_bounds}
        try:
            status, obj, vals = solve_lp_relaxation(patterns, wastes, demands, node_bounds)
        except Exception as e:
            log.write(f"{node.id} | {node.parent_id} | ERROR | {e}\n")
            continue

        status_map = {
            pywraplp.Solver.OPTIMAL: "OPTIMAL",
            pywraplp.Solver.FEASIBLE: "FEASIBLE",
            pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
            pywraplp.Solver.ABNORMAL: "ABNORMAL",
            pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED"
        }
        status_str = status_map.get(status, "UNKNOWN")

        if obj is None:
            log.write(f"{node.id} | {node.parent_id} | {status_str:<10} | {'---':<12} | ---\n")
            log.flush()
            continue

        vals_str = " ".join(f"{v:.6f}" for v in vals)
        log.write(f"{node.id} | {node.parent_id} | {status_str:<10} | {obj:<12.6f} | [{vals_str}]\n")
        log.flush()

        # poda 1: inviabilidade
        if status == pywraplp.Solver.INFEASIBLE or status == pywraplp.Solver.ABNORMAL:
            continue

        # poda 2: bound
        if best_solution is not None and obj >= best_obj - EPS:
            continue

        # poda 3: integrality
        if is_integer_solution(vals):
            int_obj = obj
            if int_obj < best_obj - EPS:
                best_obj = int_obj
                best_solution = [int(round(v)) for v in vals]
                log.write(f"--> NOVA INCUMBENTE (node {node.id}) obj={best_obj:.6f} sol={best_solution}\n")
                log.flush()
            continue

        # branching
        frac_idx, frac_val = find_fractional_var(vals)
        if frac_idx == -1:
            continue

        floor_val = math.floor(frac_val)
        ceil_val = floor_val + 1

        # criar filhos
        node_id_counter += 1
        right = node.create_child(node_id_counter)
        right.lower_bounds[frac_idx] = ceil_val

        node_id_counter += 1
        left = node.create_child(node_id_counter)
        left.upper_bounds[frac_idx] = floor_val

        # empilha: primeiro right depois left para processar left primeiro (LIFO)
        stack.append(right)
        stack.append(left)

    log.write("\n--- FIM B&B ---\n")
    log.write(f"Nodes processados: {nodes_processed}\n")
    if best_solution is None:
        log.write("Nenhuma solução inteira encontrada.\n")
    else:
        log.write(f"Melhor solução inteira: obj={best_obj} sol={best_solution}\n")
    log.close()

    return best_obj, best_solution, nodes_processed

def main():
    print("=== Branch-and-Bound (GLOP) - Corte Unidimensional ===")
    try:
        bar_length = int(input("Comprimento da barra (ex: 150): ").strip())
        m = int(input("Número de tipos de itens (ex: 3): ").strip())
        sizes = []
        demands = []
        for i in range(m):
            s = int(input(f"Tamanho do item {i+1} (ex: 80): ").strip())
            sizes.append(s)
        for i in range(m):
            d = int(input(f"Demanda do item {i+1} (ex: 70): ").strip())
            demands.append(d)
    except Exception as e:
        print("Entrada inválida:", e)
        sys.exit(1)

    patterns = generate_maximal_patterns(bar_length, sizes)
    if not patterns:
        print("Nenhum padrão gerado. Verifique os tamanhos e comprimento da barra.")
        sys.exit(1)

    wastes = []
    pattern_lengths = []
    for p in patterns:
        length = sum(p[j] * sizes[j] for j in range(len(sizes)))
        pattern_lengths.append(length)
        wastes.append(bar_length - length)

    print("\nPadrões gerados (contagens por item) e desperdício:")
    for idx, p in enumerate(patterns):
        print(f"p{idx+1}: {p}  | comprimento={pattern_lengths[idx]}  desperdicio={wastes[idx]}")

    print("\nResumo do modelo gerado. O modelo completo está em logs.txt")
    print("Iniciando Branch-and-Bound (GLOP). Veja logs.txt para todos os nós.\n")

    best_obj, best_solution, nodes = branch_and_bound(patterns, wastes, demands, log_path="logs.txt")

    if best_solution is None:
        print("Nenhuma solução inteira encontrada.")
    else:
        total_waste = sum(best_solution[i] * wastes[i] for i in range(len(best_solution)))
        print(f"Melhor solução inteira (x por padrão): {best_solution}")
        print(f"Custo (desperdício) reportado pelo LP relaxado da incumbente: {best_obj:.6f}")
        print(f"Desperdício total (inteiro calculado): {total_waste}")
        print(f"Nós processados: {nodes}")
        print("Log completo salvo em: logs.txt")

if __name__ == "__main__":
    main()
