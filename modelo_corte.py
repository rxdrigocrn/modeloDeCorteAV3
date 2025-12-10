# bb_cutting_with_model.py
# Branch-and-Bound manual para Corte Unidimensional (relaxações com GLOP)
# Usa apenas OR-Tools + bibliotecas padrão (built-ins).

from ortools.linear_solver import pywraplp

# Tolerâncias
EPS = 1e-7
INT_TOL = 1e-6

# --- CLASSE NODE ---
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

# --- GERAÇÃO DE PADRÕES ---
def generate_maximal_patterns(bar_length, sizes):
    """
    Gera padrões 'máximos' usando backtracking (recursão).
    """
    m = len(sizes)
    patterns = []
    current = [0] * m

    def backtrack(i, current_len):
        if i == m:
            if current_len == 0:
                return
            # Verifica se é maximal
            for k in range(m):
                if current_len + sizes[k] <= bar_length + EPS:
                    return
            patterns.append(current.copy())
            return

        # Quantos do item 'i' cabem no espaço restante?
        max_ci = (bar_length - current_len) // sizes[i]
        
        for c in range(int(max_ci) + 1):
            current[i] = c
            new_len = current_len + c * sizes[i]
            if new_len <= bar_length + EPS:
                backtrack(i + 1, new_len)
        current[i] = 0

    backtrack(0, 0)
    return patterns

# --- IMPRESSÃO DO MODELO MATEMÁTICO (NOVO) ---
def print_lp_model(patterns, wastes, demands):
    print("\n================ MODELO DE PROGRAMAÇÃO LINEAR INTEIRA ================")
    print("Sendo xi a quantidade produzida de cada padrão i:\n")

    # 1. Função Objetivo
    print("minimizar")
    obj_terms = []
    for i, w in enumerate(wastes):
        # Mostra o custo mesmo que seja 0, para ficar claro
        obj_terms.append(f"{w}x{i}")
    print("  " + " + ".join(obj_terms))
    print()

    # 2. Restrições
    print("sujeito a")
    for j, demand in enumerate(demands):
        lhs_terms = [] # Lado esquerdo da equação
        for p_idx, pattern in enumerate(patterns):
            coeff = pattern[j]
            # Formata como "1x0", "0x1", etc.
            lhs_terms.append(f"{coeff}x{p_idx}")
        
        lhs_str = " + ".join(lhs_terms)
        print(f"  {lhs_str}  >=  {demand}  (Item {j+1})")

    print()
    # 3. Variáveis
    var_names = [f"x{i}" for i in range(len(patterns))]
    print(f"  {', '.join(var_names)} >= 0 e inteiras")
    print("=======================================================================\n")

# --- RESOLUÇÃO LP COM GLOP ---
def solve_lp_relaxation(patterns, wastes, demands, node_bounds):
    """
    Resolve a relaxação linear usando GLOP (OR-Tools).
    """
    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver:
        return None, None, None

    infinity = solver.infinity()
    num_vars = len(patterns)
    vars = [solver.NumVar(0.0, infinity, f"x{p}") for p in range(num_vars)]

    # Aplica bounds do Branch-and-Bound
    for idx, lb in node_bounds.get('lb', {}).items():
        vars[idx].SetLb(float(lb))
    for idx, ub in node_bounds.get('ub', {}).items():
        vars[idx].SetUb(float(ub))

    # Restrições de demanda
    for j, demand_j in enumerate(demands):
        ct = solver.Constraint(demand_j, infinity, f"dem_{j}")
        for p, pattern in enumerate(patterns):
            coeff = pattern[j]
            if coeff != 0:
                ct.SetCoefficient(vars[p], coeff)

    # Objetivo: Minimizar desperdício
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

# --- VALIDAÇÕES ---
def is_integer_solution(values):
    for v in values:
        if abs(v - round(v)) > INT_TOL:
            return False
    return True

def find_fractional_var(values):
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

# --- BRANCH-AND-BOUND PRINCIPAL ---
def branch_and_bound(patterns, wastes, demands, log_path="logs.txt"):
    num_vars = len(patterns)
    best_obj = float('inf')
    best_solution = None
    nodes_processed = 0
    node_id_counter = 0

    log = open(log_path, "w", encoding="utf-8")
    
    log.write("--- LOG DE EXECUÇÃO BRANCH AND BOUND ---\n")
    log.write("Padroes (Legenda):\n")
    for idx, p in enumerate(patterns):
        log.write(f"  x{idx}: {p} (Desperdicio: {wastes[idx]})\n")
    log.write("-" * 60 + "\n")
    log.write(f"{'ID_NO':<6} | {'CUSTO(OBJ)':<12} | {'VARIAVEIS [x0, x1...]'}\n")
    log.write("-" * 60 + "\n")
    log.flush()

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
            log.write(f"{node.id:<6} | ERROR        | {e}\n")
            continue

        if obj is None:
            log.write(f"{node.id:<6} | {'Inviavel':<12} | ---\n")
        else:
            vals_str = ", ".join([f"{v:.2f}" for v in vals])
            log.write(f"{node.id:<6} | {obj:<12.4f} | [{vals_str}]\n")
        log.flush()

        if status == pywraplp.Solver.INFEASIBLE or status == pywraplp.Solver.ABNORMAL:
            continue

        if best_solution is not None and obj >= best_obj - EPS:
            continue

        if is_integer_solution(vals):
            int_obj = obj
            if int_obj < best_obj - EPS:
                best_obj = int_obj
                best_solution = [int(round(v)) for v in vals]
                log.write(f"      >>> NOVA SOLUCAO INTEIRA: obj={best_obj:.4f} sol={best_solution} <<<\n")
                log.flush()
            continue

        frac_idx, frac_val = find_fractional_var(vals)
        if frac_idx == -1:
            continue

        floor_val = int(frac_val) 
        ceil_val = floor_val + 1

        node_id_counter += 1
        right = node.create_child(node_id_counter)
        right.lower_bounds[frac_idx] = ceil_val

        node_id_counter += 1
        left = node.create_child(node_id_counter)
        left.upper_bounds[frac_idx] = floor_val

        stack.append(right)
        stack.append(left)

    log.write("-" * 60 + "\n")
    log.write(f"FIM. Nodes processados: {nodes_processed}\n")
    if best_solution:
        log.write(f"Melhor solucao inteira final: {best_solution} com custo {best_obj}\n")
    else:
        log.write("Nenhuma solucao inteira encontrada.\n")
    log.close()

    return best_obj, best_solution, nodes_processed

# --- MAIN ---
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
        return

    # Gera padrões
    patterns = generate_maximal_patterns(bar_length, sizes)
    if not patterns:
        print("Nenhum padrão gerado. Verifique os tamanhos e comprimento da barra.")
        return

    # Calcula desperdício
    wastes = []
    pattern_lengths = []
    for p in patterns:
        length = sum(p[j] * sizes[j] for j in range(len(sizes)))
        pattern_lengths.append(length)
        wastes.append(bar_length - length)

    print("\nPadrões gerados (contagens por item) e desperdício:")
    for idx, p in enumerate(patterns):
        print(f"p{idx}: {p}  | comprimento={pattern_lengths[idx]}  desperdicio={wastes[idx]}")

    print_lp_model(patterns, wastes, demands)   
 
    print("Iniciando Branch-and-Bound (GLOP)...")
    
    # Executa B&B
    best_obj, best_solution, nodes = branch_and_bound(patterns, wastes, demands, log_path="logs.txt")

    # Exibe resultados
    if best_solution is None:
        print("Nenhuma solução inteira encontrada.")
    else:
        total_waste = sum(best_solution[i] * wastes[i] for i in range(len(best_solution)))
        print(f"\nMelhor solução inteira (x por padrão): {best_solution}")
        print(f"Custo (desperdício) reportado pelo LP: {best_obj:.6f}")
        print(f"Desperdício total real: {total_waste}")
        print(f"Nós processados: {nodes}")
        print("Log limpo salvo em: logs.txt")

if __name__ == "__main__":
    main()