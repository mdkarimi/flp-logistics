# flp_logistics.py (Updated to Load CSV Files)
import numpy as np
import pulp
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import os

# Fuzzy number class
class FuzzyNumber:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def alpha_cut(self, alpha):
        lower = self.a + alpha * (self.b - self.a)
        upper = self.c - alpha * (self.c - self.b)
        return [lower, upper]

    def defuzzify(self):
        return (self.a + 4 * self.b + self.c) / 6

# FLP model for resource allocation
def solve_flp(casualties, kits, personnel, kits_coeff, personnel_coeff, utility_kits, utility_personnel):
    casualties_fuzzy = FuzzyNumber(*casualties)
    kits_coeff = FuzzyNumber(*kits_coeff)
    personnel_coeff = FuzzyNumber(*personnel_coeff)
    utility_kits = FuzzyNumber(*utility_kits)
    utility_personnel = FuzzyNumber(*utility_personnel)

    results = []
    for alpha in [0.1, 0.3, 0.5, 0.7, 1.0]:
        c_lower, c_upper = casualties_fuzzy.alpha_cut(alpha)
        k_lower, k_upper = kits_coeff.alpha_cut(alpha)
        p_lower, p_upper = personnel_coeff.alpha_cut(alpha)
        u_k_lower, u_k_upper = utility_kits.alpha_cut(alpha)
        u_p_lower, u_p_upper = utility_personnel.alpha_cut(alpha)

        prob = pulp.LpProblem("FLP_Triage", pulp.LpMaximize)
        x1 = pulp.LpVariable("kits", lowBound=0)
        x2 = pulp.LpVariable("personnel", lowBound=0)

        prob += 0.7 * (u_k_lower * x1 + u_p_lower * x2)
        prob += k_upper * x1 + p_upper * x2 <= c_lower
        prob += x1 <= kits
        prob += x2 <= personnel

        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        utility = pulp.value(prob.objective)
        kits_used = x1.varValue
        personnel_used = x2.varValue
        feasible = prob.status == 1

        results.append({
            'alpha': alpha,
            'utility': utility,
            'kits': kits_used,
            'personnel': personnel_used,
            'feasible': feasible
        })

    utility_values = [r['utility'] for r in results if r['feasible']]
    defuzzified_utility = np.mean(utility_values) if utility_values else 0
    return results, defuzzified_utility

# Fuzzy shortest path algorithm
def fuzzy_shortest_path(graph, source, target):
    G = nx.DiGraph()
    for u, v, data in graph:
        G.add_edge(u, v, weight=FuzzyNumber(*data['weight']))

    paths = {}
    costs = []
    for alpha in [0.1, 0.3, 0.5, 0.7, 1.0]:
        G_crisp = nx.DiGraph()
        for u, v, data in graph:
            w_lower, w_upper = data['weight'].alpha_cut(alpha)
            G_crisp.add_edge(u, v, weight=w_lower)

        try:
            path = nx.shortest_path(G_crisp, source, target, weight='weight')
            cost = nx.shortest_path_length(G_crisp, source, target, weight='weight')
            paths[alpha] = path
            costs.append(cost)
        except nx.NetworkXNoPath:
            paths[alpha] = None
            costs.append(float('inf'))

    defuzzified_cost = np.mean([c for c in costs if c != float('inf')]) if costs else float('inf')
    return paths, defuzzified_cost

# Plot results
def generate_plots(results, output_dir='figures'):
    os.makedirs(output_dir, exist_ok=True)
    alphas = [r['alpha'] for r in results]
    utilities = [r['utility'] for r in results]
    plt.figure(figsize=(6, 4))
    plt.plot(alphas, utilities, marker='o', label='FLP Utility')
    plt.xlabel('Confidence Level')
    plt.ylabel('Utility Score')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'utility_alpha.png'), dpi=300)
    plt.close()

# Load dataset from CSV
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    params = df[df['parameter'].notnull()]
    network = df[df['node1'].notnull()]
    
    data = {}
    for _, row in params.iterrows():
        data[row['parameter']] = (row['lower'], row['most_likely'], row['upper'])
    
    graph = []
    for _, row in network.iterrows():
        graph.append((int(row['node1']), int(row['node2']), 
                      {'weight': (row['weight_lower'], row['weight_most_likely'], row['weight_upper'])}))
    
    return data, graph

# Main function
def main():
    dataset_files = {
        'small': 'data/small_dataset.csv',
        'medium': 'data/medium_dataset.csv',
        'large': 'data/large_dataset.csv',
        'nato': 'data/nato_dataset.csv',
        'un': 'data/un_dataset.csv'
    }

    results_summary = []
    for name, file_path in dataset_files.items():
        data, graph = load_dataset(file_path)
        
        # Run FLP
        flp_results, utility = solve_flp(
            casualties=data['casualties'],
            kits=data['kits_available'][0],
            personnel=data['personnel_available'][0],
            kits_coeff=data['kits_coeff'],
            personnel_coeff=data['personnel_coeff'],
            utility_kits=data['utility_kits'],
            utility_personnel=data['utility_personnel']
        )
        
        # Run fuzzy shortest path
        paths, cost = fuzzy_shortest_path(graph, source=1, target=4)

        # Calculate metrics
        feasibility = sum(1 for r in flp_results if r['feasible']) / len(flp_results)
        robustness = 1 - np.var([r['utility'] for r in flp_results]) / max(1, np.max([r['utility'] for r in flp_results]))

        results_summary.append({
            'dataset': name,
            'utility': utility,
            'feasibility': feasibility,
            'robustness': robustness,
            'evacuation_cost': cost
        })

        generate_plots(flp_results, output_dir=f'figures/{name}')

    # Save results
    pd.DataFrame(results_summary).to_csv('results.csv', index=False)

if __name__ == "__main__":
    main()
