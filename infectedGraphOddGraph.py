"""
Odd graph O_n + threshold-2 spreading process

This script:
- Builds the odd graph O_n
- Simulates a threshold-2 infection process
- Supports:
    [1] Manual initial set
    [2] Greedy heuristic
    [3] Exact brute-force over ALL k-sized initial sets
- Optional visualization
- Reports detailed timing
"""

import itertools
import math
import sys
import time
import networkx as nx
import matplotlib.pyplot as plt


# --------------------------------------------------
# Graph construction
# --------------------------------------------------

def num_vertices(n):
    """
    Returns the number of vertices in the odd graph O_n.
    |V(O_n)| = C(2n-1, n-1)
    """
    return math.comb(2 * n - 1, n - 1)


def generate_odd_graph(n):
    """
    Constructs the odd graph O_n.

    Vertices: all (n-1)-subsets of {1, ..., 2n-1}
    Edge: two vertices are adjacent iff the subsets are disjoint
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    universe = range(1, 2 * n)
    k = n - 1

    vertices = [frozenset(c) for c in itertools.combinations(universe, k)]

    G = nx.Graph()
    G.add_nodes_from(vertices)

    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            if vertices[i].isdisjoint(vertices[j]):
                G.add_edge(vertices[i], vertices[j])

    return G


# --------------------------------------------------
# Infection process (threshold = 2)
# --------------------------------------------------

def simulate_infection(G, initial_infected):
    """
    Simulates a threshold-2 spreading process.

    A vertex becomes infected if >= 2 neighbors are infected.
    """
    infected = set(initial_infected)
    rounds = 0
    history = [set(infected)]

    while True:
        new = set()

        for v in G.nodes():
            if v in infected:
                continue

            if sum(1 for u in G.neighbors(v) if u in infected) >= 2:
                new.add(v)

        if not new:
            break

        infected |= new
        history.append(set(infected))
        rounds += 1

    return infected, rounds, history


def is_contagious(G, S):
    """
    Returns True iff S infects the entire graph.
    """
    infected, _, _ = simulate_infection(G, S)
    return len(infected) == G.number_of_nodes()


# --------------------------------------------------
# Greedy heuristic
# --------------------------------------------------

def greedy_contagious_candidate(G):
    """
    Greedy heuristic for finding a contagious set.
    """
    nodes = list(G.nodes())
    current = set()

    while True:
        if is_contagious(G, current):
            final, rounds, _ = simulate_infection(G, current)
            return current, final, rounds

        best_gain = -1
        best_node = None
        base_infected = simulate_infection(G, current)[0]

        for v in nodes:
            if v in current:
                continue

            test = current | {v}
            infected = simulate_infection(G, test)[0]
            gain = len(infected) - len(base_infected)

            if gain > best_gain:
                best_gain = gain
                best_node = v

        if best_node is None:
            final, rounds, _ = simulate_infection(G, current)
            return current, final, rounds

        current.add(best_node)


# --------------------------------------------------
# Exact brute-force over k-sized sets
# --------------------------------------------------

def test_all_initial_sets(G, k, stop_on_first=False):
    """
    Tests ALL subsets of size k as initial infected sets.

    Returns:
    - contagious_sets
    - number tested
    - elapsed time
    """
    nodes = list(G.nodes())
    contagious_sets = []

    tested = 0
    start = time.time()

    for comb in itertools.combinations(nodes, k):
        tested += 1
        S = set(comb)

        if is_contagious(G, S):
            contagious_sets.append(S)
            if stop_on_first:
                break

    elapsed = time.time() - start
    return contagious_sets, tested, elapsed


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def pretty_node_set(S):
    """
    Pretty-print a set of vertices.
    """
    return "{" + ", ".join(
        "(" + ",".join(map(str, sorted(v))) + ")"
        for v in sorted(S, key=lambda x: tuple(sorted(x)))
    ) + "}"


def parse_premade_set(raw, G):
    """
    Parses input like:
      (1,2,3),(4,5,6)
    """
    parsed = set()
    parts = raw.replace(")", "(").split("(")

    for p in parts:
        nums = [int(x) for x in p.replace(",", " ").split()]
        if nums:
            parsed.add(frozenset(nums))

    valid = parsed & set(G.nodes())
    if not valid:
        raise ValueError("No valid vertices parsed")

    return valid


# --------------------------------------------------
# Main CLI
# --------------------------------------------------

def main():
    total_start = time.time()

    try:
        n = int(input("Enter n: "))
    except Exception:
        print("Invalid input.")
        sys.exit(1)

    print(f"\nConstructing odd graph O_{n}")
    print(f"|V| = {num_vertices(n)}")

    G = generate_odd_graph(n)
    print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    mode = input(
        "\nChoose initial infected set source:\n"
        "  [1] Use premade set (manual)\n"
        "  [2] Run greedy heuristic\n"
        "  [3] Test ALL k-sized initial sets (exact)\n"
        "Enter 1, 2, or 3: "
    ).strip()

    algo_start = algo_end = None

    # -----------------------------
    # Mode 1: manual
    # -----------------------------
    if mode == "1":
        try:
            raw = input("Enter set: ")
            initial = parse_premade_set(raw, G)
        except Exception as e:
            print("Invalid set:", e)
            sys.exit(1)

    # -----------------------------
    # Mode 2: greedy
    # -----------------------------
    elif mode == "2":
        print("\nRunning greedy heuristic...")
        algo_start = time.time()
        initial, final, rounds = greedy_contagious_candidate(G)
        algo_end = time.time()

        print("\nGreedy result:")
        print(" size:", len(initial))
        print(" set :", pretty_node_set(initial))
        print(" contagious:", len(final) == G.number_of_nodes())

    # -----------------------------
    # Mode 3: exact over all k-sets
    # -----------------------------
    elif mode == "3":
        try:
            k = int(input("Enter number of initial infected vertices k: "))
        except Exception:
            print("Invalid k.")
            sys.exit(1)

        stop = input("Stop after first contagious set? [y/n]: ").strip().lower() == "y"

        print(f"\nTesting ALL subsets of size {k}...")
        contagious_sets, tested, elapsed = test_all_initial_sets(G, k, stop)

        print("\n--- Exact search results ---")
        print(" subsets tested:", tested)
        print(" contagious sets found:", len(contagious_sets))
        print(f" time elapsed: {elapsed:.3f}s")

        if not contagious_sets:
            print("\nNo contagious sets found.")
            sys.exit(0)

        initial = contagious_sets[0]
        print("\nExample contagious set:")
        print(pretty_node_set(initial))

    else:
        print("Invalid choice.")
        sys.exit(1)

    # -----------------------------
    # Simulation
    # -----------------------------
    sim_start = time.time()
    infected, rounds, _ = simulate_infection(G, initial)
    sim_end = time.time()

    print("\nSimulation result:")
    print(" initial infected:", len(initial))
    print(" rounds:", rounds)
    print(" final infected:", len(infected))

    # -----------------------------
    # Optional drawing
    # -----------------------------
    draw = input("\nDraw and save visualization? [y/n]: ").strip().lower()
    if draw == "y":
        pos = nx.circular_layout(G) if G.number_of_nodes() <= 60 else nx.spring_layout(G, seed=42)
        colors = ["red" if v in infected else "lightgray" for v in G.nodes()]

        plt.figure(figsize=(10, 10))
        nx.draw(G, pos, node_color=colors, node_size=80, with_labels=False)
        plt.title(f"O_{n} final infection")
        plt.axis("off")
        plt.savefig(f"odd_graph_O{n}_infection.png", dpi=200)
        plt.show()

    total_end = time.time()

    # -----------------------------
    # Timing summary
    # -----------------------------
    print("\n--- Timing ---")
    if algo_start is not None:
        print(f"Greedy time: {algo_end - algo_start:.3f}s")
    print(f"Simulation time: {sim_end - sim_start:.3f}s")
    print(f"TOTAL elapsed: {total_end - total_start:.3f}s")


if __name__ == "__main__":
    main()
