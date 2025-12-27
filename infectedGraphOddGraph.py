"""
Odd graph O_n + threshold-2 spreading process

This script:
- Builds the odd graph O_n
- Simulates a threshold-2 infection process on it
- Finds contagious sets using a greedy heuristic
- Optionally runs an exact (brute-force) search for small graphs
- Optionally visualizes the final infection
- Reports elapsed times
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

    Vertices of O_n are all (n-1)-subsets of {1, ..., 2n-1}.
    Count = C(2n-1, n-1).
    """
    return math.comb(2 * n - 1, n - 1)


def generate_odd_graph(n):
    """
    Constructs the odd graph O_n.

    - Vertices: all (n-1)-element subsets of {1, ..., 2n-1}
    - Edge between two vertices iff the subsets are disjoint
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    universe = range(1, 2 * n)
    k = n - 1

    # Generate all (n-1)-subsets as frozensets (hashable, usable as nodes)
    vertices = [frozenset(c) for c in itertools.combinations(universe, k)]

    G = nx.Graph()
    G.add_nodes_from(vertices)

    # Add an edge if two subsets are disjoint
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

    Rule:
    - A vertex becomes infected if at least 2 of its neighbors are infected.

    Returns:
    - final infected set
    - number of rounds until stabilization
    - history of infected sets per round
    """
    infected = set(initial_infected)
    rounds = 0
    history = [set(infected)]

    while True:
        new = set()

        # Check every uninfected vertex
        for v in G.nodes():
            if v in infected:
                continue

            # Count infected neighbors
            if sum(1 for u in G.neighbors(v) if u in infected) >= 2:
                new.add(v)

        # Stop if no new infections occur
        if not new:
            break

        infected |= new
        history.append(set(infected))
        rounds += 1

    return infected, rounds, history


def is_contagious(G, S):
    """
    Returns True if S eventually infects the entire graph.
    """
    infected, _, _ = simulate_infection(G, S)
    return len(infected) == G.number_of_nodes()


# --------------------------------------------------
# Greedy heuristic
# --------------------------------------------------

def greedy_contagious_candidate(G):
    """
    Greedy algorithm for finding a small contagious set.

    Idea:
    - Start with an empty set
    - Iteratively add the vertex that maximizes the increase
      in the number of infected vertices
    """
    nodes = list(G.nodes())
    current = set()

    while True:
        # Stop if current set already infects the entire graph
        if is_contagious(G, current):
            final, rounds, _ = simulate_infection(G, current)
            return current, final, rounds

        best_gain = -1
        best_node = None

        # Baseline infection with current set
        base_infected = simulate_infection(G, current)[0]

        # Try adding each unused vertex
        for v in nodes:
            if v in current:
                continue

            test = current | {v}
            infected = simulate_infection(G, test)[0]
            gain = len(infected) - len(base_infected)

            # Keep vertex with maximal gain
            if gain > best_gain:
                best_gain = gain
                best_node = v

        # If no improvement is possible, stop
        if best_node is None:
            final, rounds, _ = simulate_infection(G, current)
            return current, final, rounds

        current.add(best_node)


# --------------------------------------------------
# Exact search (small graphs only)
# --------------------------------------------------

def find_min_contagious_size(G, exact_limit=24, time_limit=20):
    """
    Attempts to find the minimum contagious set size.

    Strategy:
    - First compute greedy solution
    - Then try all subsets of size 1, 2, ..., |greedy|
    - Only feasible for very small graphs
    """
    result = {}

    greedy_set, greedy_final, _ = greedy_contagious_candidate(G)

    result["greedy_set"] = greedy_set
    result["greedy_size"] = len(greedy_set)
    result["greedy_contagious"] = len(greedy_final) == G.number_of_nodes()

    n = G.number_of_nodes()
    if n > exact_limit:
        result["exact_found"] = False
        result["note"] = "Graph too large for exact search"
        return result

    nodes = list(G.nodes())
    start = time.time()

    # Try subsets of increasing size
    for k in range(1, len(greedy_set) + 1):
        if time.time() - start > time_limit:
            result["exact_found"] = False
            result["note"] = "Exact search timed out"
            return result

        for comb in itertools.combinations(nodes, k):
            if is_contagious(G, set(comb)):
                result["exact_found"] = True
                result["exact_size"] = k
                result["exact_set"] = set(comb)
                return result

    result["exact_found"] = False
    return result


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def pretty_node_set(S):
    """
    Converts a set of frozenset vertices into a readable string.
    """
    return "{" + ", ".join(
        "(" + ",".join(map(str, sorted(v))) + ")"
        for v in sorted(S, key=lambda x: tuple(sorted(x)))
    ) + "}"


def parse_premade_set(raw, G):
    """
    Parses user input like:
      (1,2,3),(4,5,6)

    Returns the subset that actually exists in the graph.
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
    """
    Main command-line interface.
    Handles:
    - Input
    - Graph construction
    - Algorithm selection
    - Simulation
    - Optional visualization
    - Timing
    """
    total_start = time.time()

    try:
        n = int(input("Enter n: "))
    except Exception:
        print("Invalid input.")
        sys.exit(1)

    print(f"\nConstructing odd graph O_{n} ...")
    print(f"Number of vertices: {num_vertices(n)}")

    G = generate_odd_graph(n)
    print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # ----------------------------------------------
    # Choose initial infected set source
    # ----------------------------------------------

    mode = input(
        "\nChoose initial infected set source:\n"
        "  [1] Use premade set (skip greedy)\n"
        "  [2] Run greedy heuristic\n"
        "Enter 1 or 2: "
    ).strip()

    algo_start = algo_end = None

    if mode == "1":
        # User provides initial infected set
        try:
            raw = input("Enter set: ")
            initial = parse_premade_set(raw, G)
            print(f"Using premade set of size {len(initial)}")
        except Exception as e:
            print("Invalid premade set:", e)
            sys.exit(1)

    elif mode == "2":
        # Run greedy heuristic
        print("\nRunning greedy heuristic...")
        algo_start = time.time()
        result = find_min_contagious_size(G)
        algo_end = time.time()

        print("\nGreedy result:")
        print(" size:", result["greedy_size"])
        print(" set :", pretty_node_set(result["greedy_set"]))
        print(" contagious:", result["greedy_contagious"])

        if result.get("exact_found"):
            print("\nExact m2(G):", result["exact_size"])
            print("Exact set:", pretty_node_set(result["exact_set"]))
        else:
            print("\nExact search not completed.")
            print("Note:", result.get("note", ""))

        initial = result["greedy_set"]

    else:
        print("Invalid choice.")
        sys.exit(1)

    # ----------------------------------------------
    # Simulation
    # ----------------------------------------------

    sim_start = time.time()
    infected, rounds, _ = simulate_infection(G, initial)
    sim_end = time.time()

    print("\nSimulation result:")
    print(" initial infected:", len(initial))
    print(" rounds:", rounds)
    print(" final infected:", len(infected))

    # ----------------------------------------------
    # Optional drawing
    # ----------------------------------------------

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

    # ----------------------------------------------
    # Timing summary
    # ----------------------------------------------

    print("\n--- Timing ---")
    if algo_start is not None:
        print(f"Greedy / exact time: {algo_end - algo_start:.3f}s")
    else:
        print("Greedy skipped")

    print(f"Simulation time: {sim_end - sim_start:.3f}s")
    print(f"TOTAL elapsed: {total_end - total_start:.3f}s")


if __name__ == "__main__":
    main()
