'''
In order to run this file place it in the same folder as the .txt files, dolphins.xlsx, and the algorithm.py file.
It is required that this be run on Python 3.11.x with the newest version of the networkx and cdlib packages.

To run use the command line with the following format:
python3 splittest.py name of graph 
'''
import argparse
import sys
from algorithm import *
import pandas as pd
from datetime import date
sheets = pd.read_excel('dolphins.xlsx', sheet_name=None)

df1 = sheets['ids_and_names']
df2 = sheets['relationships']

dolphins = nx.Graph()
dolphins.add_nodes_from(df1.index)  # Using DataFrame index as node labels

edges = list(zip(df2.source, df2.target))
dolphins.add_edges_from(edges)

graphs = {
    "dolphins": dolphins,
    "blogs": convert_nodes_to_integers(nx.read_edgelist('blogs.txt', create_using=nx.DiGraph())),
    "netsci": convert_nodes_to_integers(nx.read_edgelist('net.txt', create_using=nx.Graph())),
    "facebook": convert_nodes_to_integers(nx.read_edgelist('facebook.txt',create_using=nx.DiGraph()))
}


# Define the available algorithms
available_algorithms = [
    "Walktrap",
    "Louvain",
    "S-BGACD",
    "Q-BGACD",
    "D-BGACD",
    "GA-BCD"
]

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run community detection algorithms.")
parser.add_argument("graph_name", type=str, help="Name of the graph to analyze")
parser.add_argument("algorithm", type=str, nargs="?", choices=available_algorithms + ["all"], default="all",
                    help="Specify the algorithm to run or 'all' to run all algorithms")

args = parser.parse_args()

# Get the selected graph and check if it exists
G = graphs.get(args.graph_name)
if G is None:
    print(f"Graph '{args.graph_name}' not found.")
    sys.exit(1)

# Define a function to run a specific algorithm
def run_algorithm(algorithm_name):
    start_time = time.time()
    if algorithm_name == "Walktrap":
        result = Walktrap(G),[]
    elif algorithm_name == "Louvain":
        result = Louvain(G),[]
    elif algorithm_name == "S-BGACD":
        result = BaseGA(G, 100, 100, max(Louvain(G)), 'Simpson', 10, True, 'Z', True, True, .2)
    elif algorithm_name == "Q-BGACD":
        result = BaseGA(G, 100, 100, max(Louvain(G)), 'Simpson', 10, True, 'Q', True, True, 1)
    elif algorithm_name == "D-BGACD":
        result = BaseGA(G, 100, 100, max(Louvain(G)), 'Simpson', 10, True, 'D', True, True, 1)
    elif algorithm_name == "GA-BCD":
        result = BaseGA(G, 100, 100, max(Louvain(G)), 'Simpson', 10, False, 'Z', False, False)
    else:
        return

    entry = {
        'Name': algorithm_name,
        'Result Chr': result[0],
        'Fitnesses': result[1],
        'Time': time.time() - start_time,
        'Mod': calc_mod(G, result[0]),
    }
    print(f"{entry['Name']} has a Modularity Score of {entry['Mod']:.3f} in {entry['Time']:.2f} seconds")

    
    data = {'alg': [entry['Name']], 'best_chr': [entry['Result Chr']], 'fitness': [entry['Fitnesses']],
            'times': [entry['Time']], 'mod': [entry['Mod']]}
    stats = pd.DataFrame(data=data)
    stats.to_excel(f"results_{args.graph_name}_{date.today()}_{algorithm_name}.xlsx")

# Run the selected algorithm or all algorithms
if args.algorithm == "all":
    for algorithm_name in available_algorithms:
        run_algorithm(algorithm_name)
else:
    run_algorithm(args.algorithm)
