import networkx as nx
from networkx.algorithms.community import girvan_newman
import matplotlib.pyplot as plt
import gacomm as gc
from cdlib import algorithms
import math
import random
from cdlib import evaluation
from cdlib import NodeClustering
import numpy as np
import sys

def GN(G):
    print("GN")
    G = convert_nodes_to_integers(G)
    communities = girvan_newman(G)
    print("done")
    for _ in range(11):
        next(communities)

    all_node_groups = [list(c) for c in next(communities)]


    # Convert node groups to community representation
    solution = [0 for _ in range(G.number_of_nodes())]
    for i, node_group in enumerate(all_node_groups):
        for j in node_group:
            solution[j] = i + 1

    print(solution)
    return solution

def LP(G):
    G = convert_nodes_to_integers(G)

    communities = list(nx.algorithms.community.label_propagation_communities(G))

    solution = [0 for _ in range(G.number_of_nodes())]
    for i in range(len(communities)):
        com = communities[i]
        for j in com:
            solution[j] = i + 1

    print(solution)
    return solution

# all GA-NET related code including community score is from this repo: https://github.com/hariswb/ga-community-detection
def GANet(G):
    print("GANet")
    G = convert_nodes_to_integers(G)

    nodes = list(G.nodes)
    edges = list(G.edges)

    comms = gc.community_detection(nodes, edges)

    solution = [0 for _ in range(G.number_of_nodes())]
    for i in range(len(comms)):
        com = comms[i]
        for j in com:
                solution[j] = i + 1
    print(solution)
    return solution

def Walktrap(G):
    print("walktrap")
    G = convert_nodes_to_integers(G)

    # Run walktrap algorithm
    communities = algorithms.walktrap(G)

    # Create a list where each index corresponds to a node and the value is the community label
    solution = [0] * len(G.nodes())
    for idx, community in enumerate(communities.communities):
        for node in community:
            solution[node] = idx + 1

    # Print or use the list as needed
    print(solution)
    return solution

def Louvain(G):
    print("Louvain")
    G = convert_nodes_to_integers(G)
    communities = list(nx.community.louvain_communities(G))

    solution = [0 for _ in range(G.number_of_nodes())]
    for i in range(len(communities)):
        com = communities[i]
        for j in com:
            solution[j] = i + 1

    return solution

import random
import matplotlib.pyplot as plt
import time
def mod_density(chr, G):
    coms = []
    for i in range(1,max(chr)+1):
        new_com = []
        for x in range(len(chr)):
            if chr[x] == i:
                new_com.append(x)
        if len(new_com) != 0:
            coms.append(new_com)
    mod_density = evaluation.modularity_density(G,NodeClustering(coms, G))
    return mod_density.score

def community_score(chr, G):
    Adj = nx.adjacency_matrix(G)
    locus_chr = []
    for i in chr:
        locus_chr.append(chr.index(i))
    subsets = gc.find_subsets(locus_chr)
    return gc.community_score(locus_chr,subsets,1.5,Adj)

def BaseGA(G, num_gen, pop_size, num_com, Eq, mu, uselabelprop, fitfunc, neighbour_mutation, uniform_crossover, beta = 1):
    """
    The driver function for running all genetic algorithms.

    G = input graph
    num_gen = number of generations
    pop_size = size of population
    Eq = Similarity function used by GSD also known as Z
    mu = mu value for mu + lambda selection
    uselabelprop = boolean value indicating whether or not to use label propagation based initialization
    fitfunc = string value indicating which objective function to use (Q,D,CS,Z)
    """
    A = nx.adjacency_matrix(G).toarray()
    mut_rate = .3
    cross_rate = .6
    num_cross = int(len(G.nodes())/10)
    # Uncomment the population initialization method you want to use gen_pop is the GA-BCD method
    if not uselabelprop:
        population = gen_pop(G, pop_size, num_com)
    else:
        population = labelprop_gen(G, pop_size, num_com, beta=beta)
    best_fit = -math.inf
    best_chr = []
    sd_dict = {}
    # Lists to store the best fitness values for plotting
    best_fitnesses = []
    best_mods = []

    for i in range(0, num_gen):
        fitnesses = []
        if i % 1 == 0:
            print(f'{i}: {best_fit} ')
        x = 0
        print(x)
        for chr in population:
            if fitfunc == "Z":
                fitnesses.append(GSD(chr, G,A,Eq))
            elif fitfunc == "Q":
                fitnesses.append(calc_mod(G,chr))
            elif fitfunc == "D":
                fitnesses.append(mod_density(chr,G))
            else:
                fitnesses.append(community_score(chr,G))
            sys.stdout.write("\033[F")  # Move cursor up one line
            sys.stdout.write("\033[K")  
            x += 1
            print(x)
            
        pop_fit = list(zip(population, fitnesses))
        pop_fit = sorted(pop_fit, key=lambda x: x[1])
        if pop_fit[-1][1] > best_fit:
            best_fit = pop_fit[-1][1]
            best_chr = pop_fit[-1][0]

        # Append the best fitness value to the list
        best_fitnesses.append(best_fit)
        best_mods.append(calc_mod(G, best_chr))

        # mu + lambda selection
        population = [x[0] for x in pop_fit]
        population = population[-mu:]
        new_population = []
        while len(new_population) < pop_size - mu:
            # Select parents probabilistically based on fitness
            total_fitness = sum(fitness for _, fitness in pop_fit[-mu:])
            if total_fitness == 0:
                # Handle the case when total_fitness is 0
                # You may choose to assign equal weights in this case
                weights = [1 for _ in range(mu)]
            else:
                weights = [fitness / total_fitness for _, fitness in pop_fit[-mu:]]
            parent1, parent2 = random.choices(population, weights=weights, k=2)
            if random.random() < cross_rate:
                if not uniform_crossover:
                    child = crossover(parent1, parent2, num_cross)
                else:
                    child = uniform_cross(parent1,parent2)
            else:
                child = parent1
            if random.random() < mut_rate:
                mutation_start_time = time.time()  # Start time for mutation call
                if not neighbour_mutation:
                    child = mutate(child, G, Eq, sd_dict)
                else:
                    child = neighbour_mutate(child, G)
                mutation_end_time = time.time()  # End time for mutation call
                mutation_time = mutation_end_time - mutation_start_time

            new_population.append(child)

        population.extend(new_population)


    print(best_fit)
    return best_chr, best_fitnesses, best_mods



def convert_nodes_to_integers(graph):
    mapping = {node: idx for idx, node in enumerate(graph.nodes())}
    new_graph = nx.relabel_nodes(graph, mapping)
    return new_graph


def GSD(Chromosome, G, A, Eq = "Simpson"): #Group Similarity Density

    #split graph into n subgraphs based on community
    num_communities = max(Chromosome)
    communities = {i: nx.DiGraph() for i in range (1, num_communities+1)}
    for node, com_id in zip(G.nodes(), Chromosome):
        communities[com_id].add_node(node)
    for i in range(1,num_communities+1):
        community = communities[i]
        for node in community.nodes():
            for neighbor in list(G.neighbors(node)):
                if neighbor in community.nodes():
                    community.add_edge(node, neighbor)

    #create a list of all pairs of nodes in a community
    communities_pairs_list = []
    for i in range (1, num_communities+1):
        com = communities[i]
        com_pairs = []
        for node1 in com.nodes():
            for node2 in com.nodes():
                if (node2, node1) not in com_pairs:
                    com_pairs.append((node1,node2))
        communities_pairs_list.append(com_pairs)

    #create a list of all community densities
    communities_densities = []
    for i in range(1,num_communities+1):
        sd = 0
        com = communities[i]
        if len(communities_pairs_list[i-1])>0:
            for pair in communities_pairs_list[i-1]:
                if Eq == 'Jaccard':
                    sd += Jaccard(G, pair[0], pair[1])
                elif Eq == 'Simpson':
                    sd += Simpson(G, pair[0], pair[1],A)
                elif Eq == 'Geometric':
                    sd += Geometric(G, pair[0], pair[1])
                elif Eq == 'Cosine':
                    sd += Cosine(G, pair[0], pair[1])
                elif Eq == 'Sorenson':
                    sd += Sorenson(G, pair[0], pair[1])
                else:
                    raise Exception('Similarity equation not in the possibilities')
            #sd_dict[encs[i-1]] = sd
        
            communities_densities.append(sd)
    #calculate group similarity density
    similarity_density = 1
    for i in communities_densities:
        similarity_density *= i
    similarity_density = (1 + 1/math.sqrt(num_communities))*similarity_density
    return similarity_density

def Jaccard(G, node1, node2):
    neighbors1 = set(G.neighbors(node1))
    neighbors2 = set(G.neighbors(node2))

    intersection = len(neighbors1.intersection(neighbors2))
    union = len(neighbors1.union(neighbors2))
    similarity = intersection / union if union != 0 else 0
    return similarity
def Simpson(G, node1, node2,A):
    n1 = np.array(A[node1],dtype=np.bool_)
    n2 = np.array(A[node2],dtype=np.bool_)

    intersection = np.count_nonzero(n1&n2)
    min_deg = min(np.count_nonzero(n1), np.count_nonzero(n2))
    similarity = intersection / min_deg if min_deg != 0 else 0
    return similarity
def Geometric(G, node1, node2):
    neighbors1 = set(G.neighbors(node1))
    neighbors2 = set(G.neighbors(node2))

    intersection = len(neighbors1.intersection(neighbors2))
    deg_prod = len(neighbors1)*len(neighbors2)
    similarity = intersection**2 / deg_prod if deg_prod != 0 else 0
    return similarity
def Cosine(G, node1, node2):
    neighbors1 = set(G.neighbors(node1))
    neighbors2 = set(G.neighbors(node2))

    intersection = len(neighbors1.intersection(neighbors2))
    deg_prod = len(neighbors1)*len(neighbors2)
    similarity = intersection / math.sqrt(deg_prod) if deg_prod != 0 else 0
    return similarity

def Sorenson(G, node1, node2):
    neighbors1 = set(G.neighbors(node1))
    neighbors2 = set(G.neighbors(node2))

    intersection = len(neighbors1.intersection(neighbors2))
    deg_sum = len(neighbors1)+len(neighbors2)
    similarity = 2*intersection / deg_sum if deg_sum != 0 else 0
    return similarity

def mutate(chr, G, Eq, sd_dict):
    mutated_chr = chr.copy()

    #mutate 2% of the nodes
    num_mutations = int(len(chr)/50) if len(chr)>=50 else 1
    num_communities = max(chr)
    mutated_nodes = random.sample(range(0,len(chr)), num_mutations)

    # mutate the selected nodes by selecting the chromosome with highest gsd
    for node in mutated_nodes:
        original = chr[node]
        i = random.randint(1,num_communities)
        while i == original:
            i = random.randint(1,num_communities)
        temp_chr = mutated_chr.copy()
        temp_chr[node] = i
        #mutated_sim = GSD(temp_chr, G, Eq, sd_dict)
        mutated_chr = temp_chr

    return mutated_chr

def neighbour_mutate(chr, G):
    num_mutations = int(len(chr)/50) if len(chr)>=50 else 1
    mutated_nodes = random.sample(range(0,len(chr)), num_mutations)
    for node in mutated_nodes:
        neighbours = G.to_undirected().neighbors(node)
        neighbour_labels = [chr[n] for n in neighbours]
        chr[node] = random.sample(neighbour_labels, 1)[0]
    return chr
def crossover(chr1, chr2, num_points):
    cross_points = sorted(random.sample(range(0, len(chr1)), num_points))
    child = chr1[:]  # Initialize child with the first parent's genes

    for i in range(0, num_points, 2):
        start, end = cross_points[i:i + 2] if i + 1 < num_points else (cross_points[i], len(chr1))
        child[start:end] = chr2[start:end]  # Replace genes in child with genes from the second parent
    return child

def uniform_cross(chr1,chr2):
    mask = random.choices(range(1),k=len(chr1))
    child = []
    for i in range(len(chr1)):
        if mask[i] == 0:
            child.append(chr1[i])
        else:
            child.append(chr2[i])
    return child

def is_valid_chromosome(chromosome, G):
    # Check if all nodes in the same community are connected
    if nx.is_connected(G):
        # print("connected")
        return True,[]
    # print("not connceted")
    num_communities = max(chromosome)
    communities = {i: [] for i in range(1, num_communities + 1)}
    G = G.to_undirected()
    # Group nodes into communities
    for node, com_id in zip(G.nodes(), chromosome):
        communities[com_id].append(node)

    good = True
    bad_nodes = []
    # Check if all pairs of nodes within each community have a path in the main graph
    for com_id, nodes in communities.items():
        base = nodes[0]
        com_bad_nodes = []
        random.shuffle(nodes)
        # print(base)
        for node in nodes:
            if base != node and not nx.has_path(G,base,node):
                #print("bad")
                com_bad_nodes.append(node)
                good = False
        bad_nodes.extend(com_bad_nodes)
        
    return good,bad_nodes

def gen_pop(G, pop_size, num_coms):
    G_und = G.to_undirected()
    chr_size = G_und.number_of_nodes()
    chr_list = []
    if chr_size < num_coms:
        raise Exception('More communities than nodes in the graph')
    x = 0
    while len(chr_list) < pop_size:
        chromosome = random.sample(range(1, num_coms + 1), num_coms)
        for _ in range(chr_size - num_coms):
            chromosome.append(random.randint(1, num_coms))
        random.shuffle(chromosome)
        valid = False
        while not valid:
            num_coms = max(chromosome)
            valid,bad = is_valid_chromosome(chromosome,G_und)
            sys.stdout.write("\033[F")  # Move cursor up one line
            sys.stdout.write("\033[K") 
            print(len(bad))
            if valid:
                chr_list.append(chromosome)
                print("valid")
            else:
                for badn in bad:
                    if list(G.neighbors(badn)) == []:
                        chromosome[badn]=num_coms+1
                    else:
                        chromosome[badn] = random.randint(1,num_coms)
    return chr_list

def labelprop_gen(G, pop_size, num_coms,alpha = 0.2, beta = 1):
    n = G.number_of_nodes()
    chr_list = []
    while len(chr_list)<pop_size:
        chromosome = random.sample(range(1, num_coms + 1), num_coms)
        for _ in range(n - num_coms):
            chromosome.append(random.randint(1, num_coms))
        random.shuffle(chromosome)
        seed_nodes = random.sample(range(n),int(n*alpha))
        for node in seed_nodes:
            label = chromosome[node]
            neighbours = random.sample(list(G.neighbors(node)), int(len(list(G.neighbors(node)))*beta))
            for neighbour in neighbours:
                chromosome[neighbour] = label
        chr_list.append(chromosome)
    return chr_list

def calc_mod(G, communities):
    processed_comms = []
    for i in range(max(communities)):
        processed_comms.append(set())
    for node in range(len(communities)):
        processed_comms[communities[node]-1].add(node)
    #print(processed_comms)
    # Ensure each node is assigned to only one community
    assigned_nodes = set()
    for comm in processed_comms:
        for node in comm:
            if node in assigned_nodes:
                raise ValueError(f"Node {node} is assigned to multiple communities.")
            assigned_nodes.add(node)
    
    color_map = []
    for node in G.nodes():
        for i, partition in enumerate(processed_comms):
            if node in partition:
                color_map.append(i)
                break

    # # Draw the graph with colored nodes
    # pos = nx.spring_layout(G)
    # nx.draw(G, pos, node_color=color_map, node_size=200, with_labels=True)
    # plt.show()

    return nx.community.modularity(G, processed_comms)