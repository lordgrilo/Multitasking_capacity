
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import time, sys, os
import cPickle as pk
import numpy as np
import time

def msp_associated_graph(BG,N,projection_layer=0):
    bg_graph = nx.Graph();
    for i in range(projection_layer*N,(projection_layer+1)*N):
        nei = BG.neighbors(i);
        k = nx.relabel_nodes(nx.complete_graph(len(nei)),dict(zip(range(len(nei)), nei)));
        bg_graph = nx.compose(bg_graph,k);
    return bg_graph;
    

def leaf_removal(g, verbose=False):
    G = g.copy()
    stop = 0;
    potential_mis = [];
    isolated = [x for x in g.nodes() if nx.degree(g,x)==0];
    potential_mis.extend(isolated);
    G.remove_nodes_from(isolated);
    while stop==0:
        deg = G.degree();
        if 1 in deg.values():
            for n in G.nodes_iter():
                if deg[n]==1:
                    L = n;
                    break;
            nn = nx.neighbors(G,L)[0]
            G.remove_node(L);
            G.remove_node(nn);
            potential_mis.append(L);
            isolated = [x for x in G.nodes() if nx.degree(G,x)==0];
            potential_mis.extend(isolated);
            G.remove_nodes_from(isolated);
        else:
            stop=1;
    core_mis = [];
    if G.number_of_nodes()>=1:
        core_mis = nx.maximal_independent_set(G);
        if verbose==True:
            print len(potential_mis), len(core_mis), N;
        potential_mis.extend(core_mis);
    else:
        if verbose==True:
            print len(potential_mis), len(core_mis), N;
    return potential_mis, core_mis;

def line_graph(G):
    ig = nx.Graph();
    for n in G.nodes():
        new_nodes = []
        for nn in G.neighbors(n):
            new_nodes.append(str(sorted(tuple([n,nn]))));
            ig.add_node(str(sorted(tuple([n,nn]))));
        from itertools import combinations
        for co in combinations(new_nodes,2):
            ig.add_edge(co[0],co[1])
    return ig;

def interference_graph(ig):
    g = ig.copy();
    new_edges = [];
    for i in ig.nodes():
        for j in ig.nodes():
            try:
                if nx.shortest_path_length(ig,i,j)==2:
                    new_edges.append([i,j]);
            except:
                pass;
    g.add_edges_from(new_edges);
    return g;


def create_blist(alist):
    nodes = len(alist);
    deg = np.sum(alist);
    blist = (deg/nodes) * np.ones((nodes));
    while deg>np.sum(blist):
        r = np.random.randint(0,nodes);
        blist[r]+=1;
    while deg<np.sum(blist):
        r = np.random.randint(0,nodes);
        blist[r]-=1;
    if deg==np.sum(blist):
        return blist
    else:
        print 'Fail'
        return blist;

def create_BA_bipartite_graph(N,m):
    g = nx.barabasi_albert_graph(N,m);
    alist = g.degree().values()
    blist = create_blist(alist)
    return nx.bipartite.configuration_model(alist,blist);

def create_ER_bipartite_graph(N,p):
    g = nx.erdos_renyi_graph(N,p);
    alist = g.degree().values();
    blist = create_blist(alist);
    return nx.bipartite.configuration_model(alist,blist);

def create_expected_degree_graph(degseq,selfloops=True):
    g = nx.expected_degree_graph(degseq,int(time.time()),selfloops);
    alist = g.degree().values();
    blist = create_blist(alist);
    return nx.bipartite.configuration_model(alist,blist);

def create_configuration_model_graph(degseq):
    alist = degseq;
    blist = create_blist(alist);
    return nx.bipartite.configuration_model(alist,blist);

def icml_algo(G, input_index, verbose=False):
    bg = G.copy();
    layer = nx.get_node_attributes(bg,'bipartite');
    inputs = [x for x in bg.nodes() if layer[x] == input_index]
    outputs = [x for x in bg.nodes() if layer[x] != input_index]
    M = []
    N = bg.number_of_nodes();
    while len(inputs)>0:
        x = None
        y = None
        m = N;
        
        for i in inputs:
            nei = nx.neighbors(bg,i);
            for j in nei:
                if m > nx.degree(bg,i) + nx.degree(bg,j):
                    x = i;
                    y = j;
                    m = nx.degree(bg,i) + nx.degree(bg,j);
        if x==None and y==None:
            break;
        else:
            M.append((x,y));
            removal_nodes = [];
            for n in nx.neighbors(bg,x):
                if n in outputs:
                    outputs.remove(n);
                    removal_nodes.append(n)
            for n in nx.neighbors(bg,y):
                if n in inputs:
                    inputs.remove(n);
                    removal_nodes.append(n)
        bg.remove_nodes_from(removal_nodes);
        if verbose==True:
            print len(inputs)
    return M;


def icml_mostafa(G,transpose=False):
    A = nx.adj_matrix(G).todense()
    if transpose==True:
        A = A.T;

    m , n = A.shape; 
    deg = nx.degree(G);

    done = 1;
    I = np.zeros_like(range(m));
    O = np.zeros_like(range(n))

    l = 1;

    T = [];

    while done == 1:
        #find the candidate pair
        u = -1;
        v = -1;
        min_d_uv = m+n+1;

        for i in range(m):
            if I[i] == 0:
                for j in range(n):
                    if (A[i,j] == 1) and (O[j] == 0) and ((deg[i] + deg[j]) < min_d_uv):
                        min_d_uv = deg[i] + deg[j];
                        u = i;
                        v = j;
                        I[u] = 1;
                        O[v] = 1;
                    
        if (u == -1) and (v == -1): 
            done = 0;
        else:
            # set the rows and columns to one if used
            for j in range(n):
                if A[u,j] == 1:
                    A[u,j] = 0;
                    deg[j] -= 1;
                    O[j] = 1;
                    I[j] = 1;
            
            for i in range(m):
                if A[i,v] == 1:
                    A[i,v] = 0;
                    deg[i] -= 1;
                    I[i] = 1;
                    O[i] = 1;
            I[u] = 1;
            O[v] = 1;
            
            T.append((u,v));

    return T;





def return_single_pendant(g,layer, variables_index):
    found = 0;
    for i in g.nodes():
        if layer[i]==variables_index:
            nn = nx.neighbors(g,i);
            degs = nx.degree(g,nn);
            if 1 in degs:
                nei = [i]
                nei.extend(nn);
                return nei;
        elif nx.degree(g,i)==1:
            nei = [];
            v = nx.neighbors(g,i)[0];
            nei.append(v);
            nei.extend(nx.neighbors(g,v));
            return nei;
    return [];

def GSK(BG, variables_index=0, verbose=False, add_factors=False):
    Vprime1 = [];
    Vprime2 = [];
    
    layer = nx.get_node_attributes(BG,'bipartite');
    var = [x for x in BG.nodes() if layer[x] == variables_index]
    fac = [x for x in BG.nodes() if layer[x] != variables_index]
    
    if verbose==True:
        print 'Initial variable nodes:', var;
        print 'Initial factor nodes:', fac;

    isolated_variables = [x for x in BG.nodes() if nx.degree(BG,x)==0 and layer[x]==variables_index];
    [var.remove(x) for x in isolated_variables]
    
    G = BG.copy();
    Vprime1.extend(isolated_variables);
    G.remove_nodes_from(isolated_variables)
    
    isolated_factors = [x for x in G.nodes() if nx.degree(BG,x)==0 and layer[x]!=variables_index];
    [fac.remove(x) for x in isolated_factors]
    G.remove_nodes_from(isolated_factors);
    if add_factors==True:
        Vprime1.extend(isolated_factors);

    while len(var)>0:
        if verbose==True:
            print '#var:',len(var),'#fac:', len(fac), '#nodes in depleted graph:', G.number_of_nodes(),'#original BG:',BG.number_of_nodes();

        pendant = return_single_pendant(G,layer,variables_index);
        if len(pendant)==0:
            ## if not, choose randomly and do the game. 
            if verbose==True:
                print var
            v = np.random.choice(list(var));
            pendant = []
            pendant.append(v);
            pendant.extend(nx.neighbors(G,v));
            Vprime2.append(pendant[0]);
        else:
            Vprime1.append(pendant[0]);
        augmented_pendant = []
        augmented_pendant.extend(pendant);
        for n in pendant[1:]:
            augmented_pendant.extend(nx.neighbors(G,n));
        augmented_pendant = list(set(augmented_pendant));
        G.remove_nodes_from(augmented_pendant);        
        [var.remove(x) for x in augmented_pendant if x in var];
        [fac.remove(x) for x in augmented_pendant if x in fac];

    return Vprime1,Vprime2;


def return_mindeg_pendant(g,layer, variables_index):
    found = 0;
    degs = g.degree();
    m = np.sum(degs.values())+1;
    candidate_edge = None;
    nei = []
    if g.number_of_edges()>0:
        for e in g.edges():
            if m > degs[e[0]] + degs[e[1]]:
                candidate_edge = e;
        if layer[candidate_edge[0]]!=variables_index:
            candidate_edge = candidate_edge[::-1];

        nei.append(candidate_edge[0]);
        nei.extend(nx.neighbors(g,candidate_edge[0]));
    return nei;


def mindeg_GSK(BG, variables_index=0, verbose=False):
    Vprime1 = [];
    Vprime2 = [];
    
    layer = nx.get_node_attributes(BG,'bipartite');
    var = [x for x in BG.nodes() if layer[x] == variables_index]
    fac = [x for x in BG.nodes() if layer[x] != variables_index]
    
    if verbose==True:
        print 'Initial variable nodes:', var;
        print 'Initial factor nodes:', fac;

    isolated_variables = [x for x in BG.nodes() if nx.degree(BG,x)==0 and layer[x]==variables_index];
    [var.remove(x) for x in isolated_variables]
    
    G = BG.copy();
    Vprime1.extend(isolated_variables);
    G.remove_nodes_from(isolated_variables)
    
    isolated_factors = [x for x in G.nodes() if nx.degree(BG,x)==0 and layer[x]!=variables_index];
    [fac.remove(x) for x in isolated_factors]
    G.remove_nodes_from(isolated_factors);

    while len(var)>0:
        if verbose==True:
            print '#var:',len(var),'#fac:', len(fac), '#nodes in depleted graph:', G.number_of_nodes(),'#original BG:',BG.number_of_nodes();

        pendant = return_mindeg_pendant(G,layer,variables_index);
        if len(pendant)==0:
            ## if not, choose randomly and do the game. 
            if verbose==True:
                print var
            m = G.number_of_nodes()*2;
            degs = G.degree();
            for e in G.edges():
                if degs[e[0]] + degs[e[1]] < m:
                    m = degs[e[0]] + degs[e[1]];
                    v = e;
            if e[0] in var:
                v = e[0];
            else:
                v = e[1];
            pendant = []
            pendant.append(v);
            pendant.extend(nx.neighbors(G,v));
            Vprime2.append(pendant[0]);
        else:
            Vprime1.append(pendant[0]);
        augmented_pendant = []
        augmented_pendant.extend(pendant);
        for n in pendant[1:]:
            augmented_pendant.extend(nx.neighbors(G,n));
        augmented_pendant = list(set(augmented_pendant));
        G.remove_nodes_from(augmented_pendant);        
        [var.remove(x) for x in augmented_pendant if x in var];
        [fac.remove(x) for x in augmented_pendant if x in fac];

    return Vprime1,Vprime2;



def power_law(N,exp,Range=None):
    from scipy.stats import rv_discrete
    if Range==None:
        Range=N;
    sample = rv_discrete(values=(range(1,Range+1), map(lambda x: zipf.pmf(x,exp), range(1,Range+1))))
    return sample.rvs(size=N);



def factor_graph_from_ig(ig):
    factor_graph = nx.Graph();
    bipartite_dict = {}
    for edge in ig.edges():
        bipartite_dict[edge] = 0;
        bipartite_dict[edge[0]] = 1;
        bipartite_dict[edge[1]] = 1;
        factor_graph.add_edge(edge,edge[1])
        factor_graph.add_edge(edge,edge[0])
    nx.set_node_attributes(factor_graph,'bipartite',bipartite_dict)
    return factor_graph;


def edge_factor_graph(bg,distance=1):
    edge_factor_graph = nx.Graph();
    lg = line_graph(bg);
    bipartite_dict = {};
    for edge in lg.nodes():
        nei = set(nx.single_source_shortest_path_length(lg, edge, cutoff=distance).keys());
        nei.remove(edge);
        bipartite_dict[str(edge)] = 0;
        bipartite_dict['gamma_'+str(edge)] = 1;
        edge_factor_graph.add_node(str(edge));
        edge_factor_graph.add_node('gamma_'+str(edge));
        for n in nei:
            edge_factor_graph.add_edge(str(edge),'gamma_'+str(n));
    nx.set_node_attributes(edge_factor_graph,'bipartite',bipartite_dict)
    return edge_factor_graph;

def relabel_with_integers(g, return_dict=False):
    new_nodes = dict(zip(g.nodes() , range(g.number_of_nodes())));
    if return_dict==True:
        return nx.relabel_nodes(g,new_nodes), new_nodes;
    else:
        return nx.relabel_nodes(g,new_nodes);
        
 
def mis_check(g,nodes):
    neighbors = set.union(*[set(g.neighbors(v)) for v in nodes])
    if set.intersection(neighbors, nodes):
        return 0;
    else:
        return 1;
   
def generate_graph_instance(G):
    import networkx as nx;
    from random import random;
    g = nx.Graph();
    g.add_nodes_from(G.nodes());
    w = nx.get_edge_attributes(G,'weight');
    ## this is only the trivial normalization on the maximum edge weight
    ## which implies that edge will always be present.
    ## different normalizations can easily implemented
    w = dict(zip(w.keys(), np.array(w.values()) / float(np.max(w.values()))));
    for e in w:
        if random()<w[e]:
            g.add_edge(e[0],e[1]);
    return g;


    

