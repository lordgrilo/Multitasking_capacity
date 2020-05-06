
# coding: utf-8

# In[6]:

import os, sys
import numpy as np
notebook_mode = False;

if notebook_mode==False:
    if len(sys.argv)==1:
        print 'input required: \n number of graphs to produce \n layer size \n overlap \n output directory'
        sys.exit()
    number_of_graphs = int(sys.argv[1]);
    graph_layer_size = int(sys.argv[2]);
    overlap = float(sys.argv[3]);
    output_directory = sys.argv[4]
    
else:
    number_of_graphs = 10
    graph_layer_size = 30
    overlap = 3
    output_directory = './test/'
    
if not os.path.exists(output_directory):
    os.makedirs(output_directory);



# In[7]:

import IGtools as igt
import networkx as nx


# In[13]:

subdirs = ['ER','regular','CM','CL'];
for subdir in subdirs:
    if not os.path.exists(output_directory+'/'+subdir):
        os.makedirs(output_directory+'/'+subdir);


# For a general degree sequence, one case use the following function mm create the desired bipartite graph:

# In[10]:

for i in range(number_of_graphs):
    g = igt.create_ER_bipartite_graph(graph_layer_size,overlap/float(graph_layer_size-1));
    print g.number_of_nodes(), g.number_of_edges(), np.mean(nx.degree(g).values()), overlap/float(graph_layer_size-1)
    nx.write_edgelist(g,output_directory+'/ER/edgelist-'+str(i)+'.edges',data=[]);


# In[11]:

for i in range(number_of_graphs):
    g = igt.create_configuration_model_graph(graph_layer_size*[int(overlap)]);
    print g.number_of_nodes(), g.number_of_edges(), np.mean(nx.degree(g).values())
    nx.write_edgelist(g,output_directory+'/regular/edgelist-'+str(i)+'.edges',data=[]);


# In[12]:

for i in range(number_of_graphs):
    degseq = np.random.randint(1,graph_layer_size,graph_layer_size)
    g = igt.create_configuration_model_graph(degseq);
    print g.number_of_nodes(), g.number_of_edges(), np.mean(nx.degree(g).values())
    nx.write_edgelist(g,output_directory+'/CM/edgelist-'+str(i)+'.edges',data=[]);


# In[14]:

for i in range(number_of_graphs):
    degseq = np.random.randint(1,graph_layer_size,graph_layer_size)
    g = igt.create_expected_degree_graph(degseq);
    print g.number_of_nodes(), g.number_of_edges(), np.mean(nx.degree(g).values())
    nx.write_edgelist(g,output_directory+'/CL/edgelist-'+str(i)+'.edges',data=[]);

