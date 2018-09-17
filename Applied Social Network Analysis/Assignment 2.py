
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.2** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-social-network-analysis/resources/yPcBs) course resource._
# 
# ---

# # Assignment 2 - Network Connectivity
# 
# In this assignment you will go through the process of importing and analyzing an internal email communication network between employees of a mid-sized manufacturing company. 
# Each node represents an employee and each directed edge between two nodes represents an individual email. The left node represents the sender and the right node represents the recipient.

# In[3]:

import networkx as nx

# This line must be commented out when submitting to the autograder
#!head email_network.txt


# ### Question 1
# 
# Using networkx, load up the directed multigraph from `email_network.txt`. Make sure the node names are strings.
# 
# *This function should return a directed multigraph networkx graph.*

# In[4]:

def answer_one():
    
    G = nx.read_edgelist('email_network.txt', delimiter='\t', data=[('timestamp', int)], create_using=nx.MultiDiGraph())
    
    return G
#answer_one()


# ### Question 2
# 
# How many employees and emails are represented in the graph from Question 1?
# 
# *This function should return a tuple (#employees, #emails).*

# In[5]:

def answer_two():
    G = answer_one()
    num_emp = len(G.nodes())
    num_emails = len(G.edges())
    
    return (num_emp, num_emails)
#answer_two()


# ### Question 3
# 
# * Part 1. Assume that information in this company can only be exchanged through email.
# 
#     When an employee sends an email to another employee, a communication channel has been created, allowing the sender to provide information to the receiver, but not vice versa. 
# 
#     Based on the emails sent in the data, is it possible for information to go from every employee to every other employee?
# 
# 
# * Part 2. Now assume that a communication channel established by an email allows information to be exchanged both ways. 
# 
#     Based on the emails sent in the data, is it possible for information to go from every employee to every other employee?
# 
# 
# *This function should return a tuple of bools (part1, part2).*

# In[7]:

def answer_three():
        
    # Your Code Here
    G = answer_one()
    return (nx.is_strongly_connected(G), nx.is_weakly_connected(G))
#answer_three()


# ### Question 4
# 
# How many nodes are in the largest (in terms of nodes) weakly connected component?
# 
# *This function should return an int.*

# In[27]:

def answer_four():
        
    G = answer_one()
    comps = sorted(nx.weakly_connected_components(G))

    
    return len(max(comps, key = len))
#answer_four()


# ### Question 5
# 
# How many nodes are in the largest (in terms of nodes) strongly connected component?
# 
# *This function should return an int*

# In[28]:

def answer_five():
        
    G = answer_one()
    comps = sorted(nx.strongly_connected_components(G))
    
    
    return len(max(comps,key = len))
#answer_five()


# ### Question 6
# 
# Using the NetworkX function strongly_connected_component_subgraphs, find the subgraph of nodes in a largest strongly connected component. 
# Call this graph G_sc.
# 
# *This function should return a networkx MultiDiGraph named G_sc.*

# In[30]:

def answer_six():
        
    G = answer_one()
    sub_grphs = nx.strongly_connected_component_subgraphs(G)
    G_sc = max(sub_grphs, key=len)
    
    return G_sc

#answer_six()


# ### Question 7
# 
# What is the average distance between nodes in G_sc?
# 
# *This function should return a float.*

# In[33]:

def answer_seven():
    G_sc = answer_six()
    return nx.average_shortest_path_length(G_sc)
answer_seven()


# ### Question 8
# 
# What is the largest possible distance between two employees in G_sc?
# 
# *This function should return an int.*

# In[35]:

def answer_eight():
    G_sc = answer_six()
    
    
    return nx.diameter(G_sc)# Your Answer Here
answer_eight()


# ### Question 9
# 
# What is the set of nodes in G_sc with eccentricity equal to the diameter?
# 
# *This function should return a set of the node(s).*

# In[38]:

def answer_nine():
       
    # Your Code Here
    G_sc = answer_six()
    return set(nx.periphery(G_sc))
answer_nine()


# ### Question 10
# 
# What is the set of node(s) in G_sc with eccentricity equal to the radius?
# 
# *This function should return a set of the node(s).*

# In[40]:

def answer_ten():
        
    # Your Code Here
    G_sc = answer_six()
    return set(nx.center(G_sc))
answer_ten()


# ### Question 11
# 
# Which node in G_sc is connected to the most other nodes by a shortest path of length equal to the diameter of G_sc?
# 
# How many nodes are connected to this node?
# 
# 
# *This function should return a tuple (name of node, number of satisfied connected nodes).*

# In[55]:

def answer_eleven():
        
    G_sc = answer_six()
    d = nx.diameter(G_sc)
    peri = nx.periphery(G_sc)
    max_count = -1
    for n in peri:
        sp= nx.shortest_path_length(G_sc, n)
        count = list(sp.values()).count(d)
        if count > max_count:
            max_count = count
            max_count_node = n
    
    return (max_count_node, max_count)
answer_eleven()


# ### Question 12
# 
# Suppose you want to prevent communication from flowing to the node that you found in the previous question from any node in the center of G_sc, what is the smallest number of nodes you would need to remove from the graph (you're not allowed to remove the node from the previous question or the center nodes)? 
# 
# *This function should return an integer.*

# In[70]:

def answer_twelve():
        
    G_sc = answer_six()
    center = nx.center(G_sc)
    node = answer_eleven()[0]
    rslt = set()
    for c in center : 
        tmp = nx.minimum_node_cut(G_sc, c, node)
        for i in tmp:
            if rslt is None:
                rslt = set(i)
            else:
                rslt.add(i)
    
    
    
    return len(rslt)
answer_twelve()


# ### Question 13
# 
# Construct an undirected graph G_un using G_sc (you can ignore the attributes).
# 
# *This function should return a networkx Graph.*

# In[74]:

def answer_thirteen():
    G_sc = answer_six()
    und = G_sc.to_undirected()
    G_un = nx.Graph(und)
    return G_un
answer_thirteen()


# ### Question 14
# 
# What is the transitivity and average clustering coefficient of graph G_un?
# 
# *This function should return a tuple (transitivity, avg clustering).*

# In[76]:

def answer_fourteen():
    G_un = answer_thirteen()
    trans = nx.transitivity(G_un)
    avg_clustering = nx.average_clustering(G_un)
    return (trans, avg_clustering)
answer_fourteen()


# In[ ]:



