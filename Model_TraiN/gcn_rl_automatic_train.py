import numpy as np
from collections import defaultdict
from torch.distributions import Categorical
import torch
import os
import random
import pickle
from store import embeddings
import pandas as pd
import time
import copy
import config
from torch.nn import Linear
from random import sample
from torch.nn.functional import softmax
import torch.nn as nn
from torch_geometric.nn import GCNConv, ChebConv
from torch_geometric.data import Data
import networkx as nx
from extract_substrate import extractSubstrate
from vne_u import create_vne
import torch.multiprocessing as mp


class FeatureMatrixSubstrate:
    #feature classsss
    def __init__(self,graph) -> None:
        self.sub=graph
        self.crb=list()
        self.betweeness=list()
        self.degree=list()
        self.node_max_list=list()
        self.band_max_list=list()
        self.avail_bandwidth=list()

    def calc_node_max(self):
        # print("node max feature ",self.sub.node_max)
        for i in range(self.sub.nodes):
            self.node_max_list.append(self.sub.node_max[i])
        
    def calc_crb(self):
        for i in range(self.sub.nodes):
            self.crb.append(self.sub.node_weights[i])
    
    def calc_band_max(self):
        self.sub.get_max_bandwidth()
        for i in range(self.sub.nodes):
            self.band_max_list.append(self.sub.band_max[str(i)])

    def calc_bandwidth(self):
        # self.band_max=dict()
        for a in range(self.sub.nodes):
            sum_band=0
            for b in self.sub.neighbours[a]:
                sum_band+=self.sub.edge_weights[(str(a),b)]
            self.avail_bandwidth.append(sum_band)
        
    def normalize_crb(self):
        maxi = max(self.node_max_list)
        for i in range(len(self.crb)):
            self.crb[i] = self.crb[i]/maxi
            # normalization technique
            self.node_max_list[i] /= maxi

    def normalize_bw(self):
        maxi = max(self.band_max_list)
        for i in range(len(self.band_max_list)):
            self.avail_bandwidth[i] /= maxi
            self.band_max_list[i] /= maxi


    def generate_feature_matrix(self):
        self.calc_band_max()
        self.calc_node_max()
        self.calc_crb()
        self.calc_bandwidth()
        self.normalize_crb()
        self.normalize_bw()

        feature_matrix=[self.crb,self.avail_bandwidth,self.node_max_list,self.band_max_list]
        # feature_matrix=[self.crb,self.avail_bandwidth,self.degree,self.betweeness]
        feature_matrix = torch.tensor(feature_matrix,dtype=torch.float32)
        # All rows should have nodes and columns should be feature so transposing
        feature_matrix= torch.t(feature_matrix)
        # print("Feature Matrix Substrate",feature_matrix)
        return feature_matrix

class FeatureVNR:

    def __init__(self,graph):
        self.vnr=graph
        self.crb=list()
        self.betweeness=list()
        self.degree=list()
        self.bandwidth=list()

    def calc_crb(self):
        for i in range(self.vnr.nodes):
            self.crb.append(self.vnr.node_weights[i])

    #Bandwidth= sum(bandwidth of neighbours)
    def calc_bandwidth(self):
        for a in range(self.vnr.nodes):
            sum_band=0
            for b in self.vnr.neighbours[a]:
                sum_band+=self.vnr.edge_weights[(str(a),b)]
            self.bandwidth.append(sum_band)


    def normalize_crb(self):
        maxi = max(self.crb)
        for i in range(len(self.crb)):
            self.crb[i] = self.crb[i]/maxi

    def normalize_bw(self):
        maxi = max(self.bandwidth)
        for i in range(len(self.bandwidth)):
            self.bandwidth[i] /= maxi

    #Calculates all the features
    def generate_feature_matrix(self):
        self.calc_crb()
        self.calc_bandwidth()
        self.normalize_bw()
        self.normalize_crb()
        
        feature_matrix=[self.crb,self.bandwidth]
        feature_matrix = torch.tensor(feature_matrix,dtype=torch.float32)

        # All rows should have nodes and columns should be feature..So we are transposing
        feature_matrix = torch.t(feature_matrix)
        return feature_matrix

def create_data(features,graph):
        # features is a tensor
        edge_index=[]
        rows,cols=[],[]
        #Storing the row position of each edge in row matrix
        # It's corresponding column in column edge
        for i in range(len(graph.neighbours)):
            for j in (graph.neighbours[i]):
                rows.append(i)
                cols.append(int(j))
        edge_index.append(rows)
        edge_index.append(cols)

        edge_index=torch.tensor(edge_index)
        # print("edge_indices=",edge_index)
        # Converting it into Data object which has features and edge info's
        # data=Data(x=torch.tensor(features),edge_index=edge_index)
        data  = Data(x = features.clone().detach().requires_grad_(True),edge_index=edge_index)

        return data 

# change flags
# embedding: do the action given by get_action
# give the state (appending subs and vnr graphs) in list of Data format
# store list of states

class Environment:
    # state is a list of data objects containing [sub,vnr]
    def get_state(self,substrate,vnr):
        # substrate and vnr are graph objects
        # Getting features of substrate
        x_sub = FeatureMatrixSubstrate(substrate)
        feature_subs=x_sub.generate_feature_matrix()

        #Creating the Data object for substrate and vnr inorder to pass it to GCN 
        sub_data=create_data(feature_subs,substrate)
        mask=np.array([-1]*len(sub_data.x))
        sub_data.mask = torch.from_numpy(mask)

        y=FeatureVNR(vnr)
        feature_vnr=y.generate_feature_matrix()
        vnr_data=create_data(feature_vnr,vnr)
        mask=np.array([-1]*len(vnr_data.x))
        vnr_data.mask = torch.from_numpy(mask)
        return [sub_data, vnr_data]


    def step(self,state, sub_graph, vnr_graph, vnode,snode,embeddings):
        sub = state[0]
        vnr = state[1]
        
        if(sub_graph.node_weights[snode] >= vnr_graph.node_weights[vnode] and snode!=-1):
            
            # link embedding for non-initial nodes
            for virt_node in range(vnr.x.shape[0]):
                if(vnr.mask[virt_node] != -1 and str(virt_node) in vnr_graph.neighbours[vnode] and virt_node!=vnode):
                    vweight = vnr_graph.edge_weights[(str(vnode),str(virt_node))]

                    sub1 = vnr.mask[virt_node].item()
                    sub2 = snode
                    # already filters paths that can satify the bw constraint
                    paths = sub_graph.findPathFromSrcToDst(str(sub1),str(sub2),vweight)
                    if(len(paths)!=0):
                        path = paths[0]
                        # print("Path=",path)
                        for i in range(len(path)-1):
                            sub_graph.edge_weights[str(path[i]),str(path[i+1])] -= vweight 
                            sub_graph.edge_weights[str(path[i+1]),str(path[i])] -= vweight
                        embeddings.store_link(vnode, virt_node, vweight, path)
                            # print(f"Link {vnode} to {virt_node} is put on {path[i]} to {path[i+1]}")  
            # print("VNR = ",vne_list[0].x)
                    else:
                        # print(f"Path between {vnode} and {virt_node} failed")
                        reward = self.get_reward(embeddings,vnode,sub_graph,vnr_graph.nodes,False)
                        return state, reward, False
                    
            # in case current node has no neighbours until now or has successfully mapped links with neighbours
            # print(f"vnr weight {vnr_graph.node_weights[vnode]}")
            vnr.mask[vnode] = snode
            sub.mask[snode] = vnode
            sub_graph.node_weights[snode] -= vnr_graph.node_weights[vnode]
            embeddings.store_snode(vnode,snode,vnr_graph.node_weights[vnode])
            
            # print("Masks=",vnr.mask)
            x_sub = FeatureMatrixSubstrate(sub_graph)
            feature_subs=x_sub.generate_feature_matrix()
            # print("number of sub features=",feature_subs.shape)

            #Creating the Data object for substrate and vnr inorder to pass it to GCN 
            sub_data=create_data(feature_subs,sub_graph)
            sub_data.mask = sub.mask
            
            new_state = [sub_data,vnr]
            reward = self.get_reward(embeddings,vnode,sub_graph,vnr_graph.nodes,True)
            return new_state, reward, True
                
        else:

            reward = self.get_reward(embeddings,vnode,sub_graph,vnr_graph.nodes,False)
            return state, reward, False



    def deallocate(self,embeddings, sub_graph):
        for key, value in embeddings.snode_map.items():
            vnode = key
            snode = value
            vweight = embeddings.vnode_weights[vnode]
            sub_graph.node_weights[snode] += vweight
            # print(f"Deallocating {vweight}")
        
        for key, value in embeddings.link_path.items():
            (vnode1, vnode2) = key
            vweight = embeddings.link_weights[(vnode1,vnode2)]
            path = value
            for i in range(len(path)-1):
                sub_graph.edge_weights[str(path[i]),str(path[i+1])] += vweight 
                sub_graph.edge_weights[str(path[i+1]),str(path[i])] += vweight

        embeddings.clear()


    def get_reward(self,embeddings,cur_node,sub_graph,total,done):
        sigma = (cur_node+1)/total
        if not done:
            return -100*sigma
        
        r1 = 100*sigma

        rev=cost=0
        tuple_list = list(embeddings.link_path.keys())      # pair of nodes for link
        path_list = list(embeddings.link_path.values())
        for i in range(len(embeddings.link_weights)):
            # print("Items=",embeddings.link_path.items())
            if cur_node in tuple_list[i]:
                cost += embeddings.link_weights[tuple_list[i]]*(len(path_list[i])-1)
                rev += embeddings.link_weights[tuple_list[i]]
        rev //= 2
        cost //= 2
        rev += embeddings.vnode_weights[cur_node]
        cost += embeddings.vnode_weights[cur_node]
        r2 = rev/cost

        # load balancing
        snode = embeddings.snode_map[cur_node]
        max_cpu = sub_graph.node_max[snode]
        avail_cpu = sub_graph.node_weights[snode]
        r3 = avail_cpu / max_cpu

        # congestion reward
        sum_rat = count = 0
        for i in range(len(embeddings.link_weights)):
            if cur_node in tuple_list[i]:
                for j in range(len(path_list[i])-1):
                    c = sub_graph.edge_weights[str(path_list[i][j]),str(path_list[i][j+1])]
                    c +=  embeddings.link_weights[tuple_list[i]]
                    sum_rat += embeddings.link_weights[tuple_list[i]]/ c
                    count += 1
        if(count != 0):
            r4 = sum_rat / count
        else:
            r4 = 0
                

        reward = (r1*r2*r3)*(1-r4)
        # print(f"Reward = {reward}"
        return reward
    

    def schedulablePair(self, virt_graph, subs_graph):
        # both graphs are data objects 
        candidate_subs_dict = defaultdict(list)
        # vnr_node_num: [snode1, snode2...]  vnode with list of all eligible snodes
        for i, virt_feat in enumerate(virt_graph.x):
            # if already mapped then continue to the next vnr node
            # VNR node is -1 then it is not mapped yet.
            if virt_graph.mask[i] != -1:
                continue
            # if not mapped, check if any node of the vnr is already mapped onto any node of the substrate
            for j, subs_feat in enumerate(subs_graph.x):
                if subs_graph.mask[j] != -1:
                    continue
                candidate_subs_dict[i].append(j)
        return candidate_subs_dict 

    def epsilon_greedy_action(self,probs,candidate_subs_nodes, epsilon,vnode):
        if random.random() < epsilon:
            # Choose a random action
            action = torch.randint(0, len(candidate_subs_nodes), (1,))
            action = candidate_subs_nodes[action]
        else:
            # Choose the greedy action based on the actor's output probabilities
            x = torch.clone(probs)
            action = torch.argmax(x)
            count = 0
            while(action not in candidate_subs_nodes and count<=vnode):
                x[action] = -1
                action = torch.argmax(x)
                count+=1
        return action


    def chooseAction(self, state,vnode_num,probs,epsilon):
        subs = state[0]
        vnr = state[1]
        
        candidate_subs_dict = self.schedulablePair(vnr,subs)
        candidate_subs_nodes = candidate_subs_dict[vnode_num]
        subs_node = self.epsilon_greedy_action(probs,candidate_subs_nodes, epsilon,vnode_num)
       
        subs_node = torch.tensor(subs_node)
        return vnode_num, subs_node
        
  

class GCNNBlock(torch.nn.Module):

    def __init__(self, in_features) -> None:
        super(GCNNBlock, self).__init__()
        self.in_features = in_features
        # self.k = k
        # self.args = args
        self.setup_layers()
    
    def setup_layers(self):
        # setting up all the layers
        self.gcnn_layer = ChebConv(self.in_features, 8, 3)  # in, out, k-hops
        self.init_weights()
    
    
    def init_weights(self):
        for name, param in self.gcnn_layer.named_parameters():
            if "weight" in name:
                torch.nn.init.uniform_(param.data,0.0,0.5)
        
    def forward(self, input):
        conv = self.gcnn_layer(input.x, input.edge_index)
        conv = torch.nn.functional.relu(conv)
        # print("conv = ",conv)
        return conv


class ActorCritic(nn.Module):

    def __init__(self,num_sub):
        super(ActorCritic,self).__init__()
        self.subs_gcn=GCNNBlock(4)
        self.vnr_conn = Linear(2,8)
        self.policy = Linear(2*8*num_sub,num_sub)
        self.value_head = Linear(2*8*num_sub, 1)
        self.virt_graph = torch.zeros(5,4)      # CHANGE THIS. PROBABLY NOT TENSOR

    def gcn_part(self,vnr,j):
        if(j==0):
            self.virt_graph = self.vnr_conn(vnr.x)      # vnr nodes x 128


    def forward(self,state,j):
        subs = state[0]
        vnr = state[1]
        # subs and vnr are of data format
        
        self.gcn_part(vnr,j)
        self.subs_graph = self.subs_gcn(subs)  

        # print("Virt graph = ",self.virt_graph)
        vnode_num = j
        virt_node = self.virt_graph[vnode_num]   # get the node features of the selected vnr node

        virt_node = virt_node.repeat(self.subs_graph.shape[0],1)
        fully_conn = torch.flatten(torch.cat((virt_node,self.subs_graph)))
        indices = torch.randperm(fully_conn.shape[0])
        fully_conn = fully_conn[indices]
        fully_conn = torch.nn.functional.tanh(fully_conn)
        nodes_layer = self.policy(fully_conn)
        probs = softmax(nodes_layer,dim=0)
        
        # value function
        value = self.value_head(fully_conn)
        return probs, value

class Agent(mp.Process):

    def __init__(self, substrate,global_network, optimizer, ep_id, total_eps,epsilon):
        super(Agent, self).__init__()
        self.first_substrate = substrate
        self.substrate = substrate
        self.environment = Environment()
        self.episode_id = ep_id
        self.local_network = ActorCritic(self.substrate.nodes)
        self.global_network = global_network
        self.optimizer = optimizer
        self.total_episodes = total_eps
        self.epsilon = epsilon

    def run(self):
        while(self.episode_id.value < self.total_episodes):
            self.substrate = extractSubstrate('5_uniform.pickle')       # resetting the environment
            print("Episode number = ",self.episode_id.value)
            self.clear_memory()
            episode_reward = 0
            # if(self.episode_id.value%1000==0 and self.episode_id.value!=0):
            #     torch.save(self.global_network.state_dict(), f'models/actor_critic_model_automatic{8000+self.episode_id.value}.pth')
            # Getting VNRs from create_vne file based on req_no
            req_no = random.randint(10,30)
            vne_list = create_vne(no_requests = req_no)
            print(f"total vnrs {len(vne_list)}")
            embs = []
            state = self.environment.get_state(self.substrate,vne_list[0])
            epsi = max(self.epsilon[1],self.epsilon[0]-self.epsilon[2]*self.episode_id.value)

            for i in range(req_no):
                # for each vnr create object to store embedding info for each vnr
                embs.append(embeddings(i))
                
                # for each node inside the vnr
                for j in range(len(state[1].x)):
                    probs, value = self.local_network(state,j)      # actor prob, critic value
                    vnode, snode = self.environment.chooseAction(state,j,probs,epsi)     # embed vnode onto snode
                    # print(f"probs ={probs}")
                    # print(f"vnode {vnode} onto snode {snode}")
                    # do the embedding
                    # by default, next state has (sub, current vnr), whether vnr has finished or not.
                    # will overwrite next state later if needed
                    next_state, reward, vnom_done = self.environment.step(state, self.substrate, vne_list[i],vnode,snode.item(),embs[i])

                    episode_reward += reward

                    # if node embedding not done, deallocate all embeddings until now
                    # and get the next VNR into the new state
                    if not vnom_done:
                        self.environment.deallocate(embs[i],self.substrate)
                        # print(f"Deallocated VNR = {i}")
                        if i<req_no-1:
                            next_state = self.environment.get_state(self.substrate,vne_list[i+1])
                            state = next_state.copy()
                            self.remember(next_state,probs,snode,value,reward)
                        break
                    
                    # if not terminal state, and last node embedding is done for the current vnr
                    if i<req_no-1 and j==len(state[1].x)-1:
                        next_state = self.environment.get_state(self.substrate,vne_list[i+1])
                    state = next_state.copy()

                    self.remember(next_state,probs,snode,value,reward)

                # flag to see if end of episode or not (terminal state)
                if i!=req_no-1:
                    flag = False
                else:
                    flag = True
                
                if len(self.values)==0:
                    break

                loss = self.calc_loss(flag)
                self.optimizer.zero_grad()
                self.clear_memory()
                loss.backward()

                for local_param, global_param in zip(
                                self.local_network.parameters(),
                                self.global_network.parameters()):
                    global_param._grad = local_param.grad

                self.optimizer.step()
                self.local_network.load_state_dict(
                    self.global_network.state_dict())
                self.clear_memory()
                
            with self.episode_id.get_lock():
                self.episode_id.value += 1
            
            print("Episode final reward = ",episode_reward)
   


    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.probs = []


    def remember(self, state, probs, action, value, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.probs.append(probs)


    def calcReturns(self, done):
        if not done:
            _, value = self.local_network(self.states[-1],0)
            # if not terminal state, calculate loss using the value of the next state
        else:
            value = 0
        gamma = 0.99
        R = value  # initial reward for that state
        batch_return = []
        for reward in self.rewards[::-1]:
            R = reward + gamma * R     # gamma*reward(next state)
            batch_return.append(R)
        batch_return.reverse()
        batch_return = torch.tensor(batch_return, dtype=torch.float)
        # print("Returns = ",batch_return)
        return batch_return


    def calc_loss(self, done):
        actions = torch.tensor(self.actions, dtype=torch.float)
        returns = self.calcReturns(done)
        values = []
        values = torch.cat(self.values)
        critic_loss = (returns-values)**2
        probs = torch.stack(self.probs)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        actor_loss = -log_probs*(returns-values)
        total_loss = (critic_loss + actor_loss).mean()  # good approximation
        return total_loss