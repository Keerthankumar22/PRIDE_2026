import numpy as np
from collections import defaultdict
from torch.distributions import Categorical
import torch
import random
from store import embeddings
import time
import copy
import config
from torch.nn import Linear
from torch.nn.functional import softmax
import torch.nn as nn
from torch_geometric.nn import GCNConv, ChebConv
from torch_geometric.data import Data
import networkx as nx

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

def automatic():
    def findavgLinkutil(initial_sub,final_sub):
        avglink = 0
        utilised = 0
        for edges in final_sub.edge_weights:
            if final_sub.edge_weights[edges]!=initial_sub.edge_weights[edges]:
                utilised += 1
                avglink += (initial_sub.edge_weights[edges]-final_sub.edge_weights[edges])/(initial_sub.edge_weights[edges])

        avglink /= 2
        avglink = avglink/(utilised//2)
        return avglink

    def findavgNodeUtil(initial_sub,final_sub):

        avg_node = 0
        utilised = 0
        for nodes in final_sub.node_weights:
            if final_sub.node_weights[nodes]!=initial_sub.node_weights[nodes]:
                utilised += 1
                avg_node += (initial_sub.node_weights[nodes]-final_sub.node_weights[nodes])/(initial_sub.node_weights[nodes])
        avg_node = avg_node/utilised
        return avg_node

    def findRev(embed):
        # find revenue for one vnr given all embeddings
        tot_rev = 0
        tot_cost = 0
        for emb in range(len(embed)):
            rev = 0
            cost = 0
            tuple_list = list(embed[emb].link_path.keys())      # pair of nodes for link
            path_list = list(embed[emb].link_path.values())
            for i in range(len(embed[emb].link_weights)):
                cost += embed[emb].link_weights[tuple_list[i]]*(len(path_list[i])-1)
                rev += embed[emb].link_weights[tuple_list[i]]
                # print("Path list is ", (path_list[i]))
            rev //= 2
            cost //= 2
            for j in range(len(embed[emb].vnode_weights)):
                rev += embed[emb].vnode_weights[j]
                cost += embed[emb].vnode_weights[j]
            tot_rev += rev
            tot_cost += cost
        if(tot_rev!=0 and tot_cost!=0):
            return (tot_rev,tot_cost)
        return 0,0


    def findCongestion(initial_sub,final_sub):
        
        total_links = len(initial_sub.edge_weights)//2
        avglink = 0
        for edges in final_sub.edge_weights:
            if final_sub.edge_weights[edges]!=initial_sub.edge_weights[edges]:
                avglink += (initial_sub.edge_weights[edges]-final_sub.edge_weights[edges])/(initial_sub.edge_weights[edges])
        avglink /= 2
        return avglink/total_links

    def findAvgPathLength(embedding):
        avgpathlength = 0
        vnr_accepted = 0
        for emb in range(len(embedding)):
            total_edges = 0
            path_len = 0
            path_list = list(embedding[emb].link_path.values())
            for path in path_list:
                path_len += len(path)-1
            path_len = path_len//2
            total_edges += len(embedding[emb].link_weights)//2
            if total_edges!=0:
                avgpathlength += path_len/total_edges
                vnr_accepted+=1
        if(vnr_accepted!=0):    
            return avgpathlength/vnr_accepted
        return 0
            # print(f"For {emb} , avgpathlength = {avgpathlength}")   
                    
    def consumed_resources(inital_sub,final_sub):
        ini_total_res = 0
        final_total_res = 0
        for edges in inital_sub.edge_weights:
            ini_total_res += inital_sub.edge_weights[edges]
            final_total_res += final_sub.edge_weights[edges]

        ini_total_res /=2
        final_total_res /=2

        for nodes in inital_sub.node_weights:
            ini_total_res += inital_sub.node_weights[nodes]
            final_total_res += final_sub.node_weights[nodes]
        return (ini_total_res, final_total_res)

    def avgLink(initial_sub, final_sub):
        utilized_links = 0
        for edge in initial_sub.edge_weights:
            if initial_sub.edge_weights[edge]!=final_sub.edge_weights[edge]:
                utilized_links += 1
        avg_link = utilized_links/(len(initial_sub.edge_weights))
        return utilized_links//2, avg_link/2


    def avgNode(copy_sub,substrate):
        utilized_nodes = 0
        for node in substrate.node_weights:
            if substrate.node_weights[node] != copy_sub.node_weights[node]:
                utilized_nodes += 1
        avg_node = utilized_nodes/(substrate.nodes)
        return utilized_nodes, avg_node

    def avgNodeStress(embedddings, sub):
        vnodes = count = 0
        for emb in embedddings:
            vnodes+= len(emb.vnode_weights)
            count += 1
        snodes = len(sub.node_weights)
        return vnodes/(snodes*count)

    def avgLinkStress(embeddings, sub):
        vlinks = count = 0
        for emb in embeddings:
            vlinks += len(emb.link_weights)//2
            count += 1
        slinks = len(sub.edge_weights)//2
        return vlinks/(slinks*count)
    
    def vm_embedded(embs):
        total_count = 0
        for embedding in embs:
            total_count += len(embedding.vnode_weights)
        return total_count
    
    def vl_embedded(embs):
        total_count = 0
        for embedding in embs:
            total_count += len(embedding.link_weights)
        return total_count//2


    output_dict = {
        "algorithm": [],
        "total_vm_request":[],
        "total_vl_request":[],
        "total_vm_embedded":[],
        "total_vl_embedded":[],
        "revenue": [],
        "total_cost" : [],
        "revenuetocostratio":[],
        "accepted" : [],
        "total_request": [],
        "embeddingratio":[],
        "pre_resource": [],
        "post_resource": [],
        "consumed":[],
        "avg_bw": [],
        "avg_crb": [],
        "avg_link": [],
        "No_of_Links_used": [],
        "avg_node":[],
        "No_of_Nodes_used": [],
        "avg_path": [],
        "avg_exec": [],
        "avg_node_stress" : [],
        "avg_link_stress" : [],
        "avg_congestion":[]
        # "total_nodes": [],
        # "total_links": [],
    }

    print("\tGCN-RL automatic started")
    total_rev_demand=0
    # req_list = [50,100,150]   
    # substrate = extractSubstrate('4_uniform.pickle')
    substrate = copy.deepcopy(config.substrate)
    copy_substrate = copy.deepcopy(config.substrate)

    # Getting VNRs from create vne file based on req_no
    # req_no = random.randint(2,4)
    # for it in range(3):
    # vne_list = create_vne(no_requests = req_no)
    
    # geeky_file = open(f'vne_file_{req_no}+{it}.pickle', 'wb')
    # pickle.dump(vne_list, geeky_file)
    # geeky_file.close()
    
    vne_list = copy.deepcopy(config.vne_list)
    
    total_vm_request = 0
    total_vl_request = 0
    for vnr in vne_list:
        total_rev_demand += sum(vnr.node_weights.values()) + sum(vnr.edge_weights.values())//2
        total_vm_request += len(vnr.node_weights)
        total_vl_request += len(vnr.edge_weights)//2

    embs = []
    # print(f"before {vne_list}")
    # before this sort the vnr list based on revenue
   
    environment = Environment()
    state = environment.get_state(substrate,vne_list[0])
    actor_critic_model = ActorCritic(substrate.nodes)
    actor_critic_model.load_state_dict(torch.load('automatic_gcn_rl_baseline.pth'))
    actor_critic_model.eval()
    req_no = len(vne_list)
    vnr_accepted =0
    start_time = time.time()
    for i in range(req_no):
        embs.append(embeddings(i))
        
        for j in range(len(state[1].x)):

            probs, value = actor_critic_model(state,j)      # actor prob, critic value
            vnode, snode = environment.chooseAction(state,j,probs,-1)
            # print(f"Probs= {probs} , action = {snode}")
            # print(f"Vnode {vnode} onto snode {snode} with probs {probs[snode]}")
            
            next_state, reward, vnom_done = environment.step(state, substrate, vne_list[i],vnode,snode.item(),embs[i])

            if not vnom_done:
                environment.deallocate(embs[i],substrate)
                print("Deallocated")

                if i<req_no-1:
                    next_state = environment.get_state(substrate,vne_list[i+1])
                    state = next_state.copy()
                break
            if i<req_no-1 and j==len(state[1].x)-1:
                next_state = environment.get_state(substrate,vne_list[i+1])
            state = next_state.copy()
        # print()      
        if vnom_done:
            vnr_accepted += 1
        
    end_time = time.time() - start_time
    # print("Embs = ",embs)
    (rev,cost) = findRev(embs)
    path_len = findAvgPathLength(embs)
    link_util = findavgLinkutil(copy_substrate,substrate)*100
    node_util = findavgNodeUtil(copy_substrate,substrate)*100
    pre_resource, post_resource = consumed_resources(copy_substrate,substrate)
    acc_ratio = (vnr_accepted/req_no)*100
    avg_node_stress = avgNodeStress(embs,substrate)*100
    avg_link_stress = avgLinkStress(embs,substrate)*100
    avg_congestion = findCongestion(copy_substrate,substrate)
    total_vm_emdedded = vm_embedded(embs)
    total_vl_embedded = vl_embedded(embs)
    avg_congestion = findCongestion(copy_substrate,substrate)
    # avg_congestion = sum(cong)*1000/len(cong)
    # print(f"Vnr Acceptance Ratio = {acc_ratio}")
    utilized_links, avg_link = avgLink(copy_substrate,substrate)
    utilized_nodes, avg_node = avgNode(copy_substrate,substrate)
    avg_link *= 100
    avg_node *= 100
    # print(f"Time of execution = {end_time}")
    if cost!=0:
        r2c = (rev/cost)*100
    else:
        r2c = 0

    output_dict = {
        "revenue": rev,
        "total_cost" : cost,
        "accepted" : vnr_accepted,
        "total_request": len(vne_list),
        "pre_resource": pre_resource,
        "post_resource": post_resource,
        "avg_bw": link_util,
        "avg_crb": (node_util),
        "avg_link": avg_link,
        "No_of_Links_used": utilized_links,
        "avg_node": avg_node,
        "No_of_Nodes_used": utilized_nodes,
        "avg_path": path_len,
        "avg_exec": end_time,
        "avg_node_stress": avg_node_stress,
        "avg_link_stress": avg_link_stress,
        "total_vm_request":total_vm_request,
        "total_vlinks":total_vl_request,
        "total_vm_embedded":total_vm_emdedded,
        "total_vl_embedded":total_vl_embedded,
        "avg_congestion":avg_congestion
    }
    print("\tGCN-RL automatic ended")
    return output_dict
    # excel.to_excel(f"GCN_RL_algo_normal2.xlsx")