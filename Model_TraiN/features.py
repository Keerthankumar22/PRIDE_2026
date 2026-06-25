from datetime import datetime, date
from network_attributes import NetworkAttribute
from collections import OrderedDict
import networkx as nx
import torch
from graph_u import Graph

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

    def calc_degree(self):
        for i in range(self.sub.nodes):
            self.degree.append(len(self.sub.neighbours[i]))

    def calc_bandwidth(self):
        # self.band_max=dict()
        for a in range(self.sub.nodes):
            sum_band=0
            for b in self.sub.neighbours[a]:
                sum_band+=self.sub.edge_weights[(str(a),b)]
            self.avail_bandwidth.append(sum_band)
        

    def calc_betweenesss(self):
        sn = dict()
        bet=dict()
        for i in range(self.sub.nodes):
            ls = []
            for nd in self.sub.neighbours[i]:
                ls.append(int(nd))
            sn[str(i)] = ls
        bet=nx.betweenness_centrality(nx.Graph(sn))
        for i in range(self.sub.nodes):
            self.betweeness.append(bet[str(i)])
        
    def normalize_crb(self):
        maxi = max(self.node_max_list)
        maxr = max(self.crb)
        minr = min(self.crb)
        for i in range(len(self.crb)):
            self.crb[i] = (self.crb[i]-minr)/(maxr-minr)
            # normalization technique
            self.node_max_list[i] /= maxi

    def normalize_bw(self):
        maxi = max(self.band_max_list)
        maxr = max(self.avail_bandwidth)
        minr = min(self.avail_bandwidth)
        for i in range(len(self.band_max_list)):
            self.avail_bandwidth[i] = (self.avail_bandwidth[i]-minr)/(maxr-minr)
            self.band_max_list[i] /= maxi

    def normalize_deg_betw(self):
        max_degree = max(self.degree)
        max_bet = max(self.betweeness)
        for i in range(len(self.degree)):
            self.degree[i] /= max_degree
            self.betweeness[i] /= max_bet

    def generate_feature_matrix(self):
        self.calc_band_max()
        self.calc_node_max()
        self.calc_crb()
        self.calc_bandwidth()
        self.normalize_crb()
        self.normalize_bw()
        self.calc_betweenesss()
        self.calc_degree()
        self.normalize_deg_betw()

        feature_matrix=[self.crb,self.avail_bandwidth,self.node_max_list,self.band_max_list,self.degree,self.betweeness]
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

    def calc_degree(self):
        for i in range(self.vnr.nodes):
            self.degree.append(len(self.vnr.neighbours[i]))

    #Bandwidth= sum(bandwidth of neighbours)
    def calc_bandwidth(self):
        for a in range(self.vnr.nodes):
            sum_band=0
            for b in self.vnr.neighbours[a]:
                sum_band+=self.vnr.edge_weights[(str(a),b)]
            self.bandwidth.append(sum_band)

    def calc_betweenesss(self):
        sn = dict()
        bet=dict()

        for i in range(self.vnr.nodes):
            ls = []
            for nd in self.vnr.neighbours[i]:
                ls.append(int(nd))
            sn[str(i)] = ls
        bet=nx.betweenness_centrality(nx.Graph(sn))
        for i in range(self.vnr.nodes):
            self.betweeness.append(bet[str(i)])
        

    def normalize_crb(self):
        maxi = max(self.crb)
        for i in range(len(self.crb)):
            self.crb[i] = self.crb[i]/maxi

    def normalize_bw(self):
        maxi = max(self.bandwidth)
        for i in range(len(self.bandwidth)):
            self.bandwidth[i] /= maxi

    def normalize_deg_betw(self):
        max_degree = max(self.degree)
        max_bet = max(self.betweeness)
        if(max_degree!=0):
            for i in range(len(self.degree)):
                self.degree[i] /= max_degree
        if(max_bet!=0):
            for i in range(len(self.degree)):
                self.betweeness[i] /= max_bet

    #Calculates all the features
    def generate_feature_matrix(self):
        self.calc_crb()
        self.calc_betweenesss()
        self.calc_degree()
        self.calc_bandwidth()
        self.normalize_bw()
        self.normalize_crb()
        self.normalize_deg_betw()
        
        feature_matrix=[self.crb,self.bandwidth,self.degree,self.betweeness]
        feature_matrix = torch.tensor(feature_matrix,dtype=torch.float32)

        # All rows should have nodes and columns should be feature..So we are transposing
        feature_matrix = torch.t(feature_matrix)
        return feature_matrix




    

    

        

        
    

    




    


