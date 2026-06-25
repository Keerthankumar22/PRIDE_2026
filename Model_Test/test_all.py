import os
import pickle
from time import sleep
import graph_extraction_uniform
from vne_u import create_vne
import torch
import pandas as pd
import time
from gcn_rl_normal import gcnrl
from gcn_rl_automatic import automatic
from eigen_without_cong import eigen_without_congestion as eg_without_cong
from without_cong import gcn_rl_without_cong
from automatic_w_cong import automatic_w_cong
import copy
from greedy import main as greedy
from rethinking import main as rethinking
from topsis_updated import main as topsis
from nrm import main as NRM
import config
import logging
from eigen import eigen


def generateSubstrate(for_automate, pickle_name):
    substrate, _ = for_automate(1)
    geeky_file = open(pickle_name, 'wb')
    pickle.dump(substrate, geeky_file)
    geeky_file.close()

def extractSubstrate(pickle_file):
    filehandler = open(pickle_file, 'rb')
    substrate = pickle.load(filehandler)
    return substrate

def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    # formatter = logging.Formatter('%(asctime)s %(levelname)s  : %(message)s')
    formatter = logging.Formatter('[%(levelname)s] : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler) 

output_dict = {
        "algorithm": [],
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
        "avg_node": [],
        "No_of_Nodes_used": [],
        "avg_path": [],
        "avg_exec": [],
        "avg_node_stress": [],
        "avg_link_stress": [],
        # "congestion_ratio": []
    }

def exec_greedy(tot=1):
    gred_out = greedy()
    sleep(tot*1)
    
    printToExcel(
        algorithm='GREEDY',
        revenue=gred_out['revenue'],
        total_cost=gred_out['total_cost'],
        revenuetocostratio=(gred_out['revenue']/gred_out['total_cost'])*100,
        accepted=gred_out['accepted'],
        total_request=gred_out['total_request'],
        embeddingratio=(gred_out['accepted']/gred_out['total_request'])*100,
        pre_resource=gred_out['pre_resource'],
        post_resource=gred_out['post_resource'],
        consumed=gred_out['pre_resource']-gred_out['post_resource'],
        avg_bw=gred_out['avg_bw'],
        avg_crb=gred_out['avg_crb'],
        avg_link=gred_out['avg_link'],
        No_of_Links_used=gred_out['No_of_Links_used'],
        avg_node=gred_out['avg_node'],
        No_of_Nodes_used=gred_out['No_of_Nodes_used'],
        avg_path=gred_out['avg_path'],
        avg_exec=gred_out['avg_exec'].total_seconds()*1000/gred_out['total_request'],
        avg_node_stress=gred_out['avg_node_stress']*100,
        avg_link_stress=gred_out['avg_link_stress']*100,
        avg_congestion=gred_out['avg_congestion']*100,
        total_vm_request = gred_out["total_vm_request"],
        total_vl_request = gred_out["total_vl_request"]//2,
        total_vm_embedded = gred_out["total_vm_embedded"],
        total_vl_embedded=gred_out["total_vl_embedded"]
        
    )


# Code for NORD

def exec_topsis(tot=1):
    topsis_out = topsis()
    sleep(tot*1)
    
    printToExcel(
        algorithm='TOPSIS',
        revenue=topsis_out['revenue'],
        total_cost=topsis_out['total_cost'],
        revenuetocostratio=(topsis_out['revenue']/topsis_out['total_cost'])*100,
        accepted=topsis_out['accepted'],
        total_request=topsis_out['total_request'],
        embeddingratio=(topsis_out['accepted']/topsis_out['total_request'])*100,
        pre_resource=topsis_out['pre_resource'],
        post_resource=topsis_out['post_resource'],
        consumed=topsis_out['pre_resource']-topsis_out['post_resource'],
        avg_bw=topsis_out['avg_bw'],
        avg_crb=topsis_out['avg_crb'],
        avg_link=topsis_out['avg_link'],
        No_of_Links_used=topsis_out['No_of_Links_used'],
        avg_node=topsis_out['avg_node'],
        No_of_Nodes_used=topsis_out['No_of_Nodes_used'],
        avg_path=topsis_out['avg_path'],
        avg_exec=topsis_out['avg_exec'].total_seconds()*1000/topsis_out['total_request'],
        avg_node_stress=topsis_out['avg_node_stress']*100,
        avg_link_stress=topsis_out['avg_link_stress']*100,
        avg_congestion=topsis_out['avg_congestion']*100,
        total_vm_request = topsis_out["total_vm_request"],
        total_vl_request = topsis_out["total_vl_request"]//2,
        total_vm_embedded = topsis_out["total_vm_embedded"],
        total_vl_embedded=topsis_out["total_vl_embedded"]
    )

def exec_parserr(tot=2):
    parser_out = parserr()
    sleep(tot*2)
    
    printToExcel(
        algorithm='PAGERANK-STABLE',
        revenue=parser_out[0]['revenue'],
        total_cost=parser_out[0]['total_cost'],
        revenuetocostratio=(parser_out[0]['revenue']/parser_out[0]['total_cost'])*100,
        accepted=parser_out[0]['accepted'],
        total_request=parser_out[0]['total_request'],
        embeddingratio=(parser_out[0]['accepted']/parser_out[0]['total_request'])*100,
        pre_resource=parser_out[0]['pre_resource'],
        post_resource=parser_out[0]['post_resource'],
        consumed=parser_out[0]['pre_resource']-parser_out[0]['post_resource'],
        avg_bw=parser_out[0]['avg_bw'],
        avg_crb=parser_out[0]['avg_crb'],
        avg_link=parser_out[0]['avg_link'],
        No_of_Links_used=parser_out[0]['No_of_Links_used'],
        avg_node=parser_out[0]['avg_node'],
        No_of_Nodes_used=parser_out[0]['No_of_Nodes_used'],
        avg_path=parser_out[0]['avg_path'],
        avg_exec=parser_out[0]['avg_exec'].total_seconds()*1000/parser_out[0]['total_request'],
        avg_node_stress=parser_out[0]['avg_node_stress']*100,
        avg_link_stress=parser_out[0]['avg_link_stress']*100
    )
    
    printToExcel(
        algorithm='PAGERANK-DIRECT',
        revenue=parser_out[1]['revenue'],
        total_cost=parser_out[1]['total_cost'],
        revenuetocostratio=(parser_out[1]['revenue']/parser_out[1]['total_cost'])*100,
        accepted=parser_out[1]['accepted'],
        total_request=parser_out[1]['total_request'],
        embeddingratio=(parser_out[1]['accepted']/parser_out[1]['total_request'])*100,
        pre_resource=parser_out[1]['pre_resource'],
        post_resource=parser_out[1]['post_resource'],
        consumed=parser_out[1]['pre_resource']-parser_out[1]['post_resource'],
        avg_bw=parser_out[1]['avg_bw'],
        avg_crb=parser_out[1]['avg_crb'],
        avg_link=parser_out[1]['avg_link'],
        No_of_Links_used=parser_out[1]['No_of_Links_used'],
        avg_node=parser_out[1]['avg_node'],
        No_of_Nodes_used=parser_out[1]['No_of_Nodes_used'],
        avg_path=parser_out[1]['avg_path'],
        avg_exec=parser_out[1]['avg_exec'].total_seconds()*1000/parser_out[1]['total_request'],
        avg_node_stress=parser_out[1]['avg_node_stress']*100,
        avg_link_stress=parser_out[1]['avg_link_stress']*100,
        
    )



def exec_rethinking(tot=15):
    rethinking_out = rethinking()
    sleep(tot*4)

    printToExcel(
        algorithm='RETHINKING',
        revenue=rethinking_out['revenue'],
        total_cost=rethinking_out['total_cost'],
        revenuetocostratio=(rethinking_out['revenue']/rethinking_out['total_cost'])*100,
        accepted=rethinking_out['accepted'],
        total_request=rethinking_out['total_request'],
        embeddingratio=(rethinking_out['accepted']/rethinking_out['total_request'])*100,
        pre_resource=rethinking_out['pre_resource'],
        post_resource=rethinking_out['post_resource'],
        consumed=rethinking_out['pre_resource']-rethinking_out['post_resource'],
        avg_bw=rethinking_out['avg_bw'],
        avg_crb=rethinking_out['avg_crb'],
        avg_link=rethinking_out['avg_link'],
        No_of_Links_used=rethinking_out['No_of_Links_used'],
        avg_node=rethinking_out['avg_node'],
        No_of_Nodes_used=rethinking_out['No_of_Nodes_used'],
        avg_path=rethinking_out['avg_path'],
        avg_exec=rethinking_out['avg_exec'].total_seconds()*1000/rethinking_out['total_request'],
        avg_node_stress=rethinking_out['avg_node_stress']*100,
        avg_link_stress=rethinking_out['avg_link_stress']*100,
        avg_congestion=rethinking_out['avg_congestion']*100,
        total_vm_request = rethinking_out["total_vm_request"],
        total_vl_request = rethinking_out["total_vl_request"]//2,
        total_vm_embedded = rethinking_out["total_vm_embedded"],
        total_vl_embedded=rethinking_out["total_vl_embedded"]
    )

def exec_NRM(tot=15):
    NRM_out = NRM()
    sleep(tot*4)

    printToExcel(
        algorithm='NRM',
        revenue=NRM_out['revenue'],
        total_cost=NRM_out['total_cost'],
        revenuetocostratio=(NRM_out['revenue']/NRM_out['total_cost'])*100,
        accepted=NRM_out['accepted'],
        total_request=NRM_out['total_request'],
        embeddingratio=(NRM_out['accepted']/NRM_out['total_request'])*100,
        pre_resource=NRM_out['pre_resource'],
        post_resource=NRM_out['post_resource'],
        consumed=NRM_out['pre_resource']-NRM_out['post_resource'],
        avg_bw=NRM_out['avg_bw'],
        avg_crb=NRM_out['avg_crb'],
        avg_link=NRM_out['avg_link'],
        No_of_Links_used=NRM_out['No_of_Links_used'],
        avg_node=NRM_out['avg_node'],
        No_of_Nodes_used=NRM_out['No_of_Nodes_used'],
        avg_path=NRM_out['avg_path'],
        avg_exec=NRM_out['avg_exec'].total_seconds()*1000/NRM_out['total_request'],
        avg_node_stress=NRM_out['avg_node_stress']*100,
        avg_link_stress=NRM_out['avg_link_stress']*100,
        avg_congestion=NRM_out['avg_congestion']*100,
        total_vm_request = NRM_out["total_vm_request"],
        total_vl_request = NRM_out["total_vl_request"]//2,
        total_vm_embedded = NRM_out["total_vm_embedded"],
        total_vl_embedded=NRM_out["total_vl_embedded"]
    )

def exec_gcnrl(tot=15):   
    gcnrl_out = gcnrl()
    sleep(tot*4)
    printToExcel(
        algorithm="GCN-RL_Proposed_With_Congestion",
        revenue=gcnrl_out['revenue'],
        total_cost=gcnrl_out['total_cost'],
        revenuetocostratio=(gcnrl_out['revenue']/gcnrl_out['total_cost'])*100,
        accepted=gcnrl_out['accepted'],
        total_request=gcnrl_out['total_request'],
        embeddingratio=(gcnrl_out['accepted']/gcnrl_out['total_request'])*100,
        pre_resource=gcnrl_out['pre_resource'],
        post_resource=gcnrl_out['post_resource'],
        consumed=gcnrl_out['pre_resource']-gcnrl_out['post_resource'],
        avg_bw=gcnrl_out['avg_bw'],
        avg_crb=gcnrl_out['avg_crb'],
        avg_link=gcnrl_out['avg_link'],
        No_of_Links_used=gcnrl_out['No_of_Links_used'],
        avg_node=gcnrl_out['avg_node'],
        No_of_Nodes_used=gcnrl_out['No_of_Nodes_used'],
        avg_path=gcnrl_out['avg_path'],
        avg_exec=gcnrl_out['avg_exec']/gcnrl_out['total_request'],
        avg_node_stress=gcnrl_out['avg_node_stress'],
        avg_link_stress=gcnrl_out['avg_link_stress'],
        avg_congestion=gcnrl_out['avg_congestion']*100,
        total_vm_request = gcnrl_out["total_vm_request"],
        total_vl_request = gcnrl_out["total_vlinks"],
        total_vm_embedded = gcnrl_out["total_vm_embedded"],
        total_vl_embedded=gcnrl_out["total_vl_embedded"]
    )


def exec_eigen(tot=15):   
    eigen_out = eigen()
    sleep(tot*4)
    printToExcel(algorithm="automatic_with_cong_with_Eigen_without_between",
                 revenue=eigen_out['revenue'],
        total_cost=eigen_out['total_cost'],
        revenuetocostratio=(eigen_out['revenue']/eigen_out['total_cost'])*100,
        accepted=eigen_out['accepted'],
        total_request=eigen_out['total_request'],
        embeddingratio=(eigen_out['accepted']/eigen_out['total_request'])*100,
        pre_resource=eigen_out['pre_resource'],
        post_resource=eigen_out['post_resource'],
        consumed=eigen_out['pre_resource']-eigen_out['post_resource'],
        avg_bw=eigen_out['avg_bw'],
        avg_crb=eigen_out['avg_crb'],
        avg_link=eigen_out['avg_link'],
        No_of_Links_used=eigen_out['No_of_Links_used'],
        avg_node=eigen_out['avg_node'],
        No_of_Nodes_used=eigen_out['No_of_Nodes_used'],
        avg_path=eigen_out['avg_path'],
        avg_exec=eigen_out['avg_exec']/eigen_out['total_request'],
        avg_node_stress=eigen_out['avg_node_stress'],
        avg_link_stress=eigen_out['avg_link_stress'],
         avg_congestion=eigen_out['avg_congestion']*100,
        total_vm_request = eigen_out["total_vm_request"],
        total_vl_request = eigen_out["total_vlinks"],
        total_vm_embedded = eigen_out["total_vm_embedded"],
        total_vl_embedded=eigen_out["total_vl_embedded"]
    )


def exec_automatic_with_cong(tot=15):   
    autocong_out = automatic_w_cong()
    sleep(tot*4)
    printToExcel(algorithm="automatic-with-cong_base2",
                 revenue=autocong_out['revenue'],
        total_cost=autocong_out['total_cost'],
        revenuetocostratio=(autocong_out['revenue']/autocong_out['total_cost'])*100,
        accepted=autocong_out['accepted'],
        total_request=autocong_out['total_request'],
        embeddingratio=(autocong_out['accepted']/autocong_out['total_request'])*100,
        pre_resource=autocong_out['pre_resource'],
        post_resource=autocong_out['post_resource'],
        consumed=autocong_out['pre_resource']-autocong_out['post_resource'],
        avg_bw=autocong_out['avg_bw'],
        avg_crb=autocong_out['avg_crb'],
        avg_link=autocong_out['avg_link'],
        No_of_Links_used=autocong_out['No_of_Links_used'],
        avg_node=autocong_out['avg_node'],
        No_of_Nodes_used=autocong_out['No_of_Nodes_used'],
        avg_path=autocong_out['avg_path'],
        avg_exec=autocong_out['avg_exec']/autocong_out['total_request'],
        avg_node_stress=autocong_out['avg_node_stress'],
        avg_link_stress=autocong_out['avg_link_stress'],
         avg_congestion=autocong_out['avg_congestion']*100,
        total_vm_request = autocong_out["total_vm_request"],
        total_vl_request = autocong_out["total_vlinks"],
        total_vm_embedded = autocong_out["total_vm_embedded"],
        total_vl_embedded=autocong_out["total_vl_embedded"]
    )

def exec_automatic(tot=15):
    
    automatic_out = automatic()
    sleep(tot*4)
    printToExcel(algorithm="GCN-RL-automatic_Base",
                 revenue=automatic_out['revenue'],
        total_cost=automatic_out['total_cost'],
        revenuetocostratio=(automatic_out['revenue']/automatic_out['total_cost'])*100,
        accepted=automatic_out['accepted'],
        total_request=automatic_out['total_request'],
        embeddingratio=(automatic_out['accepted']/automatic_out['total_request'])*100,
        pre_resource=automatic_out['pre_resource'],
        post_resource=automatic_out['post_resource'],
        consumed=automatic_out['pre_resource']-automatic_out['post_resource'],
        avg_bw=automatic_out['avg_bw'],
        avg_crb=automatic_out['avg_crb'],
        avg_link=automatic_out['avg_link'],
        No_of_Links_used=automatic_out['No_of_Links_used'],
        avg_node=automatic_out['avg_node'],
        No_of_Nodes_used=automatic_out['No_of_Nodes_used'],
        avg_path=automatic_out['avg_path'],
        avg_exec=automatic_out['avg_exec']/automatic_out['total_request'],
        avg_node_stress=automatic_out['avg_node_stress'],
        avg_link_stress=automatic_out['avg_link_stress'],
        avg_congestion=automatic_out['avg_congestion']*100,
        total_vm_request = automatic_out["total_vm_request"],
        total_vl_request = automatic_out["total_vlinks"],
        total_vm_embedded = automatic_out["total_vm_embedded"],
        total_vl_embedded=automatic_out["total_vl_embedded"]
    )

def exec_gcnrl_without_cong(tot=15):
    
    
    gcnrl_without_cong = gcn_rl_without_cong()
    sleep(tot*4)
    printToExcel(algorithm="GCN-RL-With_between_Without-Congestion_Eign",
                 revenue=gcnrl_without_cong['revenue'],
        total_cost=gcnrl_without_cong['total_cost'],
        revenuetocostratio=(gcnrl_without_cong['revenue']/gcnrl_without_cong['total_cost'])*100,
        accepted=gcnrl_without_cong['accepted'],
        total_request=gcnrl_without_cong['total_request'],
        embeddingratio=(gcnrl_without_cong['accepted']/gcnrl_without_cong['total_request'])*100,
        pre_resource=gcnrl_without_cong['pre_resource'],
        post_resource=gcnrl_without_cong['post_resource'],
        consumed=gcnrl_without_cong['pre_resource']-gcnrl_without_cong['post_resource'],
        avg_bw=gcnrl_without_cong['avg_bw'],
        avg_crb=gcnrl_without_cong['avg_crb'],
        avg_link=gcnrl_without_cong['avg_link'],
        No_of_Links_used=gcnrl_without_cong['No_of_Links_used'],
        avg_node=gcnrl_without_cong['avg_node'],
        No_of_Nodes_used=gcnrl_without_cong['No_of_Nodes_used'],
        avg_path=gcnrl_without_cong['avg_path'],
        avg_exec=gcnrl_without_cong['avg_exec']/gcnrl_without_cong['total_request'],
        avg_node_stress=gcnrl_without_cong['avg_node_stress'],
        avg_link_stress=gcnrl_without_cong['avg_link_stress'],
        avg_congestion=gcnrl_without_cong['avg_congestion']*100,
        total_vm_request = gcnrl_without_cong["total_vm_request"],
        total_vl_request = gcnrl_without_cong["total_vlinks"],
        total_vm_embedded = gcnrl_without_cong["total_vm_embedded"],
        total_vl_embedded=gcnrl_without_cong["total_vl_embedded"]
    )

def exec_eigen_without_cong(tot=15):
    
    
    eigen_without_congestion = eg_without_cong()
    sleep(tot*4)
    printToExcel(algorithm="GCN-RL-Eigen-Without-Congestion_without_bet",
                 revenue=eigen_without_congestion['revenue'],
        total_cost=eigen_without_congestion['total_cost'],
        revenuetocostratio=(eigen_without_congestion['revenue']/eigen_without_congestion['total_cost'])*100,
        accepted=eigen_without_congestion['accepted'],
        total_request=eigen_without_congestion['total_request'],
        embeddingratio=(eigen_without_congestion['accepted']/eigen_without_congestion['total_request'])*100,
        pre_resource=eigen_without_congestion['pre_resource'],
        post_resource=eigen_without_congestion['post_resource'],
        consumed=eigen_without_congestion['pre_resource']-eigen_without_congestion['post_resource'],
        avg_bw=eigen_without_congestion['avg_bw'],
        avg_crb=eigen_without_congestion['avg_crb'],
        avg_link=eigen_without_congestion['avg_link'],
        No_of_Links_used=eigen_without_congestion['No_of_Links_used'],
        avg_node=eigen_without_congestion['avg_node'],
        No_of_Nodes_used=eigen_without_congestion['No_of_Nodes_used'],
        avg_path=eigen_without_congestion['avg_path'],
        avg_exec=eigen_without_congestion['avg_exec']/eigen_without_congestion['total_request'],
        avg_node_stress=eigen_without_congestion['avg_node_stress'],
        avg_link_stress=eigen_without_congestion['avg_link_stress'],
        avg_congestion=eigen_without_congestion['avg_congestion']*100,
        total_vm_request = eigen_without_congestion["total_vm_request"],
        total_vl_request = eigen_without_congestion["total_vlinks"],
        total_vm_embedded = eigen_without_congestion["total_vm_embedded"],
        total_vl_embedded=eigen_without_congestion["total_vl_embedded"]
    )

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

def printToExcel(algorithm='', revenue='', total_cost='', revenuetocostratio='', accepted='', total_request='', 
embeddingratio='', pre_resource='', post_resource='',consumed='',avg_bw='',avg_crb='',avg_link='',No_of_Links_used='',
avg_node='',No_of_Nodes_used='',avg_path='',avg_exec='', total_nodes='', total_links='',avg_node_stress='', avg_link_stress='',
avg_congestion='',total_vm_request='',total_vl_request='',total_vm_embedded='',total_vl_embedded=''):
    output_dict["algorithm"].append(algorithm)
    output_dict["revenue"].append(revenue)
    output_dict["total_cost"].append(total_cost)
    output_dict["revenuetocostratio"].append(revenuetocostratio)
    output_dict["accepted"].append(accepted)
    output_dict["total_request"].append(total_request)
    output_dict["embeddingratio"].append(embeddingratio)
    output_dict["pre_resource"].append(pre_resource)
    output_dict["post_resource"].append(post_resource)
    output_dict["consumed"].append(consumed)
    output_dict["avg_bw"].append(avg_bw)
    output_dict["avg_crb"].append(avg_crb)
    output_dict["avg_link"].append(avg_link)
    output_dict["avg_node_stress"].append(avg_node_stress)
    output_dict["avg_link_stress"].append(avg_link_stress)
    output_dict["No_of_Links_used"].append(No_of_Links_used)
    output_dict["avg_node"].append(avg_node)
    output_dict["No_of_Nodes_used"].append(No_of_Nodes_used)
    output_dict["avg_path"].append(avg_path)
    output_dict["avg_exec"].append(avg_exec)
    output_dict["total_vm_request"].append(total_vm_request)
    output_dict["total_vl_request"].append(total_vl_request)
    output_dict["total_vm_embedded"].append(total_vm_embedded)
    output_dict["total_vl_embedded"].append(total_vl_embedded)
    output_dict["avg_congestion"].append(avg_congestion)
    # output_dict["total_nodes"].append(total_nodes)
    # output_dict["total_links"].append(total_links)

            
if __name__ == "__main__":

    file_exists = os.path.exists('1_random.pickle') or os.path.exists('3_uniform.pickle') or os.path.exists(
        '1_poission.pickle') or os.path.exists('1_normal.pickle')
    # file_exists = False       #Manually set, if want to update a substrate pickle
    if(file_exists == False):
        generateSubstrate(graph_extraction_uniform.for_automate, str(3) + '_uniform.pickle')  # Uniform Distribution
    # algos = [Automatic(),Eigen_without_congestion(),Eigen_with_congestion(),Gcn_Rl(),Gcn_Rl_without_Congestion()]
    algo= []
    total_rev_demand=0
    substrate = extractSubstrate('3_uniform.pickle')
    req_list = [250, 500, 750, 1000]
    iterations = 10
    tot=0
    for req_no in req_list:
        tot += 1
        cnt=0
        print(f"\n\tRequest number: {req_no}\n")
        while cnt<iterations:
                vne_list = create_vne(no_requests=req_no)
                    # geeky_file = open(f'vne_file_{req_no}+{cnt}.pickle', 'rb')
                    # vne_list = pickle.load(geeky_file)
                config.substrate = copy.deepcopy(substrate)
                config.vne_list = copy.deepcopy(vne_list)
                        # Uncomment those functions to run, comment all other. for ex if want to run greedy algorithm only leave

                exec_gcnrl() # Proposed with betweenness and no eigen ,with congestion  ****
                exec_automatic(tot) # baseline as it is  *****
                    # exec_eigen(tot) # gcn_rl with eigen , without betweenness and with congestion (automatic with congestion)
                    # exec_eigen_without_cong(tot) # gcn_rl with eigen , without betweenness and without congestion error
                    #exec_gcnrl_without_cong(tot) # gcn_rl without eigen , with betweenness and without congestion error
                    # exec_automatic_with_cong(tot) # Baseline Modified automatic_with_cong  * B2 error

                exec_greedy() # uncommented and comment all other (exec_topsis(), exec_parser(),  exec_rethinking())
                    # exec_rethinking(tot)    # problem!
                    # exec_greedy(tot)        #Runs GREEDY algorithm
                exec_topsis(tot)       #Runs NORD algorithm
                exec_NRM(tot)

                
                if((cnt+1)%2==0):
                    print(f'\n\tREQUEST {req_no}, {cnt+1} ITERATIONS COMPLETED\n\n')
                printToExcel()
                cnt += 1

    # print(output_dict)
    excel = pd.DataFrame(output_dict)
    excel.to_excel("Results.xlsx")
