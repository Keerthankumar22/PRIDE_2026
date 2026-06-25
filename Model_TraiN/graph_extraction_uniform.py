import os
import pickle
import sys
import graph_u
from vne_u import create_vne

class Extract:
    def get_graphs(self, req_no = 5):     # USE THIS DEFINITION FOR AUTOMATION & comment line no 10
        current = os.path.dirname(os.path.realpath(__file__))
        print(current)
        sys.path.append(os.path.join(os.path.dirname(current), "P3_ALIB_MASTER"))
        current = os.path.join(
            os.path.dirname(current),
            "P3_ALIB_MASTER",
            "input",
            "senario_RedBestel.pickle", # senario_RedBestel # KK_Aarnet
        )
        print(current)
        with open(current, "rb") as f:
            data = pickle.load(f)

        #Setting lower and upper bound for the parameters such as CRB,BW etc
        para = graph_u.Parameters(10,500,10,500,0,100,0,100,1,1)# BW,CRB, X,Y Delay

        # (10,40, 20,40, 100, 0, 100, 0, 100, 1, 1) for TOY example
        #Parameters for subsrate graph BW ,CRB, Location,Delay
        #para = graph_u.Parameters(1000, 1000, 1000, 1000, 0, 100, 0, 100, 1, 1) for DAA and VRMAP

        # para = graph_u.Parameters(50,1000,200,1000,0,100,0,100,1,1) FOR TOPSIS and VIKOR
        print(len(data.scenario_list[0].substrate.nodes))
        try:
            # This returns substrate nodes with random CRB's , BW between edges
            substrate = graph_u.Graph(
                len(data.scenario_list[0].substrate.nodes),
                data.scenario_list[0].substrate.edges,
                para,
            )
        except:
            substrate = graph_u.Graph(
                data.get("substrate").nodes,
                data.get("substrate").edges,
                para,
            )
        
        #THis creates VNR's based on req_no
        #    # USE THIS STATEMENT FOR AUTOMATION & comment line no 28
        vne_list=[]
        print("Substrate=", substrate)
        return substrate,vne_list

def for_automate(req_no = 5):
    x = Extract()
    #Getting the substrate network from alib
    # Getting VNr's from create vne file based on req_no
    substrate, vne_list = x.get_graphs(req_no)

    # Getting features of substrate
    # x_sub = FeatureMatrixSubstrate(substrate)
    # feature_subs=x_sub.generate_feature_matrix()

    # #Getting features of VNR
    # y=FeatureVNR(vne_list[0])
    # feature_vnr=y.generate_feature_matrix()

    # #Creating the Data object for substrate and vnr inorder to pass it to GCN 
    # sub_data=create_data(feature_subs,substrate)
    # vnr_data=create_data(feature_vnr,vne_list[0])

    # print("substrate ",sub_data)
    # print("Vnr data ",vnr_data)

    # gcn_layer=GCNNBlock(6)
    # output_substrate=gcn_layer(sub_data)
    # print("Output substrate=",output_substrate)

    # gcn_vnr = GCNNBlock(4)
    # output_vnr = gcn_vnr(vnr_data)
    # print("Output VNr = ",output_vnr)
    
    

    return substrate, vne_list

# if _name_ == "_main_":
#     for_automate(req_no=1)