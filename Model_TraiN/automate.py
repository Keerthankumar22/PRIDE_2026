import graph_extraction_uniform
from time import sleep
import os
import pickle
import torch
from gcn_rl_normal_train import Agent as normal_Agent
from gcn_rl_normal_train import ActorCritic as normal_AC
from gcn_rl_automatic_train import Agent as AutomaticAgent
from gcn_rl_automatic_train import ActorCritic as Automatic_AC
from gcn_rl_eign_train import Agent as Gcn_Rl_Eigen_Agent
from gcn_rl_eign_train import ActorCritic as Gcn_Rl_Eigen_AC
from gcn_rl_eigen_without_cong_train import Agent as Eigen_without_congestion_Agent
from gcn_rl_eigen_without_cong_train import ActorCritic as Eigen_without_cong_AC
from gcn_rl_wihtout_cong_train import Agent as Gcn_Rl_without_cong_Agent 
from gcn_rl_wihtout_cong_train import ActorCritic as Gcn_rl_without_cong_AC
from automatic_with_congestion import ActorCritic as automatic_with_cong_AC
from automatic_with_congestion import Agent as automatic_with_cong_Agent
from shared_adam import SharedAdam
import torch.multiprocessing as mp
from extract_substrate import extractSubstrate

# all agents running simultaneously

def generateSubstrate(for_automate, pickle_name):
    substrate, _ = for_automate(1)
    geeky_file = open(pickle_name, 'wb')
    pickle.dump(substrate, geeky_file)
    geeky_file.close()

def extractSubstrate(pickle_file):
    filehandler = open(pickle_file, 'rb')
    substrate = pickle.load(filehandler)
    return substrate

#testunjdqwd
def runUniformExtraction(pickle_name):
    sleep(10)
    substrate = extractSubstrate(str(pickle_name))
    print("\nUNIFORM Extraction\n")
    # print("Substrate ",substrate.substrate)
    # main(substrate, vne_u)

def normal_gcn_rl(substrate):
    lr = 7e-61
    total_eps = 3000
    epsilon_initial = 0.88    # Initial exploration rate
    epsilon_final = 0.2      # Final exploration rate
    epsilon_decay = 0.001   # Rate at which epsilon decreases per training step
    epsil = [epsilon_initial,epsilon_final,epsilon_decay]
    global_actor_critic = normal_AC(substrate.nodes)
    # global_actor_critic.load_state_dict(torch.load('C:/GCN_Congestion_Aware/all_model_test/cyan5_gcn_rl1000.pth'))
    # torch.save(global_actor_critic.state_dict(), 'C:/GCN_Congestion_Aware/all_model_test/normal_gcn_rl.pth')

    # print("Global paramaters",global_actor_critic.parameters)
    global_actor_critic.share_memory()
    optim = SharedAdam(global_actor_critic.parameters(), lr=lr, 
                        betas=(0.92, 0.999))
    global_ep = mp.Value('i', 0)

    workers = [normal_Agent(substrate,global_actor_critic,
                    optim, global_ep, total_eps,epsil) for i in range(3)]
    
    [w.start() for w in workers]
    [w.join() for w in workers]


    torch.save(global_actor_critic.state_dict(), 'C:/GCN_Congestion_Aware/all_model_test/normal_gcn_rl_proposed.pth')
    # teal has leaky relu in GCN. No randperm in fullyconn. 30-60 VNRs
    # cyan has linear VNR instead of GCN

    print("saved the model")

def automatic_gcn_rl(substrate):
    lr = 1e-6
    total_eps = 500
    epsilon_initial = 0.88   # Initial exploration rate
    epsilon_final = 0.2      # Final exploration rate
    epsilon_decay = 0.0   # Rate at which epsilon decreases per training step
    epsil = [epsilon_initial,epsilon_final,epsilon_decay]
    global_actor_critic = Automatic_AC(substrate.nodes)
    # global_actor_critic.load_state_dict(torch.load('models/actor_critic_modelfresh3000.pth'))
    # print("Global paramaters",global_actor_critic.parameters)
    global_actor_critic.share_memory()
    optim = SharedAdam(global_actor_critic.parameters(), lr=lr, 
                        betas=(0.92, 0.999))
    global_ep = mp.Value('i', 0)

    workers = [AutomaticAgent(substrate,global_actor_critic,
                    optim, global_ep, total_eps,epsil) for i in range(3)]
    
    [w.start() for w in workers]
    [w.join() for w in workers]


    torch.save(global_actor_critic.state_dict(), 'C:/GCN_Congestion_Aware/all_model_test/automatic_gcn_rl_baseline.pth')

def gcn_rl_eigen(substrate):
    lr = 1e-6
    total_eps = 2000
    epsilon_initial = 1    # Initial exploration rate
    epsilon_final = 0.2      # Final exploration rate
    epsilon_decay = 0.0   # Rate at which epsilon decreases per training step
    epsil = [epsilon_initial,epsilon_final,epsilon_decay]
    global_actor_critic = Gcn_Rl_Eigen_AC(substrate.nodes)
    # global_actor_critic.load_state_dict(torch.load('models/actor_critic_modelfresh3000.pth'))
    # print("Global paramaters",global_actor_critic.parameters)
    global_actor_critic.share_memory()
    optim = SharedAdam(global_actor_critic.parameters(), lr=lr, 
                        betas=(0.92, 0.999))
    global_ep = mp.Value('i', 0)

    workers = [Gcn_Rl_Eigen_Agent(substrate,global_actor_critic,
                    optim, global_ep, total_eps,epsil) for i in range(6)]
    
    [w.start() for w in workers]
    [w.join() for w in workers]
    torch.save(global_actor_critic.state_dict(), 'C:/GCN_Congestion_Aware/all_model_test/gcn_rl_eigen_with_congestion.pth')
    # torch.save(global_actor_critic.state_dict(), 'models/eigen1000.pth')
    return

def gcn_rl_eigen_without_congestion(substrate):
    lr = 1e-6
    total_eps = 500
    epsilon_initial = 1    # Initial exploration rate
    epsilon_final = 0.2      # Final exploration rate
    epsilon_decay = 0.0   # Rate at which epsilon decreases per training step
    epsil = [epsilon_initial,epsilon_final,epsilon_decay]
    global_actor_critic = Eigen_without_cong_AC(substrate.nodes)
    # global_actor_critic.load_state_dict(torch.load('models/actor_critic_modelfresh3000.pth'))
    # print("Global paramaters",global_actor_critic.parameters)
    global_actor_critic.share_memory()
    optim = SharedAdam(global_actor_critic.parameters(), lr=lr, 
                        betas=(0.92, 0.999))
    global_ep = mp.Value('i', 0)

    workers = [Eigen_without_congestion_Agent(substrate,global_actor_critic,
                    optim, global_ep, total_eps,epsil) for i in range(6)]
    
    [w.start() for w in workers]
    [w.join() for w in workers]
    torch.save(global_actor_critic.state_dict(), 'C:/GCN_Congestion_Aware/all_model_test/gcn_rl_eigen_without_congestion.pth')
    # torch.save(global_actor_critic.state_dict(), 'models/eigen_without_congestion1000.pth')
    return
    
def gcn_rl_without_congestion(substrate):
    lr = 1e-6
    total_eps = 500
    epsilon_initial = 1    # Initial exploration rate
    epsilon_final = 0.2      # Final exploration rate
    epsilon_decay = 0.0   # Rate at which epsilon decreases per training step
    epsil = [epsilon_initial,epsilon_final,epsilon_decay]
    global_actor_critic = Gcn_rl_without_cong_AC(substrate.nodes)
    # global_actor_critic.load_state_dict(torch.load('models/actor_critic_modelfresh3000.pth'))
    # print("Global paramaters",global_actor_critic.parameters)
    global_actor_critic.share_memory()
    optim = SharedAdam(global_actor_critic.parameters(), lr=lr, 
                        betas=(0.92, 0.999))
    global_ep = mp.Value('i', 0)

    workers = [Gcn_Rl_without_cong_Agent(substrate,global_actor_critic,
                    optim, global_ep, total_eps,epsil) for i in range(6)]
    
    [w.start() for w in workers]
    [w.join() for w in workers]
    torch.save(global_actor_critic.state_dict(), 'C:/GCN_Congestion_Aware/all_model_test/normal_gcn_rl_without_congestion.pth')
    # torch.save(global_actor_critic.state_dict(), 'models/normal_gcn_rl_without_congestion1000.pth')
    return

def automatic_with_congestion(substrate):
    lr = 1e-6
    total_eps = 1000
    epsilon_initial = 0.8    # Initial exploration rate
    epsilon_final = 0.2      # Final exploration rate
    epsilon_decay = 0.0   # Rate at which epsilon decreases per training step
    epsil = [epsilon_initial,epsilon_final,epsilon_decay]
    global_actor_critic = automatic_with_cong_AC(substrate.nodes)
    # global_actor_critic = automatic_with_congestion(substrate.nodes)
    # global_actor_critic.load_state_dict(torch.load('models/automatic_w_cong_1100.pth'))
    # torch.save(global_actor_critic.state_dict(), 'C:/GCN_Congestion_Aware/all_model_test/automatic_w_cong_1100.pth')
    # print("Global paramaters",global_actor_critic.parameters)
    global_actor_critic.share_memory()
    optim = SharedAdam(global_actor_critic.parameters(), lr=lr, 
                        betas=(0.92, 0.999))
    global_ep = mp.Value('i', 0)

    workers = [automatic_with_cong_Agent(substrate,global_actor_critic,
                    optim, global_ep, total_eps,epsil) for i in range(6)]
    
    [w.start() for w in workers]
    [w.join() for w in workers]
    # torch.save(global_actor_critic.state_dict(), 'models/automatic_w_cong_2000.pth')
    torch.save(global_actor_critic.state_dict(), 'C:/GCN_Congestion_Aware/all_model_test/automatic_with_cong.pth')

    return


if __name__ == "__main__":

    file_exists = os.path.exists('1_random.pickle') or os.path.exists('5_uniform.pickle') or os.path.exists(
        '1_poission.pickle') or os.path.exists('1_normal.pickle')
    # file_exists = False       #Manually set, if you want to update a substrate pickle
    if(file_exists == False):
        generateSubstrate(graph_extraction_uniform.for_automate, str(5) + '_uniform.pickle')  # Uniform Distribution

    # CHANGE SUBSTRATE HERE
    substrate = extractSubstrate('5_uniform.pickle')
    while True:
        print("\nTrain this algorithm:")
        print("1. Normal GCN-RL")
        print("2. Automatic GCN-RL") # BAseline withiout congestion
        print("3. GCN-RL Eigen") # with congestion
        print("4. GCN-RL Eigen without Congestion")
        print("5. GCN-RL without Congestion") # With Betweenness
        print("6. Automatic with congestion") # Baseline
        print("0. Exit")
        
        choice = input("Select an algorithm (0-6): ")
        
        if choice == "1":
            normal_gcn_rl(substrate)
        elif choice == "2":
            automatic_gcn_rl(substrate)
        elif choice == "3":
            gcn_rl_eigen(substrate)
        elif choice == "4":
            gcn_rl_eigen_without_congestion(substrate)
        elif choice == "5":
            gcn_rl_without_congestion(substrate)
        elif choice == "6":
            automatic_with_congestion(substrate)
        elif choice == "0":
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please select a valid option.")
            continue
        break
    # lr = 1e-6
    # total_eps = 1000
    # epsilon_initial = 1    # Initial exploration rate
    # epsilon_final = 0.2      # Final exploration rate
    # epsilon_decay = 0.0   # Rate at which epsilon decreases per training step
    # epsil = [epsilon_initial,epsilon_final,epsilon_decay]
    # global_actor_critic = normal_AC(substrate.nodes)
    # # global_actor_critic.load_state_dict(torch.load('models/actor_critic_modelfresh3000.pth'))
    # # print("Global paramaters",global_actor_critic.parameters)
    # global_actor_critic.share_memory()
    # optim = SharedAdam(global_actor_critic.parameters(), lr=lr, 
    #                     betas=(0.92, 0.999))
    # global_ep = mp.Value('i', 0)

    # workers = [normal_Agent(substrate,global_actor_critic,
    #                 optim, global_ep, total_eps,epsil) for i in range(6)]
    
    # [w.start() for w in workers]
    # [w.join() for w in workers]

    # torch.save(global_actor_critic.state_dict(), 'models/throw.pth')

    # print("saved the model")
    # excel = pd.DataFrame(output_dict)
    # excel.to_excel("Results.xlsx")
    # x = Extract()
    # runUniformExtraction('1_uniform.pickle')

    

       
    '''
    5: 1000 episodes. Without dropout after gcn. 32->16->8 layers
    6: same but with dropout after last GCN layer
    7: 500 eps after 6  (all ~100 nodes)

    9: 100+300+600 eps with 17 nodes sub network
    10: with learning rate 1e-1, 200 episodes, 17 nodes
    12: 300 eps with Chebconv and new virt_graph that goes only once per vnr
    model: model 12 + 500 + 2000 eps
    20 : 4000 episodes approx with addition policy and value layer added
    21 : 1000 eps. With new congestion reward changed to (1-r4)*r3*r2*r1. 
        (Gives good spread values but they are not changing)
    22: 21st + 1000 eps with rand_perm uncommented 
    23: load from 22 and removed softmax/10 because policy has good values. 1000 episodes
        wASTE
    23: load from 22, added back softmax/10. 1000 episodes

    24: softmax/5

    1: uniform pickle taken from NORD (4_uniform). 1000 eps
    2: same as 1, but loss for every VNR instead of end of network
    3: update local parameters after every VNR, global params after one episode
    4: update all params after one episode only
    model: with normalization. so temperature softmax
    5: model, but update loss after every VNR. And GCN with RELU
    8: 300 episodes ran
    9:changed the self.first_substrate part to extracting directly from pickle file and 
    changed vnr settings to (10,30,10,35) so that deallocation happens in each episode , 210 episodes

    2000plus1.0: 2100 eps
    2000plus6: 2500 eps
    x3: 3000 eps
    '''


    
