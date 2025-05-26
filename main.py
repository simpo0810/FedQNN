

import numpy as np
from server import FederatedServer
from client import FederatedClient
from config import Config
from utils import load_data, split_data
import matplotlib.pyplot as plt
import argparse
import logging

logger = logging.getLogger(__name__)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Federated QNN Training")
    parser.add_argument('--data', default='ch_data_adr.npy', help='Path to channel data')
    parser.add_argument('--rounds', type=int, help='Number of federated rounds')
    args = parser.parse_args()

    # Load configuration
    cfg = Config()
    if args.rounds:
        cfg.num_rounds = args.rounds
    cfg_dict = cfg.to_dict()
    cfg_dict['SNR'] = 10**(np.array(cfg_dict['SNR_dB'])/10)
    logger.info(f"Configuration loaded: {cfg_dict}")

    # Initial weights
    # initial_weights = [
    #     923.84533672, 4112.04464518, 310.63783701, -349.96561575, 3358.22978568,
    #     420.72663005, -4054.73969949, 392.60148109, 2561.27378612, 3934.61425277,
    #     1852.7232888, -1890.4371678, 794.60803502, -785.74486591, 87.86202179,
    #     -229.25771119
    # ]
    initial_weights =[
        1.57079633, 1.57079633, 1.57079633, 1.57079633, 1.57079633, 1.57079633,
        1.57079633, 1.57079633, 1.57079633, 1.57079633, 1.57079633, 1.57079633,
        1.57079633, 1.57079633, 1.57079633, 1.57079633]

    # Load and split data
    # full_data = load_data(args.data)
    # clients_data = split_data(full_data, cfg.num_clients)
    # clients = [FederatedClient(i, data, cfg_dict) for i, data in enumerate(clients_data)]
    clients = FederatedClient.create_clients(cfg)
    logger.info(f"Data split among {cfg.num_clients} clients")
    
    H_eval = FederatedClient.ch(cfg.N_Tx, cfg.M_Tx, cfg.N_Rx, cfg.N_user, cfg.N_path)
    print('test channel size ', H_eval.shape)
    
    # Run federated learning
    server = FederatedServer(cfg.num_clients, initial_weights, cfg.num_rounds)
    # print(' WTF are the federated server initialized well ', clients)
    
    final_weights,loss_history = server.run_federated_learning(clients,plot_progress=True)

    # Evaluate

    # R = []
    # for s in range(len(cfg_dict['SNR'])):
    #     R_data = []
    #     for client in clients:
    #         client.weights = np.array(final_weights)
            
    #         for i_data in range(len(H_eval)):
    #             h_reshape = client.reshape(H_eval[i_data])
    #             qc1 = client.Q_cloud_original(len(h_reshape), len(h_reshape), h_reshape, final_weights, cfg.N_out)
    #             qnn_output = client.QNN_decode_cloud(qc1, cfg.N_out)
    #             H_grouping = np.concatenate((client.clusterheader(H_eval[i_data]), 
    #                                        client.usergrouping(H_eval[i_data], client.clusterheader(H_eval[i_data]))))
    #             powers = client.power_alloc((qnn_output[0], qnn_output[1]))
    #             precoding = client.precoding(H_grouping, qnn_output[2:6], cfg_dict['SNR'][s],powers )
    #             comb = client.init_combiner(H_eval[i_data].shape)
    #             sinrinv = client.invSINR(H_grouping, comb, precoding, powers, cfg_dict['SNR'][s])
    #             combiner = client.combiner(H_grouping, qnn_output[6:10], sinrinv)
    #             sinr = client.SINR(H_grouping, combiner, precoding, powers, cfg_dict['SNR'][s])
    #             R_data.append(client.rsum_(sinr))
            
    #     R.append(np.mean(R_data) / cfg.B)
    #     logger.info(f"SNR {cfg_dict['SNR_dB'][s]} dB: Spectral Efficiency = {R[-1]}")

    # # Plot
    # plt.plot(cfg_dict['SNR_dB'], R, label='Federated QNN')
    # plt.legend()
    # plt.xlabel('SNR (dB)')
    # plt.ylabel('Spectral Efficiency (bits/s/Hz)')
    # plt.grid()
    # plt.savefig('results.png')
    # plt.show()
    
    #let's check individual clients performance 
    R_per_client = {client_id: [] for client_id in range(len(clients))}  
    R_combined = []  # this just to compare with the original weights from previous Ap

    for s in range(len(cfg_dict['SNR'])):
        R_data_per_client = {client_id: [] for client_id in range(len(clients))}
        R_data_combined = []
        
        for client_idx, client in enumerate(clients):
            client.weights = np.array(final_weights)
        
            for i_data in range(len(H_eval)):
                h_reshape = client.reshape(H_eval[i_data])
                qc1 = client.Q_cloud_original(len(h_reshape), len(h_reshape), h_reshape, final_weights, cfg.N_out)
                qnn_output = client.QNN_decode_cloud(qc1, cfg.N_out)
                H_grouping = np.concatenate((client.clusterheader(H_eval[i_data]),
                                        client.usergrouping(H_eval[i_data], client.clusterheader(H_eval[i_data]))))
                powers = client.power_alloc((qnn_output[0], qnn_output[1]))
                precoding = client.precoding(H_grouping, qnn_output[2:6], cfg_dict['SNR'][s], powers)
                comb = client.init_combiner(H_eval[i_data].shape)
                sinrinv = client.invSINR(H_grouping, comb, precoding, powers, cfg_dict['SNR'][s])
                combiner = client.combiner(H_grouping, qnn_output[6:10], sinrinv)
                sinr = client.SINR(H_grouping, combiner, precoding, powers, cfg_dict['SNR'][s])
                
                # Store per-client rate
                r_value = client.rsum_(sinr)
                R_data_per_client[client_idx].append(r_value)
                R_data_combined.append(r_value)
        
        # Store average rate for each client at this SNR
        for client_idx in range(len(clients)):
            client_avg_rate = np.mean(R_data_per_client[client_idx]) / cfg.B
            R_per_client[client_idx].append(client_avg_rate)
            logger.info(f"Client {client_idx}, SNR {cfg_dict['SNR_dB'][s]} dB: Spectral Efficiency = {client_avg_rate}")
        
        # Also keep the combined rate calculation like in the previous stuff 
        R_combined.append(np.mean(R_data_combined) / cfg.B)
        logger.info(f"Combined, SNR {cfg_dict['SNR_dB'][s]} dB: Spectral Efficiency = {R_combined[-1]}")

    # Plot individual client performance
    print(f"Initial loss: {loss_history[0]:.6f}")
    print(f"Final loss: {loss_history[-1]:.6f}")
    print(f"Improvement: {(1 - loss_history[-1]/loss_history[0])*100:.2f}%")
    plt.figure(figsize=(10, 6))
    for client_idx in range(len(clients)):
        plt.plot(cfg_dict['SNR_dB'], R_per_client[client_idx], label=f'Client {client_idx}')

    # Plot combined performance
    plt.plot(cfg_dict['SNR_dB'], R_combined, label='Federated QNN (Combined)', linestyle='--', linewidth=2)

    plt.legend()
    plt.xlabel('SNR (dB)')
    plt.ylabel('Spectral Efficiency (bits/s/Hz)')
    plt.grid()
    plt.title('Performance of Individual Clients vs Combined')
    plt.savefig('client_comparison_results.png')
    plt.show()


    #Alright what if the weights were being used for by a different client that did not participate in the training 
    
 
    new_client = FederatedClient(client_id="eval", H_data=H_eval, config=cfg_dict)

    # Step 2: Evaluate the global model on the new client's data
    # R = []
    # for s in range(len(cfg_dict['SNR'])):  # Loop over SNR levels
    #     R_data = []
    #     new_client.weights = np.array(final_weights)  # Assign the global model weights
    #     for i_data in range(len(new_client.H_data)):  # Loop over the new client's data samples
            
    #         h_reshape = new_client.reshape(new_client.H_data[i_data])
            
            
    #         qc1 = new_client.Q_cloud_original(len(h_reshape), len(h_reshape), h_reshape, final_weights, cfg.N_out)
    #         qnn_output = new_client.QNN_decode_cloud(qc1, cfg.N_out)
            
            
    #         H_grouping = np.concatenate((new_client.clusterheader(new_client.H_data[i_data]), 
    #                                 new_client.usergrouping(new_client.H_data[i_data], 
    #                                                         new_client.clusterheader(new_client.H_data[i_data]))))
    #         powers = new_client.power_alloc((qnn_output[0], qnn_output[1]))
    #         precoding = new_client.precoding(H_grouping, qnn_output[2:6], cfg_dict['SNR'][s], powers)
    #         comb = new_client.init_combiner(new_client.H_data[i_data].shape)
    #         sinrinv = new_client.invSINR(H_grouping, comb, precoding, powers, cfg_dict['SNR'][s])
    #         combiner = new_client.combiner(H_grouping, qnn_output[6:10], sinrinv)
    #         sinr = new_client.SINR(H_grouping, combiner, precoding, powers, cfg_dict['SNR'][s])
            
    #         R_data.append(new_client.rsum_(sinr))
        
        
    #     R.append(np.mean(R_data) / cfg.B)
    #     logger.info(f"SNR {cfg_dict['SNR_dB'][s]} dB: Spectral Efficiency = {R[-1]}")

    # # Step 3: Plot the results
    # plt.plot(cfg_dict['SNR_dB'], R, label='Federated QNN on New Client')
    # plt.legend()
    # plt.xlabel('SNR (dB)')
    # plt.ylabel('Spectral Efficiency (bits/s/Hz)')
    # plt.grid()
    # plt.savefig('results.png')
    # plt.show()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()