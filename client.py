

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.providers.basic_provider import BasicProvider
from qiskit.result import marginal_counts
from tqdm import trange
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class FederatedClient:
    def __init__(self, client_id: int, H_data: np.ndarray, config: dict):
        """Initialize a client with local data and configuration."""
        if not isinstance(H_data, np.ndarray) or H_data.size == 0:
            raise ValueError("H_data must be a non-empty NumPy array")
        self.client_id = client_id
        self.H_data = H_data
        self.config = config
        self.weights = None
        
        self.N_Tx = config['N_Tx']
        self.M_Tx = config['M_Tx']
        self.N_Rx = config['N_Rx']
        self.N_user = config['N_user']
        self.N_path = config['N_path']
        self.N_out = config['N_out']
        self.SNR = config['SNR']
        self.alpha = config.get('alpha', 0.01)
        self.local_epochs = config.get('local_epochs', 1)
        logger.info(f"Client {client_id} initialized with {len(H_data)} data samples")

    def Q_cloud_original(self, N_Tx: int, Nrx_u: int, H: np.ndarray, array_w: List[float], out_c: int) -> QuantumCircuit:
        """Create the QNN circuit (cached structure could be added)."""
        # print(N_Tx)
        q = QuantumRegister(out_c if out_c > 2*N_Tx else 2*N_Tx, 'q')
        c = ClassicalRegister(out_c, 'c')
        qc1 = QuantumCircuit(q, c)

        for i in range(2*N_Tx):
            qc1.h(i)
        qc1.barrier()
        
        # m_ch = np.array([[H[i][j].real, H[i][j].imag] for i in range(len(H)) for j in range(len(H[0]))]).T
        m_ch = []
        for i in range(len(H)):
            m_temp = []
            for j in range(len(H[0])):
                m_temp.append(H[i][j].real)
                m_temp.append(H[i][j].imag)
            m_ch.append(m_temp)
        
        m_ch = np.array(m_ch).T
        # print('the size of m_ch', m_ch.shape)
        # print('the size of H', H.shape)
        for m in range(2*N_Tx):
            
            for k in range(Nrx_u):
                qc1.ry(np.linalg.norm(m_ch[m][k]), q[m])
        qc1.barrier()

        for i in range((out_c if out_c > 2*N_Tx else 2*N_Tx) - 1):
            qc1.cz(q[i], q[i+1])
        qc1.barrier()

        for i in range(out_c if out_c > 2*N_Tx else 2*N_Tx):
            qc1.ry(array_w[i], q[i])
        qc1.barrier()

        for i in range((out_c if out_c > 2*N_Tx else 2*N_Tx) - 1):
            qc1.cz(q[-1-i], q[-2-i])
        qc1.barrier()

        for i in range(out_c):
            qc1.measure(q[i], c[i])

        return qc1

    def QNN_decode_cloud(self, qc1: QuantumCircuit, N_out: int) -> List[float]:
        """Decode QNN measurements."""
        backend = BasicProvider().get_backend("basic_simulator")
        job = backend.run(qc1, shots=1024)
        output = []
        for i in range(N_out):
            counts = marginal_counts(job.result(), indices=[i]).get_counts()
            z_QNN = counts.get('1', 0) / 1024  # Default to 0 if '1' not present
            output.append(z_QNN)
        return output

    # def train_local_model(self, global_model: Dict) -> Dict:
    #     """Perform local training using the global model."""
    #     self.weights = np.array(global_model['weights'])
    #     prev_loss = float('inf')
        
    #     for epoch in range(self.local_epochs):
    #         gradients = np.zeros_like(self.weights)
    #         total_loss = 0
            
    #         for i_data in trange(len(self.H_data), desc=f"Client {self.client_id} Epoch {epoch+1}"):
    #             # try:
    #                 #print('train channel size ', self.H_data.shape)
    #                 h_reshape = self.reshape(self.H_data[i_data])
    #                 # print(h_reshape.shape)
    #                 qc1 = self.Q_cloud_original(len(h_reshape), len(h_reshape), h_reshape, self.weights, self.N_out)
    #                 qnn_output = self.QNN_decode_cloud(qc1, self.N_out)
                    
    #                 H_grouping = np.concatenate((self.clusterheader(self.H_data[i_data]), 
    #                                            self.usergrouping(self.H_data[i_data], self.clusterheader(self.H_data[i_data]))))
    #                 powers = self.power_alloc((qnn_output[0], qnn_output[1]))
    #                 precoding = self.precoding(H_grouping, qnn_output[2:6], self.SNR[0],powers )
    #                 # print('proof that the first precoding is computed ')
    #                 comb = self.init_combiner(self.H_data[i_data].shape)
    #                 sinrinv = self.invSINR(H_grouping, comb, precoding, powers, self.SNR[0])
    #                 # print('the inverse shape',sinrinv)
    #                 combiner = self.combiner(H_grouping,  qnn_output[6:10], sinrinv)
    #                 sinr = self.SINR(H_grouping, combiner, precoding, powers, self.SNR[0])
    #                 rsum = self.rsum_(sinr)
                    
    #                 # Simplified gradient: maximize sum-rate (negative loss)
    #                 #loss = -rsum / self.config['B']
    #                 loss = -rsum 
    #                 total_loss += loss
    #                 # Finite difference approximation for gradient (placeholder for actual QNN gradient)
    #                 eps = 2*np.sinh(np.pi / 0.5 )
    #                 for i in range(len(self.weights)):
    #                     w_plus = self.weights.copy()
    #                     w_plus[i] += eps
    #                     qc_plus = self.Q_cloud_original(len(h_reshape), len(h_reshape), h_reshape, w_plus, self.N_out)
    #                     out_plus = self.QNN_decode_cloud(qc_plus, self.N_out)
    #                     rsum_plus = self.rsum_(self.SINR(H_grouping, combiner, 
    #                                                    self.precoding(H_grouping, out_plus[2:6], self.SNR[0], powers), 
    #                                                    powers, self.SNR[0]))
    #                     gradients[i] += (rsum_plus - rsum) / eps
                    
    #                 self.weights -= self.alpha * gradients / len(self.H_data)
    #             # except Exception as e:
    #             #     logger.error(f"Error in data {i_data}: {e}")
            
    #         avg_loss = total_loss / len(self.H_data)
    #         logger.info(f"Client {self.client_id} Epoch {epoch+1}: Loss = {avg_loss}")
    #         if abs(prev_loss - avg_loss) < 1e-4:  # Early stopping
    #             logger.info(f"Client {self.client_id} stopped early at epoch {epoch+1}")
    #             break
    #         prev_loss = avg_loss

    #     return {
    #         'client_id': self.client_id,
    #         'weights': self.weights.tolist(),
    #         'samples': len(self.H_data),
    #         'version': global_model['version']
    #     }
    def train_local_model(self, global_model: Dict) -> Dict:
        """Perform local training using the global model with parameter-shift gradients."""
        self.weights = np.array(global_model['weights'])
        prev_loss = float('inf')
        
        # Track all epoch losses
        all_epoch_losses = []
        
        for epoch in range(self.local_epochs):
            gradients = np.zeros_like(self.weights)
            total_loss = 0
            epoch_losses = []
            
            for i_data in trange(len(self.H_data), desc=f"Client {self.client_id} Epoch {epoch+1}"):
                h_reshape = self.reshape(self.H_data[i_data])
                qc1 = self.Q_cloud_original(len(h_reshape), len(h_reshape), h_reshape, self.weights, self.N_out)
                qnn_output = self.QNN_decode_cloud(qc1, self.N_out)
                
                H_grouping = np.concatenate((self.clusterheader(self.H_data[i_data]),
                                        self.usergrouping(self.H_data[i_data], self.clusterheader(self.H_data[i_data]))))
                powers = self.power_alloc((qnn_output[0], qnn_output[1]))
                precoding = self.precoding(H_grouping, qnn_output[2:6], self.SNR[0], powers)
                comb = self.init_combiner(self.H_data[i_data].shape)
                sinrinv = self.invSINR(H_grouping, comb, precoding, powers, self.SNR[0])
                combiner = self.combiner(H_grouping, qnn_output[6:10], sinrinv)
                sinr = self.SINR(H_grouping, combiner, precoding, powers, self.SNR[0])
                rsum = self.rsum_(sinr)
                
                loss = -rsum  # Maximize sum-rate
                total_loss += loss
                epoch_losses.append(loss)
                
                # Parameter-shift rule for gradients (simplified)
                shift = np.pi / 2
                for i in range(len(self.weights)):
                    w_plus = self.weights.copy()
                    w_minus = self.weights.copy()
                    w_plus[i] += shift
                    w_minus[i] -= shift
                    
                    qc_plus = self.Q_cloud_original(len(h_reshape), len(h_reshape), h_reshape, w_plus, self.N_out)
                    qc_minus = self.Q_cloud_original(len(h_reshape), len(h_reshape), h_reshape, w_minus, self.N_out)
                    out_plus = self.QNN_decode_cloud(qc_plus, self.N_out)
                    out_minus = self.QNN_decode_cloud(qc_minus, self.N_out)
                    
                    rsum_plus = self.rsum_(self.SINR(H_grouping, combiner,
                                                self.precoding(H_grouping, out_plus[2:6], self.SNR[0], powers),
                                                powers, self.SNR[0]))
                    rsum_minus = self.rsum_(self.SINR(H_grouping, combiner,
                                                    self.precoding(H_grouping, out_minus[2:6], self.SNR[0], powers),
                                                    powers, self.SNR[0]))
                    gradients[i] += (rsum_plus - rsum_minus) / (2 * np.sin(shift))
                
                self.weights -= self.alpha * gradients / len(self.H_data)
                
            avg_loss = total_loss / len(self.H_data)
            all_epoch_losses.append(avg_loss)
            logger.info(f"Client {self.client_id} Epoch {epoch+1}: Loss = {avg_loss}")
            
            if abs(prev_loss - avg_loss) < 1e-4:
                logger.info(f"Client {self.client_id} stopped early at epoch {epoch+1}")
                break
                
            prev_loss = avg_loss
        
        # Calculate final loss after training
        final_loss = self.compute_loss(self.weights)
        
        return {
            'client_id': self.client_id,
            'weights': self.weights.tolist(),
            'samples': len(self.H_data),
            'version': global_model['version'],
            'loss': final_loss,  # Add the loss to the return dict
            'epoch_losses': all_epoch_losses  # Optionally track all epoch losses
        }

    def compute_loss(self, weights) -> float:
        """
        Compute the average loss across all data samples for given weights
        
        Returns:
            Average negative sum-rate across all samples (lower is better)
        """
        total_loss = 0
        
        for i_data in range(len(self.H_data)):
            h_reshape = self.reshape(self.H_data[i_data])
            qc1 = self.Q_cloud_original(len(h_reshape), len(h_reshape), h_reshape, weights, self.N_out)
            qnn_output = self.QNN_decode_cloud(qc1, self.N_out)
            
            H_grouping = np.concatenate((self.clusterheader(self.H_data[i_data]),
                                    self.usergrouping(self.H_data[i_data], self.clusterheader(self.H_data[i_data]))))
            powers = self.power_alloc((qnn_output[0], qnn_output[1]))
            precoding = self.precoding(H_grouping, qnn_output[2:6], self.SNR[0], powers)
            comb = self.init_combiner(self.H_data[i_data].shape)
            sinrinv = self.invSINR(H_grouping, comb, precoding, powers, self.SNR[0])
            combiner = self.combiner(H_grouping, qnn_output[6:10], sinrinv)
            sinr = self.SINR(H_grouping, combiner, precoding, powers, self.SNR[0])
            rsum = self.rsum_(sinr)
            
            loss = -rsum  # Maximize sum-rate
            total_loss += loss
        
        return total_loss / len(self.H_data)

    def evaluate_model(self, global_model: Dict) -> float:
        """
        Evaluate the global model on this client's data without training
        
        Args:
            global_model: Dictionary with model weights
            
        Returns:
            Loss value (negative sum-rate, lower is better)
        """
        weights = np.array(global_model['weights'])
        return self.compute_loss(weights)
    # def train_local_model(self, global_model: Dict) -> Dict:
    #         """Perform local training using the global model with parameter-shift gradients."""
    #         self.weights = np.array(global_model['weights'])
    #         prev_loss = float('inf')
            
    #         for epoch in range(self.local_epochs):
    #             gradients = np.zeros_like(self.weights)
    #             total_loss = 0
    #             epoch_losses = []
    #             for i_data in trange(len(self.H_data), desc=f"Client {self.client_id} Epoch {epoch+1}"):
    #                 h_reshape = self.reshape(self.H_data[i_data])
    #                 qc1 = self.Q_cloud_original(len(h_reshape), len(h_reshape), h_reshape, self.weights, self.N_out)
    #                 qnn_output = self.QNN_decode_cloud(qc1, self.N_out)
                    
    #                 H_grouping = np.concatenate((self.clusterheader(self.H_data[i_data]), 
    #                                         self.usergrouping(self.H_data[i_data], self.clusterheader(self.H_data[i_data]))))
    #                 powers = self.power_alloc((qnn_output[0], qnn_output[1]))
    #                 precoding = self.precoding(H_grouping, qnn_output[2:6], self.SNR[0], powers)
    #                 comb = self.init_combiner(self.H_data[i_data].shape)
    #                 sinrinv = self.invSINR(H_grouping, comb, precoding, powers, self.SNR[0])
    #                 combiner = self.combiner(H_grouping, qnn_output[6:10], sinrinv)
    #                 sinr = self.SINR(H_grouping, combiner, precoding, powers, self.SNR[0])
    #                 rsum = self.rsum_(sinr)
                    
    #                 loss = -rsum  # Maximize sum-rate
    #                 total_loss += loss
                    
    #                 # Parameter-shift rule for gradients (simplified)
    #                 shift = np.pi / 2
    #                 for i in range(len(self.weights)):
    #                     w_plus = self.weights.copy()
    #                     w_minus = self.weights.copy()
    #                     w_plus[i] += shift
    #                     w_minus[i] -= shift
                        
    #                     qc_plus = self.Q_cloud_original(len(h_reshape), len(h_reshape), h_reshape, w_plus, self.N_out)
    #                     qc_minus = self.Q_cloud_original(len(h_reshape), len(h_reshape), h_reshape, w_minus, self.N_out)
    #                     out_plus = self.QNN_decode_cloud(qc_plus, self.N_out)
    #                     out_minus = self.QNN_decode_cloud(qc_minus, self.N_out)
                        
    #                     rsum_plus = self.rsum_(self.SINR(H_grouping, combiner, 
    #                                                     self.precoding(H_grouping, out_plus[2:6], self.SNR[0], powers), 
    #                                                     powers, self.SNR[0]))
    #                     rsum_minus = self.rsum_(self.SINR(H_grouping, combiner, 
    #                                                     self.precoding(H_grouping, out_minus[2:6], self.SNR[0], powers), 
    #                                                     powers, self.SNR[0]))
    #                     gradients[i] += (rsum_plus - rsum_minus) / (2 * np.sin(shift))
                    
    #                 self.weights -= self.alpha * gradients / len(self.H_data)
                
    #             avg_loss = total_loss / len(self.H_data)
    #             logger.info(f"Client {self.client_id} Epoch {epoch+1}: Loss = {avg_loss}")
    #             if abs(prev_loss - avg_loss) < 1e-4:
    #                 logger.info(f"Client {self.client_id} stopped early at epoch {epoch+1}")
    #                 break
    #             prev_loss = avg_loss

    #         return {
    #             'client_id': self.client_id,
    #             'weights': self.weights.tolist(),
    #             'samples': len(self.H_data),
    #             'version': global_model['version']
    #         }

    def reshape(self, H): return np.reshape(H, (len(H[1]), len(H[1])))
    def clusterheader(self, H): 
        norms = [np.linalg.norm(H[i])**2 for i in range(4)]
        idx = np.argsort(norms)[-2:]
        return H[idx[1]], H[idx[0]]
    def usergrouping(self, H, clusterheader): 
        norms = [np.linalg.norm(H[i])**2 for i in range(4)]
        cluster_norms = [np.linalg.norm(clusterheader[0])**2, np.linalg.norm(clusterheader[1])**2]
        remaining = [H[i] for i in range(4) if norms[i] not in cluster_norms]
        return remaining[0], remaining[1]
    def power_alloc(self, power_var): 
        total_power = 1
        p_weak_g = 0.8
        p_strong_g = total_power - p_weak_g
        return [0.2 * p_strong_g, 0.2 * p_weak_g, 0.8 * p_strong_g, 0.8 * p_weak_g]
    # def precoding(self, H, precoder_var, Tx_SNR, PA):
    #     N_user = len(H)  # 4
    #     p = []
    #     H_con = [h.conj().T for h in H]  # Each H_con[i] is (2, 8)
    #     for i in range(N_user):
    #         sum_fk = sum(precoder_var[j] * H[j] @ H_con[j] for j in range(N_user) if j != i)
    #         f_k = np.identity(len(sum_fk)) + sum_fk  # (8, 8)
    #         # Use SVD to get dominant precoding vector
    #         U, S, Vh = np.linalg.svd(H[i].T)  # H[i].T is (2, 8)
    #         fk = Vh[0, :].conj().T  # Dominant right singular vector, (8,)
    #         fk = fk.reshape(-1, 1)  # (8, 1)
    #         fk_ue = np.sqrt(PA[i] * Tx_SNR) * (fk / np.linalg.norm(fk))
    #         p.append(fk_ue)
    #     return p
    def precoding(self, H, precoder_var, Tx_SNR, PA):
        N_user = len(H)  # 4
        noise_var = (10**(-16.9)) * 180
        H_con = []
        for i in range(N_user):
            H_con.append(H[i].conj().T)
        H_con = np.array(H_con)
        sum_fk_list = []
        p = []
        for i in range (N_user):
            for j in range (N_user):
                fk_op_v = precoder_var[j] * H[j] @ H_con[j]
                sum_fk_list.append(fk_op_v)

            fk_op = sum(sum_fk_list)
            sum_fk = (1/(noise_var))*(fk_op)
            f_k = (np.identity(len(sum_fk)) + sum_fk)
            
            fk = np.linalg.pinv(f_k) @ H[i] 
            
            fk_ue = np.sqrt(PA[i]*Tx_SNR) * (fk /  np.linalg.norm(fk))
            p.append(fk_ue)

        return np.array(p)
    def init_combiner(self, shape): 
        return np.random.uniform(0, 1, shape) + 1.j * np.random.uniform(0, 1, shape)
    def invSINR(self, H, comb, pre, PA, Tx_SNR): 
        noise = 1.9952623149688797e-17 * 180
        sinr = [(PA[i] * Tx_SNR) * np.linalg.norm(comb[i] @ H[i].conj().T @ pre[i])**2 / 
                (noise if i <= 1 else ((PA[i-2] * Tx_SNR) * np.linalg.norm(comb[i-2] @ H[i-2].conj().T @ pre[i-2])**2 + noise))
                for i in range(len(H))]
        return np.array(sinr)
    # def combiner(self, H, combiner_var, sinr_inv):
    #     N_user = len(H)
    #     comb = []
    #     I = np.identity(self.N_Rx)  # (2, 2)
    #     for i in range(N_user):
    #         h_sn = np.linalg.pinv((H[i].conj().T @ H[i]) + (sinr_inv * I)) @ H[i].conj().T  # (2, 2) @ (2, 8) = (2, 8)
    #         # Take the first column or use SVD for a single combiner
    #         U, _, _ = np.linalg.svd(h_sn.T)  # h_sn.T is (8, 2)
    #         c = U[:, 0].reshape(-1, 1)  # (2, 1)
    #         c = combiner_var[i] * (c / np.linalg.norm(c))
    #         comb.append(c)
    #     return comb
    def combiner(self, H, combiner_var, sinrinv):
        N_user = len(H)
        comb = []

        for i in range(N_user):
            
            sinr_inv = sinrinv[i]
            I = np.identity(2)
            h_sn = np.linalg.pinv((H[i].conj().T @ H[i]) + (sinr_inv*I)) @ H[i].conj().T
            c = combiner_var[i] * h_sn
            comb.append(c.T) # not sure, i extend the transpose to match the matrix, but still not yet explainable

        comb = np.array(comb)

        return comb
    def SINR(self, H, comb, pre, PA, Tx_SNR): 
        noise = 1.9952623149688797e-17 * 180
        sinr = []
        for i in range(len(H)):
            P_ch_gain = (PA[i] * Tx_SNR) * np.linalg.norm(comb[i] @ H[i].conj().T @ pre[i])**2
            if i > 1:
                intra = (PA[i-2] * Tx_SNR) * np.linalg.norm(comb[i-2] @ H[i-2].conj().T @ pre[i-2])**2 if i-2 >= 0 else 0
                inter = (PA[i-1] * Tx_SNR) * np.linalg.norm(comb[i-1] @ H[i-1].conj().T @ pre[i-1])**2 if i-1 >= 0 else 0
                s = P_ch_gain / (intra + inter + noise)
            else:
                s = P_ch_gain / noise
            sinr.append(s)
        return np.array(sinr)
    def rsum_(self, sinr): 
        return sum(self.config['B'] * np.log2(1 + s) for s in sinr)
    def create_clients(cfg):
        """
        Create federated clients where each client generates its own data
        with a different number of taps (N_path)
        """
        clients = []
        
        # Define the range of N_path values for different clients
        # You can customize this based on your requirements
        min_paths = cfg.min_paths if hasattr(cfg, 'min_paths') else 1
        max_paths = cfg.max_paths if hasattr(cfg, 'max_paths') else 10
        
        # Generate N_path values for each client
        if hasattr(cfg, 'path_distribution') and cfg.path_distribution == 'uniform':
            # Uniform distribution of N_path values
            client_paths = [min_paths + i % (max_paths - min_paths + 1) for i in range(cfg.num_clients)]
        elif hasattr(cfg, 'path_distribution') and cfg.path_distribution == 'random':
            # Random distribution of N_path values
            import random
            client_paths = [random.randint(min_paths, max_paths) for _ in range(cfg.num_clients)]
        elif hasattr(cfg, 'custom_paths') and len(cfg.custom_paths) == cfg.num_clients:
            # Use custom-defined paths if provided
            client_paths = cfg.custom_paths
        else:
            # Default: linear distribution
            import numpy as np
            client_paths = np.linspace(min_paths, max_paths, cfg.num_clients, dtype=int).tolist()
        
        # Create clients with their specific N_path values
        for i in range(cfg.num_clients):
        # Create a copy of the configuration
            client_cfg = vars(cfg).copy() if hasattr(cfg, '__dict__') else cfg.copy()
            
            # Update N_path for this specific client
            client_cfg['N_path'] = client_paths[i]
            client_cfg['SNR'] = 10**(np.array(client_cfg['SNR_dB'])/10)
            # Generate H_data for this client
            H_data = FederatedClient.ch(
                client_cfg['N_Tx'], 
                client_cfg['M_Tx'], 
                client_cfg['N_Rx'], 
                client_cfg['N_user'], 
                client_paths[i]
            )
            
            # Create client with the generated data
            client = FederatedClient(i, H_data, client_cfg)
            clients.append(client)
            
            logger.info(f"Client {i} created with N_path = {client_paths[i]}")

        logger.info(f"Created {cfg.num_clients} clients with varying path configurations")
        return clients    
    @staticmethod
    def ch(N_Tx, M_Tx, N_Rx, N_user, N_path):
        n_samples = 10
        H_all_samples = []  # List to hold all samples

        for sample in range(n_samples):
            H_k_all = []  # List for one sample's channels
            for k in range(N_user):
                h_k_path = []
                for l_path in range(N_path):
                    # Tx UPA array
                    arrayTx = []
                    for indexNTx in range(N_Tx):
                        for indexMTx in range(M_Tx):
                            angleNTx = np.random.normal(0, 2 * np.pi)
                            angleMTx = np.random.normal(0, 2 * np.pi)
                            arr = np.exp(-1.j * np.pi * np.sin(angleNTx) * 
                                        (((indexMTx - 1) * np.cos(angleMTx)) + (indexNTx - 1) * np.sin(angleMTx)))
                            arrayTx.append(arr)
                    arrayTx = (1 / np.sqrt(N_Tx * M_Tx)) * np.array(arrayTx)  # Shape: (8,)

                    # Rx ULA array
                    arrayRx = []
                    for indexRx in range(N_Rx):
                        angleRx = np.random.normal(0, 2 * np.pi)
                        arr = np.exp(-1.j * np.pi * np.cos(angleRx) * indexRx)
                        arrayRx.append(arr)
                    arrayRx = (1 / np.sqrt(N_Rx)) * np.array(arrayRx)  # Shape: (2,)

                    # Outer product for channel matrix
                    h_path = np.random.randn() * np.outer(arrayRx, arrayTx)  # (2, 8)
                    h_k_path.append(h_path)

                h_k = sum(h_k_path)  # Sum over paths, still (2, 8)
                H_coeff = np.sqrt((N_Tx * M_Tx * N_Rx) / N_path) * h_k  # (2, 8)
                H_k_all.append(H_coeff)

            H_k_all = np.array(H_k_all)  # (4, 2, 8) for one sample
            H_all_samples.append(H_k_all.transpose(0, 2, 1))  # Transpose to (4, 8, 2)

        H_all_samples = np.array(H_all_samples)  # (10, 4, 8, 2)
        return H_all_samples