

# import numpy as np
# from typing import List, Dict
# import pickle
# from client import FederatedClient

# class FederatedServer:
#     def __init__(self, num_clients: int, initial_weights: List[float], num_rounds: int):
#         """
#         Args:
#             num_clients: Number of base stations participating
#             initial_weights: Initial QNN weights from the original implementation
#             num_rounds: Number of federated learning rounds
#         """
#         self.num_clients = num_clients
#         self.global_weights = np.array(initial_weights)  # Initialize with original weights
#         self.num_rounds = num_rounds
#         self.model_version = 0
#         self.client_updates: List[Dict] = []
       
#     def distribute_model(self) -> Dict:
#         """
#         Package the current global model for distribution to clients
#         Returns:
#             Dictionary containing the global weights
#         """
#         return {
#             'weights': self.global_weights.tolist(),
#             'round': self.current_round,
#             'version': self.model_version
#         }

#     def aggregate_updates(self) -> None:
#         """
#         Perform federated averaging on collected client updates
#         Uses simple averaging as in the original FedAvg algorithm
#         """
#         if not self.client_updates:
#             return
        
#         # Average weights from all clients
#         aggregated_weights = np.mean([update['weights'] for update in self.client_updates], axis=0)
#         self.global_weights = aggregated_weights
#         self.client_updates = []  # Clear updates after aggregation

#     def run_federated_learning(self, clients: List['FederatedClient']) -> List[float]:
#         """
#         Main federated learning loop
#         Args:
#             clients: List of client instances participating in training
#         Returns:
#             Final global weights after all rounds
#         """
#         self.current_round = 0
#         for round_num in range(self.num_rounds):
#             self.current_round = round_num
#             print(f"Server: Starting round {round_num + 1}/{self.num_rounds}")
            
#             # Distribute model to all clients
#             global_model = self.distribute_model()
#             # print('the global model initial is like this ',global_model)
            
#             # Collect updates from all clients
#             self.client_updates = []
#             for client in clients:
#                 update = client.train_local_model(global_model)
#                 self.client_updates.append(update)
            
#             # Aggregate updates
#             self.aggregate_updates()
            
#             # Save intermediate model (optional)
#             self.save_model(f"global_model_round_{round_num}.pkl")
        
#         return self.global_weights.tolist()

#     def save_model(self, filename: str) -> None:
#         """Save the current global model to disk"""
#         with open(filename, 'wb') as f:
#             pickle.dump(self.global_weights, f)

import numpy as np
from typing import List, Dict, Optional, Tuple
import pickle
import matplotlib.pyplot as plt
from client import FederatedClient

class FederatedServer:
    def __init__(self, num_clients: int, initial_weights: List[float], num_rounds: int):
        """
        Args:
            num_clients: Number of base stations participating
            initial_weights: Initial QNN weights from the original implementation
            num_rounds: Number of federated learning rounds
        """
        self.num_clients = num_clients
        self.global_weights = np.array(initial_weights)  # Initialize with original weights
        self.num_rounds = num_rounds
        self.model_version = 0
        self.client_updates: List[Dict] = []
        
        # Add tracking for global model loss
        self.global_loss_history = []
        
    def distribute_model(self) -> Dict:
        """
        Package the current global model for distribution to clients
        Returns:
            Dictionary containing the global weights
        """
        return {
            'weights': self.global_weights.tolist(),
            'round': self.current_round,
            'version': self.model_version
        }

    def aggregate_updates(self) -> float:
        """
        Perform federated averaging on collected client updates
        Uses simple averaging as in the original FedAvg algorithm
        
        Returns:
            Average loss across all clients for this round
        """
        if not self.client_updates:
            return float('inf')
        
        # Average weights from all clients
        aggregated_weights = np.mean([update['weights'] for update in self.client_updates], axis=0)
        self.global_weights = aggregated_weights
        
        # Calculate and store average loss
        avg_loss = np.mean([update.get('loss', 0.0) for update in self.client_updates])
        self.global_loss_history.append(avg_loss)
        
        self.client_updates = []  # Clear updates after aggregation
        return avg_loss

    def evaluate_global_model(self, clients: List['FederatedClient']) -> float:
        """
        Evaluate global model on all clients without updating weights
        
        Args:
            clients: List of client instances for evaluation
            
        Returns:
            Average loss across all clients
        """
        global_model = self.distribute_model()
        losses = []
        
        for client in clients:
            # Assume client has an evaluate method that returns loss without training
            loss = client.evaluate_model(global_model)
            losses.append(loss)
            
        avg_loss = np.mean(losses)
        return avg_loss

    def run_federated_learning(self, clients: List['FederatedClient'], 
                              eval_every: int = 1, 
                              plot_progress: bool = True) -> Tuple[List[float], List[float]]:
        """
        Main federated learning loop
        
        Args:
            clients: List of client instances participating in training
            eval_every: Evaluate global model every N rounds (set to 0 to disable)
            plot_progress: Whether to plot the loss trend after training
            
        Returns:
            Tuple of (final global weights, loss history)
        """
        self.current_round = 0
        self.global_loss_history = []  # Reset loss history
        
        for round_num in range(self.num_rounds):
            self.current_round = round_num
            print(f"Server: Starting round {round_num + 1}/{self.num_rounds}")
            
            # Distribute model to all clients
            global_model = self.distribute_model()
            
            # Collect updates from all clients
            self.client_updates = []
            for client in clients:
                update = client.train_local_model(global_model)
                # Ensure client returns loss in the update
                self.client_updates.append(update)
            
            # Aggregate updates and get average loss
            avg_loss = self.aggregate_updates()
            print(f"Round {round_num + 1} completed. Average loss: {avg_loss:.6f}")
            
            # Optional: Evaluate global model across all clients
            if eval_every > 0 and (round_num + 1) % eval_every == 0:
                eval_loss = self.evaluate_global_model(clients)
                print(f"Global model evaluation at round {round_num + 1}: Loss = {eval_loss:.6f}")
            
            # Save intermediate model (optional)
            self.save_model(f"global_model_round_{round_num}.pkl")
        
        # Plot loss trend if requested
        if plot_progress:
            self.plot_loss_trend()
        
        return self.global_weights.tolist(), self.global_loss_history

    def save_model(self, filename: str) -> None:
        """Save the current global model to disk"""
        with open(filename, 'wb') as f:
            pickle.dump(self.global_weights, f)
    
    def plot_loss_trend(self, save_path: Optional[str] = None) -> None:
        """
        Plot the global model loss trend
        
        Args:
            save_path: Optional path to save the plot image
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.global_loss_history) + 1), self.global_loss_history, 'b-', marker='o')
        plt.title('Global Model Loss Trend')
        plt.xlabel('Training Round')
        plt.ylabel('Average Loss')
        plt.grid(True)
        
        # Add annotations for initial and final loss values
        if self.global_loss_history:
            plt.annotate(f'Initial: {self.global_loss_history[0]:.4f}', 
                        xy=(1, self.global_loss_history[0]),
                        xytext=(5, 10), textcoords='offset points')
            
            plt.annotate(f'Final: {self.global_loss_history[-1]:.4f}', 
                        xy=(len(self.global_loss_history), self.global_loss_history[-1]),
                        xytext=(-70, -15), textcoords='offset points')
        
        # Save the plot if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Loss trend plot saved to {save_path}")
        
        plt.show()
    
    def save_loss_history(self, filename: str) -> None:
        """
        Save the loss history to a file
        
        Args:
            filename: Path to save the loss history
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.global_loss_history, f)
        
        # Also save as CSV for easier analysis
        csv_filename = filename.replace('.pkl', '.csv')
        rounds = list(range(1, len(self.global_loss_history) + 1))
        with open(csv_filename, 'w') as f:
            f.write("Round,Loss\n")
            for round_num, loss in zip(rounds, self.global_loss_history):
                f.write(f"{round_num},{loss}\n")
        
        print(f"Loss history saved to {filename} and {csv_filename}")