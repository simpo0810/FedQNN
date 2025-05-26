
import numpy as np
import logging

logger = logging.getLogger(__name__)

def load_data(filename: str) -> np.ndarray:
    """Load channel data from file."""
    try:
        data = np.load(filename)
        logger.info(f"Loaded data from {filename} with shape {data.shape}")
        return data
    except FileNotFoundError:
        logger.error(f"Data file {filename} not found")
        raise

def split_data(data: np.ndarray, num_clients: int) -> list:
    """Split data among clients."""
    if len(data) < num_clients:
        raise ValueError(f"Data length {len(data)} less than number of clients {num_clients}")
    data_per_client = len(data) // num_clients
    remainder = len(data) % num_clients
    clients_data = []
    start_idx = 0
    for i in range(num_clients):
        end_idx = start_idx + data_per_client + (1 if i < remainder else 0)
        clients_data.append(data[start_idx:end_idx])
        start_idx = end_idx
    return clients_data

def save_results(filename: str, data: dict) -> None:
    """Save results to file."""
    try:
        np.save(filename, data)
        logger.info(f"Results saved to {filename}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")