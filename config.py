

from dataclasses import dataclass, field
import numpy as np

@dataclass
class Config:
    N_Tx: int = 2
    M_Tx: int = 4
    N_Rx: int = 2
    N_user: int = 4
    N_path: int = 3
    N_out: int = 10
    N_data: int = 50
    SNR_dB: list = field(default_factory=lambda: list(range(0, 20, 2)))
    B: float = 8.64e6 * 0.5
    alpha: float = 0.01
    local_epochs: int =3
    num_rounds: int = 3
    num_clients: int = 3

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {k: getattr(self, k) for k in self.__dataclass_fields__}