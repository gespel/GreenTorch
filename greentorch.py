import time
import logging
import socket
import random
import json
from contextlib import ContextDecorator
from termcolor import colored
import torch
import torch.nn as nn
import os
import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import requests
from core.devicemanager import DeviceManager


class GreenTorch(ContextDecorator):
    def __init__(self, key: float = 0, gpu_id: str = "1002:73AF-1EAE:6905-0000:09:00.0"):
        self.logger = logging.getLogger("GreenTorch")
        logging.basicConfig(level=logging.DEBUG, format=colored("%(asctime)s", "yellow") + "|" + colored("%(name)s", "green") + "|%(levelname)s: %(message)s")
        self.logger.info("GreenTorch initialized")
        self.key = key
        self.gpu_id = gpu_id
        self.last_timestamp = 0
        self.last_timediff = 0
        self.last_epsilon = None
        self.last_energy = 0
        self.devicemanager = DeviceManager(gpu_id)

        self.devicemanager.set_gpu_max_frequency(2600)

    def __enter__(self):
        self.logger.info("Entering dynamic frequency scaling part!")
        self.logger.info(f"initial key={self.key}")
        return self

    def __exit__(self, exc_type, exc, tb):
        self.logger.info(f"Exiting dynamic frequency part! final key={self.key}")

    def calc_optimize_frequency(self, energy: float) -> float:
        if self.last_epsilon == None:
            self.last_epsilon = self.key / energy
        else:
            epsilon = self.key / energy
            if self.last_epsilon > epsilon:
                self.logger.info(f"Last ε {self.last_epsilon} was {colored("BETTER", "red")} than current ε {epsilon}. {colored("CLOCKING DOWN", "green")}")

                curr_freq = self.devicemanager.get_gpu_max_frequency()
                
                self.last_epsilon = epsilon
                return curr_freq - 100

            elif self.last_epsilon < epsilon:
                self.logger.info(f"Last ε {self.last_epsilon} was {colored("WORSE", "green")} than current ε {epsilon}. {colored("CLOCKING UP", "red")}")

                curr_freq = self.devicemanager.get_gpu_max_frequency()

                self.last_epsilon = epsilon
                return curr_freq + 100
            else:
                self.logger.info(f"Epsilon is equal at {epsilon}...")
                return self.devicemanager.get_gpu_max_frequency()
            

    def optimize(self):
        self.logger.info(f"Called energy optimizer")

        if self.last_timestamp == 0:
            self.last_timestamp = time.time()
        else:
            now = time.time()
            time_diff = now - self.last_timestamp
            self.last_timediff = time_diff
            curr_power = self.devicemanager.get_power_value()
            self.logger.info(f"Measured power: {curr_power} W")
            energy = curr_power * time_diff
            self.logger.info(f"Time since last call {time_diff} s. Energy used {energy:0.4f} Joule with key {self.key:0.4f}: epsilon is {self.key / energy}")

            new_frequency = self.calc_optimize_frequency(energy)
            self.logger.info(f"Setting new GPU Frequency: {new_frequency}")

            self.devicemanager.set_gpu_max_frequency(new_frequency)
            self.last_timestamp = now

    

