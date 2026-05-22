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
from .core.devicemanager import DeviceManager


class GreenTorch(ContextDecorator):
    def __init__(self, key: float = 0, gpu_id: str = "1002:73AF-1EAE:6905-0000:09:00.0"):
        self.logger = logging.getLogger("GreenTorch")
        logging.basicConfig(level=logging.INFO, format=colored("%(asctime)s", "yellow") + "|" + colored("%(name)s", "green") + "|%(levelname)s: %(message)s")
        self.logger.info("GreenTorch initialized")
        self.key = key
        self.gpu_id = gpu_id
        self.last_timestamp = 0
        self.last_timediff = 0
        self.last_epsilon = None
        self.last_energy = 0
        self.last_freq = None

        self.direction = -1 
        self.step_mhz = 50
        self.epsilon_ema = None
        self.epsilon_alpha = 0.3 
        self.epsilon_tolerance = 0.005
        self.devicemanager = DeviceManager(gpu_id)

        self.devicemanager.set_gpu_max_frequency(2600)

    def __enter__(self):
        self.logger.debug("Entering dynamic frequency scaling part!")
        self.logger.debug(f"initial key={self.key}")
        return self

    def __exit__(self, exc_type, exc, tb):
        self.logger.debug(f"Exiting dynamic frequency part! final key={self.key}")

    def _clamp_frequency(self, freq: int) -> int:
        return max(100, int(freq))

    def calc_optimize_frequency(self, power_w: float) -> float:
        """Simple directional optimizer for ε = key/power (higher is better)."""

        curr_freq = self.devicemanager.get_gpu_max_frequency()
        self.last_freq = curr_freq
        if curr_freq <= 0:
            self.logger.info("Could not read current GPU frequency; skipping adjustment.")
            return curr_freq

        if power_w <= 0:
            self.logger.info("Power is non-positive; keeping current frequency.")
            return curr_freq

        epsilon_raw = self.key / power_w

        if self.epsilon_ema is None:
            epsilon = epsilon_raw
            self.epsilon_ema = epsilon_raw
        else:
            self.epsilon_ema = (self.epsilon_alpha * epsilon_raw) + ((1.0 - self.epsilon_alpha) * self.epsilon_ema)
            epsilon = self.epsilon_ema

        if self.last_epsilon is None:
            self.last_epsilon = epsilon
            self.logger.debug(
                f"Baseline ε initialized to {epsilon:0.4f} (raw {epsilon_raw:0.4f}) at {curr_freq:0.4f} MHz."
            )
            next_freq = curr_freq + (self.direction * self.step_mhz)
            return self._clamp_frequency(next_freq)

        prev = self.last_epsilon

        if prev == 0:
            prev = 1e-12

        rel_change = (epsilon - prev) / abs(prev)

        if rel_change > self.epsilon_tolerance:
            self.logger.info(
                f"ε improved {prev:0.4f} -> {epsilon:0.4f} (raw {epsilon_raw:0.4f}, Δ={rel_change:+.2%}). "
                f"{colored('CONTINUE', 'green')} direction {self.direction:+d}"
            )
        elif rel_change < -self.epsilon_tolerance:
            self.direction *= -1
            self.logger.info(
                f"ε worsened {prev:0.4f} -> {epsilon:0.4f} (raw {epsilon_raw:0.4f}, Δ={rel_change:+.2%}). "
                f"{colored('REVERSE', 'red')} direction to {self.direction:+d}"
            )
        else:
            self.logger.info(
                f"ε change within deadband {prev:0.4f} -> {epsilon:0.4f} (raw {epsilon_raw:0.4f}, Δ={rel_change:+.2%}). "
                f"{colored('HOLD', 'yellow')} direction to {self.direction:+d}"
            )

        self.last_epsilon = epsilon
        next_freq = curr_freq + (self.direction * self.step_mhz)
        return self._clamp_frequency(next_freq)
            

    def optimize(self):
        self.logger.debug(f"Called energy optimizer")
        if self.last_timestamp == 0:
            self.last_timestamp = time.time()
        else:
            now = time.time()
            time_diff = now - self.last_timestamp
            self.last_timediff = time_diff
            curr_power = self.devicemanager.get_power_value()
            self.logger.info(f"Measured power: {curr_power} W")

            epsilon = (self.key / curr_power) if curr_power > 0 else 0.0
            self.logger.debug(f"Time since last call {time_diff} s. Key {self.key:0.4f}: epsilon is {epsilon} (key/power)")

            new_frequency = self.calc_optimize_frequency(curr_power)

            if self.last_freq is not None and int(new_frequency) == int(self.last_freq):
                self.logger.info(f"Holding GPU Frequency: {new_frequency}")
            else:
                self.logger.info(f"Setting new GPU Frequency: {new_frequency}")
                self.devicemanager.set_gpu_max_frequency(new_frequency)
            self.last_timestamp = now

    

