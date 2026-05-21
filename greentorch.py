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
        self.last_freq = None

        # Simple directional search state
        self.direction = -1  # start by clocking down from initial frequency
        self.step_mhz = 100
        self.epsilon_ema = None
        self.epsilon_alpha = 0.3 
        self.epsilon_tolerance = 0.01  # 1% deadband to avoid direction-flapping
        self.devicemanager = DeviceManager(gpu_id)

        self.devicemanager.set_gpu_max_frequency(2600)

    def __enter__(self):
        self.logger.info("Entering dynamic frequency scaling part!")
        self.logger.info(f"initial key={self.key}")
        return self

    def __exit__(self, exc_type, exc, tb):
        self.logger.info(f"Exiting dynamic frequency part! final key={self.key}")

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
            self.logger.info(
                f"Baseline ε initialized to {epsilon} (raw {epsilon_raw}) at {curr_freq} MHz."
            )
            next_freq = curr_freq + (self.direction * self.step_mhz)
            return self._clamp_frequency(next_freq)

        # Relative-change deadband to avoid flapping on noise.
        prev = self.last_epsilon
        denom = max(abs(prev), 1e-12)
        rel_change = (epsilon - prev) / denom

        if rel_change > self.epsilon_tolerance:
            self.logger.info(
                f"ε improved {prev} -> {epsilon} (raw {epsilon_raw}, Δ={rel_change:+.2%}). "
                f"{colored('CONTINUE', 'green')} direction {self.direction:+d}"
            )
        elif rel_change < -self.epsilon_tolerance:
            self.direction *= -1
            self.logger.info(
                f"ε worsened {prev} -> {epsilon} (raw {epsilon_raw}, Δ={rel_change:+.2%}). "
                f"{colored('REVERSE', 'red')} direction to {self.direction:+d}"
            )
        else:
            self.logger.info(
                f"ε change within deadband {prev} -> {epsilon} (raw {epsilon_raw}, Δ={rel_change:+.2%}). "
                f"{colored('HOLD', 'yellow')}"
            )

        self.last_epsilon = epsilon
        next_freq = curr_freq + (self.direction * self.step_mhz)
        return self._clamp_frequency(next_freq)
            

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

            # `key` is expected to be throughput (e.g. batches/s), so efficiency is key/power.
            epsilon = (self.key / curr_power) if curr_power > 0 else 0.0
            self.logger.info(
                f"Time since last call {time_diff} s. Key {self.key:0.4f}: epsilon is {epsilon} (key/power)"
            )

            new_frequency = self.calc_optimize_frequency(curr_power)

            if self.last_freq is not None and int(new_frequency) == int(self.last_freq):
                self.logger.info(f"Holding GPU Frequency: {new_frequency}")
            else:
                self.logger.info(f"Setting new GPU Frequency: {new_frequency}")
                self.devicemanager.set_gpu_max_frequency(new_frequency)
            self.last_timestamp = now

    

