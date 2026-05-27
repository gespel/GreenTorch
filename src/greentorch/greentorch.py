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
import greentorch.rust_core as rust_core


class GreenTorch(ContextDecorator):
    def __init__(self, key: float = 0, gpu_ids: list = ["1002:73AF-1EAE:6905-0000:09:00.0"]):
        self.logger = logging.getLogger("GreenTorch")
        logging.basicConfig(level=logging.INFO, format=colored("%(asctime)s", "yellow") + "|" + colored("%(name)s", "green") + "|%(levelname)s: %(message)s")
        self.logger.info("GreenTorch initialized")
        self.key = key
        self.gpu_ids = gpu_ids
        self.last_timestamp = 0
        self.last_timediff = 0

        self.devicemanager = DeviceManager(gpu_ids=self.gpu_ids)

        for gpu_id in self.gpu_ids:
            self.logger.info(f"Monitoring GPU {gpu_id} with initial frequency {self.devicemanager.gpu_devices[gpu_id]['backend'].get_gpu_max_frequency()} MHz")

    def __enter__(self):
        self.logger.debug("Entering dynamic frequency scaling part!")
        self.logger.debug(f"initial key={self.key}")
        return self

    def __exit__(self, exc_type, exc, tb):
        self.logger.debug(f"Exiting dynamic frequency part! final key={self.key}")

    def _clamp_frequency(self, freq: int) -> int:
        return max(100, int(freq))

    def calc_optimize_frequency(self, gpu_id: str, curr_power: float) -> float:
        """Simple directional optimizer for ε = key/power (higher is better)."""

        curr_freq = self.devicemanager.gpu_devices[gpu_id]["backend"].get_gpu_max_frequency()
        self.last_freq = curr_freq
        if curr_freq <= 0:
            self.logger.info("Could not read current GPU frequency; skipping adjustment.")
            return curr_freq

        if curr_power <= 0:
            self.logger.info("Power is non-positive; keeping current frequency.")
            return curr_freq

        epsilon_raw = self.key / curr_power
        if self.devicemanager.gpu_devices[gpu_id]["epsilon_ema"] is None:
            epsilon = epsilon_raw
            self.devicemanager.gpu_devices[gpu_id]["epsilon_ema"] = epsilon_raw
        else:
            self.devicemanager.gpu_devices[gpu_id]["epsilon_ema"] = (self.devicemanager.gpu_devices[gpu_id]["epsilon_alpha"] * epsilon_raw) + ((1.0 - self.devicemanager.gpu_devices[gpu_id]["epsilon_alpha"]) * self.devicemanager.gpu_devices[gpu_id]["epsilon_ema"])
            epsilon = self.devicemanager.gpu_devices[gpu_id]["epsilon_ema"]

        if self.devicemanager.gpu_devices[gpu_id]["last_epsilon"] is None:
            self.devicemanager.gpu_devices[gpu_id]["last_epsilon"] = epsilon
            self.logger.debug(
                f"Baseline ε initialized to {epsilon:0.4f} (raw {epsilon_raw:0.4f}) at {curr_freq:0.4f} MHz."
            )
            next_freq = curr_freq + (self.devicemanager.gpu_devices[gpu_id]["direction"] * self.devicemanager.gpu_devices[gpu_id]["step_mhz"])
            return self._clamp_frequency(next_freq)

        prev = self.devicemanager.gpu_devices[gpu_id]["last_epsilon"]

        if prev == 0:
            prev = 1e-12

        rel_change = (epsilon - prev) / abs(prev)

        if rel_change > self.devicemanager.gpu_devices[gpu_id]["epsilon_tolerance"]:
            self.logger.info(
                f"ε improved {prev:0.4f} -> {epsilon:0.4f} (raw {epsilon_raw:0.4f}, Δ={rel_change:+.2%}). "
                f"{colored('CONTINUE', 'green')} direction {self.devicemanager.gpu_devices[gpu_id]['direction']:+d}"
            )
        elif rel_change < -self.devicemanager.gpu_devices[gpu_id]["epsilon_tolerance"]:
            self.devicemanager.gpu_devices[gpu_id]["direction"] *= -1
            self.logger.info(
                f"ε worsened {prev:0.4f} -> {epsilon:0.4f} (raw {epsilon_raw:0.4f}, Δ={rel_change:+.2%}). "
                f"{colored('REVERSE', 'red')} direction to {self.devicemanager.gpu_devices[gpu_id]['direction']:+d}"
            )
        else:
            self.logger.info(
                f"ε change within deadband {prev:0.4f} -> {epsilon:0.4f} (raw {epsilon_raw:0.4f}, Δ={rel_change:+.2%}). "
                f"{colored('HOLD', 'yellow')} direction to {self.devicemanager.gpu_devices[gpu_id]['direction']:+d}"
            )

        self.devicemanager.gpu_devices[gpu_id]["last_epsilon"] = epsilon
        next_freq = curr_freq + (self.devicemanager.gpu_devices[gpu_id]["direction"] * self.devicemanager.gpu_devices[gpu_id]["step_mhz"])
        return self._clamp_frequency(next_freq)
            

    def optimize(self):
        self.logger.debug(f"Called energy optimizer")
        if self.last_timestamp == 0:
            self.last_timestamp = time.time()
        else:
            now = time.time()
            time_diff = now - self.last_timestamp
            self.last_timediff = time_diff
            for gpu_id in self.gpu_ids:
                curr_power = self.devicemanager.get_power_value(gpu_id)
                self.logger.info(f"Measured power: {curr_power} W")

                epsilon = (self.key / curr_power) if curr_power > 0 else 0.0
                self.logger.debug(f"Time since last call {time_diff} s. Key {self.key:0.4f}: epsilon is {epsilon} (key/power)")

                new_frequency = self.calc_optimize_frequency(gpu_id=gpu_id, curr_power=curr_power)

                if self.last_freq is not None and int(new_frequency) == int(self.last_freq):
                    self.logger.info(f"Holding GPU Frequency: {new_frequency}")
                else:
                    self.logger.info(f"Setting new GPU Frequency: {new_frequency}")
                    self.devicemanager.gpu_devices[gpu_id]["backend"].set_gpu_max_frequency(new_frequency)
            self.last_timestamp = now

    

