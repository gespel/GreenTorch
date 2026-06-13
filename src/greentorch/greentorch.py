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
from .core.optimizer import SimpleDirectionalOptimizer
from .core.tools import *
from . import optimizer


class GreenTorch(ContextDecorator):
    def __init__(self, key: float = 0, gpu_ids: list = ["1002:73AF-1EAE:6905-0000:09:00.0"]):
        self.logger = logging.getLogger("GreenTorch")
        logging.basicConfig(level=logging.INFO, format=colored("%(asctime)s", "yellow") + "|" + colored("%(name)s", "green") + "|%(levelname)s: %(message)s")
        self.logger.info("GreenTorch initialized")
        self.key = key
        self.gpu_ids = gpu_ids
        self.last_timestamp = 0
        self.last_timediff = 0
        self.profiler_index = 0
        self.profiler_values = {}

        self.devicemanager = DeviceManager(gpu_ids=self.gpu_ids)

        self.devicemanager.gpu_devices[gpu_ids[0]]["backend"].set_gpu_max_frequency(2600)

        for gpu_id in self.gpu_ids:
            self.logger.info(f"Monitoring GPU {gpu_id} with initial frequency {self.devicemanager.gpu_devices[gpu_id]['backend'].get_gpu_max_frequency()} MHz")

        self.logger.info(f"{optimizer.sum_as_string(1, 2)}")
        self.soptimizer = SimpleDirectionalOptimizer(self.logger)

    def __enter__(self):
        self.logger.debug("Entering dynamic frequency scaling part!")
        self.logger.debug(f"initial key={self.key}")
        return self

    def __exit__(self, exc_type, exc, tb):
        self.logger.debug(f"Exiting dynamic frequency part! final key={self.key}")

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

                new_frequency = self.soptimizer.calc_optimize_frequency(
                    self.devicemanager.gpu_devices[gpu_id]["backend"].get_gpu_max_frequency(),
                    self.key,
                    curr_power
                )

                self.logger.info(f"Setting new GPU Frequency: {new_frequency}")
                self.devicemanager.gpu_devices[gpu_id]["backend"].set_gpu_max_frequency(new_frequency)
            self.last_timestamp = now

    def profile(self, max_profiler_measurements: int = 25):
        self.logger.debug("Called energy profiler")

        for gpu_id in self.gpu_ids:
            if gpu_id not in self.profiler_values:
                self.profiler_values[gpu_id] = {
                    "max_frequency": [],
                    "key": [],
                    "power": []
                }

            self.profiler_values[gpu_id]["max_frequency"].append(self.devicemanager.gpu_devices[gpu_id]["backend"].get_gpu_max_frequency())
            self.profiler_values[gpu_id]["key"].append(self.key)
            self.profiler_values[gpu_id]["power"].append(self.devicemanager.get_power_value(gpu_id))

            self.devicemanager.gpu_devices[gpu_id]["backend"].set_gpu_max_frequency(self.devicemanager.gpu_devices[gpu_id]["backend"].get_gpu_max_frequency() - 50)
        self.logger.info(f"Measured: {self.profiler_values}")

        self.profiler_index += 1

        if self.profiler_index >= max_profiler_measurements:
            print(self.profiler_values)
            print_profiling(self.profiler_values)
            exit(0)
