import time
import multiprocessing

from .backend import LACTDeviceBackend
from . import *
from collections import deque


class DeviceManager:
    def __init__(self, gpu_ids: list, logger):
        self.logger = logger
        try:
            self._mp_ctx = multiprocessing.get_context("fork")
        except ValueError:
            self._mp_ctx = multiprocessing.get_context()

        self.gpu_ids = gpu_ids
        self.gpu_devices = {}

        for gpu_id in gpu_ids:
            self.gpu_devices[gpu_id] = {
                "backend": LACTDeviceBackend(gpu_id=gpu_id),
                "power_value": self._mp_ctx.Value("d", 0.0),
                "power_lock": self._mp_ctx.Lock(),
                "power_stop_event": self._mp_ctx.Event(),
                "power_process": None,
                "epsilon": None,
                "last_epsilon": None,
                "last_energy": None,
                "last_frequency": None,
                "direction": -1,
                "step_mhz": 50,
                "epsilon_ema": None,
                "epsilon_alpha": 0.3,
                "epsilon_tolerance": 0.005,
            }

            self.start_power_monitor(gpu_id)


    def power_monitor_process_handle(self, gpu_id: str, value, lock, stop_event, interval: float = 0.1, max_samples: int = 100):
        past_values = []
        running_sum = 0.0
        while not stop_event.is_set():
            v = self.gpu_devices[gpu_id]["backend"].get_power_usage()

            if v is None or v <= 0:
                time.sleep(interval)
                continue

            if len(past_values) >= max_samples:
                running_sum -= past_values[0]
                past_values.pop(0)

            past_values.append(v)
            running_sum += v

            avg = running_sum / max(1, len(past_values))

            #print(f"current: {v} avg: {avg} pastvalues.len: {len(past_values)}")

            with lock:
                value.value = avg

            time.sleep(interval)

    def start_power_monitor(self, gpu_id: str, interval: float = 0.01, max_samples: int = 100) -> None:
        self.gpu_devices[gpu_id]["power_stop_event"].clear()
        self.gpu_devices[gpu_id]["power_process"] = self._mp_ctx.Process(
            target=self.power_monitor_process_handle,
            args=(
                gpu_id,
                self.gpu_devices[gpu_id]["power_value"],
                self.gpu_devices[gpu_id]["power_lock"],
                self.gpu_devices[gpu_id]["power_stop_event"],
                interval,
                max_samples
            ),
            daemon=True,
        )
        self.gpu_devices[gpu_id]["power_process"].start()

    def stop_power_monitor(self, gpu_id: str) -> None:
        if not self.gpu_devices[gpu_id]["power_process"]:
            return

        self.gpu_devices[gpu_id]["power_stop_event"].set()
        self.gpu_devices[gpu_id]["power_process"].join(timeout=1.0)

    def get_power_value(self, gpu_id: str) -> float:
        with self.gpu_devices[gpu_id]["power_lock"]:
            value = float(self.gpu_devices[gpu_id]["power_value"].value)
        if value <= 0:
            direct = self.gpu_devices[gpu_id]["backend"].get_power_usage()
            if direct is not None and direct > 0:
                with self.gpu_devices[gpu_id]["power_lock"]:
                    self.gpu_devices[gpu_id]["power_value"].value = float(direct)
                return float(direct)

        return value
