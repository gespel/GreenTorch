import socket
import json
import time
import multiprocessing
from collections import deque

class DeviceBackend:
    def __init__(self):
        pass

class LACTDeviceBackend(DeviceBackend):
    def __init__(self):
        self.super().__init__()

class DeviceManager:
    def __init__(self, gpu_id: str):
        self.gpu_id = gpu_id
        self._power_value = multiprocessing.Value("d", 0.0)
        self._power_lock = multiprocessing.Lock()
        self._power_stop_event = multiprocessing.Event()
        self._power_process = None
        self.start_power_monitor(interval=0.1, max_samples=100)

    def lact_request(self, payload: dict) -> dict:
        try:
            with socket.create_connection(("127.0.0.1", 12853), timeout=2.0) as sock:
                message = (json.dumps(payload) + "\n").encode()
                sock.sendall(message)

                buffer = b""
                while b"\n" not in buffer:
                    chunk = sock.recv(65536)
                    if not chunk:
                        break
                    buffer += chunk

                line = buffer.split(b"\n", 1)[0].strip()
                if not line:
                    return None

                return json.loads(line.decode())

        except Exception as e:
            print(f"Error while connecting to LACT: {e}")
            return None
        
    def get_gpu_max_frequency(self) -> float:
        response = self.lact_request({
            "command": "device_clocks_info",
            "args": {
                "id": self.gpu_id
            }
        })

        if response == None:
            return 0
        
        if response["status"] != "ok":
            print(response)

        return response["data"]["max_sclk"]

    def set_gpu_max_frequency(self, freq: int) -> bool:
        response = self.lact_request({
            "command": "set_clocks_value",
            "args": {
                "id": self.gpu_id,
                "command": {
                    "type": "max_core_clock",
                    "value": freq
                }
            }
        })
        
        if response == None:
            return False

        if response["status"] != "ok":
            print("set:", response)

        confirm_response = self.lact_request({
            "command": "confirm_pending_config",
            "args": {
                "command": "confirm"
            }
        })

        if confirm_response["status"] != "ok":
            print("confirm:", confirm_response)

        return True

    def get_power_usage(self) -> float:
        response = self.lact_request({
            "command": "device_stats",
            "args": {
                "id": self.gpu_id
            }
        })

        if response == None:
            return None

        if response["status"] != "ok":
            print(response)
            return None

        try:
            return float(response["data"]["power"]["average"])
        except Exception:
            return None

    def power_monitor_process_handle(self, value, lock, stop_event, interval: float = 0.1, max_samples: int = 1000):
        past_values = deque(maxlen=max_samples)
        running_sum = 0.0
        while not stop_event.is_set():
            v = self.get_power_usage()

            if v is None or v <= 0:
                time.sleep(interval)
                continue

            if len(past_values) >= past_values.maxlen:
                running_sum -= past_values[0]

            past_values.append(v)
            running_sum += v

            avg = running_sum / max(1, len(past_values))

            with lock:
                value.value = avg

            time.sleep(interval)

    def start_power_monitor(self, interval: float = 0.1, max_samples: int = 3000) -> None:
        if self._power_process and self._power_process.is_alive():
            return

        self._power_stop_event.clear()
        self._power_process = multiprocessing.Process(
            target=self.power_monitor_process_handle,
            args=(self._power_value, self._power_lock, self._power_stop_event, interval, max_samples),
            daemon=True,
        )
        self._power_process.start()

    def stop_power_monitor(self) -> None:
        if not self._power_process:
            return

        self._power_stop_event.set()
        self._power_process.join(timeout=1.0)
        self._power_process = None

    def get_power_value(self) -> float:
        with self._power_lock:
            value = float(self._power_value.value)

        if value <= 0:
            direct = self.get_power_usage()
            if direct is not None and direct > 0:
                with self._power_lock:
                    self._power_value.value = float(direct)
                return float(direct)

        return value
            