import socket
import json
import time
import multiprocessing

class DeviceManager:
    def __init__(self, gpu_id: str):
        self.gpu_id = gpu_id
        self._power_value = multiprocessing.Value("d", 0.0)
        self._power_lock = multiprocessing.Lock()
        self._power_stop_event = multiprocessing.Event()
        self._power_process = None
        self.start_power_monitor()

    def lact_request(self, payload: dict) -> dict:
        try:
            sock = socket.create_connection(("127.0.0.1", 12853))
            sock.sendall((json.dumps(payload) + "\n").encode())
            
            response = sock.recv(163840)

            #print(json.loads(response.decode()))

            return json.loads(response.decode())

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
            return 0

        if response["status"] != "ok":
            print(response)
        return response["data"]["power"]["average"]

    def power_monitor_process_handle(self, value, lock, stop_event, interval: float = 0.1, max_samples: int = 1000):
        past_values = []
        while not stop_event.is_set():
            past_values.append(self.get_power_usage())
            if len(past_values) > max_samples:
                past_values.pop(0)

            total = 0.0
            for v in past_values:
                total += v

            avg = total / len(past_values)

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
            return float(self._power_value.value)
            