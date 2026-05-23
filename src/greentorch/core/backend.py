import socket
import json
from collections import deque

class DeviceBackend:
    def __init__(self, gpu_id: str):
        self.gpu_id = gpu_id

    def get_gpu_max_frequency(self) -> float:
        raise NotImplementedError
    
    def set_gpu_max_frequency(self, freq: int) -> bool:
        raise NotImplementedError
    
    def get_power_usage(self) -> float:
        raise NotImplementedError
    
    

class LACTDeviceBackend(DeviceBackend):
    def __init__(self, gpu_id: str):
        super().__init__(gpu_id=gpu_id)

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
