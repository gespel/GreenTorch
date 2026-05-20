import time
import logging
import socket
import json
from contextlib import ContextDecorator
from termcolor import colored

class GreenTorch(ContextDecorator):
    def __init__(self, key: float = 0, gpu_id: str = "1002:73AF-1EAE:6905-0000:09:00.0"):
        self.logger = logging.getLogger("GreenTorch")
        logging.basicConfig(level=logging.DEBUG, format=colored("%(asctime)s", "yellow") + "|" + colored("%(name)s", "green") + "|%(levelname)s: %(message)s")
        self.logger.info("GreenTorch initialized")
        self.key = key
        self.gpu_id = gpu_id
        self.last_timestamp = 0
        self.last_timediff = 0

    def __enter__(self):
        self.logger.info("Entering dynamic frequency scaling part!")
        self.logger.info(f"initial key={self.key}")
        return self

    def __exit__(self, exc_type, exc, tb):
        self.logger.info(f"Exiting dynamic frequency part! final key={self.key}")

    def calc_optimize_frequency(self, energy: float):
        pass

    def optimize(self):
        self.logger.info(f"Called energy optimizer")

        if self.last_timestamp == 0:
            self.last_timestamp = time.time()
        
        else:
            time_diff = time.time() - self.last_timestamp
            self.last_timediff = time_diff
            self.logger.info(f"Time since last call {time_diff} s")

            energy = self.get_power_usage() * time_diff

            self.set_gpu_max_frequency(self.calc_optimize_frequency(energy))



    def lact_request(self, payload: dict) -> dict:
        try:
            sock = socket.create_connection(("127.0.0.1", 12853))
            sock.sendall((json.dumps(payload) + "\n").encode())

            response = sock.recv(4096)

            return json.loads(response.decode())
        
        except:
            print("Error while connecting to LACT!")
            return None

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



if __name__ == "__main__":
    x = 1
    with GreenTorch() as gt:
        gt.key = 7
        print("This is a test!")
        gt.optimize()
        time.sleep(3)
        gt.optimize()