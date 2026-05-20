import logging
from contextlib import ContextDecorator

class GreenTorch(ContextDecorator):
    def __init__(self, key=0):
        self.logger = logging.getLogger("GreenTorch")
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s|%(name)s|%(levelname)s: %(message)s")
        self.logger.info("GreenTorch initialized")
        self.key = key

    def __enter__(self):
        self.logger.info("Entering dynamic frequency scaling part!")
        self.logger.info(f"initial key={self.key}")
        return self

    def __exit__(self, exc_type, exc, tb):
        self.logger.info(f"Exiting dynamic frequency part! final key={self.key}")

    def optimize(self):
        self.logger.info(f"Called energy optimizer")

    def lact_request(self, payload: dict) -> dict:
        with socket.create_connection(("127.0.0.1", 12853)) as sock:
            sock.sendall((json.dumps(payload) + "\n").encode())

            response = sock.recv(4096)

        return json.loads(response.decode())

    def set_gpu_max_frequency(self, freq: int) -> bool:
        gpu_id = "1002:73AF-1EAE:6905-0000:09:00.0"

        response = lact_request({
            "command": "set_clocks_value",
            "args": {
                "id": gpu_id,
                "command": {
                    "type": "max_core_clock",
                    "value": freq
                }
            }
        })
        
        if response["status"] != "ok":
            print("set:", response)

        confirm_response = lact_request({
            "command": "confirm_pending_config",
            "args": {
                "command": "confirm"
            }
        })

        if confirm_response["status"] != "ok":
            print("confirm:", confirm_response)

        return True

    def get_power_usage(self) -> float:
        response = lact_request({
            "command": "device_stats",
            "args": {
                "id": "1002:73AF-1EAE:6905-0000:09:00.0"
            }
        })

        if response["status"] != "ok":
            print(response)
        return response["data"]["power"]["average"]



if __name__ == "__main__":
    x = 1
    with GreenTorch() as gt:
        gt.key = 7
        print("This is a test!")
        gt.optimize()