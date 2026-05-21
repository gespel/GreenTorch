class DeviceManager:
    def __init__(self, gpu_id: str):
        self.gpu_id = gpu_id

    def lact_request(self, payload: dict) -> dict:
        try:
            sock = socket.create_connection(("127.0.0.1", 12853))
            sock.sendall((json.dumps(payload) + "\n").encode())
            
            response = sock.recv(163840)

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