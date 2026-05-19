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



if __name__ == "__main__":
    x = 1
    with GreenTorch() as gt:
        gt.key = 7
        print("This is a test!")
        gt.optimize()