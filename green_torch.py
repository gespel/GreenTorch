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
        self.logger.info(self.key)

    def __exit__(self, exc_type, exc, tb):
        self.logger.info("Exiting dynamic frequency part!")



if __name__ == "__main__":
    x = 1
    with GreenTorch():
        print(f"This is a test!")