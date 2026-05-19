from contextlib import ContextDecorator

class GreenTorch(ContextDecorator):
    def __init__(self):
        print("GreenTorch initialized")

    def __enter__(self):
        print("Entering dynamic frequency scaling part!")

    def __exit__(self, exc_type, exc, tb):
        print("Exiting dynamic frequency part!")


if __name__ == "__main__":
    with GreenTorch():
        print("This is a test!")