import os
import sys
from main import main
from icecream import ic

def list_full_dir(path: str):
    return [os.path.join(path, file) for file in os.listdir(path)]

if __name__ == '__main__':
    for config in list_full_dir(sys.argv[1]):
        ic(config)
        main(config)