from Gender_main_file import *
from sys import argv
import os

def Find_gender(FOLDER, FILE, *args):
    final(FOLDER, FILE)
    print("Get current path:", os.getcwd())

if __name__ == "__main__":
    if len(argv) != 3:
        print("Usage: script.py <FOLDER> <FILE>")
    else:
        Find_gender(argv[1], argv[2])