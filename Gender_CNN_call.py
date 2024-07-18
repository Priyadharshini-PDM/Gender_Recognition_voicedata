from Gender_CNN import *
from sys import argv
import os

def Find_gender(FOLDER,FILE,*args):
    final(FOLDER, FILE)
    print("Get current path:", os.getcwd())


Find_gender(*argv[1:])



  