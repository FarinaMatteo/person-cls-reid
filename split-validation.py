import os 
import shutil
import random


#create another directory with labeled queries (froma validation)
def create_queries_from_validation():
    try:
        shutil.rmtree('queries_val/')
    except FileNotFoundError as exc:
        print("No validation directory found on your system, creating one.")
    try:      
        os.mkdir('queries_val/')
    except FileExistsError as exc:
        print(exc)

    val_list = os.listdir("validation_directory")
    random.shuffle(val_list)
    length = len(val_list)


    for i in range(int(length*0.1)):
        shutil.copy("validation_directory/"+val_list[i], "queries_val")
        print(val_list[i])

