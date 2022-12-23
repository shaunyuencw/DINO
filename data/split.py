import os
import shutil
import numpy as np

def split_two(lst, ratio=[0.5, 0.5]):
    """ Split a list into partions
        Parameters
        ---------
        lst : List
            List of Data
        ratio : List[Int, Int]
            Ratio to split, default to [0.5, 0.5]

        Return
        ------
        Returns 2 List of Custom Partitions
    
    """
    assert(np.sum(ratio) == 1.0)  # makes sure the splits make sense
    train_ratio = ratio[0]
    # note this function needs only the "middle" index to split, the remaining is the rest of the split
    indices_for_splittin = [int(len(lst) * train_ratio)]
    train, test = np.split(lst, indices_for_splittin)
    return train, test

CLASSES_TO_INCLUDE = ['ContainerShip', 'Cruise', 'Tanker', 'Warship']
DATASET_PERCENTAGE = 0.5
IS_SPLIT = True
PERCENTAGE = False
NUM_TRAIN = 4000
NUM_VAL = 100
NEW_DATASET_NAME = "ship_experiment"
TRAIN_VAL_SPLIT = [0.78, 0.22]

BASE_FOLDER = f"ship_spotting"
classes = [f.path for f in os.scandir(BASE_FOLDER) if f.is_dir()]
matched = [s for s in classes if any(xs in s for xs in CLASSES_TO_INCLUDE)]

# Makes sure every class needed is found...
assert len(CLASSES_TO_INCLUDE) == len(matched)

# Make sure percentage is less than 1
assert DATASET_PERCENTAGE <= 1

# If assertions passed, generate the folder for processing
if os.path.exists(NEW_DATASET_NAME):
    shutil.rmtree(NEW_DATASET_NAME)

os.mkdir(NEW_DATASET_NAME)

if IS_SPLIT:
    os.mkdir(f"{NEW_DATASET_NAME}/train")
    os.mkdir(f"{NEW_DATASET_NAME}/val")
    os.mkdir(f"{NEW_DATASET_NAME}/test")
    for class_path in matched:
        class_name = class_path.replace(f"{BASE_FOLDER}/", "")
        images = [f.name for f in os.scandir(class_path) if f.is_file()]

        os.mkdir(f"{NEW_DATASET_NAME}/train/{class_name}")
        os.mkdir(f"{NEW_DATASET_NAME}/val/{class_name}")
        os.mkdir(f"{NEW_DATASET_NAME}/test/{class_name}")

        # Split by percentage
        if PERCENTAGE == True:
            train_lst, val_lst = split_two(images, TRAIN_VAL_SPLIT)
            for image_name in train_lst:
                try:
                    shutil.copy(f"{BASE_FOLDER}/{class_name}/{image_name}", f"{NEW_DATASET_NAME}/train/{class_name}/{image_name}")

                # For other errors
                except:
                    print("Error occurred while copying file.")
                    break
            
            for image_name in val_lst:
                try:
                    shutil.copy(f"{BASE_FOLDER}/{class_name}/{image_name}", f"{NEW_DATASET_NAME}/val/{class_name}/{image_name}")
                
                # For other errors
                except:
                    print("Error occurred while copying file.")
                    break
        
        # Split by number
        else:
            train_lst, val_lst = split_two(images, TRAIN_VAL_SPLIT)
            for image_name in train_lst[0:NUM_TRAIN]:
                try:
                    shutil.copy(f"{BASE_FOLDER}/{class_name}/{image_name}", f"{NEW_DATASET_NAME}/train/{class_name}/{image_name}")
                # For other errors
                except:
                    print("Error occurred while copying file.")
                    break
            
            for image_name in val_lst[0:NUM_VAL]:
                try:
                    shutil.copy(f"{BASE_FOLDER}/{class_name}/{image_name}", f"{NEW_DATASET_NAME}/val/{class_name}/{image_name}")
                # For other errors
                except:
                    print("Error occurred while copying file.")
                    break

            for image_name in val_lst[len(val_lst):len(val_lst) - 201: -1]:
                try:
                    shutil.copy(f"{BASE_FOLDER}/{class_name}/{image_name}", f"{NEW_DATASET_NAME}/test/{class_name}/{image_name}")
                # For other errors
                except:
                    print("Error occurred while copying file.")
                    break



else:
    for class_path in matched:
        class_name = class_path.replace(f"{BASE_FOLDER}/", "")
        os.mkdir(f"{NEW_DATASET_NAME}/{class_name}")
        images = [f.name for f in os.scandir(class_path) if f.is_file()]
        max_images = round(len(images) * DATASET_PERCENTAGE)

        for image_name in images[0:max_images]:
            try:
                shutil.copy(f"{BASE_FOLDER}/{class_name}/{image_name}", f"{NEW_DATASET_NAME}/{class_name}/{image_name}")
            # If source and destination are same
            except shutil.SameFileError:
                print("Source and destination represents the same file.")
            
            # If there is any permission issue
            except PermissionError:
                print("Permission denied.")
            
            # For other errors
            except:
                print("Error occurred while copying file.")
        

        

