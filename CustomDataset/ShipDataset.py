import os.path
import pandas as pd
import numpy as np
from typing import Any, Callable, Iterable, Optional, Tuple, TypeVar, cast
from torch.utils.data import Dataset
from PIL import Image
from CustomDataset.VisionDataset import VisionDataset
from tqdm.auto import tqdm

class ShipDataset(VisionDataset):

    base_folder = "sub_ship_spotting_single"
    ds_types = ("pt_unlabeled", "pt_labeled", "ft_train", "ft_test", "full")
    methods = ("self-supervised", "supervised", "unsupervised")

    """ Creates a Dataset Object for various purposes
        
        Parameters
        ----------
        root : str
            Root path of the images
        
        ds_type : str
            Dataset type -> pt_unlabeled, pt_labeled, ft_trian or ft_test
        
        train_ratio: float
            Percentage of dataset to be training data
        
        method: str
            Method Dataset will be used for -> self-supervised, supervised, unsupervised
        
        transform : Optional[Callable]
            Apply transformation to images

        target_transform: Optional[Callable]
            Secondary trainsformations
        
        Attributes
        ----------
        None

        Return
        ------
        None

    """

    def __init__(
        self,
        root: str,
        ds_type: str,
        train_ratio: float,
        method: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.root = root
        self.base_folder = f"{root}/{self.base_folder}"
        self.ds_type = ds_type
        assert ds_type in self.ds_types, "Invalid Dataset Type"
        assert method in self.methods, "Invalid Dataaset Type"
        
        if self.ds_type == "pt_unlabeled":
            self.data, _ = self.__loadfile(train_ratio)
            self.labels = np.asarray([-1] * self.data.shape[0])
        else:
            self.data, self.labels = self.__loadfile(train_ratio)



    def __len__(self):
        """ Returns number of data entry in data
            Parameters
            ----------
            None

            Return
            ------
            int
                Number of data entries in Dataset
        """
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """ Get nth item in Dataset
            Parameters
            ----------
            index : int 
                Index of item to return
            
            Return
            ------
            img, target : Tuple[Any, Any] 
                Image and label of the entry
        """
        
        target: Optional[int]
        if self.labels is not None:
            img_path, target = self.data.iloc[index][0], int(self.labels[index])
        else:
            img,path, target = self.data.iloc[index][0], None

        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target

    
    def __loadfile(self, train_ratio:float):
        """ Loads image files and labels
            
            Parameters
            ----------
            train_ratio : float
                Percentage of image to use for this segment of Data
        """
        image_df = pd.DataFrame(columns=["image", "label"])
        labels = []

        require_labels = False if self.ds_type == "pt_unlabeled" else True
        
        # ? Number of images for each class to use
        self.classes = [f.path for f in os.scandir(self.base_folder) if f.is_dir()]
        for cidx, class_path in enumerate(tqdm(self.classes, unit="class", leave=False)):
            # print(f"{cidx} : {class_path}")
            images = [f.path for f in os.scandir(class_path) if f.is_file()]
            # print(f"{len(images)} images in this class")

            pt_types = ("pt_unlabeled", "pt_labeled")
            ft_types = ("ft_train", "ft_test")

            pt_lst, ft_lst = split_two(images, [train_ratio, 1-train_ratio])
            target_list = images


            if self.ds_type in pt_types:
                pt_unlabeled, pt_labeled = split_two(pt_lst, [train_ratio, 1-train_ratio])
                target_list = pt_unlabeled if self.ds_type == "pt_unlabeled" else pt_labeled
                
            elif self.ds_type in ft_types:
                ft_train, ft_test = split_two(ft_lst, [0.8, 0.2])
                target_list = ft_train if self.ds_type == "ft_train" else ft_test
                
            elif self.ds_type == "full":
                target_list = images
    
            # print(f"Taking {len(target_list)} images...")
            # print(f"Processing images...")

            for image_path in tqdm(target_list, unit="images", miniters=250, leave=False):
                if require_labels:
                    image_df.loc[len(image_df.index)] = [image_path, cidx]
                    labels.append(cidx)
                else:
                    image_df.loc[len(image_df.index)] = [image_path, -1]
                    labels.append(-1)
                    
        print(f"{self.ds_type} loaded")
        return image_df, np.array(labels)

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