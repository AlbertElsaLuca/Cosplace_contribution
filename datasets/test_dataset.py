
import os
import numpy as np
from glob import glob
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from sklearn.neighbors import NearestNeighbors


def open_image(path):
    return Image.open(path).convert("RGB")


class TestDataset(data.Dataset):
    def __init__(self, dataset_folder, database_folder="database",
                 queries_folder="queries", positive_dist_threshold=25):
        """Dataset with images from database and queries, used for validation and test.
        Parameters
        ----------
        dataset_folder : str, should contain the path to the val or test set,
            which contains the folders {database_folder} and {queries_folder}.
        database_folder : str, name of folder with the database.
        queries_folder : str, name of folder with the queries.
        positive_dist_threshold : int, distance in meters for a prediction to
            be considered a positive.
        """
        super().__init__()
        self.dataset_folder = dataset_folder
        print(f"data set folder -->: {self.dataset_folder}") #added
        print(f"database folder -->: {database_folder}") #added
        self.database_folder = os.path.join(dataset_folder, database_folder)
        print(f"database folder updated -->: {database_folder}") #added
        self.queries_folder = os.path.join(dataset_folder, queries_folder)
        self.dataset_name = os.path.basename(dataset_folder)
        print(f"-->: {self.database_folder}") #added
        if not os.path.exists(self.dataset_folder):
            raise FileNotFoundError(f"Folder {self.dataset_folder} does not exist")
        if not os.path.exists(self.database_folder):
            raise FileNotFoundError(f"Folder {self.database_folder} does not exist")
        if not os.path.exists(self.queries_folder):
            raise FileNotFoundError(f"Folder {self.queries_folder} does not exist")
        
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        #### Read paths and UTM coordinates for all images.
        self.database_paths = sorted(glob(os.path.join(self.database_folder, "**", "*.jpg"), recursive=True))
        self.queries_paths  = sorted(glob(os.path.join(self.queries_folder, "**", "*.jpg"),  recursive=True))
        print(f"the shape of database_paths is {len(self.database_paths)}") #added
        # The format must be path/to/file/@utm_easting@utm_northing@...@.jpg
        self.database_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in self.database_paths]).astype(np.float)
        self.queries_utms  = np.array([(path.split("@")[1], path.split("@")[2]) for path in self.queries_paths]).astype(np.float)
        print(f"The Lenght of the small/test/database is {self.database_utms.shape}")
        print(f"The Lenght of the small/test/queries is {self.queries_utms.shape}")
      
        
        # Find positives_per_query, which are within positive_dist_threshold (default 25 meters)
        knn = NearestNeighbors(n_jobs=-1)
        print(self.database_utms.shape)  #added
        knn.fit(self.database_utms)
        self.positives_per_query = knn.radius_neighbors(self.queries_utms, 
                                                             radius=positive_dist_threshold,
                                                             return_distance=False)                                           
        self.images_paths  = [p for p in self.database_paths]
        self.images_paths += [p for p in self.queries_paths]
        print(f"The lenght of images_path is {len(self.images_paths)}")
        
        self.database_num = len(self.database_paths)
        self.queries_num  = len(self.queries_paths)
        print(f"database_num : {self.database_num}\n queries_num : {self.queries_num}")
    
    def __getitem__(self, index):
        image_path = self.images_paths[index]
        #we still need to change the image size even here in the test set
        pil_img = open_image(image_path)
        new_w=pil_img.size[0]*1
        new_h=pil_img.size[0]*1
        normalized_img = self.base_transform(pil_img.resize((new_w,new_h)))
        return normalized_img, index
    
    def __len__(self):
        return len(self.images_paths)
    
    def __repr__(self):
        return  (f"< {self.dataset_name} - #q: {self.queries_num}; #db: {self.database_num} >")
    
    def get_positives(self):
        return self.positives_per_query

