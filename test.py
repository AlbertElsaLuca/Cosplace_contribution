
import faiss
import torch
import logging
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
import torchvision.transforms as T
import multiScale 
import MyAugmentation as A


# Compute R@1, R@5, R@10, R@20
RECALL_VALUES = [1, 5, 10, 20]


def test(args, eval_ds, model):
    """Compute descriptors of the given dataset and compute the recalls."""
    
    model = model.eval()
    with torch.no_grad():
        logging.debug("Extracting database descriptors for evaluation/testing")
        database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
        database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                        batch_size=args.infer_batch_size, pin_memory=(args.device=="cuda"))
        #args.infer_batch_size = 16, so since database contain 27191 images we will get 1700 iterations.
        all_descriptors = np.empty((len(eval_ds), args.fc_output_dim), dtype="float32")
        #all descriptors: is of shape (28191,512) and contain all values ==0
        #28191 = 27191 (database) + 1000 (queries) 
        #each image's descriptor is args.fc_output_dim = 512 long
        #Let's define two transformer
        #to_tensor(to_PIL(images).resize([256,256])).shape
        for images, indices in tqdm(database_dataloader, ncols=100):
            #form each tensor batch of image in the test/database extract descriptors
            #may be i should pass here the the image.resize(new_h,new_w) and set
            #descriptors2 = model(images.ToPILImage().resize([new_h,new_w]).To_tensor().to(args.device))...
            img = multiScale.image_resize(images)
            images=A.DefaultTransformation(images)
            desc1= model(images.to(args.device))
            desc2=model(img.to(args.device))
            desc1 = desc1.cpu().numpy()
            desc2 = desc2.cpu().numpy()
            descriptors = (desc1 + desc2)/2 #average both descriptors from to diff resolution images
            #descriptors=np.array([np.array([i,j]).max() for i,j in zip(desc1.flatten(),desc2.flatten())]).reshape((-1,512))
            #descriptors = desc2.cpu().numpy()
            #images=A.DefaultTransformation(images)
            #descriptors = model(images.to(args.device))
            #descriptors = descriptors.cpu().numpy()
            all_descriptors[indices.numpy(), :] = descriptors
        logging.debug("Extracting queries descriptors for evaluation/testing using batch size 1")
        queries_infer_batch_size = 1
        queries_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num, eval_ds.database_num+eval_ds.queries_num)))
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                        batch_size=queries_infer_batch_size, pin_memory=(args.device=="cuda"))
        for images, indices in tqdm(queries_dataloader, ncols=100):
            #for each image in test/queries extract descriptors
            # img=[to_tensor(to_PIL(i).resize([256,256])) for i in images]
            # img=[np.array(i) for i in img]
            # img=np.array(img)
            # img=torch.from_numpy(img)
            img = multiScale.image_resize(images)
            desc1 = model(images.to(args.device))
            desc2 = model(img.to(args.device))
            desc1 = desc1.cpu().numpy()
            desc2 = desc2.cpu().numpy()
            #images=A.DefaultTransformation(images)
            #descriptors = model(images.to(args.device))
            #descriptors = descriptors.cpu().numpy()
            descriptors = (desc1 + desc2)/2
            #descriptors=np.array([np.array([i,j]).max() for i,j in zip(desc1.flatten(),desc2.flatten())]).reshape((-1,512))
            all_descriptors[indices.numpy(), :] = descriptors
    
    queries_descriptors = all_descriptors[eval_ds.database_num:]
    database_descriptors = all_descriptors[:eval_ds.database_num]
    # print(f"queries descriptor type {type(queries_descriptors[0])} and shape is : {queries_descriptors.shape}")
    # print(f"database descriptor type {type(database_descriptors[0])} and shape is : {database_descriptors.shape}")
    # print(f"Example of queries descriptors: {queries_descriptors[:3]}")
    # print(f"Example of database descriptors: {database_descriptors[:3]}")
    # Use a kNN to find predictions
    faiss_index = faiss.IndexFlatL2(args.fc_output_dim)
    faiss_index.add(database_descriptors)
    del database_descriptors, all_descriptors
    
    logging.debug("Calculating recalls")
    _, predictions = faiss_index.search(queries_descriptors, max(RECALL_VALUES))
    
    #### For each query, check if the predictions are correct
    positives_per_query = eval_ds.get_positives()
    recalls = np.zeros(len(RECALL_VALUES))
    for query_index, preds in enumerate(predictions):
        for i, n in enumerate(RECALL_VALUES):
            if np.any(np.in1d(preds[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                break
    # Divide by queries_num and multiply by 100, so the recalls are in percentages
    recalls = recalls / eval_ds.queries_num * 100
    recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(RECALL_VALUES, recalls)])
    return recalls, recalls_str

