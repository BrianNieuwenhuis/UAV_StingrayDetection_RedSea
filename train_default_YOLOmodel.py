from ultralytics import YOLO
import os

#enable wanb logging
from ultralytics import settings
settings.update({"wandb": True})

# Load a model
model = YOLO("PROJECTDIRECTORY/yolo11x.pt")

# Training with custom augmentation parameters
model.train(data="data.yaml", project='./train', name='default_b16',
            epochs=100,
            batch = 16,
            imgsz = 1344,
            device = [0,1,2,3],
            workers = 8,
            patience = 100,
            save = True,
            save_period = 10,
            optimizer = 'AdamW',
            verbose = False,
             
            degrees = 180,
            flipud = 0.5,
            fliplr = 0.5,
            )
            
