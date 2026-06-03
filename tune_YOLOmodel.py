from ultralytics import YOLO
import wandb
from ultralytics import settings
settings.update({"wandb": True})

import argparse

def main():
    # parse command-line args if needed
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    # Define a YOLO model
    model = YOLO("yolo11x.pt")
    
    # Dynamically assign GPUs based on --gpus argument
    device = list(range(args.gpus))  # e.g., [0, 1, 2, 3] if args.gpus = 4

    # Define search space
    search_space = {
        "lr0": (1e-5, 1e-1),
        "lrf": (0.01, 1.0),
        "momentum": (0.6, 0.98),
        "weight_decay": (0.0, 0.001),
        "warmup_epochs": (0, 5),
        "warmup_momentum": (0.0,0.95),
        "box": (0.02, 0.2),
        "cls": (0.2, 4.0),
        "hsv_h": (0.0, 0.1),
        "hsv_s": (0.0, 0.9),
        "hsv_v": (0.0, 0.9),
        "translate":(0.0, 0.9),
        "scale": (0.0, 0.9),
        "shear": (0.0, 10.0),
        "perspective": (0.0, 0.001),
        "mosaic": (0.0,1.0),
        "mixup": (0.0, 1.0),
        "copy_paste": (0.0, 1.0),
    }

    # Manually initialize certain hyperparameters that would not be tuned otherwise because they are set to 0 by default.
    model.overrides.update({
        'shear': 1,
        'perspective': 0.001,
        'mixup': 0.01,
        'copy_paste' : 0.01,
    
    })
    
    # tune the model
    results = model.tune(project='./tune/PROJECTDIRECTORY',name="PROJECTNAME", use_ray = False,
        data="data.yaml", 
        batch = 16,
        imgsz = 1344,    
        device = device,
        workers = args.workers,
    
        degrees = 180, #these parameters are fixed in the model tune
        flipud = 0.5,
        fliplr = 0.5,
        bgr = 0,
        auto_augment = None,
                         
        epochs = 100,
        iterations=200, 
        space=search_space,
        optimizer="AdamW", 
        resume=True,  #start over or resume previous checkpoint
        plots=False,  # don't make plots etc to save time until last timepoint
        save=False,
        val=False,
        verbose=False)

if __name__ == "__main__":
    main()
