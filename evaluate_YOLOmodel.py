from ultralytics import YOLO
import os
import numpy as np
import pandas as pd
from openpyxl import load_workbook


# adapt these 3 lines to run different evaluations
model = YOLO("PROJECTDIRECTORY/PROJECTNAME/weights/best.pt")
project = ('PROJECTDIRECTORY/eval_test/PROJECTNAME')
data = ('data.yaml') 
imagesize = 2432 #sliced image = [1632,2432], depends on how the model was trained


# Evaluate on test set
metrics_test = model.val(data= data, split='test',
   imgsz=imagesize, 
   conf=0.001, # default = 0.001
   batch=1, plots=True,save_json=True,
   project = project,
   name='6_classes')

save_dir = metrics_test.save_dir

output_csv1 = os.path.join(save_dir, "overall_metrics.csv")
output_csv2 = os.path.join(save_dir, "class_metrics.csv")
output_csv3 = os.path.join(save_dir, "all_ap.csv")

# 1. Overall Mean Metrics
mp, mr, map50, map_overall = metrics_test.box.mean_results()
overall_metrics_data = {
    'Metric': ['Mean Precision (mp)', 'Mean Recall (mr)', 'mAP@0.5 (map50)', 'mAP@0.5:0.95 (map)', 'Fitness'],
    'Value': [mp, mr, map50, map_overall, metrics_test.box.fitness()]
}
df_overall = pd.DataFrame(overall_metrics_data)

# 2. Per-Class Metrics
per_class_data = []
for i in range(metrics_test.box.nc):
    p_i, r_i, ap50_i, ap_i = metrics_test.box.class_result(i)
    per_class_data.append({
        'Class': model.names[i],
        'Precision': p_i,
        'Recall': r_i,
        'F1 Score': metrics_test.box.f1[i],
        'AP@0.5': ap50_i,
        'AP@0.5:0.95': ap_i # This is the mean AP over all 10 IoU thresholds for the class
    })
df_per_class = pd.DataFrame(per_class_data).set_index('Class')


# 3. All AP Scores (per class, per IoU threshold)
iou_thresholds = [f'IoU_{0.5 + 0.05 * i:.2f}' for i in range(10)]
df_all_ap = pd.DataFrame(
    metrics_test.box.all_ap,
    index=model.names,
    columns=iou_thresholds
)
df_all_ap.index.name = 'Class'

df_overall.to_csv(output_csv1, index=False)
df_per_class.to_csv(output_csv2,index=True)
df_all_ap.to_csv(output_csv3,index=True)

# Class-agnostic evaluation
metrics_class_agnostic = model.val(data = data, split='test',
   imgsz=imagesize,
   conf=0.001, # default = 0.001
   single_cls = True,
   batch=1, plots=True,save_json=True,
   project = project,
   name='class_agnostic')

# 4. Overall metrics class agnostic
mp, mr, map50, map_overall = metrics_class_agnostic.box.mean_results()
overall_metrics_agnostic = {
    'Metric': ['Mean Precision (mp)', 'Mean Recall (mr)', 'mAP@0.5 (map50)', 'mAP@0.5:0.95 (map)', 'Fitness'],
    'Value': [mp, mr, map50, map_overall, metrics_class_agnostic.box.fitness()]
}
df_agnostic = pd.DataFrame(overall_metrics_agnostic)

save_dir = metrics_class_agnostic.save_dir
output_csv4 = os.path.join(save_dir, "agnostic_metrics.csv")
df_agnostic.to_csv(output_csv4, index = False)
