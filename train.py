import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
# 训练参数官方详解链接：https://docs.ultralytics.com/modes/train/#resuming-interrupted-trainings:~:text=a%20training%20run.-,Train%20Settings,-The%20training%20settings

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/yolov8-dyhead.yaml')
    #model = YOLO('ultralytics/cfg/models/v10/yolov10s.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    #model.train(data='../VisDrone2019/VisDrone.yaml',
    model.train(data='E:/boshi/VisDrone2019/VisDrone.yaml',
                cache=False,
                imgsz=640,
                epochs=230,
                batch=1,
                close_mosaic=0,
                workers=4,
                #device='0',
                #device='0',
                optimizer='SGD', # using SGD
                # patience=0, # close earlystop
                # resume=True, # 断点续训,YOLO初始化时选择last.pt
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='exp',
                )

