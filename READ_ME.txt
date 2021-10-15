################################################################################

Before using this tool, please install requirements mentioned below.

################################################################################

0. Make sure that
       
       CUDA, CUDNN, pytorch, torchvision
       
  are correctly installed and then install the following requirements

################################################################################

1. If you want to run this tool in a spesific conda enviroment, first activate it

     $~: source activate $YOUR_CONDA_ENV_NAME

################################################################################

2. Then run setup.sh

     $~: ./setup.sh

follow the step and then the icon of this software will be created in your desktop applications, click it and you can run this file

################################################################################

3. Download Weight File

     $~: wget -O packages/yolov5/models/pt/yolov5s.pt https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt
     $~: wget -O packages/yolov5/models/pt/yolov5m.pt https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5m.pt
     $~: wget -O packages/yolov5/models/pt/yolov5l.pt https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5l.pt
     $~: wget -O packages/yolov5/models/pt/yolov5x.pt https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5x.pt
     
if you get a extremly slow speed downloading these files, you can try to replace "github.com" with "github.com.cnpmjs.org"

################################################################################

4. fix error

there maybe some errors occured when you start or run this program, if you want to fix them, you can open terminal and change to this dir, then run

     $~: python3 YOLOv5TrainGuide

and see what error appears in terminal.

################################################################################
