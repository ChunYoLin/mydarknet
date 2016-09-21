# Elastic_for_ADAS
Elastic for ADAS is a computing framework to enable elastic computing for Advanced Driver Assistant Systems (ADAS). The project is hosted by the Dr. Chi-Sheng Shih's research group at National Taiwan University and sponsored by MediaTek Inc. and MOST, Taiwan.

The software is implemented and tested on nVidia GPU 970 card and Intel i7 CPU. It should work on similar hardware platforms but is not fully tested.

YOLO engine is used in the framework and is NOT owned by our team.
#Prerequest
NVIDIA VGA with CUDA toolkit
Linux based OS
#Build
```
make all
```
#Run
```
./darknet yolo demo_vid cfg/YourCfg.cfg YourWeightFile.weights TestVideo
```
for example:
```
 ./darknet yolo demo_vid cfg/yolo.cfg ../demo_darknet/yolo_weight/yolo.weights ../../video_1920_1080/ForDemo/demo0826_origin.mov framework
```
#Cfg And Weights File
You have to first go to YOLO website to see how to generate these two file
[http://pjreddie.com/darknet/yolo/]
If you want to train your own customize model, you can refer to [http://guanghan.info/blog/en/my-works/train-yolo/]
