# 利用yolov5进行台标检测识别，将各个台标转化为目标识别，该方法只适用于有限个目标签名
1. 在有一些台标图像和一些背景图之后，可在项目[build_sign_dataset_yolo](https://github.com/JiangHejun/build_sign_dataset_yolo)中进行手写体数据集生成，手写体如：
<p align="center">
    <img src="https://github.com/JiangHejun/build_sign_dataset_yolo/blob/main/dataset/signature/hccl/1.png?raw=true" width="280"\>
</p>

2. 如需在生成数据集的时候效果更好，可在项目[gan_for_sign](https://github.com/JiangHejun/gan_for_sign)中训练GAN网络，训练效果如：
<p align="center">
    <img src="https://github.com/JiangHejun/gan_for_sign/blob/main/show/train.gif?raw=true" width="320"\>
</p>

3. 在生成完成数据集之后，运行如：
```
python train.py --data xx/coco_sign.yaml --device 0
```

4. 训练完成之后，运行如：
```
python detect.py --source xx/test_sign --weights runs/train/exp/weight/best.pt --conf 0.25
```

5. 训练效果如下：
<p align="center">
    <img src="./show/sign_11.jpg" width="320"\>
</p>