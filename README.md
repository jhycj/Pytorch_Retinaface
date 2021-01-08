# Detect 5 landmarks with RetinaFace & Face Align with FER 2013 Dataset.

## Reference  
- [RetinaFace in PyTorch](https://github.com/biubug6/Pytorch_Retinaface)
- [FaceBoxes](https://github.com/zisianw/FaceBoxes.PyTorch)
- [Retinaface (mxnet)](https://github.com/deepinsight/insightface/tree/master/RetinaFace)

## Data

FER 2013 data

1. Download the [FER2013 dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data), train.csv and icml_face_data.csv. train.csv is for training and validation. icml_face_data.csv is for test. 

2. Organize FER2013 dataset as follows. 
'''Shell
  ./data/FDDB/csv/
    train/
      train.csv
    test/  
      icml_face_data.csv 
'''

3. Convert csv type file into jpg type images.   

  1. run 
    ```Shell
    python3 ./data/FDDB/csv/train/csv2image.py
    python3 ./data/FDDB/csv/test/icml_csv2image.py
    ```
  2. Then you can find .jpg image files in 
      ./data/FDDB/images/train/{emotion_class_number} 
      or in 
      ./data/FDDB/images/test/{emotion_class_number}  

4. You should make a text file ./data/FDDB/img_list.txt 
  - To make img_list.txt with folder {class_num} in train image folder, please run 
  '''Shell
  python3 ./data/get_image_list.py --phase='train' --class_num={class_num}
  '''

  - To make img_list.txt with {class_num} in test image folder, please run 
  '''Shell
  python3 ./data/get_image_list.py --phase='test' --class_num={class_num}
  '''

5. To detect 5 landmarks, you should download the weights from [google cloud](https://drive.google.com/drive/folders/1oZRSG0ZegbVkVwUd8wUIQx8W7yfZ_ki1). This model weight should be put as follows: 
'''Shell
  ./weights/
    mobilenet0.25_Final.pth
    mobilenetV1X0.25_pretrain.tar
    Resnet50_Final.pth
'''

## TEST  


- If you want to use mobile0.25 as backbone network, please run 
```Shell
python test.py --trained_model weight_file --network mobile0.25
```

- Or if you want to use resnet50 as backbone network, please run 
```Shell
python test.py --trained_model weight_file --network resnet50
```


## References
- [Retinaface in Pytorch]()
- [FaceBoxes](https://github.com/zisianw/FaceBoxes.PyTorch)
- [Retinaface (mxnet)](https://github.com/deepinsight/insightface/tree/master/RetinaFace)
```
@inproceedings{deng2019retinaface,
title={RetinaFace: Single-stage Dense Face Localisation in the Wild},
author={Deng, Jiankang and Guo, Jia and Yuxiang, Zhou and Jinke Yu and Irene Kotsia and Zafeiriou, Stefanos},
booktitle={arxiv},
year={2019}
```
