# CMPUT414

This is the 3D style transfer of Team 04. 

To run our code, please follow the instruciton

1. Clone the repo from Github
```
git clone https://github.com/YeeSkywalker/CMPUT414.git
```

2. Download data set from [Stanford.edu](https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip) and put it under the root file of the repo

3. To train our model, use command below in the terminal, replace ```$Path``` with your own root path of the cloned repo. Set the object to the point cloud data set as desired (Chair as default)

```
python train.py --dataset=/{$Path}/CMPUT414/shapenetcore_partanno_segmentation_benchmark_v0 --object=Chair
```

4. To do the style transfer, use command below in the terminal, replace ```$Path``` with own root path of the cloned repo. 
Set the object to the point cloud data set as desired (Chair as default). Set indexs of the content point cloud and the style point cloud as desired (By default content is set to 99 and style is set to 199)

```
ython decoder.py --model=/{$Path}/segmentation/segmentation.pth --dataset=/{$Path}/shapenetcore_partanno_segmentation_benchmark_v0/ -—object=Chair —-content=99 —-style=199
```