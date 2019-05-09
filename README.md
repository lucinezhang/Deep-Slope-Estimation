# Deep Slope Estimation with Formal Verification

This project aims to estimate the drivability of the road for autonomous vehicles given a point cloud of the surroundings and verify the robustness of the model with formal verification techniques. We use the estimated point normals to 
represent the slope or the condition of the road. Since formal verification tools requires the model size to be as small as possible, we have also tried weight pruning and knowledge distilaltion to compress the model.

The project homepage is [here](https://mscvprojects.ri.cmu.edu/2019teamf/).


## Data preparation
We select the first four classes and some samples for training. code can be downloaded and processed by running:
```
sh raw_data_downloader.sh
python prep_kitti_data.py
```
This will give you four h5py files wich contain the training and testing data.

## Train a model

You can train your normla estimation model by running:
```
python train.py 
```
We've also provided a trained model in model_0.pth with rms angle error 32.56.

## Weight prune and re-train
Test the weight pruning results on one test sample. The following code will give you the number of pruned parameters and the rms angle error after weight pruning.
The threshold controls the number of prameters being pruned.
```
python test.py --thres 1.5 --model model.pth
```
We have provided test.h5 which contains one full scene testing sample. you can also test other samples by modifying prep_test_data.py
Re-train your model: 
```
python train.py --model "PATH TO THE TRAINED MODEL" --thres 1.5 --prune
```
Genrally, training several epochs should be enough to give you acceptable reaults. The trained model will be saved in kitti_output folder. You can test and compare the results after retraining. model_90.pth is the retrained model after pruning 90% parameters with rms angle error 34.18.

## Visualization
By runnning the testing code, you will get two .npz files with normal prediction results of the original model and the pruned model.
Qulititative results can be demonstrated by running:
```
python vis.py --file 'PATH TO .npz FILE'
```
We've also provided res_0.npz and res_90.npz which is the prediction results before and after pruning 90% parameters.

ps: For the mentioned rms angle error, we refer to the error on the test.h5 full scene sample instead of the whole test set, beacause the ground truth (geometric method labeled) might not be correct.

## Reference
[1] The codes are developed based on [this repo](https://github.com/fxia22/pointnet.pytorch).
[2] R. Q. Charles, H. Su, M. Kaichun, and L. J.Guibas. Pointnet: Deep learning on point setsfor 3d classification and segmentation.2017IEEE Conference on Computer Vision andPattern Recognition (CVPR), Jul 2017.

[3] C. R. Qi, L. Yi, H. Su, and L. J. Guibas. Point-net++: Deep hierarchical feature learning onpoint sets in a metric space, 2017.

[4] A. Romero, N. Ballas, S. E. Kahou, A. Chas-sang, C. Gatta, and Y. Bengio. Fitnets: Hintsfor thin deep nets, 2014