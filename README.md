# AFD
## Association with Formation Distance (AFD)
AFD is a MOT method using relative spatial relationship of objects. 

## Setup Instructions

* Clone this repo, and we'll call the directory that you cloned as {AFD Root}
* Install dependencies.
```
conda create -n AFD python=3.7
conda activate AFD

# Install pytorch with the proper cuda version to suit your machine
# We are using torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 with cuda==11.6
# Sorry about folder name is DeepEIoU, it's named like that because this code is based on Deep-EIoU. i will change it later.

cd DeepEIoU/reid
pip install -r requirements.txt
pip install cython_bbox
python setup.py develop
cd ..
```

## Download SporsMOT dataset
Please Sign up in [codalab](https://codalab.lisn.upsaclay.fr/competitions/12424#participate-get-data)

Download links are available in Participate/Get Data named "sportsmot_publish"

And put that folder to your project like following directory structure.
```
{AFD Root}
   |——————Deep-EIoU
   |——————SportsMOT(folder for evaluation)
   |——————sportsmot_publish
```

## Download checkpoint file
Download the detector and ReID model from [drive](https://drive.google.com/drive/folders/1wItcb0yeGaxOS08_G9yRWBTnpVf0vZ2w) and put them in the corresponding folder.

```
{AFD Root}
   └——————Deep-EIoU
            └——————checkpoints
                └——————best_ckpt.pth.tar (YOLOX Detector)
                └——————sports_model.pth.tar-60 (OSNet ReID Model)
```

## Demo on your original video
Run the following commands.
```
python tools/demo.py --path <your video path>
```

## Run AFD on SportsMOT video
Run the following commands.
```
python tools/AFD_T.py --img_folder <img_folder path of SportsMOT> --output_dir <Output folder path> 
```

img_folder is a folder named [VIDE_NAME] that you can see in dataset directry like this.
```
sportsmot_publish
└—dataset
     └—train
          |——VIDEO_NAME1
          |    |——gt
          |    |——img1
          |    |    └—000001.jpg
          |    |    └—000002.jpg
          |    |                :
          |    |
          |    |——seqinfo.ini
          |
          |——VIDEO_NAME2 
          |——VIDEO_NAME3
                   :     
```

If you want to use GT BBOX, run the following command.
```
python tools/AFD_T_wgt.py --img_folder <img_folder path of SportsMOT> --output_dir <Output folder path> 
```

If you want video results, run the following command (this code use GT BBOX)
```
python tools/AFD_TandV_wgt.py --img_folder <img_folder path of SportsMOT> --output_dir <Output folder path> 
```

## Run AFD on validation or traning data of SportsMOT
Run the following commands to get tracking results of validation or training data.

```
./tools/run_all_train_AFD.sh --output_dir <OUTPUT FOLDER PATH>
./tools/run_all_val_AFD.sh --output_dir <OUTPUT FOLDER PATH>
```
Don't forget to change the path for dataset and checkpoints file in those code.

## Evaluate your tracking results
Put your tracking results in the corresponding folder.

Folder which includes tracking datas must be following directory structure.
```
sportsmot-val
   └———folder named [tracker name] # like DeepEIoU
            └———data
                  └——————v_-9kabh1K8UA_c008.txt
                  └——————v_-9kabh1K8UA_c009.txt
                  └——————...
```

After putting your tracking resutlts to directory as above, run the following commands.

Then you will get evaluation results in output directory(SportsMOT/codes/evaluation/TrackEval/output)
```
cd AFD/SportsMOT/codes/evaluation/TrackEval
python evaluate.sh
```

## Acknowledgements
The code is based on [Deep-EIoU](https://github.com/hsiangwei0903/Deep-EIoU)

## Contact
Shozaburo Hirano (hiranozaburo@gmail.com)