# Video-Style-Transferring-CycleGAN
Convert your video to anime style, utilizing CycleGAN. This is the record of my project in 2023.

## Dataset
- [COCO2017](https://cocodataset.org/#download)
- [Dataset from AnimeGAN](https://github.com/TachibanaYoshino/AnimeGANv2/releases/download/1.0/Shinkai.tar.gz), [repo of AnimeGAN](https://github.com/TachibanaYoshino/AnimeGANv3)

## Data location

Download the content and style dataset to the corresponding folders in `data/train/`. In this project, COCO is the content dataset and Shinkai dataset is the style dataset.

```
----- Video-Style-Transferring-CycleGAN
    ----- data
        -----train
            ----- unlabeled2017
            ----- Shinkai
        -----test
```
And you can put the images you want to apply as test data in `data/test/` folder.

## Usage
Change directory to project folder

`cd Video-Style-Transferring-CycleGAN`

You can all train program

`python train.py`

or test program

`python test.py`

Apply the video style transferring program

`python main.py --video_path --output_path`

## Catalog

- [ ] Add trained model.
- [ ] Add example videos.
- [ ] Google Colab inference example.
