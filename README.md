# super-resolution-detection
End-to-End Super Resolution Object Detection Networks

## Models
https://drive.google.com/drive/u/1/folders/1qXuCJbsDp7Hno4JUwanviMIWDJG1-eck


### Progress

##### Data Set VOC0712

- Training: VOC12 (17128 images)
- mAP Test: VOC07 ( 4952 images)

##### DBPN: Original DBPN-4x model (DIV2K pretrained). Used for 3rd column, and as initial weight for 4th

- SSD300 mAP

| 300x300 HR   |  300x300 LR (75x75 Bicubic Upscale)  |  75x75 LR => DBPNx4 |  75x75 LR => SROD (4000 end-2-end iters)  |
|   :---:      |                :---:                 |        :---:        |              :---:                        |
|   76.99%     |                48.65%                |        39.84%       |              41.42%                       |



##### DBPN: Retrained (50 Epochs / VOC12) DBPN-4x model (start with DIV2K pretrained). Used for 3rd column, and as initial weight for 4th

- SSD300 mAP

| 300x300 HR   |  300x300 LR (75x75 Bicubic Upscale)  |  75x75 LR => DBPNx4 |  75x75 LR => SROD (pending end-2-end )  |
|   :---:      |                :---:                 |        :---:        |              :---:                      |
|   76.99%     |                48.65%                |        48.52%       |                                         |



#### New input Images scaled by imageScale300.ipynb

##### DBPN: Retrained (50 Epochs / VOC12) DBPN-4x model (start with DIV2K pretrained). Used for 5th column, and as initial weight for 6th

- SSD300 mAP

| 300x300 HR   | 300x300 LR (75x75 Bicubic) | 300x300 LR (75x75 Bilinear) | 300x300 LR (75x75 Nearest) | 75x75 LR => DBPNx4 |  75x75 LR => SROD (end-2-end ) |
|   :---:      |          :---:             |          :---:              |            :---:           |       :---:        |            :---:               |
|   77.34%     |          48.91%            |          46.92%             |             7.17%          |       49.59%       |             TBD                |



##### DBPN: Intensively Retrained (100 Epochs / VOC12) DBPN-4x base model (start with DIV2K pretrained). Used for 5th column, and as SROD's initial S-Net weight for 6th

- SSD300 mAP

| 300x300 HR   | 300x300 LR (75x75 Bicubic) | 300x300 LR (75x75 Bilinear) |  75x75 LR (in 300x300 Bgrd) | 300x300 LR (75x75 Nearest) | 75x75 LR => DBPNx4 |  75x75 LR => SROD (e2e Retrain 16 Epochs) |
|   :---:      |          :---:             |          :---:              |            :---:            |            :---:           |       :---:        |                   :---:                   |
|   77.34%     |          48.91%            |          46.92%             |            40.35%           |             7.17%          |       56.08%       |                   62.89%                  |




## Authorship

This project is equally contributed by [Hai Xiao](https://github.com/haishaw) and [Kegang Xu](https://github.com/tosmaster), with extra code components from following authors:


## Citation

    @article{SROD2018,
        Author = {Hai Xiao and Kegang Xu},
        Title = {A Pytorch Implementation of Super Resolution Object Detection},
        Journal = {https://github.com/cs231x/super-resolution-detection},
        Year = {2018}
    }
 
    @inproceedings{DBPN2018,
        title={Deep Back-Projection Networks for Super-Resolution},
        author={Haris, Muhammad and Shakhnarovich, Greg and Ukita, Norimichi},
        booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        year={2018}
    }

    @article{SSD300,
        Author = {Max deGroot and Ellis Brown},
        Title = {SSD: Single Shot MultiBox Object Detector, in PyTorch},
        Journal = {https://github.com/amdegroot/ssd.pytorch},
        Year = {2017}
    }
