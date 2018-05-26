# super-resolution-detection
End-to-End Super Resolution Object Detection Networks

## Models
https://drive.google.com/drive/u/1/folders/1qXuCJbsDp7Hno4JUwanviMIWDJG1-eck


### Progress

##### Data Set VOC0712

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



##### DBPN: Retrained (100 Epochs / VOC12) DBPN-4x model (start with DIV2K pretrained). Used for 5th column, and as initial weight for 6th

- SSD300 mAP

| 300x300 HR   | 300x300 LR (75x75 Bicubic) | 300x300 LR (75x75 Bilinear) | 300x300 LR (75x75 Nearest) | 75x75 LR => DBPNx4 |  75x75 LR => SROD (end-2-end ) |
|   :---:      |          :---:             |          :---:              |            :---:           |       :---:        |            :---:               |
|   77.34%     |          48.91%            |          46.92%             |             7.17%          |       56.08%       |             TBD                |





