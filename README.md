# DopUS-Net: Quality-Aware Robotic Ultrasound Imaging Based on Doppler Signal

[Data](https://syncandshare.lrz.de/getlink/fi3EGowa2yGEkUr3yZFUwn/)

Pytorch implementation of segmentation based on Ultrasound B-Mode and Doppler images.<br><br>

[DopUS-Net: Quality-Aware Robotic Ultrasound Imaging Based on Doppler Signal](https://ieeexplore.ieee.org/abstract/document/10152472)  
 [Zhongliang Jiang](https://www.cs.cit.tum.de/camp/members/zhongliang-jiang/)\*<sup>1</sup>,
 [Felix Dülmer](https://www.cs.cit.tum.de/camp/members/felix-duelmer/)\*<sup>1</sup>,
 [Nassir Navab](https://www.professoren.tum.de/en/navab-nassir)<sup>1</sup> <br>
 <sup>1</sup>Technical University of Munich (TUM)
\*denotes equal contribution  
Accepted at IEEE Transactions on Automation Science and Engineering.


## Installation

pip install -r requirements.txt

## Training

To run the latest version of the DopUs-Network please run:
- train_segmentation.py -c configs/config_dopus_v4.json 

## Visualization

To visualize the training process please launch a local visdom server:
 - python -m visdom.server