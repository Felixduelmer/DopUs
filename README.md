# DopUS-Net: Quality-Aware Robotic Ultrasound Imaging Based on Doppler Signal

Pytorch implementation of segmentation based on Ultrasound B-Mode and Doppler images. Accepted at IEEE Transactions on Automation Science and Engineering. <br><br>

[DopUS-Net: Quality-Aware Robotic Ultrasound Imaging Based on Doppler Signal](https://ieeexplore.ieee.org/abstract/document/10152472)  
 [Zhongliang Jiang](https://www.cs.cit.tum.de/camp/members/zhongliang-jiang/)\*,
 [Felix Dülmer](https://www.cs.cit.tum.de/camp/members/felix-duelmer/)\*,
 [Nassir Navab](https://www.professoren.tum.de/en/navab-nassir) <br>
 Technical University of Munich (TUM)
 
 \*denotes equal contribution  


## Installation

pip install -r requirements.txt

## Training

To run the latest version of the DopUs-Network please run:
- train_segmentation.py -c configs/config_dopus_v4.json 

## Visualization

To visualize the training process please launch a local visdom server:
 - python -m visdom.server