# ImOV3D-Distill Modification

## 2025/04/09whc更新注意
  注意代码中新增加了tqdm库，可能需要自己安装一下
## 2025/04/13ry参考文献

PointDistiller: Structured Knowledge Distillation Towards Efficient and Compact 3D Detection  
https://arxiv.org/pdf/2205.11098


DOT: A Distillation-Oriented Trainer  
https://openaccess.thecvf.com/content/ICCV2023/papers/Zhao_DOT_A_Distillation-Oriented_Trainer_ICCV_2023_paper.pdf


Decoupled Knowledge Distillation  
https://arxiv.org/pdf/2203.08679


LabelDistill: Label-guided Cross-modal Knowledge Distillation for Camera-based 3D Object Detection  
https://arxiv.org/pdf/2407.10164


Fusion-then-Distillation: Toward Cross-modal Positive Distillation for Domain Adaptive 3D Semantic Segmentation  
https://arxiv.org/pdf/2410.19446


A Comprehensive Survey on Knowledge Distillation(2025最新综述)  
https://arxiv.org/pdf/2503.12067

## 2025/04/23修改注意
我们ImvoteNet.py的forward前向传播里面的特征提取代码重复了... 怪不得会有问题，需要删掉重复的地方

## 2025/04/27
由于操作不当，我把上传时间给重置了。。。adapter部分一共更新了imvotenet_new,distill_rao（主程序）,adapter_net（新）三个文件
  
