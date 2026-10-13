# Models and citations

Every model registered in CFM-Model-Zoo, the papers to cite for it, the credits for the code adapted in
`cfm_model_zoo/legacy/`, and the BibTeX entries. Installing, using and naming the models: the
[main README](../../README.md). The `.py` files next to this one are the configs the package ships.

The **registered name** is `<Package>Split<Method><Backbone>`, derived from the model's config. The **legacy name**
is the short name the model had before. It resolves to the same model (`cfm_model_zoo.directory.LEGACY_NAMES`) and is kept for
backward compatibility (the paper and its code use it); new code should use the registered name.

Any split model not listed below should be cited by the user independently: most OpenMMLab configs give the
citation on their config page. For example, TOOD in MMDetection
([configs/tood](https://github.com/open-mmlab/mmdetection/blob/v3.3.0/configs/tood/README.md), `feng2021tood`) and
EMANet in MMSegmentation
([configs/emanet](https://github.com/open-mmlab/mmsegmentation/blob/v1.2.2/configs/emanet/README.md),
`li2019expectation`); each page's "Citation" section has the BibTeX.

<!-- attribution:begin (generated from the model registry; do not edit by hand) -->
## Attribution

This library builds on OpenMMLab. Every model in the registry belongs to its authors, and this library distributes no weights: they are downloaded from their publishers. They are OpenMMLab's, resolved through mim or, for the two DDQ-DETR models, downloaded from OpenMMLab's server; the RepPointsV2 weights are the Mask RepPoints V2 Swin-T 3x release of Microsoft's Swin-Transformer-Object-Detection; GLIP and GroundingDINO's language model, bert-base-uncased, comes from the Hugging Face Hub. If you use a model through this library, please cite the papers listed for it below, and the OpenMMLab toolboxes.

**OpenMMLab:** MMCV (`mmcv`), MMEngine (`mmengine2022`), MMDetection (`mmdetection`), MMSegmentation (`mmseg2020`) and MIM (`mim`).

**Models in the registry:**

| registered name | legacy name | package | cite |
|---|---|---|---|
| `MMDetSplitMaskRCNNSwinT` | `MaskRCNN` | mmdet | `he2017mask`, `liu2021swin`, `lin2017feature` |
| `MMDetSplitRetinaNetR50` | `RetinaNet` | mmdet | `lin2017focal`, `he2016deep`, `lin2017feature` |
| `MMDetSplitCascadeRCNNR50` | `CascadeMaskRCNN` | mmdet | `cai2018cascade`, `he2017mask`, `he2016deep`, `lin2017feature` |
| `MMDetSplitMask2FormerSwinT` | `Mask2FormerSWIN` | mmdet | `cheng2022masked`, `liu2021swin` |
| `MMDetSplitMask2FormerR50` | `Mask2FormerResNet` | mmdet | `cheng2022masked`, `he2016deep` |
| `MMDetSplitGroundingDINOSwinT` | `GroundingDINO` | mmdet | `liu2024grounding`, `liu2021swin`, `devlin2019bert` |
| `MMDetSplitGLIPSwinT` | `GLIP` | mmdet | `li2022grounded`, `zhang2020bridging`, `dai2021dynamic`, `liu2021swin`, `devlin2019bert`, `lin2017feature` |
| `MMDetSplitRepPointsV2MaskDetectorSwinT` | `RepPointsV2` | mmdet | `chen2020reppointsv2`, `yang2019reppoints`, `liu2021swin`, `lin2017feature` |
| `MMDetSplitATSSR50` | `ATSSResNet` | mmdet | `zhang2020bridging`, `dai2021dynamic`, `he2016deep`, `lin2017feature` |
| `MMDetSplitATSSSwinL` | `ATSSSWIN-L` | mmdet | `zhang2020bridging`, `dai2021dynamic`, `liu2021swin`, `lin2017feature` |
| `MMDetSplitFCOSR50` | `FCOS` | mmdet | `tian2019fcos`, `zhu2019deformable`, `wu2018group`, `he2016deep`, `lin2017feature` |
| `MMDetSplitFasterRCNNR50` | `FasterRCNN` | mmdet | `ren2015faster`, `he2016deep`, `lin2017feature` |
| `MMDetSplitDDQDETRR50` | `DDQDETRResNet` | mmdet | `zhang2023dense`, `zhu2021deformable`, `he2016deep` |
| `MMDetSplitDDQDETRSwinL` | `DDQDETRSWIN-L` | mmdet | `zhang2023dense`, `zhu2021deformable`, `liu2021swin` |
| `MMDetSplitDINOR50` | `DINOResNet` | mmdet | `zhang2023dino`, `zhu2021deformable`, `he2016deep` |
| `MMDetSplitDINOSwinL` | `DINOSWIN-L` | mmdet | `zhang2023dino`, `zhu2021deformable`, `liu2021swin` |
| `MMDetSplitDeformableDETRR50` | `DeformableDETR` | mmdet | `zhu2021deformable`, `he2016deep` |
| `MMSegSplitANNR50` | `SegANNResNet` | mmseg | `zhu2019asymmetric`, `he2016deep` |
| `MMSegSplitAPCNetR50` | `SegAPCNetResNet` | mmseg | `he2019adaptive`, `he2016deep` |
| `MMSegSplitCCNetR50` | `SegCCNetResNet` | mmseg | `huang2019ccnet`, `he2016deep` |
| `MMSegSplitDANetR50` | `SegDANetResNet` | mmseg | `fu2019dual`, `he2016deep` |
| `MMSegSplitDeepLabV3R50` | `SegDeepLabV3ResNet` | mmseg | `chen2017rethinking`, `he2016deep` |
| `MMSegSplitDeepLabV3PlusR50` | `SegDeepLabV3PlusResNet` | mmseg | `chen2018encoder`, `he2016deep` |
| `MMSegSplitDMNetR50` | `SegDMNetResNet` | mmseg | `he2019dynamic`, `he2016deep` |
| `MMSegSplitDNLNetR50` | `SegDNLNetResNet` | mmseg | `yin2020disentangled`, `he2016deep` |
| `MMSegSplitEncNetR50` | `SegENCNetResNet` | mmseg | `zhang2018context`, `he2016deep` |
| `MMSegSplitFastFCNASPPR50` | `SegFastFCNASPPResNet` | mmseg | `wu2019fastfcn`, `chen2017rethinking`, `he2016deep` |
| `MMSegSplitFastFCNEncR50` | `SegFastFCNENCResNet` | mmseg | `wu2019fastfcn`, `zhang2018context`, `he2016deep` |
| `MMSegSplitFastFCNPSPR50` | `SegFastFCNPSPResNet` | mmseg | `wu2019fastfcn`, `zhao2017pyramid`, `he2016deep` |
| `MMSegSplitFCNR50` | `SegFCNResNet` | mmseg | `long2015fully`, `he2016deep` |
| `MMSegSplitFPNR50` | `SegFPNResNet` | mmseg | `kirillov2019panoptic`, `lin2017feature`, `he2016deep` |
| `MMSegSplitGCNetR50` | `SegGCNetResNet` | mmseg | `cao2019gcnet`, `he2016deep` |
| `MMSegSplitISANetR50` | `SegISANetResNet` | mmseg | `huang2019interlaced`, `he2016deep` |
| `MMSegSplitNonLocalNetR50` | `SegNonLocalNetResNet` | mmseg | `wang2018non`, `he2016deep` |
| `MMSegSplitPSANetR50` | `SegPSANetResNet` | mmseg | `zhao2018psanet`, `he2016deep` |
| `MMSegSplitPSPNetR50` | `SegPSPNetResNet` | mmseg | `zhao2017pyramid`, `he2016deep` |
| `MMSegSplitMaskFormerR50` | `SegMaskFormerResNet` | mmseg | `cheng2021per`, `he2016deep` |
| `MMSegSplitMaskFormerSwinT` | `SegMaskFormerSWIN` | mmseg | `cheng2021per`, `liu2021swin` |
| `MMSegSplitMask2FormerR50` | `SegMask2FormerResNet` | mmseg | `cheng2022masked`, `he2016deep` |
| `MMSegSplitMask2FormerSwinT` | `SegMask2FormerSWIN` | mmseg | `cheng2022masked`, `liu2021swin` |
| `MMSegSplitKNetASPPR50` | `SegKNetDeepLabResNet` | mmseg | `zhang2021k`, `chen2017rethinking`, `he2016deep` |
| `MMSegSplitKNetFCNR50` | `SegKNetFCNResNet` | mmseg | `zhang2021k`, `long2015fully`, `he2016deep` |
| `MMSegSplitKNetPSPR50` | `SegKNetPSPResNet` | mmseg | `zhang2021k`, `zhao2017pyramid`, `he2016deep` |
| `MMSegSplitKNetUPerR50` | `SegKNetUPerResNet` | mmseg | `zhang2021k`, `xiao2018unified`, `he2016deep` |
| `MMSegSplitKNetUPerSwinT` | `SegKNetSWIN` | mmseg | `zhang2021k`, `xiao2018unified`, `liu2021swin` |
| `MMSegSplitUPerNetR50` | `SegUperNetResNet` | mmseg | `xiao2018unified`, `he2016deep` |
| `MMSegSplitUPerNetSwinT` | `SegUperNetSWIN` | mmseg | `xiao2018unified`, `liu2021swin` |

**Adapted code in `cfm_model_zoo/legacy/`:**

- the Mask RepPoints V2 Swin-T code this port is taken from: Microsoft's Swin-Transformer-Object-Detection (Apache-2.0), which also publishes the RepPointsV2 weights: `liu2021swin`
- the RepPoints V2 detector, head and losses (from the official implementation): `chen2020reppointsv2`
- the dynamic mask branch in `condconv.py`, modified from AdelaiDet (CondInst) by Swin-Transformer-Object-Detection: `tian2019adelaidet`
- corner pooling and the Gaussian focal loss (CornerNet): `law2018cornernet`
- the ATSS assigner in `assigners.py` (ATSSAssignerV2), adapted from ATSS by Swin-Transformer-Object-Detection: `zhang2020bridging`
- MMDetection v2 code the port builds on: `mmdetection`

**BibTeX:**

```bibtex
@misc{mmcv,
  title = {{MMCV}: {OpenMMLab} Computer Vision Foundation},
  author = {{MMCV Contributors}},
  howpublished = {\url{https://github.com/open-mmlab/mmcv}},
  year = {2018}
}

@misc{mmengine2022,
  title = {{MMEngine}: {OpenMMLab} Foundational Library for Training Deep Learning Models},
  author = {{MMEngine Contributors}},
  howpublished = {\url{https://github.com/open-mmlab/mmengine}},
  year = {2022}
}

@article{mmdetection,
  title = {{MMDetection}: Open {MMLab} Detection Toolbox and Benchmark},
  author = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
  journal = {arXiv preprint arXiv:1906.07155},
  year = {2019}
}

@misc{mmseg2020,
  title = {{MMSegmentation}: {OpenMMLab} Semantic Segmentation Toolbox and Benchmark},
  author = {{MMSegmentation Contributors}},
  howpublished = {\url{https://github.com/open-mmlab/mmsegmentation}},
  year = {2020}
}

@misc{mim,
  title = {{MIM}: {MIM} Installs {OpenMMLab} Packages},
  author = {{MIM Contributors}},
  howpublished = {\url{https://github.com/open-mmlab/mim}},
  year = {2021}
}

@inproceedings{he2017mask,
  title = {Mask {R-CNN}},
  author = {He, Kaiming and Gkioxari, Georgia and Doll{\'a}r, Piotr and Girshick, Ross},
  booktitle = {Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  year = {2017}
}

@inproceedings{liu2021swin,
  title = {Swin Transformer: Hierarchical Vision Transformer using Shifted Windows},
  author = {Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year = {2021}
}

@inproceedings{lin2017feature,
  title = {Feature Pyramid Networks for Object Detection},
  author = {Lin, Tsung-Yi and Doll{\'a}r, Piotr and Girshick, Ross and He, Kaiming and Hariharan, Bharath and Belongie, Serge},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2017}
}

@inproceedings{lin2017focal,
  title = {Focal Loss for Dense Object Detection},
  author = {Lin, Tsung-Yi and Goyal, Priya and Girshick, Ross and He, Kaiming and Doll{\'a}r, Piotr},
  booktitle = {Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  year = {2017}
}

@inproceedings{he2016deep,
  title = {Deep Residual Learning for Image Recognition},
  author = {He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2016}
}

@inproceedings{cai2018cascade,
  title = {Cascade {R-CNN}: Delving into High Quality Object Detection},
  author = {Cai, Zhaowei and Vasconcelos, Nuno},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2018}
}

@inproceedings{cheng2022masked,
  title = {Masked-attention Mask Transformer for Universal Image Segmentation},
  author = {Cheng, Bowen and Misra, Ishan and Schwing, Alexander G. and Kirillov, Alexander and Girdhar, Rohit},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2022}
}

@inproceedings{liu2024grounding,
  title = {Grounding {DINO}: Marrying {DINO} with Grounded Pre-Training for Open-Set Object Detection},
  author = {Liu, Shilong and Zeng, Zhaoyang and Ren, Tianhe and Li, Feng and Zhang, Hao and Yang, Jie and Jiang, Qing and Li, Chunyuan and Yang, Jianwei and Su, Hang and Zhu, Jun and Zhang, Lei},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year = {2024}
}

@inproceedings{devlin2019bert,
  title = {{BERT}: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author = {Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  booktitle = {Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)},
  year = {2019}
}

@inproceedings{li2022grounded,
  title = {Grounded Language-Image Pre-training},
  author = {Li, Liunian Harold and Zhang, Pengchuan and Zhang, Haotian and Yang, Jianwei and Li, Chunyuan and Zhong, Yiwu and Wang, Lijuan and Yuan, Lu and Zhang, Lei and Hwang, Jenq-Neng and Chang, Kai-Wei and Gao, Jianfeng},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2022}
}

@inproceedings{zhang2020bridging,
  title = {Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection},
  author = {Zhang, Shifeng and Chi, Cheng and Yao, Yongqiang and Lei, Zhen and Li, Stan Z.},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2020}
}

@inproceedings{dai2021dynamic,
  title = {Dynamic Head: Unifying Object Detection Heads with Attentions},
  author = {Dai, Xiyang and Chen, Yinpeng and Xiao, Bin and Chen, Dongdong and Liu, Mengchen and Yuan, Lu and Zhang, Lei},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2021}
}

@inproceedings{chen2020reppointsv2,
  title = {{RepPoints} v2: Verification Meets Regression for Object Detection},
  author = {Chen, Yihong and Zhang, Zheng and Cao, Yue and Wang, Liwei and Lin, Stephen and Hu, Han},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year = {2020}
}

@inproceedings{yang2019reppoints,
  title = {{RepPoints}: Point Set Representation for Object Detection},
  author = {Yang, Ze and Liu, Shaohui and Hu, Han and Wang, Liwei and Lin, Stephen},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year = {2019}
}

@inproceedings{tian2019fcos,
  title = {{FCOS}: Fully Convolutional One-Stage Object Detection},
  author = {Tian, Zhi and Shen, Chunhua and Chen, Hao and He, Tong},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year = {2019}
}

@inproceedings{zhu2019deformable,
  title = {Deformable {ConvNets} v2: More Deformable, Better Results},
  author = {Zhu, Xizhou and Hu, Han and Lin, Stephen and Dai, Jifeng},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2019}
}

@inproceedings{wu2018group,
  title = {Group Normalization},
  author = {Wu, Yuxin and He, Kaiming},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year = {2018}
}

@inproceedings{ren2015faster,
  title = {Faster {R-CNN}: Towards Real-Time Object Detection with Region Proposal Networks},
  author = {Ren, Shaoqing and He, Kaiming and Girshick, Ross and Sun, Jian},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year = {2015}
}

@inproceedings{zhang2023dense,
  title = {Dense Distinct Query for End-to-End Object Detection},
  author = {Zhang, Shilong and Wang, Xinjiang and Wang, Jiaqi and Pang, Jiangmiao and Lyu, Chengqi and Zhang, Wenwei and Luo, Ping and Chen, Kai},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2023}
}

@inproceedings{zhu2021deformable,
  title = {Deformable {DETR}: Deformable Transformers for End-to-End Object Detection},
  author = {Zhu, Xizhou and Su, Weijie and Lu, Lewei and Li, Bin and Wang, Xiaogang and Dai, Jifeng},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year = {2021}
}

@inproceedings{zhang2023dino,
  title = {{DINO}: {DETR} with Improved DeNoising Anchor Boxes for End-to-End Object Detection},
  author = {Zhang, Hao and Li, Feng and Liu, Shilong and Zhang, Lei and Su, Hang and Zhu, Jun and Ni, Lionel M. and Shum, Heung-Yeung},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year = {2023}
}

@inproceedings{zhu2019asymmetric,
  title = {Asymmetric Non-local Neural Networks for Semantic Segmentation},
  author = {Zhu, Zhen and Xu, Mengde and Bai, Song and Huang, Tengteng and Bai, Xiang},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year = {2019}
}

@inproceedings{he2019adaptive,
  title = {Adaptive Pyramid Context Network for Semantic Segmentation},
  author = {He, Junjun and Deng, Zhongying and Zhou, Lei and Wang, Yali and Qiao, Yu},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2019}
}

@inproceedings{huang2019ccnet,
  title = {{CCNet}: Criss-Cross Attention for Semantic Segmentation},
  author = {Huang, Zilong and Wang, Xinggang and Huang, Lichao and Huang, Chang and Wei, Yunchao and Liu, Wenyu},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year = {2019}
}

@inproceedings{fu2019dual,
  title = {Dual Attention Network for Scene Segmentation},
  author = {Fu, Jun and Liu, Jing and Tian, Haijie and Li, Yong and Bao, Yongjun and Fang, Zhiwei and Lu, Hanqing},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2019}
}

@article{chen2017rethinking,
  title = {Rethinking Atrous Convolution for Semantic Image Segmentation},
  author = {Chen, Liang-Chieh and Papandreou, George and Schroff, Florian and Adam, Hartwig},
  journal = {arXiv preprint arXiv:1706.05587},
  year = {2017}
}

@inproceedings{chen2018encoder,
  title = {Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation},
  author = {Chen, Liang-Chieh and Zhu, Yukun and Papandreou, George and Schroff, Florian and Adam, Hartwig},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year = {2018}
}

@inproceedings{he2019dynamic,
  title = {Dynamic Multi-Scale Filters for Semantic Segmentation},
  author = {He, Junjun and Deng, Zhongying and Qiao, Yu},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year = {2019}
}

@inproceedings{yin2020disentangled,
  title = {Disentangled Non-Local Neural Networks},
  author = {Yin, Minghao and Yao, Zhuliang and Cao, Yue and Li, Xiu and Zhang, Zheng and Lin, Stephen and Hu, Han},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year = {2020}
}

@inproceedings{zhang2018context,
  title = {Context Encoding for Semantic Segmentation},
  author = {Zhang, Hang and Dana, Kristin and Shi, Jianping and Zhang, Zhongyue and Wang, Xiaogang and Tyagi, Ambrish and Agrawal, Amit},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2018}
}

@article{wu2019fastfcn,
  title = {{FastFCN}: Rethinking Dilated Convolution in the Backbone for Semantic Segmentation},
  author = {Wu, Huikai and Zhang, Junge and Huang, Kaiqi and Liang, Kongming and Yu, Yizhou},
  journal = {arXiv preprint arXiv:1903.11816},
  year = {2019}
}

@inproceedings{zhao2017pyramid,
  title = {Pyramid Scene Parsing Network},
  author = {Zhao, Hengshuang and Shi, Jianping and Qi, Xiaojuan and Wang, Xiaogang and Jia, Jiaya},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2017}
}

@inproceedings{long2015fully,
  title = {Fully Convolutional Networks for Semantic Segmentation},
  author = {Long, Jonathan and Shelhamer, Evan and Darrell, Trevor},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2015}
}

@inproceedings{kirillov2019panoptic,
  title = {Panoptic Feature Pyramid Networks},
  author = {Kirillov, Alexander and Girshick, Ross and He, Kaiming and Doll{\'a}r, Piotr},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2019}
}

@inproceedings{cao2019gcnet,
  title = {{GCNet}: Non-local Networks Meet Squeeze-Excitation Networks and Beyond},
  author = {Cao, Yue and Xu, Jiarui and Lin, Stephen and Wei, Fangyun and Hu, Han},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
  year = {2019}
}

@article{huang2019interlaced,
  title = {Interlaced Sparse Self-Attention for Semantic Segmentation},
  author = {Huang, Lang and Yuan, Yuhui and Guo, Jianyuan and Zhang, Chao and Chen, Xilin and Wang, Jingdong},
  journal = {arXiv preprint arXiv:1907.12273},
  year = {2019}
}

@inproceedings{wang2018non,
  title = {Non-local Neural Networks},
  author = {Wang, Xiaolong and Girshick, Ross and Gupta, Abhinav and He, Kaiming},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2018}
}

@inproceedings{zhao2018psanet,
  title = {{PSANet}: Point-wise Spatial Attention Network for Scene Parsing},
  author = {Zhao, Hengshuang and Zhang, Yi and Liu, Shu and Shi, Jianping and Loy, Chen Change and Lin, Dahua and Jia, Jiaya},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year = {2018}
}

@inproceedings{cheng2021per,
  title = {Per-Pixel Classification is Not All You Need for Semantic Segmentation},
  author = {Cheng, Bowen and Schwing, Alexander G. and Kirillov, Alexander},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year = {2021}
}

@inproceedings{zhang2021k,
  title = {{K-Net}: Towards Unified Image Segmentation},
  author = {Zhang, Wenwei and Pang, Jiangmiao and Chen, Kai and Loy, Chen Change},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year = {2021}
}

@inproceedings{xiao2018unified,
  title = {Unified Perceptual Parsing for Scene Understanding},
  author = {Xiao, Tete and Liu, Yingcheng and Zhou, Bolei and Jiang, Yuning and Sun, Jian},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year = {2018}
}

@misc{tian2019adelaidet,
  title = {{AdelaiDet}: A Toolbox for Instance-level Recognition Tasks},
  author = {Tian, Zhi and Chen, Hao and Wang, Xinlong and Liu, Yuliang and Shen, Chunhua},
  howpublished = {\url{https://git.io/adelaidet}},
  year = {2019}
}

@inproceedings{law2018cornernet,
  title = {{CornerNet}: Detecting Objects as Paired Keypoints},
  author = {Law, Hei and Deng, Jia},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year = {2018}
}
```
<!-- attribution:end -->
