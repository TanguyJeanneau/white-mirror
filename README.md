# README

This repository contains the result of my Master Thesis, along with the thesis itself. It was then re-used as  an entrepreneurial project, whose goal was to create "Magical Mirrors" which reflects reality as painted by any famous author, to be displayed in Museums.

### Setup

* Download VGG19 model

`$ python vgg.py`

* Execute program

`$ python main.py`

* on vast.ai GPU instances, it is necessary to :

`
$ apt-get update
$ apt-get install libglib2.0-0
$ apt-get install libgtk2.0-dev
`
### Next steps

In order to improve the quality of the results, the following path could be explored:
* To improve the temporal coherency: change the optical flow computation method. Huang et al [1] used a deep learning based method, which yields better resutls than the current opencv dense opticalflow method.
* Pre-compute optical flows on the training dataset for better performances
* Improve the training dataset: It is currently made of royalty-free videos from stock videos websites (180k frames), as done in several source papers. Using dataset of a higher quality, such as the Hollywood2 scenes dataset and build a synthetic dataset from images (e.g. the Microsoft COCO dataset), as done by M.Ruder et. al [2] could improve the dataset and therefore the quality of the results.

### Bibliography

An exhaustive bibliography is available in the Master Thesis. Here are the most influencial papers for this work:

[1]: Huang et al, Real-time neural style transfer for videos, 2017 http://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Real-Time_Neural_Style_CVPR_2017_paper.pdf.

[2]: Ruder, M., Dosovitskiy, A., Brow, T., Artistic style transfer for videos and spherical images, 2018 https://arxiv.org/pdf/1708.04538.pdf.

[3]: Johnson, J., Alahi, A., Fei-Fei, L., Perceptual losses for real-time style transfer and super-resolution, Stanford University, 2016 https://arxiv.org/pdf/1603.08155.pdf.

[4]: Gatys, L., Ecker, A., Bethge, M., A neural algorithm of artistic style, 2015, https://arxiv.org/pdf/1508.06576.pdf.
