使用方法：只需要使用“切割机加识别”文件夹中的文件即可

1.运行"first_img.py"文件，input_path 为需要处理的文件的文件路径，处理后的图片将在同文件夹输出，weigths_path 是训练好的模型路径，现在直接使用即可

2.使用单词识别功能，只需要使用"run.py",即可使用



模型选择比较数据集：EMNIST

选用Alexnet后使用数据集：改良后的Chars74K（分类并将大小写相似的数据集合并，如S，s；O，o）

图片预处理数据集及参考：K. Sadekar, A. Tiwari, P. Singh and S. Raman, "LS-HDIB: A Large Scale Handwritten Document Image Binarization Dataset," 2022 26th International Conference on Pattern Recognition (ICPR), 2022, pp. 1678-1684, doi: 10.1109/ICPR56361.2022.9956447.