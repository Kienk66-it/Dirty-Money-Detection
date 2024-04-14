#include <opencv2/core/core.hpp>
#include <codecvt>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <stdio.h>
#include <vector>
#include<opencv2/opencv.hpp>
//#include "Channels.h"
//#include "YUV_space.h"
#include "Equal_His.h"
//#include "MedianFilter.h"
#include "MeanFilter.h"
#include "Sobel_Filter.h"
#include "Canny_Filter.h"
//#include "DCT_Transform.h"
#include "Gaussian_Filter.h"
#include "USM.h"
//#include "Segmentation.h"
//#include "Remove_Drawing.h"
#include "Serial_Detection.h"
#include "Anomaly.h"
#include <opencv2/features2d.hpp>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
typedef int ll;

using namespace cv;
using namespace std;

void Rotated_Img_Processing();
Mat rotate(Mat src, double angle);