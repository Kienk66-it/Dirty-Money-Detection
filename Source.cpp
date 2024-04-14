#include <opencv2/core/core.hpp>
#include <codecvt>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <map>
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
#include "Rotated_img_Processing.h"
#include "Two_Same_Images.h"
#include "Different_Images.h"
#include "Serial_Different_Processing.h"
#include "Torn_Money.h"
#include <opencv2/features2d.hpp>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "SIFT.h"
typedef int ll;

using namespace cv;
using namespace std;


void Display_Histogram(Mat img, string Diagram_Name) {
	int histSize = 256;
	float range[] = { 0, 255 }; //Pham vi gia tri muon do
	const float* ranges[] = { range };
	// Tinh histogram
	MatND hist;
	calcHist(&img, 1, 0, Mat(), hist, 1, &histSize, ranges, true, false);
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);
	Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(0, 0, 0));
	normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
	}
	imshow(Diagram_Name, histImage);
}

ll is_Original(Mat img, ll x, ll y) {
	for (ll i = (720 - x) / 2 - 1; i < (720 - x) / 2 + x + 1; i++) {
		if (img.at<uchar>(i, (720 - y) / 2 - 1) != 0) return 0;
		if (img.at<uchar>(i, (720 - y) / 2 + y) != 0) return 0;
	}
	for (ll j = (720 - y) / 2 - 1; j < (720 - y) / 2 + y + 1; j++) {
		if (img.at<uchar>((720 - x) / 2 - 1, j) != 0) return 0;
		if (img.at<uchar>((720 - x) / 2 + x, j) != 0) return 0;
	}
	return 1;
}

int main()
{
	//Same_Img_Processing();
	//Diff_Img_Processing();
	//Rotated_Img_Processing();
	//Serial_Diff_Processing();
	Torn_Money_Processing();
	//SIFT_Processing();
	waitKey(0);
}