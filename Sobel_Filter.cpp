#include "Sobel_Filter.h"

Mat Display_Sobel(Mat img) {
	//cvtColor(img, img, COLOR_BGR2GRAY);
	int Gx[9] = { -1, -5, -1, 0,0,0, 1,5,1 };
	int Gy[9] = { -1,0,1,-5,0,5,-1,0,1 };
	Mat dst = img.clone();
	for (int y = 0; y < img.rows; y++)
		for (int x = 0; x < img.cols; x++)
			dst.at<uchar>(y, x) = 0.0;
	double AGx = 0.0, AGy = 0.0;
	for (int y = 1; y < img.rows - 1; y++) {
		for (int x = 1; x < img.cols - 1; x++) {
			AGx = img.at<uchar>(y - 1, x - 1) * Gx[0]
				+ img.at<uchar>(y - 1, x) * Gx[1]
				+ img.at<uchar>(y - 1, x + 1) * Gx[2]
				+ img.at<uchar>(y, x - 1) * Gx[3]
				+ img.at<uchar>(y, x) * Gx[4]
				+ img.at<uchar>(y, x + 1) * Gx[5]
				+ img.at<uchar>(y + 1, x - 1) * Gx[6]
				+ img.at<uchar>(y + 1, x) * Gx[7]
				+ img.at<uchar>(y + 1, x + 1) * Gx[8];
			AGy = img.at<uchar>(y - 1, x - 1) * Gy[0]
				+ img.at<uchar>(y - 1, x) * Gy[1]
				+ img.at<uchar>(y - 1, x + 1) * Gy[2]
				+ img.at<uchar>(y, x - 1) * Gy[3]
				+ img.at<uchar>(y, x) * Gy[4]
				+ img.at<uchar>(y, x + 1) * Gy[5]
				+ img.at<uchar>(y + 1, x - 1) * Gy[6]
				+ img.at<uchar>(y + 1, x) * Gy[7]
				+ img.at<uchar>(y + 1, x + 1) * Gy[8];
			double val = AGx + AGy;
			if (val > 127)
				val = 255;
			else
				val = 0;
			dst.at<uchar>(y, x) = val;
		}
	}
	return dst;
}