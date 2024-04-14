#include "MeanFilter.h"

Mat meanFilter(Mat img) {
	int sizeMatrix = 9;
	Mat dst = img.clone();
	for (int y = 0; y < img.rows; y++)
		for (int x = 0; x < img.cols; x++)
			dst.at<uchar>(y, x) = 0.0;
	double mean = 0.0;
	for (int y = 1; y < img.rows - 1; y++) {
		for (int x = 1; x < img.cols - 1; x++) {
			mean = (img.at<uchar>(y - 1, x - 1)
				+ img.at<uchar>(y, x - 1)
				+ img.at<uchar>(y + 1, x - 1)
				+ img.at<uchar>(y - 1, x)
				+ img.at<uchar>(y, x)
				+ img.at<uchar>(y + 1, x)
				+ img.at<uchar>(y - 1, x + 1)
				+ img.at<uchar>(y, x + 1)
				+ img.at<uchar>(y + 1, x + 1)) / sizeMatrix;
			dst.at<uchar>(y, x) = mean;
		}
	}
	/*imshow("Anh goc", img);
	imshow("Sau Mean Filter", dst);*/
	return dst;
}