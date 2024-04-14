#include "MedianFilter.h"

void insertionSort(int window[])
{
	int temp, i, j;
	for (i = 0; i < 9; i++) {
		temp = window[i];
		for (j = i - 1; j >= 0 && temp < window[j]; j--)
		{
			window[j + 1] = window[j];
		}
		window[j + 1] = temp;
	}
}

void
medianFilter(Mat img) {
	int window[9];
	Mat dst = img.clone();
	for (int y = 0; y < img.rows; y++)
		for (int x = 0; x < img.cols; x++)
			dst.at<uchar>(y, x) = 0.0;
	for (int y = 1; y < img.rows - 1; y++) {
		for (int x = 1; x < img.cols - 1; x++) {
			window[0] = img.at<uchar>(y - 1, x - 1);
			window[1] = img.at<uchar>(y, x - 1);
			window[2] = img.at<uchar>(y + 1, x - 1);
			window[3] = img.at<uchar>(y - 1, x);
			window[4] = img.at<uchar>(y, x);
			window[5] = img.at<uchar>(y + 1, x);
			window[6] = img.at<uchar>(y - 1, x + 1);
			window[7] = img.at<uchar>(y, x + 1);
			window[8] = img.at<uchar>(y + 1, x + 1);
			insertionSort(window);
			dst.at<uchar>(y, x) = window[4];
		}
		
	}
	imshow("Anh goc", img);
	imshow("Sau Median Filter", dst);
}