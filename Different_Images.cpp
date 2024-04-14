#include "Different_Images.h"

void Diff_Img_Processing() {
	Mat img = imread("1.jpg", IMREAD_COLOR);
	Mat cop = imread("7.jpg", IMREAD_COLOR);
	imshow("Original_Image", img);
	imshow("Input_Image", cop);
	resize(img, img, Size(cop.cols, cop.rows), INTER_LINEAR);
	cvtColor(img, img, COLOR_BGR2GRAY);
	cvtColor(cop, cop, COLOR_BGR2GRAY);
	//imshow("img", img);
	ll h = img.rows, w = img.cols;
	Mat tmp = cop.clone();
	threshold(img, img, 89, 255, THRESH_BINARY);
	threshold(cop, cop, 50, 255, THRESH_BINARY);
	for (ll i = 0; i < h; i++) {
		for (ll j = 0; j < w; j++) {
			tmp.at<uchar>(i, j) = abs(cop.at<uchar>(i, j) - img.at<uchar>(i, j));
		}
	}
	for (ll i = 1; i < h - 1; i++) {// lọc đơn chấm
		for (ll j = 1; j < w - 1; j++) {
			if (tmp.at<uchar>(i - 1, j - 1) == 0 && tmp.at<uchar>(i - 1, j) == 0 && tmp.at<uchar>(i - 1, j + 1) == 0 &&
				tmp.at<uchar>(i, j - 1) == 0 && tmp.at<uchar>(i, j) == 255 && tmp.at<uchar>(i, j + 1) == 0 &&
				tmp.at<uchar>(i + 1, j - 1) == 0 && tmp.at<uchar>(i + 1, j) == 0 && tmp.at<uchar>(i + 1, j + 1) == 0)
				tmp.at<uchar>(i, j) = 0;
		}
	}
	Mat white_img = img.clone();
	for (ll i = 0; i < h; i++) {
		for (ll j =0; j < w; j++) {
			white_img.at<uchar>(i, j) = 255;
		}
	}
	Mat obj = imread("7.jpg", IMREAD_COLOR);
	Mat matElement = getStructuringElement(MORPH_RECT, Size(3, 3));//đánh dấu vết bẩn
	morphologyEx(tmp, tmp, MORPH_ERODE, matElement);

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(tmp, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);
	drawContours(white_img, contours, -1, Scalar(0, 255, 0), 21);

	tmp = white_img.clone();
	tmp = Display_Canny(tmp);// lọc cạnh vết bẩn
	//imshow("2.5", tmp);
	matElement = getStructuringElement(MORPH_RECT, Size(1, 1));
	morphologyEx(tmp, tmp, MORPH_OPEN, matElement);
	vector<vector<Point>> contours1;
	vector<Vec4i> v;
	findContours(tmp, contours1, v, RETR_TREE, CHAIN_APPROX_NONE);
	drawContours(obj, contours1, -1, Scalar(0, 255, 0), 7);// vẽ đường bao quanh vết bẩn
	imshow("Result", obj);
}