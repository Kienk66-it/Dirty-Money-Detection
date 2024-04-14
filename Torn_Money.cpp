#include "Torn_Money.h"

ll fre[256];

void Torn_Money_Processing() {
	Mat img = imread("To-tien-50-nghin-HD.jpg", IMREAD_GRAYSCALE);
	Mat cop = imread("50k_rach1.jpg", IMREAD_GRAYSCALE);
	//img = Equalize_Hist_Red(img);
	//cop = Equalize_Hist_Red(cop);
	//threshold(cop, cop, 67, 255, THRESH_BINARY);
	Mat tmp = cop.clone();
	for (ll i = 0; i < img.rows; i++) {// dem so lan xuat hien tung pixel cua img
		for (ll j = 0; j < img.cols; j++) {
			fre[img.at<uchar>(i, j)] += 1;
		}
	}
	////////tach background
	for (ll i = 0; i < cop.rows; i++) {
		for (ll j = 0; j < cop.cols; j++) {
			if (fre[cop.at<uchar>(i, j)] < 1000) cop.at<uchar>(i, j) = 255;
			else break;
		}
	}
	for (ll i = cop.rows - 1; i >= 0; i--) {
		for (ll j = cop.cols - 1; j >= 0; j--) {
			if (fre[cop.at<uchar>(i, j)] < 1000) cop.at<uchar>(i, j) = 255;
			else break;
		}
	}//////
	
	////loc ra nhung pixel ma it xuat hien hoac khong co trong img
	for (ll i = 0; i < cop.rows; i++) {
		for (ll j = 0; j < cop.cols; j++) {
			if (fre[cop.at<uchar>(i, j)] == 0 || fre[cop.at<uchar>(i, j)] < 50) tmp.at<uchar>(i, j) = cop.at<uchar>(i, j);
			else {
				tmp.at<uchar>(i, j) = 255;
			}
		}
	}
	/////////

	//////chon ra nhung pixel xuat hien nhieu nhat trong tmp
	memset(fre, 0, sizeof(fre));
	for (ll i = 0; i < cop.rows; i++) {
		for (ll j = 0; j < cop.cols; j++) {
			fre[tmp.at<uchar>(i, j)]++;
		}
	}
	for (ll i = 0; i < tmp.rows; i++) {
		for (ll j = 0; j < tmp.cols; j++) {
			if (fre[tmp.at<uchar>(i, j)] < 700) tmp.at<uchar>(i, j) = 255;
		}
	}
	///////

	threshold(tmp, tmp, 150, 255, THRESH_BINARY);
	tmp = Display_Canny(tmp);

	Mat res = imread("50k_rach1.jpg", IMREAD_COLOR);// bat dau highlight len anh goc

	Mat matElement = getStructuringElement(MORPH_RECT, Size(1, 1));
	morphologyEx(tmp, tmp, MORPH_ERODE, matElement);
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(tmp, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);
	drawContours(res, contours, -1, Scalar(0, 255, 0), 5);
	///////

	img = imread("To-tien-50-nghin-HD.jpg", IMREAD_COLOR);
	cop = imread("50k_rach1.jpg", IMREAD_COLOR);
	imshow("Original_Money_Image", img);
	imshow("Torn_Money_Image", cop);
	imshow("My result", res);
}