#include "Serial_Different_Processing.h"

ll cnt1[256], cnt2[256], cnt3[256], cnt[256];

void Serial_Diff_Processing() {
	Mat img = imread("1.jpg", IMREAD_COLOR);
	Mat cop = imread("6.jpg", IMREAD_COLOR);
	imshow("Original_img", img);
	imshow("Input_img", cop);
	Mat v = cop.clone();
	cvtColor(img, img, COLOR_BGR2GRAY);
	cvtColor(cop, cop, COLOR_BGR2GRAY);
	Mat tmp = img.clone();
	ll h = img.rows, w = img.cols, max1 = INT_MIN, max2 = INT_MIN;
	for (ll i = 0; i < h; i++) {
		for (ll j = 0; j < w; j++) {
			ll v = img.at<uchar>(i, j);
			cnt1[v]++;
			ll k = cnt1[v];
			max1 = max(max1, k);
			tmp.at<uchar>(i, j) = 255;
		}
	}
	for (ll i = 0; i < h; i++) {
		for (ll j = 0; j < w; j++) {
			ll v = cop.at<uchar>(i, j);
			cnt2[v]++;
			ll k = cnt2[v];
			max2 = max(max2, k);
		}
	}
	/*ll sub = 39;
	for (ll i = 0; i < h; i++) {
		for (ll j = 0; j < w; j++) {
			if (cop.at <uchar> (i, j) < 39) cop.at <uchar> (i, j) = 0;
			else cop.at <uchar> (i, j) -= 39;
		}
	}*/
	cop = Equalize_Hist_Red(cop);
	for (ll i = 0; i < h; i++) {
		for (ll j = 0; j < w; j++) {
			if (cnt1[cop.at <uchar>(i, j)] == 0) tmp.at<uchar>(i, j) = cop.at <uchar>(i, j);
		}
	}
	for (ll i = 0; i < h; i++) {
		for (ll j = 0; j < w; j++) {
			cnt[tmp.at<uchar>(i, j)]++;
		}
	}
	//for (ll i = 0; i < 256; i++) cout << cnt[i] << " ";
	for (ll i = 0; i < h; i++) {
		for (ll j = 0; j < w; j++) {
			if (tmp.at<uchar>(i, j) != 5 && tmp.at<uchar>(i, j) != 255 && tmp.at<uchar>(i, j) != 254) tmp.at<uchar>(i, j) = 255;
		}
	}
	for (ll i = 0; i < h; i++) {
		for (ll j = 0; j < w; j++) {
			if (tmp.at<uchar>(i - 1, j - 1) == 255 && tmp.at<uchar>(i - 1, j) == 255 && tmp.at<uchar>(i - 1, j + 1) == 255 &&
				tmp.at<uchar>(i, j - 1) == 255 && tmp.at<uchar>(i, j) == 0 && tmp.at<uchar>(i, j + 1) == 255 &&
				tmp.at<uchar>(i + 1, j - 1) == 255 && tmp.at<uchar>(i + 1, j) == 255 && tmp.at<uchar>(i + 1, j + 1) == 255)
				tmp.at<uchar>(i, j) = 255;
		}
	}
	Mat matElement = getStructuringElement(MORPH_RECT, Size(1, 1));
	morphologyEx(tmp, tmp, MORPH_CLOSE, matElement);
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(tmp, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);
	drawContours(cop, contours, -1, Scalar(0, 255, 0), 3);
	//tmp = Display_Canny(tmp);
	threshold(tmp, tmp, 127, 255, THRESH_BINARY_INV);
	matElement = getStructuringElement(MORPH_RECT, Size(1, 3));
	morphologyEx(tmp, tmp, MORPH_ERODE, matElement);
	vector<vector<Point>> contours1;
	vector<Vec4i> hierarchy1;
	findContours(tmp, contours1, hierarchy1, RETR_TREE, CHAIN_APPROX_NONE);
	//drawContours(v, contours1, -1, Scalar(0, 255, 0), 3);
	Mat var = imread("6.jpg", IMREAD_GRAYSCALE);
	threshold(var, var, 69, 255, THRESH_BINARY);
	for (ll i = 0; i < h; i++) {
		for (ll j = 0; j < w; j++) {
			if (tmp.at<uchar>(i, j) != 0) {
				if (var.at<uchar>(i, j) == var.at<uchar>(i - 1, j - 1)) tmp.at<uchar>(i - 1, j - 1) = 255;
				if (var.at<uchar>(i, j) == var.at<uchar>(i - 1, j)) tmp.at<uchar>(i - 1, j) = 255;
				if (var.at<uchar>(i, j) == var.at<uchar>(i - 1, j + 1)) tmp.at<uchar>(i - 1, j + 1) = 255;
				if (var.at<uchar>(i, j) == var.at<uchar>(i, j - 1)) tmp.at<uchar>(i, j - 1) = 255;
				if (var.at<uchar>(i, j) == var.at<uchar>(i, j + 1)) tmp.at<uchar>(i, j + 1) = 255;
				if (var.at<uchar>(i, j) == var.at<uchar>(i + 1, j - 1)) tmp.at<uchar>(i + 1, j - 1) = 255;
				if (var.at<uchar>(i, j) == var.at<uchar>(i + 1, j)) tmp.at<uchar>(i + 1, j) = 255;
				if (var.at<uchar>(i, j) == var.at<uchar>(i + 1, j + 1)) tmp.at<uchar>(i + 1, j + 1) = 255;
			}

		}
	}
	for (ll i = h - 1; i >= 0; i--) {
		for (ll j = w - 1; j >= 0; j--) {
			if (tmp.at<uchar>(i, j) != 0) {
				if (var.at<uchar>(i, j) == var.at<uchar>(i - 1, j - 1)) tmp.at<uchar>(i - 1, j - 1) = 255;
				if (var.at<uchar>(i, j) == var.at<uchar>(i - 1, j)) tmp.at<uchar>(i - 1, j) = 255;
				if (var.at<uchar>(i, j) == var.at<uchar>(i - 1, j + 1)) tmp.at<uchar>(i - 1, j + 1) = 255;
				if (var.at<uchar>(i, j) == var.at<uchar>(i, j - 1)) tmp.at<uchar>(i, j - 1) = 255;
				if (var.at<uchar>(i, j) == var.at<uchar>(i, j + 1)) tmp.at<uchar>(i, j + 1) = 255;
				if (var.at<uchar>(i, j) == var.at<uchar>(i + 1, j - 1)) tmp.at<uchar>(i + 1, j - 1) = 255;
				if (var.at<uchar>(i, j) == var.at<uchar>(i + 1, j)) tmp.at<uchar>(i + 1, j) = 255;
				if (var.at<uchar>(i, j) == var.at<uchar>(i + 1, j + 1)) tmp.at<uchar>(i + 1, j + 1) = 255;
			}

		}
	}
	matElement = getStructuringElement(MORPH_RECT, Size(1, 1));
	morphologyEx(tmp, tmp, MORPH_ERODE, matElement);
	vector<vector<Point>> contours2;
	vector<Vec4i> hierarchy2;
	findContours(tmp, contours2, hierarchy2, RETR_TREE, CHAIN_APPROX_NONE);
	drawContours(v, contours2, -1, Scalar(0, 255, 0), 3);
	//imshow("tmp", tmp);
	imshow("Result", v);
}