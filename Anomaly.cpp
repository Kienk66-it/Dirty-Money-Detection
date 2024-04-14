#include "Anomaly.h"

void Anomaly_Process() {
	Mat Original_img = imread("1.jpg", IMREAD_COLOR);
	Mat Dirty_img = imread("3.jpg", IMREAD_COLOR);
	cvtColor(Original_img, Original_img, COLOR_BGR2GRAY);
	cvtColor(Dirty_img, Dirty_img, COLOR_BGR2GRAY);
	ll h = Original_img.rows, w = Original_img.cols;
	double u, o2;
	double const pi = 3.14159, e = 2.71828;
	double px = 1;
	for (ll i = 0; i < h; i++) {
		for (ll j = 0; j < w; j++) {
			u = sqrt((ll)Original_img.at<uchar>(i, j));
			o2 = (ll)Original_img.at<uchar>(i, j) - u;

			px *= 1 / sqrt(2 * pi) / o2 / 1.64872;
		}
	}
	cout << px;
}