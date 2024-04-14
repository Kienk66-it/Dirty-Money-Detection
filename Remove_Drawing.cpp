#include "Remove_Drawing.h"
#include <map>

void Display_After_Removing_Drawing() {
	Mat Original_img = imread("1.jpg", IMREAD_COLOR);
	Mat Dirty_img = imread("4.jpg", IMREAD_COLOR);
	Mat tmp = Dirty_img.clone();
	ll h = Original_img.rows, w = Original_img.cols;
	for (ll i = 1; i < h - 1; i++) {
		for (ll j = 1; j < w - 1; j++) {
			if (Original_img.at<Vec3b>(i, j)[0] != Dirty_img.at<Vec3b>(i, j)[0] ||
				Original_img.at<Vec3b>(i, j)[1] != Dirty_img.at<Vec3b>(i, j)[1] ||
				Original_img.at<Vec3b>(i, j)[2] != Dirty_img.at<Vec3b>(i, j)[2]) {
				if (Original_img.at<Vec3b>(i - 1, j - 1)[0] == Dirty_img.at<Vec3b>(i - 1, j - 1)[0] &&
					Original_img.at<Vec3b>(i - 1, j - 1)[1] == Dirty_img.at<Vec3b>(i - 1, j - 1)[1] &&
					Original_img.at<Vec3b>(i - 1, j - 1)[2] == Dirty_img.at<Vec3b>(i - 1, j - 1)[2]) {
					tmp.at<Vec3b>(i - 1, j - 1)[0] = 0;
					tmp.at<Vec3b>(i - 1, j - 1)[1] = 0;
					tmp.at<Vec3b>(i - 1, j - 1)[2] = 0;
				}
				if (Original_img.at<Vec3b>(i - 1, j)[0] == Dirty_img.at<Vec3b>(i - 1, j)[0] &&
					Original_img.at<Vec3b>(i - 1, j)[1] == Dirty_img.at<Vec3b>(i - 1, j)[1] &&
					Original_img.at<Vec3b>(i - 1, j)[2] == Dirty_img.at<Vec3b>(i - 1, j)[2]) {
					tmp.at<Vec3b>(i - 1, j)[0] = 0;
					tmp.at<Vec3b>(i - 1, j)[1] = 0;
					tmp.at<Vec3b>(i - 1, j)[2] = 0;
				}
				if (Original_img.at<Vec3b>(i - 1, j + 1)[0] == Dirty_img.at<Vec3b>(i - 1, j + 1)[0] &&
					Original_img.at<Vec3b>(i - 1, j + 1)[1] == Dirty_img.at<Vec3b>(i - 1, j + 1)[1] &&
					Original_img.at<Vec3b>(i - 1, j + 1)[2] == Dirty_img.at<Vec3b>(i - 1, j + 1)[2]) {
					tmp.at<Vec3b>(i - 1, j + 1)[0] = 0;
					tmp.at<Vec3b>(i - 1, j + 1)[1] = 0;
					tmp.at<Vec3b>(i - 1, j + 1)[2] = 0;
				}
				if (Original_img.at<Vec3b>(i, j - 1)[0] == Dirty_img.at<Vec3b>(i, j - 1)[0] &&
					Original_img.at<Vec3b>(i, j - 1)[1] == Dirty_img.at<Vec3b>(i, j - 1)[1] &&
					Original_img.at<Vec3b>(i, j - 1)[2] == Dirty_img.at<Vec3b>(i, j - 1)[2]) {
					tmp.at<Vec3b>(i, j - 1)[0] = 0;
					tmp.at<Vec3b>(i, j - 1)[1] = 0;
					tmp.at<Vec3b>(i, j - 1)[2] = 0;
				}
				if (Original_img.at<Vec3b>(i, j + 1)[0] == Dirty_img.at<Vec3b>(i, j + 1)[0] &&
					Original_img.at<Vec3b>(i, j + 1)[1] == Dirty_img.at<Vec3b>(i, j + 1)[1] &&
					Original_img.at<Vec3b>(i, j + 1)[2] == Dirty_img.at<Vec3b>(i, j + 1)[2]) {
					tmp.at<Vec3b>(i, j + 1)[0] = 0;
					tmp.at<Vec3b>(i, j + 1)[1] = 0;
					tmp.at<Vec3b>(i, j + 1)[2] = 0;
				}
				if (Original_img.at<Vec3b>(i, j)[0] == Dirty_img.at<Vec3b>(i, j)[0] &&
					Original_img.at<Vec3b>(i, j)[1] == Dirty_img.at<Vec3b>(i, j)[1] &&
					Original_img.at<Vec3b>(i, j)[2] == Dirty_img.at<Vec3b>(i, j)[2]) {
					tmp.at<Vec3b>(i, j)[0] = 0;
					tmp.at<Vec3b>(i, j)[1] = 0;
					tmp.at<Vec3b>(i, j)[2] = 0;
				}
				if (Original_img.at<Vec3b>(i + 1, j - 1)[0] == Dirty_img.at<Vec3b>(i + 1, j - 1)[0] &&
					Original_img.at<Vec3b>(i + 1, j - 1)[1] == Dirty_img.at<Vec3b>(i + 1, j - 1)[1] &&
					Original_img.at<Vec3b>(i + 1, j - 1)[2] == Dirty_img.at<Vec3b>(i + 1, j - 1)[2]) {
					tmp.at<Vec3b>(i + 1, j - 1)[0] = 0;
					tmp.at<Vec3b>(i + 1, j - 1)[1] = 0;
					tmp.at<Vec3b>(i + 1, j - 1)[2] = 0;
				}
				if (Original_img.at<Vec3b>(i + 1, j)[0] == Dirty_img.at<Vec3b>(i + 1, j)[0] &&
					Original_img.at<Vec3b>(i + 1, j)[1] == Dirty_img.at<Vec3b>(i + 1, j)[1] &&
					Original_img.at<Vec3b>(i + 1, j)[2] == Dirty_img.at<Vec3b>(i + 1, j)[2]) {
					tmp.at<Vec3b>(i + 1, j)[0] = 0;
					tmp.at<Vec3b>(i + 1, j)[1] = 0;
					tmp.at<Vec3b>(i + 1, j)[2] = 0;
				}
				if (Original_img.at<Vec3b>(i + 1, j + 1)[0] == Dirty_img.at<Vec3b>(i + 1, j + 1)[0] &&
					Original_img.at<Vec3b>(i + 1, j + 1)[1] == Dirty_img.at<Vec3b>(i + 1, j + 1)[1] &&
					Original_img.at<Vec3b>(i + 1, j + 1)[2] == Dirty_img.at<Vec3b>(i + 1, j + 1)[2]) {
					tmp.at<Vec3b>(i + 1, j + 1)[0] = 0;
					tmp.at<Vec3b>(i + 1, j + 1)[1] = 0;
					tmp.at<Vec3b>(i + 1, j + 1)[2] = 0;
				}
			}
		}
	}
	ll d = 2, k = 0;
	Mat tmp1 = tmp.clone();
	while (d != 0) {
		map<ll, ll> v;
		for (ll i = 1; i < h - 1; i++) {
			for (ll j = 1; j < w - 1; j++) {
				if (tmp1.at<Vec3b>(i, j)[0] == 0 && tmp1.at<Vec3b>(i, j)[1] == 0 && tmp1.at<Vec3b>(i, j)[2] == 0 && v[k] == 0) {
					tmp1.at<Vec3b>(i - 1, j - 1)[0] = 0;
					tmp1.at<Vec3b>(i - 1, j - 1)[1] = 0;
					tmp1.at<Vec3b>(i - 1, j - 1)[2] = 0;
					v[k - w + 1] = 1;

					tmp1.at<Vec3b>(i - 1, j)[0] = 0;
					tmp1.at<Vec3b>(i - 1, j)[1] = 0;
					tmp1.at<Vec3b>(i - 1, j)[2] = 0;
					v[k - w + 2] = 1;

					tmp1.at<Vec3b>(i - 1, j + 1)[0] = 0;
					tmp1.at<Vec3b>(i - 1, j + 1)[1] = 0;
					tmp1.at<Vec3b>(i - 1, j + 1)[2] = 0;
					v[k - w + 3] = 1;

					tmp1.at<Vec3b>(i, j - 1)[0] = 0;
					tmp1.at<Vec3b>(i, j - 1)[1] = 0;
					tmp1.at<Vec3b>(i, j - 1)[2] = 0;
					v[k - 1] = 1;

					tmp1.at<Vec3b>(i, j)[0] = 0;
					tmp1.at<Vec3b>(i, j)[1] = 0;
					tmp1.at<Vec3b>(i, j)[2] = 0;
					v[k] = 1;

					tmp1.at<Vec3b>(i, j + 1)[0] = 0;
					tmp1.at<Vec3b>(i, j + 1)[1] = 0;
					tmp1.at<Vec3b>(i, j + 1)[2] = 0;
					v[k + 1] = 1;

					tmp1.at<Vec3b>(i + 1, j - 1)[0] = 0;
					tmp1.at<Vec3b>(i + 1, j - 1)[1] = 0;
					tmp1.at<Vec3b>(i + 1, j - 1)[2] = 0;
					v[k + w - 3] = 1;

					tmp1.at<Vec3b>(i + 1, j)[0] = 0;
					tmp1.at<Vec3b>(i + 1, j)[1] = 0;
					tmp1.at<Vec3b>(i + 1, j)[2] = 0;
					v[k + w - 2] = 1;

					tmp1.at<Vec3b>(i + 1, j + 1)[0] = 0;
					tmp1.at<Vec3b>(i + 1, j + 1)[1] = 0;
					tmp1.at<Vec3b>(i + 1, j + 1)[2] = 0;
					v[k + w - 1] = 1;
				}
				k++;
			}
		}
		d--;
	}
	imshow("White board", tmp1);
}