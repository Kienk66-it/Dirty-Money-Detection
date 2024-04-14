#include "Equal_His.h"

ll Fre0[256], Fre1[256], Fre2[256], v;

Mat Equalize_Hist_Red(Mat img) {
	ll h = img.rows, w = img.cols, s0 = 0, s1 = 0, s2 = 0;

	for (ll i = 0; i < h; i++) {
		for (ll j = 0; j < w; j++) {
			/*v = img.at<Vec3b>(i, j)[2];
			Fre2[v]++;
			v = img.at<Vec3b>(i, j)[1];
			Fre1[v]++;
			v = img.at<Vec3b>(i, j)[0];
			Fre0[v]++;*/
			v = (ll)img.at<uchar>(i, j);
			Fre0[v]++;
		}
	}
	for (ll i = 0; i < 256; i++) {
		/*s0 += Fre0[i];
		Fre0[i] = s0 * 255 / (w * h);
		s1 += Fre1[i];
		Fre1[i] = s1 * 255 / (w * h);
		s2 += Fre2[i];
		Fre2[i] = s2 * 255 / (w * h);*/
		s0 += Fre0[i];
		Fre0[i] = s0 * 255 / (w * h);
	}
	for (ll i = 0; i < h; i++) {
		for (ll j = 0; j < w; j++) {
			/*v = img.at<Vec3b>(i, j)[2];
			img.at<Vec3b>(i, j)[2] = Fre2[v];
			v = img.at<Vec3b>(i, j)[1];
			img.at<Vec3b>(i, j)[1] = Fre1[v];
			v = img.at<Vec3b>(i, j)[0];
			img.at<Vec3b>(i, j)[0] = Fre0[v];*/
			v = (ll)img.at<uchar>(i, j);
			img.at<uchar>(i, j) = Fre0[v];
		}
	}
	return img;
}