#include "YUV_space.h"

void YUV_img(Mat img) {
	Mat rgb[3];
	split(img, rgb);

	Mat y(img.rows, img.cols, CV_8UC1);
	Mat u(img.rows, img.cols, CV_8UC1);
	Mat v(img.rows, img.cols, CV_8UC1);

	ll R, G, B, Y, U, V;
	for (ll i = 0; i < img.rows; i++)
	{
		for (ll j = 0; j < img.cols; j++)
		{
			R = rgb[2].at<uchar>(i, j);
			G = rgb[1].at<uchar>(i, j);
			B = rgb[0].at<uchar>(i, j);

			Y = 0.299 * R + 0.587 * G + 0.114 * B;
			U = 128 - 0.168736 * R - 0.331264 * G + 0.5 * B;
			V = 128 + 0.5 * R - 0.418688 * G - 0.081312 * B;
			y.at<uchar>(i, j) = Y;
			u.at<uchar>(i, j) = U;
			v.at<uchar>(i, j) = V;
		}
	}
	imshow("V_space", v);
	imshow("U_space", u);
	imshow("Y_space", y);
}