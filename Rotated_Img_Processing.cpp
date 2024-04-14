#include "Rotated_img_Processing.h"

Mat rotate(Mat src, double angle)   //rotate function returning mat object with parametres imagefile and angle    
{
	Mat dst;      //Mat object for output image file
	Point2f pt(src.cols / 2., src.rows / 2.);          //point from where to rotate    
	Mat r = getRotationMatrix2D(pt, angle, 1.0);      //Mat object for storing after rotation
	warpAffine(src, dst, r, Size(src.cols, src.rows));  ///applie an affine transforation to image.
	return dst;         //returning Mat object for output image file
}

void Rotated_Img_Processing() {
	Mat img = imread("Original.jpg", IMREAD_GRAYSCALE);
	Mat cop = imread("Rotated.jpg", IMREAD_GRAYSCALE);
	//ghep hinh vuong
	Mat sq = imread("black_img.jpg", IMREAD_GRAYSCALE);// khung hình vuông màu đen
	ll h = img.rows, w = img.cols;
	/*resize(img, img, Size(w / 4, h / 4), INTER_LINEAR);
	resize(cop, cop, Size(w / 4, h / 4), INTER_LINEAR);
	h = img.rows, w = img.cols;
	ll x = h, y = w;

	for (ll i = (720 - h) / 2; i < (720 - h) / 2 + h; i++) {
		for (ll j = (720 - w) / 2; j < (720 - w) / 2 + w; j++) {
			ll temp = img.at<uchar>(i - (720 - h) / 2, j - (720 - w) / 2);
			sq.at<uchar>(i, j) = temp;
		}
	}
	img = sq.clone();
	sq = imread("black_img.jpg", IMREAD_GRAYSCALE);
	for (ll i = (720 - h) / 2; i < (720 - h) / 2 + h; i++) {
		for (ll j = (720 - w) / 2; j < (720 - w) / 2 + w; j++) {
			ll temp = cop.at<uchar>(i - (720 - h) / 2, j - (720 - w) / 2);
			sq.at<uchar>(i, j) = temp;
		}
	}
	cop = sq.clone();
	cop = rotate(cop, 30);*/
	threshold(img, img, 89, 255, THRESH_BINARY);
	threshold(cop, cop, 50, 255, THRESH_BINARY);
	//imshow("3", cop);
	ll angle, d = 0, maxx = INT_MIN;
	for (ll k = 0; k <= 360; k++) {// Thuật toán tìm góc
		Mat tmp = img.clone();
		tmp = rotate(tmp, k);
		d = 0;
		for (ll i = 0; i < 720; i++) {
			for (ll j = 0; j < 720; j++) {
				if (tmp.at<uchar>(i, j) == cop.at<uchar>(i, j)) d++;
			}
		}
		maxx = max(maxx, d);
	}
	for (ll k = 0; k <= 360; k++) {
		Mat tmp = img.clone();
		tmp = rotate(tmp, k);
		d = 0;
		for (ll i = 0; i < 720; i++) {
			for (ll j = 0; j < 720; j++) {
				if (tmp.at<uchar>(i, j) == cop.at<uchar>(i, j)) d++;
			}
		}
		if (d == maxx) {
			img = rotate(img, k);
			angle = k;// góc cần tìm đây
			break;
		}
	}
	Mat tmp = cop.clone();
	for (ll i = 0; i < 720; i++) {// trừ 2 ảnh
		for (ll j = 0; j < 720; j++) {
			tmp.at<uchar>(i, j) = abs(cop.at<uchar>(i, j) - img.at<uchar>(i, j));
		}
	}
	for (ll i = 1; i < 720 - 1; i++) {// Xử lí vết chấm
		for (ll j = 1; j < 720 - 1; j++) {
			if (tmp.at<uchar>(i - 1, j - 1) == 0 && tmp.at<uchar>(i - 1, j) == 0 && tmp.at<uchar>(i - 1, j + 1) == 0 &&
				tmp.at<uchar>(i, j - 1) == 0 && tmp.at<uchar>(i, j) == 255 && tmp.at<uchar>(i, j + 1) == 0 &&
				tmp.at<uchar>(i + 1, j - 1) == 0 && tmp.at<uchar>(i + 1, j) == 0 && tmp.at<uchar>(i + 1, j + 1) == 0)
				tmp.at<uchar>(i, j) = 0;
		}
	}
	/*for (ll i = 1; i < 720 - 1; i++) {
		for (ll j = 1; j < 720 - 1; j++) {
			if (tmp.at<uchar>(i - 1, j - 1) == 0 && tmp.at<uchar>(i - 1, j) == 0 && tmp.at<uchar>(i - 1, j + 1) == 0 &&
				tmp.at<uchar>(i, j - 1) == 0 && tmp.at<uchar>(i, j) == 255 && tmp.at<uchar>(i, j + 1) == 0 &&
				tmp.at<uchar>(i + 1, j - 1) == 0 && tmp.at<uchar>(i + 1, j) == 0 && tmp.at<uchar>(i + 1, j + 1) == 0)
				tmp.at<uchar>(i, j) = 0;
		}
	}
	for (ll i = 1; i < 720 - 1; i++) {
		for (ll j = 1; j < 720 - 1; j++) {
			if (tmp.at<uchar>(i - 1, j - 1) == 0 && tmp.at<uchar>(i - 1, j) == 0 && tmp.at<uchar>(i - 1, j + 1) == 0 &&
				tmp.at<uchar>(i, j - 1) == 0 && tmp.at<uchar>(i, j) == 255 && tmp.at<uchar>(i, j + 1) == 0 &&
				tmp.at<uchar>(i + 1, j - 1) == 0 && tmp.at<uchar>(i + 1, j) == 0 && tmp.at<uchar>(i + 1, j + 1) == 0)
				tmp.at<uchar>(i, j) = 0;
		}
	}*/


	Mat org = imread("1.jpg", IMREAD_COLOR);//ghép ảnh gốc vào khung
	h = org.rows, w = org.cols;
	resize(org, org, Size(w / 4, h / 4), INTER_LINEAR);
	h /= 4;
	w /= 4;
	sq = imread("black_img.jpg", IMREAD_COLOR);
	for (ll i = (720 - h) / 2; i < (720 - h) / 2 + h; i++) {
		for (ll j = (720 - w) / 2; j < (720 - w) / 2 + w; j++) {
			ll temp = org.at<Vec3b>(i - (720 - h) / 2, j - (720 - w) / 2)[0];
			sq.at<Vec3b>(i, j)[0] = temp;
			temp = org.at<Vec3b>(i - (720 - h) / 2, j - (720 - w) / 2)[1];
			sq.at<Vec3b>(i, j)[1] = temp;
			temp = org.at<Vec3b>(i - (720 - h) / 2, j - (720 - w) / 2)[2];
			sq.at<Vec3b>(i, j)[2] = temp;
		}
	}
	imshow("Original", sq);
	Mat rot = imread("4.jpg", IMREAD_COLOR);//ghép ảnh bị xoay vào khung
	h = rot.rows, w = rot.cols;
	resize(rot, rot, Size(w / 4, h / 4), INTER_LINEAR);
	h /= 4;
	w /= 4;
	sq = imread("black_img.jpg", IMREAD_COLOR);
	for (ll i = (720 - h) / 2; i < (720 - h) / 2 + h; i++) {
		for (ll j = (720 - w) / 2; j < (720 - w) / 2 + w; j++) {
			ll temp = rot.at<Vec3b>(i - (720 - h) / 2, j - (720 - w) / 2)[0];
			sq.at<Vec3b>(i, j)[0] = temp;
			temp = rot.at<Vec3b>(i - (720 - h) / 2, j - (720 - w) / 2)[1];
			sq.at<Vec3b>(i, j)[1] = temp;
			temp = rot.at<Vec3b>(i - (720 - h) / 2, j - (720 - w) / 2)[2];
			sq.at<Vec3b>(i, j)[2] = temp;
		}
	}
	imshow("Rotated", rotate(sq, 30));


	Mat obj = imread("4.jpg", IMREAD_COLOR);//ghép ảnh bị xoay vào khung lần nữa
	h = obj.rows, w = obj.cols;
	resize(obj, obj, Size(w / 4, h / 4), INTER_LINEAR);
	h /= 4;
	w /= 4;
	sq = imread("black_img.jpg", IMREAD_COLOR);
	for (ll i = (720 - h) / 2; i < (720 - h) / 2 + h; i++) {
		for (ll j = (720 - w) / 2; j < (720 - w) / 2 + w; j++) {
			ll temp = obj.at<Vec3b>(i - (720 - h) / 2, j - (720 - w) / 2)[0];
			sq.at<Vec3b>(i, j)[0] = temp;
			temp = obj.at<Vec3b>(i - (720 - h) / 2, j - (720 - w) / 2)[1];
			sq.at<Vec3b>(i, j)[1] = temp;
			temp = obj.at<Vec3b>(i - (720 - h) / 2, j - (720 - w) / 2)[2];
			sq.at<Vec3b>(i, j)[2] = temp;
		}
	}
	sq = rotate(sq, angle);
	Mat white_img = img.clone();//tạo ảnh trắng
	for (ll i = 0; i < 720; i++) {
		for (ll j = 0; j < 720; j++) {
			white_img.at<uchar>(i, j) = 255;
		}
	}
	tmp = Display_Serial(white_img, tmp);//lấy viền vết bẩn
	tmp = Display_Canny(tmp);//lọc cho viền mịn hơn
	//imshow("tmp", tmp);
	Mat matElement = getStructuringElement(MORPH_RECT, Size(1, 1));// công đoạn khoanh vùng vết bẩn
	morphologyEx(tmp, tmp, MORPH_OPEN, matElement);
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(tmp, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);
	drawContours(sq, contours, -1, Scalar(0, 255, 0), 6);
	sq = rotate(sq, -angle);
	imshow("My result", sq);
}