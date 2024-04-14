#include "Channels.h"
void display_channels(Mat img) {
	Mat rgb[3];
	split(img, rgb); // ham chia anh thanh tung thanh phan

	//tao cu so hien thi anh
	//Hien thi anh
	imshow("Original img", img);
	imshow("blue channel", rgb[0]);
	imshow("green channel", rgb[1]);
	imshow("red channel", rgb[2]);
}