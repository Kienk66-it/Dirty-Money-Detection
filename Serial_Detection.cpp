#include "Serial_Detection.h"
#include "Sobel_Filter.h"
#include "Canny_Filter.h"

Mat Display_Serial(Mat img, Mat cop) {
	//Mat image_copy = img.clone();
	//cvtColor(img, img, COLOR_BGR2GRAY);
	//threshold
	//cop = Display_Sobel(cop);
	//threshold(cop, cop, 0, 255, THRESH_OTSU + THRESH_BINARY);

	//close morphology
	Mat matElement = getStructuringElement(MORPH_RECT, Size(7, 7));
	morphologyEx(cop, cop, MORPH_ERODE, matElement);

	//contours
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(cop, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);

	// draw contours on the original image
	drawContours(img, contours, -1, Scalar(0, 255, 0), 23);
	//imshow("None approximation", img);
	return img;
	waitKey(0);
	//imwrite("contours_none_image1.jpg", img);
}