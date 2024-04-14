#include "Segmentation.h"

Mat Display_Segmentation(Mat image) {
	Mat image_gray(image.rows, image.cols, CV_8UC1);
	//cvtColor(image, image_gray, COLOR_BGR2GRAY);

	Mat binary(image.rows, image.cols, CV_8UC1);
	threshold(image_gray, binary, 40, 255, THRESH_BINARY | THRESH_OTSU);
	// Eliminate noise and smaller objects
	Mat fg;
	erode(binary, fg, Mat(), Point(-1, -1), 2);

	// Identify image pixels without objects
	Mat bg;
	dilate(binary, bg, Mat(), Point(-1, -1), 3);
	threshold(bg, bg, 1, 128, THRESH_BINARY_INV);

	// Create markers image
	Mat markers(binary.size(), CV_8U, Scalar(0));
	markers = fg + bg;

	markers.convertTo(markers, CV_32S);
	watershed(image, markers);

	markers.convertTo(markers, CV_8U);
	threshold(markers, markers, 40, 255, THRESH_BINARY | THRESH_OTSU);
	//imshow("Org", image);
	//imshow("Sau Segmentation", markers);
	return markers;
}