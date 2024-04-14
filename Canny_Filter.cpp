#include "Canny_Filter.h"

Mat Display_Canny(Mat img) {
	//cvtColor(img, img, COLOR_BGR2GRAY);
	Canny(img, img, 255 / 3, 255);
	return img;
}