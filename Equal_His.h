#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <math.h>
#include<opencv2/opencv.hpp>
typedef int ll;

using namespace cv;
using namespace std;

void Frequency(Mat img);
void Frequency_Red(Mat img);
Mat Equalize_Hist_Red(Mat img);
void Frequency_Green(Mat img);
void Equalize_Hist_Greeb(Mat img);
void Frequency_Blue(Mat img);
void Equalize_Hist_Blue(Mat img);
void Equalization_Histogram(Mat img);