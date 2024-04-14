#include "SIFT.h"

void SIFT_Processing() {
	//SIFT algorithm
	//-- Step 1: Detect the keypoints using SURF Detector
	Mat input1 = imread("Screenshot 2023-04-04 134100.png", IMREAD_GRAYSCALE);
	Mat input2 = imread("Screenshot 2023-04-04 134120.png", IMREAD_GRAYSCALE);
	ll h = input1.rows, w = input1.cols;
	resize(input1, input1, Size(w / 2, h / 2), INTER_LINEAR);
	resize(input2, input2, Size(w / 2, h / 2), INTER_LINEAR);

	Ptr<SiftFeatureDetector> detector1 = SiftFeatureDetector::create(), detector2 = SiftFeatureDetector::create();
	vector<KeyPoint> keypoints1, keypoints2;
	detector1->detect(input1, keypoints1);
	detector2->detect(input2, keypoints2);

	// Add results to image and save.
	Mat output1, output2;
	drawKeypoints(input1, keypoints1, output1);
	drawKeypoints(input2, keypoints2, output2);
	imshow("sift_result1.jpg", output1);
	imshow("sift_result2.jpg", output2);
	
	//Step2
	Ptr<SiftDescriptorExtractor> extractor = SiftDescriptorExtractor::create();
	Mat descriptors_1, descriptors_2;
	extractor->compute(input1, keypoints1, descriptors_1);
	extractor->compute(input2, keypoints2, descriptors_2);
	/*BFMatcher matcher(NORM_L2);
	vector< DMatch > matches;
	matcher.match(descriptors_1, descriptors_2, matches);*/

	//-- Draw matches
	/*Mat img_matches;
	drawMatches(input1, keypoints1, input2, keypoints2, matches, img_matches);*/

	////-- Show detected matches
	//imshow("Matches", img_matches);

	//step3
	FlannBasedMatcher matcher;
	vector< DMatch > matches;
	matcher.match(descriptors_1, descriptors_2, matches);

	double max_dist = 0; double min_dist = 100;

	//-- Quick calculation of max and min distances between keypoints
	for (int i = 0; i < descriptors_1.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	printf("-- Max dist : %f \n", max_dist);
	printf("-- Min dist : %f \n", min_dist);

	//-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
	//-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
	//-- small)
	//-- PS.- radiusMatch can also be used here.
	std::vector< DMatch > good_matches;

	for (int i = 0; i < descriptors_1.rows; i++)
	{
		if (matches[i].distance <= max(2 * min_dist, 0.02))
		{
			good_matches.push_back(matches[i]);
		}
	}

	//-- Draw only "good" matches
	Mat img_matches;
	drawMatches(input1, keypoints1, input2, keypoints2,
		good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	//-- Show detected matches
	imshow("Good Matches", img_matches);

	for (int i = 0; i < (int)good_matches.size(); i++)
	{
		printf("-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx);
	}
}