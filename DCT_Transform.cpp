#include "DCT_Transform.h"
void DCT(Mat img) {
	const int m = 8, n = 8;
	int val, matrix[m][n];
	printf("Ma tran block truoc DCT \n");
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			matrix[i][j] = img.at<uchar>(i, j);
			printf("%d \t", matrix[i][j]);
		}
		printf("\n");
	}
	int i, j, k, l;
	float dct[m][n];
	float ci, cj, dct1, sum;
	Mat img_dct8(8, 8, CV_32F);
	printf("\nMa tran block sau DCT \n");
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			if (i == 0)
				ci = 1 / sqrt(m);
			else
				ci = sqrt(2) / sqrt(m);
			if (j == 0)
				cj = 1 / sqrt(n);
			else
				cj = sqrt(2) / sqrt(n);
			sum = 0;
			for (k = 0; k < m; k++) {
				for (l = 0; l < n; l++) {
					dct1 = matrix[k][l] *
						cos((2 * k + 1) * i * pi / (2 * m)) *
						cos((2 * l + 1) * j * pi / (2 * n));
					sum = sum + dct1;
				}
			}
			dct[i][j] = ci * cj * sum;
			img_dct8.at<float>(i, j) = dct[i][j];
			printf("%d\t", (int)dct[i][j]);
		}
		printf("\n");
	}
}