//Code by: Artur Fabrício, ROB4 Student @ AAU

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include "Coins.h"
#include <fstream>
#include <sstream>

//Here we declare the functions used in the two classes.

using namespace cv;
using namespace std;

void collectData::roughdata(Mat img) {
	//First we check if the image is loaded properly
	if (img.empty()) cout << "Failed loading image. Check for errors." << endl;		
	else cout << "Image loaded succesfully!" << endl;

	//It is necessary to resize the images so that they fit on the screen. They are
	//reduced to 20% of their original size. The images are particularly big as they were
	//captured by using a smartphone.
	Mat coins_resized = Mat(img.size(), CV_8U);
	resize(img, coins_resized, Size(img.cols * 0.20, img.rows * 0.20), 0, 0, INTER_LINEAR);

	Mat coins_Grey = Mat(coins_resized.size(), CV_8U);
	Mat coins_Thresh = Mat(coins_resized.size(), CV_8U);
	cvtColor(coins_resized, coins_Grey, COLOR_BGR2GRAY);
	threshold(coins_Grey, coins_Thresh, 70, 255, THRESH_BINARY);

	//Here an opening (erosion followed by dilation) is performed. This method
	//is used in order to eliminate some of the excess light present in the thresholding
	//previously performed. The value of the Structuring Element (SE) is small (1x1) in
	//order to avoid eroding the coins themselves two much, perserving their circularity.
	//This is one of the features later used.
	Mat coins_morph = Mat(coins_resized.size(), CV_8U);
	Mat ElemOpencoins = getStructuringElement(MORPH_ELLIPSE, Size(1, 1));
	morphologyEx(coins_Thresh, coins_morph, MORPH_OPEN, ElemOpencoins);

	//A dilate is now performed, in order to increase the size of the coins.
	//The SE size of 5x5 was the one that yielded the most satisfactory results.
	Mat coins_morph1 = Mat(coins_resized.size(), CV_8U);
	Mat ElemDilatecoins = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
	morphologyEx(coins_morph, coins_morph1, MORPH_DILATE, ElemDilatecoins);

	//Sometimes, even after the opening, not all the noise is removed. With that in mind,
	//after applying the dilate operation, that surviving noise becomes bigger. Even thought
	//this rarely afects the coins themselves (as this noise is usually present in the corners)
	//an erote operation is performed in order to get rid of the majority of this noise, providing
	//(hopefully) an image with only three BLOBs: one for each coin.
	Mat coins_morph2 = Mat(coins_resized.size(), CV_8U);
	Mat ElemErotecoins = getStructuringElement(MORPH_ELLIPSE, Size(9, 9));
	morphologyEx(coins_morph1, coins_morph2, MORPH_ERODE, ElemErotecoins);

	Mat Contour = Mat(coins_resized.size(), CV_8U);
	Mat ContourIMG = Mat(coins_resized.size(), CV_8U, Scalar(255, 255, 255));

	vector<Vec4i> hierarchy;

	//The countors are now drawn for the BLOBs present in the final image.
	findContours(coins_morph2, Gcontours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);
	drawContours(Contour, Gcontours, -1, Scalar(0, 0, 0), 1);

	for (int i = 0; i < Gcontours.size(); i++)
	{
		if (hierarchy[i][3] == -1)
		{
			//Now some of the features are extracted, being that the most relevant ones
			//are the area, perimeter and circularity.

			CoinFeatures G;
			G.contourIndex = i;
			G.area = contourArea(Gcontours[i]);
			G.perimeter = arcLength(Gcontours[i], true);
			G.circularity = (4 * 3.14 * G.area) / pow(G.perimeter, 2);
			G.hasHole = (hierarchy[i][2] == -1) ? false : true;

			RotatedRect box = minAreaRect(Gcontours[i]);

			//cout << "G.Area:" << G.area << "; G.Perimeter:" << G.perimeter << "; G.Circularity:" << G.circularity << "; G.Hole:" << G.hasHole << endl;

			float circularity = G.circularity;
			int area = G.area;
			int perimeter = G.perimeter;
			featVec.push_back(G);
			
			//All the relevant features are stored in three distinct vectors.
			circularityVec.push_back(G.circularity);
			areaVec.push_back(G.area);
			perimeterVec.push_back(G.perimeter);
		}
	}
		/*
		imshow("Resized Image", coins_resized);
		waitKey(0);
		destroyAllWindows;
		imshow("Thresholding Image", coins_Thresh);
		waitKey(0);
		destroyAllWindows;
		imshow("Morphology Image", coins_morph);
		waitKey(0);
		destroyAllWindows;
		imshow("Morphology Image1", coins_morph1);
		waitKey(0);
		destroyAllWindows;
		imshow("Morphology Image2", coins_morph2);
		waitKey(0);
		destroyAllWindows;
		imshow("Contours", Contour);
		waitKey(0);
		destroyAllWindows;
		system("CLS");
		*/
};

void classifyData::loadDataEuro() {
	//This function is responsible with loading the features from a text file, where they where stored.
	std::ifstream in("DataEuro.txt");
	for (int i = 0; i < numRows; i++) {
		for (int j = 0; j < numCols; j++) {
			//The area was devided by the perimeter, yielding a new feature: ratio.
			//This data is stored in the array euro;
			in >> euro[i][j];
		}
		//Here the data is normalized, since the circularity value varies between
		//0.7 and 1, and the ratio between 25-50.
		double value = euro[i][1] / 45;
		euro[i][1] = value;
		double value2 = ((euro[i][0] - 0.827443) / (0.900933 - 0.827443));
		euro[i][0] = value2;
	}
	//The same logic is applied to the other loadData functions, just for the kroner and the pence.
}

void classifyData::loadDataKroner() {
	std::ifstream in("DataKroner.txt");
	for (int i = 0; i < numRows; i++) {
		for (int j = 0; j < numCols; j++) {
			in >> kroner[i][j];
		}
		double value = kroner[i][1] / 45;
		kroner[i][1] = value;
		double value2 = ((kroner[i][0] - 0.827443) / (0.900933 - 0.827443));
		kroner[i][0] = value2;
	}
}

void classifyData::loadDataPence() {
	std::ifstream in("DataPence.txt");
	for (int i = 0; i < numRows; i++) {
		for (int j = 0; j < numCols; j++) {
			in >> pence[i][j];
		}
		double value = pence[i][1] / 45;
		pence[i][1] = value;
		double value2 = ((pence[i][0] - 0.827443) / (0.900933 - 0.827443));
		pence[i][0] = value2;
	}
}

void classifyData::findCoins(Mat coins) {

	//Almost the entirity of this function performs exactly the same operations
	//as the roughdata function from the previous class. The only difference 
	//can be found at the end of the function.

	if (coins.empty()) cout << "Failed loading image. Check for errors." << endl;
	else cout << "Image loaded succesfully!" << endl;
	Mat coins_resized = Mat(coins.size(), CV_8U);
	resize(coins, coins_resized, Size(coins.cols * 0.20, coins.rows * 0.20), 0, 0, INTER_LINEAR);

	Mat coins_Grey = Mat(coins_resized.size(), CV_8U);
	Mat coins_Thresh = Mat(coins_resized.size(), CV_8U);
	cvtColor(coins_resized, coins_Grey, COLOR_BGR2GRAY);
	threshold(coins_Grey, coins_Thresh, 70, 255, THRESH_BINARY);

	Mat coins_morph = Mat(coins_resized.size(), CV_8U);
	Mat ElemOpencoins = getStructuringElement(MORPH_ELLIPSE, Size(1, 1));
	morphologyEx(coins_Thresh, coins_morph, MORPH_OPEN, ElemOpencoins);

	Mat coins_morph1 = Mat(coins_resized.size(), CV_8U);
	Mat ElemDilatecoins = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
	morphologyEx(coins_morph, coins_morph1, MORPH_DILATE, ElemDilatecoins);

	Mat coins_morph2 = Mat(coins_resized.size(), CV_8U);
	Mat ElemErotecoins = getStructuringElement(MORPH_ELLIPSE, Size(9, 9));
	morphologyEx(coins_morph1, coins_morph2, MORPH_ERODE, ElemErotecoins);

	Mat Contour = Mat(coins_resized.size(), CV_8U);
	Mat ContourIMG = Mat(coins_resized.size(), CV_8U, Scalar(255, 255, 255));

	vector<Vec4i> hierarchy;

	findContours(coins_morph2, Gcontours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);
	drawContours(Contour, Gcontours, -1, Scalar(0, 0, 0), 1);

	for (int i = 0; i < Gcontours.size(); i++)
	{
		if (hierarchy[i][3] == -1)
		{
			CoinFeatures G;
			G.contourIndex = i;
			G.area = contourArea(Gcontours[i]);
			G.perimeter = arcLength(Gcontours[i], true);
			G.circularity = (4 * 3.14 * G.area) / pow(G.perimeter, 2);
			G.hasHole = (hierarchy[i][2] == -1) ? false : true;

			RotatedRect box = minAreaRect(Gcontours[i]);

			//cout << "G.Area:" << G.area << "; G.Perimeter:" << G.perimeter << "; G.Circularity:" << G.circularity << "; G.Hole:" << G.hasHole << endl;

			float circularity = G.circularity;
			int area = G.area;
			int perimeter = G.perimeter;
			featVec.push_back(G);
			circularityVec.push_back(G.circularity);
			areaVec.push_back(G.area);
			perimeterVec.push_back(G.perimeter);
		}
	}

	//Here we select the BLOBs that present a Circularity superior to 0.75
	//and an area superior to 300. These conditions are valid for all the
	//coins/data collected, and allow the program to know which values to
	//use in the k-nearest neighbour algorithm.
	for (int i = 0; i < featVec.size(); i++) {
		if (circularityVec[i] > 0.75 && areaVec[i] > 300) {
			double ratio = areaVec[i] / perimeterVec[i];
			double circularity = circularityVec[i];
			circVec.push_back(circularity);
			ratioVec.push_back(ratio);
		}
	}

	/*
		imshow("Resized Image", coins_resized);
		waitKey(0);
		destroyAllWindows;
		imshow("Thresholding Image", coins_Thresh);
		waitKey(0);
		destroyAllWindows;
		imshow("Morphology Image", coins_morph);
		waitKey(0);
		destroyAllWindows;
		imshow("Morphology Image1", coins_morph1);
		waitKey(0);
		destroyAllWindows;
		imshow("Morphology Image2", coins_morph2);
		waitKey(0);
		destroyAllWindows;
		imshow("Contours", Contour);
		waitKey(0);
		destroyAllWindows;
	*/
};

void classifyData::Classify(Mat coins) {

	//This is the function in which we use the k-nearest algorithm. We also
	//reduce by 80% the loaded image.

	Mat coins_resized = Mat(coins.size(), CV_8U);
	resize(coins, coins_resized, Size(coins.cols * 0.20, coins.rows * 0.20), 0, 0, INTER_LINEAR);
	Mat everything = Mat(coins_resized.size(), CV_8UC3);
	Mat drawing = Mat::zeros(coins_resized.size(), CV_8UC3);
	int k;

	//This function allows for the user to select the k value for which he would desire the algorithm
	//to use. That k value can be any int between 1 and 11.

	cout << "To how many neigbhbours would you like to look at? (Choose a number between 1 and 11): ";
	cin >> k;
	cout << " " << endl;
	if (k > 11 || k < 1) {
		cout << "Please try again with a number between 1 and 11: ";
		cin >> k;
		cout << " " << endl;
	}
	else {
		cout << "The algorithm will look at the " << k << " nearest neighbours." << endl;
		cout << "  " << endl;
	}
	
	//After that we enter in a for loop for each BLOB previously detected.
	for (int o = 0; o < ratioVec.size(); o++) {

		//All the values are normalized.
		double x = ((circVec[o] - 0.827443) / (0.900933 - 0.827443));
		double y = ratioVec[o] / 45;

		vector<double> distancesEuro;
		vector<double> distancesKroner;
		vector<double> distancesPence;
		vector<double> all;
		vector<int> allks;

		//The distance between the inputed features is calculated between itself and the loaded data from
		//the euros, kroner and pence. They are all stored in a vector for each type of coin, and a general
		//which contains all the distances.

		for (int i = 0; i < 86; i++) {
			double dist;
			dist = sqrt(((x - euro[i][0]) * (x - euro[i][0])) + ((y - euro[i][1]) * (y - euro[i][1])));
			all.push_back(dist);
			distancesEuro.push_back(dist);
		};

		for (int i = 0; i < 86; i++) {
			double dist;
			dist = sqrt(((x - kroner[i][0]) * (x - kroner[i][0])) + ((y - kroner[i][1]) * (y - kroner[i][1])));
			distancesKroner.push_back(dist);
			all.push_back(dist);
		};

		for (int i = 0; i < 86; i++) {
			double dist;
			dist = sqrt(((x - pence[i][0]) * (x - pence[i][0])) + ((y - pence[i][1]) * (y - pence[i][1])));
			distancesPence.push_back(dist);
			all.push_back(dist);
		};

		int i;
		double first, second, third, fourth, fifth, sixth, seventh, eight, ninth, tenth, eleventh;
		first, second, third, fourth, fifth, sixth, seventh, eight, ninth, tenth, eleventh = INT_MAX;

		//Taking into account the value k inputed by the user, the code will enter the following if statements.
		if (k >= 1) {
			first = INT_MAX;
			for (i = 0; i < all.size(); i++)
			{
				if (all[i] < first)
				{
					//Inside this if statement, the smallest distance from the calculated before is found. 
					first = all[i];
				}
			};
			//That value is then deleted from the general vector.
			vector<double>::iterator it = remove(all.begin(), all.end(), first);
			all.erase(it, all.end());

			//The value is then compared to the distances for each specific coin, and these bool determine
			//to which coin vector that distance belong.
			bool existsEuro = std::find(std::begin(distancesEuro), std::end(distancesEuro), first) != std::end(distancesEuro);
			bool existsKroner = std::find(std::begin(distancesKroner), std::end(distancesKroner), first) != std::end(distancesKroner);
			bool existsPence = std::find(std::begin(distancesPence), std::end(distancesPence), first) != std::end(distancesPence);

			//If the distance found belongs to the euro vector, a 0 is pushed into the allks vector.
			//If it belongs to kroner, a 1 is inputed, and if its a pence, a 2 is inputed.
			if (existsEuro == 1) {
				allks.push_back(0);
			}

			if (existsKroner == 1) {
				allks.push_back(1);
			}

			if (existsPence == 1) {
				allks.push_back(2);
			}
			//This process is then repeated however many times the value k corresponds to.
		};

		if (k >= 2) {
			second = INT_MAX;
			for (i = 0; i < all.size(); i++)
			{
				if (all[i] < second)
				{
					second = all[i];
				}
			};
			vector<double>::iterator it = remove(all.begin(), all.end(), second);
			all.erase(it, all.end());
			bool existsEuro = std::find(std::begin(distancesEuro), std::end(distancesEuro), second) != std::end(distancesEuro);
			bool existsKroner = std::find(std::begin(distancesKroner), std::end(distancesKroner), second) != std::end(distancesKroner);
			bool existsPence = std::find(std::begin(distancesPence), std::end(distancesPence), second) != std::end(distancesPence);

			if (existsEuro == 1) {
				allks.push_back(0);
			}

			if (existsKroner == 1) {
				allks.push_back(1);
			}

			if (existsPence == 1) {
				allks.push_back(2);
			}
		};

		if (k >= 3) {
			third = INT_MAX;
			for (i = 0; i < all.size(); i++)
			{
				if (all[i] < third)
				{
					third = all[i];
				}
			};
			vector<double>::iterator it = remove(all.begin(), all.end(), third);
			all.erase(it, all.end());
			bool existsEuro = std::find(std::begin(distancesEuro), std::end(distancesEuro), third) != std::end(distancesEuro);
			bool existsKroner = std::find(std::begin(distancesKroner), std::end(distancesKroner), third) != std::end(distancesKroner);
			bool existsPence = std::find(std::begin(distancesPence), std::end(distancesPence), third) != std::end(distancesPence);

			if (existsEuro == 1) {
				allks.push_back(0);
			}

			if (existsKroner == 1) {
				allks.push_back(1);
			}

			if (existsPence == 1) {
				allks.push_back(2);
			}
		};

		if (k >= 4) {
			fourth = INT_MAX;
			for (i = 0; i < all.size(); i++)
			{
				if (all[i] < fourth)
				{
					fourth = all[i];
				}
			};
			vector<double>::iterator it = remove(all.begin(), all.end(), fourth);
			all.erase(it, all.end());
			bool existsEuro = std::find(std::begin(distancesEuro), std::end(distancesEuro), fourth) != std::end(distancesEuro);
			bool existsKroner = std::find(std::begin(distancesKroner), std::end(distancesKroner), fourth) != std::end(distancesKroner);
			bool existsPence = std::find(std::begin(distancesPence), std::end(distancesPence), fourth) != std::end(distancesPence);

			if (existsEuro == 1) {
				allks.push_back(0);
			}

			if (existsKroner == 1) {
				allks.push_back(1);
			}

			if (existsPence == 1) {
				allks.push_back(2);
			}
		};

		if (k >= 5) {
			fifth = INT_MAX;
			for (i = 0; i < all.size(); i++)
			{
				if (all[i] < fifth)
				{
					fifth = all[i];
				}
			};
			vector<double>::iterator it = remove(all.begin(), all.end(), fifth);
			all.erase(it, all.end());
			bool existsEuro = std::find(std::begin(distancesEuro), std::end(distancesEuro), fifth) != std::end(distancesEuro);
			bool existsKroner = std::find(std::begin(distancesKroner), std::end(distancesKroner), fifth) != std::end(distancesKroner);
			bool existsPence = std::find(std::begin(distancesPence), std::end(distancesPence), fifth) != std::end(distancesPence);

			if (existsEuro == 1) {
				allks.push_back(0);
			}

			if (existsKroner == 1) {
				allks.push_back(1);
			}

			if (existsPence == 1) {
				allks.push_back(2);
			}
		};

		if (k >= 6) {
			sixth = INT_MAX;
			for (i = 0; i < all.size(); i++)
			{
				if (all[i] < sixth)
				{
					sixth = all[i];
				}
			};
			vector<double>::iterator it = remove(all.begin(), all.end(), sixth);
			all.erase(it, all.end());
			bool existsEuro = std::find(std::begin(distancesEuro), std::end(distancesEuro), sixth) != std::end(distancesEuro);
			bool existsKroner = std::find(std::begin(distancesKroner), std::end(distancesKroner), sixth) != std::end(distancesKroner);
			bool existsPence = std::find(std::begin(distancesPence), std::end(distancesPence), sixth) != std::end(distancesPence);

			if (existsEuro == 1) {
				allks.push_back(0);
			}

			if (existsKroner == 1) {
				allks.push_back(1);
			}

			if (existsPence == 1) {
				allks.push_back(2);
			}
		};

		if (k >= 7) {
			seventh = INT_MAX;
			for (i = 0; i < all.size(); i++)
			{
				if (all[i] < seventh)
				{
					seventh = all[i];
				}
			};
			vector<double>::iterator it = remove(all.begin(), all.end(), seventh);
			all.erase(it, all.end());
			bool existsEuro = std::find(std::begin(distancesEuro), std::end(distancesEuro), seventh) != std::end(distancesEuro);
			bool existsKroner = std::find(std::begin(distancesKroner), std::end(distancesKroner), seventh) != std::end(distancesKroner);
			bool existsPence = std::find(std::begin(distancesPence), std::end(distancesPence), seventh) != std::end(distancesPence);

			if (existsEuro == 1) {
				allks.push_back(0);
			}

			if (existsKroner == 1) {
				allks.push_back(1);
			}

			if (existsPence == 1) {
				allks.push_back(2);
			}
		};

		if (k >= 8) {
			eight = INT_MAX;
			for (i = 0; i < all.size(); i++)
			{
				if (all[i] < eight)
				{
					eight = all[i];
				}
			};
			vector<double>::iterator it = remove(all.begin(), all.end(), eight);
			all.erase(it, all.end());
			bool existsEuro = std::find(std::begin(distancesEuro), std::end(distancesEuro), eight) != std::end(distancesEuro);
			bool existsKroner = std::find(std::begin(distancesKroner), std::end(distancesKroner), eight) != std::end(distancesKroner);
			bool existsPence = std::find(std::begin(distancesPence), std::end(distancesPence), eight) != std::end(distancesPence);

			if (existsEuro == 1) {
				allks.push_back(0);
			}

			if (existsKroner == 1) {
				allks.push_back(1);
			}

			if (existsPence == 1) {
				allks.push_back(2);
			}
		};

		if (k >= 9) {
			ninth = INT_MAX;
			for (i = 0; i < all.size(); i++)
			{
				if (all[i] < ninth)
				{
					ninth = all[i];
				}
			};
			vector<double>::iterator it = remove(all.begin(), all.end(), ninth);
			all.erase(it, all.end());
			bool existsEuro = std::find(std::begin(distancesEuro), std::end(distancesEuro), ninth) != std::end(distancesEuro);
			bool existsKroner = std::find(std::begin(distancesKroner), std::end(distancesKroner), ninth) != std::end(distancesKroner);
			bool existsPence = std::find(std::begin(distancesPence), std::end(distancesPence), ninth) != std::end(distancesPence);

			if (existsEuro == 1) {
				allks.push_back(0);
			}

			if (existsKroner == 1) {
				allks.push_back(1);
			}

			if (existsPence == 1) {
				allks.push_back(2);
			}
		};

		if (k >= 10) {
			tenth = INT_MAX;
			for (i = 0; i < all.size(); i++)
			{
				if (all[i] < tenth)
				{
					tenth = all[i];
				}
			};
			vector<double>::iterator it = remove(all.begin(), all.end(), tenth);
			all.erase(it, all.end());
			bool existsEuro = std::find(std::begin(distancesEuro), std::end(distancesEuro), tenth) != std::end(distancesEuro);
			bool existsKroner = std::find(std::begin(distancesKroner), std::end(distancesKroner), tenth) != std::end(distancesKroner);
			bool existsPence = std::find(std::begin(distancesPence), std::end(distancesPence), tenth) != std::end(distancesPence);

			if (existsEuro == 1) {
				allks.push_back(0);
			}

			if (existsKroner == 1) {
				allks.push_back(1);
			}

			if (existsPence == 1) {
				allks.push_back(2);
			}
		};

		if (k >= 11) {
			eleventh = INT_MAX;
			for (i = 0; i < all.size(); i++)
			{
				if (all[i] < eleventh)
				{
					eleventh = all[i];
				}
			};
			vector<double>::iterator it = remove(all.begin(), all.end(), eleventh);
			all.erase(it, all.end());
			bool existsEuro = std::find(std::begin(distancesEuro), std::end(distancesEuro), eleventh) != std::end(distancesEuro);
			bool existsKroner = std::find(std::begin(distancesKroner), std::end(distancesKroner), eleventh) != std::end(distancesKroner);
			bool existsPence = std::find(std::begin(distancesPence), std::end(distancesPence), eleventh) != std::end(distancesPence);

			if (existsEuro == 1) {
				allks.push_back(0);
			}

			if (existsKroner == 1) {
				allks.push_back(1);
			}

			if (existsPence == 1) {
				allks.push_back(2);
			}
		};


		//After this, the number of times the values appear in the allks vector are detected.
		int eurocount = std::count(allks.begin(), allks.end(), 0);
		int kronercount = std::count(allks.begin(), allks.end(), 1);
		int pencecount = std::count(allks.begin(), allks.end(), 2);

		int kroner;
		int euro;
		int pence;

		euro = kroner = pence = INT_MAX;

		//If the euro is the most frequent, it enters the following if statement.
		if (eurocount > kronercount && eurocount > pencecount) {
			cout << "For values " << circVec[o] << " and " << ratioVec[o] << " the coin is an euro! (Red box)" << endl;

			for (int i = 0; i < featVec.size(); i++)
			{
				if (featVec[i].circularity == circVec[o])
				{
					//The coin has been identified as an euro. 
					drawContours(ContourIMG, Gcontours, featVec[i].contourIndex, Scalar(0, 255, 0), 1);
					euro = featVec[i].contourIndex;
					vector<Rect> boundeuro(featVec.size());
					for (int i = 0; i < featVec.size(); i++)
					{
						boundeuro[i] = boundingRect(Mat(Gcontours[euro]));
					}
					for (int i = 0; i < featVec.size(); i++)
					{
						//As such, a bounding box is drawn around the corresponding coin, as well as
						//some text that identifies said coin.
						Scalar red = Scalar(0, 0, 255);
						rectangle(drawing, boundeuro[i].tl(), boundeuro[i].br(), red, 1, 8, 0);
						cv::putText(drawing, "Euro", Point(boundeuro[i].x, boundeuro[i].y + boundeuro[i].width + 25), FONT_HERSHEY_SIMPLEX, 1, CV_RGB(255, 0, 0), 2);

					}
				}

			}
			//The same logic is applied for the other two coin types.
		}

		if (kronercount > eurocount && kronercount > pencecount) {
			cout << "For values " << circVec[o] << " and " << ratioVec[o] << " the coin is a kroner! (Green Box)" << endl;

			for (int i = 0; i < featVec.size(); i++)
			{
				if (featVec[i].circularity == circVec[o])
				{
					drawContours(ContourIMG, Gcontours, featVec[i].contourIndex, Scalar(0, 255, 0), 1);
					kroner = featVec[i].contourIndex;
					vector<Rect> boundkroner(featVec.size());
					for (int i = 0; i < featVec.size(); i++)
					{
						boundkroner[i] = boundingRect(Mat(Gcontours[kroner]));
					}
					for (int i = 0; i < featVec.size(); i++)
					{
						Scalar green = Scalar(0, 255, 0);
						rectangle(drawing, boundkroner[i].tl(), boundkroner[i].br(), green, 1, 8, 0);
						cv::putText(drawing, "Kroner", Point(boundkroner[i].x, boundkroner[i].y + boundkroner[i].width + 25), FONT_HERSHEY_SIMPLEX, 1, CV_RGB(0, 255, 0), 2);
					}
				}

			}

		}

		if (pencecount > kronercount && pencecount > eurocount) {
			cout << "For values " << circVec[o] << " and " << ratioVec[o] << " the coin is a pence! (Blue Box)" << endl;

			for (int i = 0; i < featVec.size(); i++)
			{
				if (featVec[i].circularity == circVec[o])
				{
					drawContours(ContourIMG, Gcontours, featVec[i].contourIndex, Scalar(0, 255, 0), 1);
					pence = featVec[i].contourIndex;
					vector<Rect> boundpence(featVec.size());
					for (int i = 0; i < featVec.size(); i++) {
						boundpence[i] = boundingRect(Mat(Gcontours[pence]));
					}
					for (int i = 0; i < featVec.size(); i++)
					{
						Scalar blue = Scalar(255, 0, 0);
						rectangle(drawing, boundpence[i].tl(), boundpence[i].br(), blue, 1, 8, 0);
						cv::putText(drawing, "Pence", Point(boundpence[i].x, boundpence[i].y + boundpence[i].width + 25), FONT_HERSHEY_SIMPLEX, 1, CV_RGB(0, 0, 255), 2);
					}
				}

			}

		}

		everything = coins_resized + drawing;
	}
	//In the end, we output eveything (the original image with the bounding boxes classifying each coin).
	imshow("Final Result", everything);
	waitKey(0);
	destroyAllWindows;
}
