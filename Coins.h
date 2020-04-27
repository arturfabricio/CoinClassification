//Code by: Artur Fabrício, ROB4 Student @ AAU

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

//The main aim of this project is to sucessfully implement a k-nearest neighbour
//algorithm that can sucessfully classify 3 different coins: fifty pence, 5 kroner
//and 20 cents. In order to simpligy the code, these coins are refered to as pence,
//kroner and euro, respectively.

//For this code, I decided to organize the functions in two different classes.
//The first class, collectData, is concerned with performing some operations
//on the image, as well as outputting some relevant features that will be used
//further ahead in the code. The second class repeats the same image processing 
//methods of the function found in the previous class, but takes a diferent turn, 
//as it implements the k-nearest algorithm to detect which coin is which.

#ifndef COINS_H 
#define COINS_H

class collectData
{
public:
	//This function is concerned with processing the image in order to obtain features.
    void roughdata(Mat img);			 

	//Here we can find some of the variables used in the fuction above.
	struct CoinFeatures
	{
		int contourIndex;
		int area;
		int perimeter;
		float circularity;
		bool hasHole;
	};

	vector<vector<Point>> Gcontours;
	vector<CoinFeatures> featVec;
	vector<float> circularityVec;
	vector<float> areaVec;
	vector<float> perimeterVec;
	vector<float> circVec;
	vector<float> ratioVec;

private:
	/*
	Mat fig1 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\1.jpg");
	Mat fig2 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\2.jpg");
	Mat fig3 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\3.jpg");
	Mat fig4 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\4.jpg");
	Mat fig5 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\5.jpg");
	Mat fig6 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\6.jpg");
	Mat fig7 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\7.jpg");
	Mat fig8 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\8.jpg");
	Mat fig9 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\9.jpg");
	Mat fig10 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\10.jpg");
	Mat fig11 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\11.jpg");
	Mat fig12 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\12.jpg");
	Mat fig13 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\13.jpg");
	Mat fig14 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\14.jpg");
	Mat fig15 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\15.jpg");
	Mat fig16 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\16.jpg");
	Mat fig17 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\17.jpg");
	Mat fig18 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\18.jpg");
	Mat fig19 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\19.jpg");
	Mat fig20 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\20.jpg");
	Mat fig21 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\21.jpg");
	Mat fig22 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\22.jpg");
	Mat fig23 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\23.jpg");
	Mat fig24 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\24.jpg");
	Mat fig25 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\25.jpg");
	Mat fig26 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\26.jpg");
	Mat fig27 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\27.jpg");
	Mat fig28 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\28.jpg");
	Mat fig29 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\29.jpg");
	Mat fig30 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\30.jpg");
	Mat fig31 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\31.jpg");
	Mat fig32 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\32.jpg");
	Mat fig33 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\33.jpg");
	Mat fig34 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\34.jpg");
	Mat fig35 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\35.jpg");
	Mat fig36 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\36.jpg");
	Mat fig37 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\37.jpg");
	Mat fig38 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\38.jpg");
	Mat fig39 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\39.jpg");
	Mat fig40 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\40.jpg");
	Mat fig41 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\41.jpg");
	Mat fig42 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\42.jpg");
	Mat fig43 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\43.jpg");
	Mat fig44 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\44.jpg");
	Mat fig45 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\45.jpg");
	Mat fig46 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\46.jpg");
	Mat fig47 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\47.jpg");
	Mat fig48 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\48.jpg");
	Mat fig49 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\49.jpg");
	Mat fig50 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\50.jpg");
	Mat fig51 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\51.jpg");
	Mat fig52 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\52.jpg");
	Mat fig53 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\53.jpg");
	Mat fig54 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\54.jpg");
	Mat fig55 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\55.jpg");
	Mat fig56 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\56.jpg");
	Mat fig57 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\57.jpg");
	Mat fig58 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\58.jpg");
	Mat fig59 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\59.jpg");
	Mat fig60 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\60.jpg");
	Mat fig61 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\61.jpg");
	Mat fig62 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\62.jpg");
	Mat fig63 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\63.jpg");
	Mat fig64 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\64.jpg");
	Mat fig65 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\65.jpg");
	Mat fig66 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\66.jpg");
	Mat fig67 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\67.jpg");
	Mat fig68 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\68.jpg");
	Mat fig69 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\69.jpg");
	Mat fig70 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\70.jpg");
	Mat fig71 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\71.jpg");
	Mat fig72 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\72.jpg");
	Mat fig73 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\73.jpg");
	Mat fig74 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\74.jpg");
	Mat fig75 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\75.jpg");
	Mat fig76 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\76.jpg");
	Mat fig77 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\77.jpg");
	Mat fig78 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\78.jpg");
	Mat fig79 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\79.jpg");
	Mat fig80 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\80.jpg");
	Mat fig81 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\81.jpg");
	Mat fig82 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\82.jpg");
	Mat fig83 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\83.jpg");
	Mat fig84 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\84.jpg");
	Mat fig85 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\85.jpg");
	Mat fig86 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\86.jpg");
	Mat fig87 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\87.jpg");
	Mat fig88 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\88.jpg");
	Mat fig89 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\89.jpg");
	Mat fig90 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\90.jpg");
	Mat fig91 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\91.jpg");
	Mat fig92 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\92.jpg");
	Mat fig93 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\93.jpg");
	Mat fig94 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\94.jpg");
	Mat fig95 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\95.jpg");
	Mat fig96 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\96.jpg");
	Mat fig97 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\97.jpg");
	Mat fig98 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\98.jpg");
	Mat fig99 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\99.jpg");
	Mat fig100 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\100.jpg");
	Mat fig101 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\101.jpg");
	Mat fig102 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\102.jpg");
	Mat fig103 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\103.jpg");
	Mat fig104 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\104.jpg");
	Mat fig105 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\105.jpg");
	Mat fig106 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\106.jpg");
	Mat fig107 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\107.jpg");
	Mat fig108 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\108.jpg");
	Mat fig109 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\109.jpg");
	Mat fig110 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\110.jpg");
	Mat fig111 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\111.jpg");
	Mat fig112 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\112.jpg");
	Mat fig113 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\113.jpg");
	*/
	/*
	collectData data;
		 data.roughdata(data.fig1);
		 data.roughdata(data.fig2);
		 data.roughdata(data.fig3);
		 data.roughdata(data.fig4);
		 data.roughdata(data.fig5);
		 data.roughdata(data.fig6);
		 data.roughdata(data.fig7);
		 data.roughdata(data.fig8);
		 data.roughdata(data.fig9);
		 data.roughdata(data.fig10);
		 data.roughdata(data.fig11);
		 data.roughdata(data.fig12);
		 data.roughdata(data.fig13);
		 data.roughdata(data.fig14);
		 data.roughdata(data.fig15);
		 data.roughdata(data.fig16);
		 data.roughdata(data.fig17);
		 data.roughdata(data.fig18);
		 data.roughdata(data.fig19);
		 data.roughdata(data.fig20);
		 data.roughdata(data.fig21);
		 data.roughdata(data.fig22);
		 data.roughdata(data.fig23);
		 data.roughdata(data.fig24);
		 data.roughdata(data.fig25);
		 data.roughdata(data.fig26);
		 data.roughdata(data.fig27);
		 data.roughdata(data.fig28);
		 data.roughdata(data.fig29);
		 data.roughdata(data.fig30);
		 data.roughdata(data.fig31);
		 data.roughdata(data.fig32);
		 data.roughdata(data.fig33);
		 data.roughdata(data.fig34);
		 data.roughdata(data.fig35);
		 data.roughdata(data.fig36);
		 data.roughdata(data.fig37);
		 data.roughdata(data.fig38);
 */
};

class classifyData{

public:

	//Here we have 5 functions: the first three are responsible with loading the
	//respective data for each coin from text files which contain these values.
	//The function findCoins is similar to the function roughdata from the class above.
	//Finally, the Classify function implements a k-nearest algorithm to detect which coin
	//is which.

	void loadDataEuro();
	void loadDataKroner();
	void loadDataPence();
	void findCoins(Mat coins);
	void Classify(Mat coins);

private:

	//As with the other class, here we can find some of the variables
	//used whithin the functions above.

	struct CoinFeatures
	{
		int contourIndex;
		int area;
		int perimeter;
		float circularity;
		bool hasHole;
	};

	vector<vector<Point>> Gcontours;
	vector<CoinFeatures> featVec;
	vector<float> circularityVec;
	vector<float> areaVec;
	vector<float> perimeterVec;
	float coins3[3][3];
	float coinsArray[3][2];
	vector<float> circVec;
	vector<float> ratioVec;

	const int numRows = 117;
	const int numCols = 2;
	double euro[117][2];
	double kroner[117][2];
	double pence[117][2];

	Mat coins_resized;
	Mat Contour;
	Mat ContourIMG;
};

#endif 