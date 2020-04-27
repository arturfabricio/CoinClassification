//Code by: Artur Fabrício, ROB4 Student @ AAU

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <iterator>
#include <algorithm>
#include "Coins.h"

using namespace cv;
using namespace std;

//Here we have two test images to run the program through.
Mat test1 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\TestData\\test1.jpg");
Mat test2 = imread("C:\\Users\\artur\\Desktop\\TrainingData\\TestData\\test2.jpg");

void main() {
	classifyData coins;
	coins.loadDataEuro();
	coins.loadDataKroner();
	coins.loadDataPence();
	coins.findCoins(test1);
	coins.Classify(test1);
};