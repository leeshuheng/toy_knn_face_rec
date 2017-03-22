// 2017年 03月 21日 星期二 15:49:57 CST
// author: 李小丹(Li Shao Dan) 字 殊恒(shuheng)
// K.I.S.S
// S.P.O.T

// credit: Opencv Manual: tutorial_face_main.html
// create csv: j=0;while [ $j -le 39 ]; do i=1;while [ $i -le 10 ]; do echo "/home/shuheng/文档/att_faces/s$((j+1))/$i.pgm;$j"; i=$((i+1)); done; j=$((j+1)); done > a.txt


#include <iostream>
#include <fstream>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>

using namespace std;
using namespace cv;
using namespace cv::ml;

void data_reshape(Mat &im)
{
	cvtColor(im, im, COLOR_BGR2GRAY);
	im.convertTo(im, CV_32F);
	im = im.reshape(0, 1);
}


// credit: http://docs.opencv.org/3.1.0/da/d60/tutorial_face_main.html
static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
	std::ifstream file(filename.c_str(), ifstream::in);
	if (!file) {
		string error_message = "No valid input file was given, please check the given filename.";
		CV_Error(Error::StsBadArg, error_message);
	}
	string line, path, classlabel;
	while (getline(file, line)) {
		stringstream liness(line);
		getline(liness, path, separator);
		getline(liness, classlabel);
		if(!path.empty() && !classlabel.empty()) {
			Mat im = imread(path);
			data_reshape(im);
			images.push_back(im);
			labels.push_back(atoi(classlabel.c_str()));
		}
	}
}

int main()
{
	vector<Mat> ims;
	vector<int> lbs;
	read_csv("/home/shuheng/data/att_faces/a.txt", ims, lbs);
	Mat lbv(lbs);
	lbv.convertTo(lbv, CV_32F);
	Mat imv(ims.size(), ims[0].cols, ims[0].type());

	for(int i = 0; i < ims.size(); ++i)
		ims[i].copyTo(imv.row(i));

	Ptr<KNearest> knnp = KNearest::create();
	Ptr<TrainData> tdp = TrainData::create(imv, ROW_SAMPLE, lbv);

	tdp->setTrainTestSplitRatio(0.9);
	std::cout << "Test/Train: " << tdp->getNTestSamples()
		<< "/" << tdp->getNTrainSamples() << endl;

	knnp->train(tdp);

	knnp->setDefaultK(5);
	knnp->setIsClassifier(true);

    Mat test = tdp->getTestSampleIdx();

	for(int i = 0; i < tdp->getNTestSamples(); ++i) {
		int idx = test.at<int>(i);
		int pred = knnp->predict(ims[idx]);
		cout << "Predict:   " << pred;
		cout << "  Actual: " << lbs[idx] << endl;
	}
	return 0;
}
