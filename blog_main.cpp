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

class face_rec {
public:
	face_rec();
	face_rec(const char *);

public:
	int load(const char *);
	int model_save(const char *);
	int model_load(const char *);
	int set_train_ratio(double = 0.8);
	int get_sample_count() const;
	int get_train_count() const;
	Mat get_sample_idx() const;
	int set_default_k(int = 3);
	int train();
	inline float predict(InputArray, OutputArray = noArray(), int flags = 0);
	int test();

private:
	int __load(const char *);

private:
	Ptr<KNearest> knnp;
	Ptr<TrainData> tdp;
};

face_rec::face_rec()
:knnp()
{
}

face_rec::face_rec(const char *label_file)
:knnp(KNearest::create())
{
	knnp->setIsClassifier(true);
	knnp->setDefaultK(3);

	__load(label_file);
}

int face_rec::load(const char *label_file)
{
	if(!knnp.empty()) knnp.release();

	knnp = KNearest::create();

	knnp->setIsClassifier(true);
	knnp->setDefaultK(3);

	__load(label_file);
	return(0);
}

int face_rec::__load(const char *label_file)
{
	vector<Mat> ims;
	vector<int> lbs;

	read_csv(label_file, ims, lbs);

	Mat lbv(lbs);
	lbv.convertTo(lbv, CV_32F);
	Mat imv(ims.size(), ims[0].cols, ims[0].type());

	for(int i = 0; i < ims.size(); ++i)
		ims[i].copyTo(imv.row(i));

	tdp = TrainData::create(imv, ROW_SAMPLE, lbv);
	return 0;
}

int face_rec::model_save(const char *fn)
{
	knnp->save(fn);
	return 0;
}

int face_rec::model_load(const char *fn)
{
	if(!knnp.empty()) knnp.release();

	knnp = Algorithm::load<KNearest>(fn);
	return 0;
}

int face_rec::set_train_ratio(double train_ratio)
{
	tdp->setTrainTestSplitRatio(train_ratio);
	return 0;
}

int face_rec::set_default_k(int k)
{
	knnp->setDefaultK(k);
	return 0;
}

int face_rec::train()
{
	knnp->train(tdp);
	return 0;
}

int face_rec::get_sample_count() const
{
	return tdp->getNTestSamples();
}

int face_rec::get_train_count() const
{
	return tdp->getNTrainSamples();
}

Mat face_rec::get_sample_idx() const
{
	return tdp->getTestSampleIdx();
}

float face_rec::predict(InputArray in, OutputArray out, int flags)
{
	return knnp->predict(in, out, flags);
}


int face_rec::test()
{
	Mat test = this->get_sample_idx();
	for(int i = 0; i < this->get_sample_count(); ++i) {
		int idx = test.at<int>(i);
		int pred = this->predict(tdp->getSamples().row(idx));
		cout << "Predict:   " << pred;
		cout << "  Actual: " << tdp->getResponses().row(idx).at<float>() << endl;
	}
	return 0;
}

int main(int argc, char *argv[])
{
	if(argc < 2) return 1;

	face_rec fc(argv[1]);
	fc.set_train_ratio(0.9);

	std::cout << "Test/Train: " << fc.get_sample_count()
		<< "/" << fc.get_train_count() << endl;

	fc.train();

	fc.set_default_k(5);

	fc.test();

	fc.model_save("./model.xml");

	cout << "============================\n";

	face_rec frc;
	frc.model_load("./model.xml");
	// fc.model_load("./model.xml");

	vector<Mat> ims;
	vector<int> lbs;
	read_csv(argv[1], ims, lbs);

    Mat test = fc.get_sample_idx();

	for(int i = 0; i < test.cols; ++i) {
		cout << "Actual:  " << lbs[test.at<int>(i)]
			<< "   Predict:   " << frc.predict(ims[test.at<int>(i)]) << endl;
	}
	return 0;
}
