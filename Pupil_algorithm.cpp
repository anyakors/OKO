#if 0

#include"opencv2/opencv.hpp"
#include"opencv2/core/core.hpp"
#include"opencv2/highgui/highgui.hpp"
#include"opencv2/imgproc/imgproc.hpp"
#include"iostream"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

using namespace cv;
using namespace std;

const cv::Scalar blue(255, 0, 0),  green(0, 255, 0);

inline int GetElapsedMillis(int64 start, int64 end = 0) {
	static const double freq = cv::getTickFrequency() / 1000;
	if (end == 0)
		end = cv::getTickCount();
	return round((end - start) / freq);
}

std::string FormatTime(double millis);

inline std::string GetElapsedTime(int64 start, int64 end = 0) {
	return FormatTime(GetElapsedMillis(start, end));
}

std::string FormatTime(double millis) {
	const int kSecond = 1000, kMinute = 60 * kSecond, kHour = 60 * kMinute;
	if (millis >= kHour)
		return cv::format("%d h %d m", int(millis / kHour), (int(millis) % kHour) / kMinute);
	else if (millis >= kMinute)
		return cv::format("%.3f m", millis / kMinute);
	else if (millis >= kSecond)
		return cv::format("%.2f s", millis / kSecond);
	else
		return cv::format("%.1f ms", millis);
}


Mat norm_0_255(const Mat& src) {
    // Create and return normalized image:
    Mat dst;
    switch(src.channels()) {
    case 1:
        cv::normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
        break;
    case 3:
        cv::normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
        break;
    default:
        src.copyTo(dst);
        break;
    }
    return dst;
}

int main()
{
	Mat src, src_gray, mask_1;
	vector<Mat> arr;
	list<Mat> masks;
	Rect rect;
	int focus_par = 10, width, height, mask_bool = 0, minsize = 40, maxsize =
			200, thrsh = 100;

	Point cent;

	//for ( int device = 0; device < 10; device++ ) {
	cv::VideoCapture cap(1 /*device*/);
	if (!cap.isOpened())
		return (0);
	//}
	cap >> src;
	double iratio = 100.0 / src.rows;

	width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
	height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);

	namedWindow("Preview", CV_WINDOW_AUTOSIZE);
	namedWindow("Preview1", CV_WINDOW_AUTOSIZE);
	namedWindow("Mask", CV_WINDOW_AUTOSIZE);
	//namedWindow("Correlation", CV_WINDOW_AUTOSIZE);
	cv::Mat result;
	cv::Mat mask = cv::imread("/home/gunmachine/workspace/pca_1/1.png", CV_LOAD_IMAGE_GRAYSCALE);
	imshow("Mask", mask);
	double mratio = 100.0 / mask.rows;

	cv::resize(mask, mask_1, Size(0.5*mask.cols, 0.5*mask.rows), 0, 0, INTER_AREA);

	/// Localizing the best match with minMaxLoc
	vector<double> minVal(6);
	vector<double> maxVal(6);
	vector<Point> minLoc(6);
	vector<Point> maxLoc(6);
	Point matchLoc;
	int minCorr[2], maxCorr[2];
	double min, max;

	for (;;) {

		cap >> src;
		cv::Mat src_1;

		/// Convert it to gray
		cvtColor(src, src_gray, CV_BGR2GRAY);
		//cvtColor( src, src_gray, CV_RGB2HSV);

		cv::resize(src_gray, src_1, Size(0.5*src_gray.cols, 0.5*src_gray.rows), 0, 0, INTER_AREA);

		int64 start = cv::getTickCount();
		string time;

		//cv::split(src, arr);
		//src_gray = arr[2];
		//equalizeHist( src_gray, src_gray );

		//GaussianBlur( src_gray, src_gray, Size(9, 9), 2, 2 );

		for (int ratio = 10; ratio > 4; ratio--) {

			float fratio = float(ratio/10.0);
			cv::Mat mask_;
			cv::resize(mask_1, mask_, Size(), fratio, fratio, INTER_AREA);

			int result_cols = src_1.cols - mask_.cols + 1;
			int result_rows = src_1.rows - mask_.rows + 1;

			result.create(result_cols, result_rows, CV_32FC1);

			/// Do the Matching and Normalize
			matchTemplate(src_1, mask_, result, CV_TM_CCOEFF_NORMED);

			minMaxLoc(result, &(minVal[10-ratio]), &(maxVal[10-ratio]), &(minLoc[10-ratio]), &(maxLoc[10-ratio]));
			//minLoc[10-ratio].x = (minLoc[10-ratio].x)/fratio; minLoc[10-ratio].y = (minLoc[10-ratio].y)/fratio;
			//maxLoc[10-ratio].x = (maxLoc[10-ratio].x)/fratio; maxLoc[10-ratio].y = (maxLoc[10-ratio].y)/fratio;

		}

	  minMaxIdx(maxVal, &min, &max, minCorr, maxCorr);
	  //cout << "0 comp - " << maxCorr[0] << " ; ------ 1 comp - " << maxCorr[1] << endl;
	  /// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
	  matchLoc = maxLoc[maxCorr[1]];
	  float fratio = (10.0 - maxCorr[1])/10.0;
	  float coeff = (src.cols-mask.cols+1)/(src.cols*iratio-mask.cols*mratio*fratio+1);
	  //matchLoc.x = matchLoc.x*coeff; matchLoc.y = matchLoc.y/coeff;
	  //cout << result.at<float>(matchLoc.y, matchLoc.x) << endl;

	  /// Show me what you got
	  if (max>0.60) {
		  rectangle(src_1, matchLoc, Point( matchLoc.x + mask_1.cols*fratio , matchLoc.y + mask_1.rows*fratio ), green, 2, 8, 0 );
	  }
	  else {
		  rectangle(src_1, Point(0,0), Point( src_1.cols, src_1.rows), green, 4, 8, 0 );
	  }
	  //rectangle( result, matchLoc, Point( matchLoc.x + mask.cols , matchLoc.y + mask.rows ), green, 2, 8, 0 );

	  //imshow( "Preview", src );
	  imshow( "Preview", norm_0_255(result));
	  imshow( "Preview1", src_1);
	  //imshow( "Gray", result );

	  int64 end = cv::getTickCount();
	  time = GetElapsedTime( start, end );
	  cout << time << endl;

  	  int key = cv::waitKey(1) & 255;
  	  if (key == 27)
			break;

  }

  for (;;) {

	  cap >> src;

	  //cv::split(src, arr);
	  //src_gray = arr[2];
	  cvtColor( src, src_gray, CV_BGR2GRAY );

	  int result_cols =  src_gray.cols - mask.cols + 1;
	  int result_rows = src_gray.rows - mask.rows + 1;

	  result.create( result_cols, result_rows, CV_32FC1 );

	  /// Do the Matching and Normalize
	  matchTemplate( src_gray, mask, result, CV_TM_CCOEFF_NORMED );

	  /// Localizing the best match with minMaxLoc
	  double minVal; double maxVal; Point minLoc; Point maxLoc;
	  Point matchLoc;

	  minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc );

	  /// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
	  matchLoc = maxLoc;
	  //cout << result.at<float>(matchLoc.y, matchLoc.x) << endl;

	  /// Show me what you got
	  rectangle( src, matchLoc, Point( matchLoc.x + mask.cols , matchLoc.y + mask.rows ), blue, 2, 8, 0 );
	  rectangle( result, matchLoc, Point( matchLoc.x + mask.cols , matchLoc.y + mask.rows ), blue, 2, 8, 0 );

	  imshow( "Preview", src );
	  //imshow( "Gray", result );

  	  int key = cv::waitKey(1) & 255;
		if (key == 27)
			break;
  }

  return 0;

}

#endif

#if 1

#include"opencv2/opencv.hpp"
#include"opencv2/core/core.hpp"
#include"opencv2/highgui/highgui.hpp"
#include"opencv2/imgproc/imgproc.hpp"
#include"iostream"
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <math.h>
#include <time.h>

using namespace cv;
using namespace std;

const cv::Scalar blue(255, 0, 0),  green(0, 255, 0);

inline int GetElapsedMillis(int64 start, int64 end = 0) {
	static const double freq = cv::getTickFrequency() / 1000;
	if (end == 0)
		end = cv::getTickCount();
	return round((end - start) / freq);
}

std::string FormatTime(double millis);

inline std::string GetElapsedTime(int64 start, int64 end = 0) {
	return FormatTime(GetElapsedMillis(start, end));
}

std::string FormatTime(double millis) {
	const int kSecond = 1000, kMinute = 60 * kSecond, kHour = 60 * kMinute;
	if (millis >= kHour)
		return cv::format("%d h %d m", int(millis / kHour), (int(millis) % kHour) / kMinute);
	else if (millis >= kMinute)
		return cv::format("%.3f m", millis / kMinute);
	else if (millis >= kSecond)
		return cv::format("%.2f s", millis / kSecond);
	else
		return cv::format("%.1f ms", millis);
}


Mat norm_0_255(const Mat& src) {
    // Create and return normalized image:
    Mat dst;
    switch(src.channels()) {
    case 1:
        cv::normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
        break;
    case 3:
        cv::normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
        break;
    default:
        src.copyTo(dst);
        break;
    }
    return dst;
}

void on_trackbar( int focus, void* cap )
{
	static_cast<cv::VideoCapture*>(cap)->set(CV_CAP_PROP_FOCUS, focus);
}


int main()
{
	Mat src, src_gray, mask_1;
	vector<Mat> arr;
	list<Mat> masks;
	Rect rect;
	int width, height, track_flag = 0, minsize = 40, maxsize =
			200, thrsh = 100, focus_par;

	Point cent;

	//for ( int device = 0; device < 10; device++ ) {
	cv::VideoCapture cap(0 /*device*/);
	if (!cap.isOpened())
		return (0);
	//}
	cap >> src;
	double iratio = 100.0 / src.rows;

	width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
	height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);

	namedWindow("Preview", CV_WINDOW_AUTOSIZE);
	namedWindow("Preview1", CV_WINDOW_AUTOSIZE);
	namedWindow("Mask", CV_WINDOW_AUTOSIZE);
	focus_par = 10;
	cv::createTrackbar("Focus", "Preview", &focus_par, 100, on_trackbar, &cap);
	on_trackbar(focus_par, &cap);
	//namedWindow("Correlation", CV_WINDOW_AUTOSIZE);
	cv::Mat result, src_1;
	cv::Mat mask = cv::imread("/home/gunmachine/workspace/pca_1/1.png", CV_LOAD_IMAGE_GRAYSCALE);
	imshow("Mask", mask);
	double mratio = 100.0 / mask.rows;

	cv::resize(mask, mask_1, Size(0.5*mask.cols, 0.5*mask.rows), 0, 0, INTER_AREA);

	/// Localizing the best match with minMaxLoc
	vector<double> minVal(6);
	vector<double> maxVal(6);
	vector<Point> minLoc(6);
	vector<Point> maxLoc(6);
	Point matchLoc;
	int minCorr[2], maxCorr[2];
	double min, max;
	float fratio;

	for (;;) {

		cap >> src;

		/// Convert it to gray
		cvtColor(src, src_gray, CV_BGR2GRAY);
		//cvtColor( src, src_gray, CV_RGB2HSV);

		cv::resize(src_gray, src_1, Size(0.5*src_gray.cols, 0.5*src_gray.rows), 0, 0, INTER_AREA);
		int64 start = cv::getTickCount();
		string time;

		//cv::split(src, arr);
		//src_gray = arr[2];
		//equalizeHist( src_gray, src_gray );

		//GaussianBlur( src_gray, src_gray, Size(9, 9), 2, 2 );

		for (int ratio = 10; ratio > 4; ratio--) {

			float fratio = float(ratio/10.0);
			cv::Mat mask_;
			cv::resize(mask_1, mask_, Size(), fratio, fratio, INTER_AREA);

			int result_cols = src_1.cols - mask_.cols + 1;
			int result_rows = src_1.rows - mask_.rows + 1;

			result.create(result_cols, result_rows, CV_32FC1);

			/// Do the Matching and Normalize
			matchTemplate(src_1, mask_, result, CV_TM_CCOEFF_NORMED);

			minMaxLoc(result, &(minVal[10-ratio]), &(maxVal[10-ratio]), &(minLoc[10-ratio]), &(maxLoc[10-ratio]));
			//minLoc[10-ratio].x = (minLoc[10-ratio].x)/fratio; minLoc[10-ratio].y = (minLoc[10-ratio].y)/fratio;
			//maxLoc[10-ratio].x = (maxLoc[10-ratio].x)/fratio; maxLoc[10-ratio].y = (maxLoc[10-ratio].y)/fratio;

		}

	  minMaxIdx(maxVal, &min, &max, minCorr, maxCorr);
	  //cout << "0 comp - " << maxCorr[0] << " ; ------ 1 comp - " << maxCorr[1] << endl;
	  /// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
	  matchLoc = maxLoc[maxCorr[1]];
	  fratio = (10.0 - maxCorr[1])/10.0;
	  float coeff = (src.cols-mask.cols+1)/(src.cols*iratio-mask.cols*mratio*fratio+1);
	  //matchLoc.x = matchLoc.x*coeff; matchLoc.y = matchLoc.y/coeff;
	  //cout << result.at<float>(matchLoc.y, matchLoc.x) << endl;

  	  int key = cv::waitKey(1) & 255;

	  /// Show me what you got
	  if (max>0.75) {
//		  if (key == 27) {
//			  int x = rand();
//		  }
		  rectangle(src_1, matchLoc, Point( matchLoc.x + mask_1.cols*fratio , matchLoc.y + mask_1.rows*fratio ), green, 2, 8, 0 );
		  rect = Rect( matchLoc.x, matchLoc.y, int (mask_1.cols*fratio) , int (mask_1.rows*fratio) );
	  }
	  else {
		  rectangle(src_1, Point(0,0), Point( src_1.cols, src_1.rows), green, 4, 8, 0 );
	  }
	  mask = src_1(rect).clone();
	  //rectangle( result, matchLoc, Point( matchLoc.x + mask.cols , matchLoc.y + mask.rows ), green, 2, 8, 0 );

	  //imshow( "Preview", src );
	  imshow( "Preview", norm_0_255(result));
	  imshow( "Preview1", src_1);

	  //imshow( "Gray", result );

	  int64 end = cv::getTickCount();
	  time = GetElapsedTime( start, end );
	  cout << time << endl;

  	  if (key == 27)
			break;

	}

	//mask_1 = src_1(rect).clone();
	//cv::resize(mask_1, mask, Size(), 2*fratio, 2*fratio, INTER_AREA);
	//cap >> src;

	//cv::split(src, arr);
	//src_gray = arr[2];
	//cvtColor(src, src_gray, CV_BGR2GRAY);
	cv::Mat mask_2;
	cv::resize(mask, mask_2, Size(), 2, 2, INTER_AREA);


  for (;;) {

	  cap >> src;

	  //cv::split(src, arr);
	  //src_gray = arr[2];
	  cvtColor( src, src_gray, CV_BGR2GRAY );

	  int result_cols =  src_gray.cols - mask_2.cols + 1;
	  int result_rows = src_gray.rows - mask_2.rows + 1;

	  result.create( result_cols, result_rows, CV_32FC1 );

	  /// Do the Matching and Normalize
	  matchTemplate( src_gray, mask_2, result, CV_TM_CCOEFF_NORMED );

	  /// Localizing the best match with minMaxLoc
	  double minVal; double maxVal; Point minLoc; Point maxLoc;
	  Point matchLoc;

	  minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc );

	  /// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
	  matchLoc = maxLoc;
	  cout << maxVal << endl;

	  /// Show me what you got
	  rectangle( src, matchLoc, Point( matchLoc.x + mask.cols , matchLoc.y + mask.rows ), blue, 2, 8, 0 );
	  rectangle( result, matchLoc, Point( matchLoc.x + mask.cols , matchLoc.y + mask.rows ), blue, 2, 8, 0 );

	  rect = Rect( matchLoc.x, matchLoc.y, mask_2.cols, mask_2.rows);
	  mask = src_gray(rect).clone();
	  cv::Mat dst;
	  cv::addWeighted(mask_2, 0.9, mask, 0.1, 0, dst, -1);
	  mask_2 = dst.clone();
	  imshow( "Preview", src );
	  imshow("Mask", mask_2);
	  //imshow( "Gray", result );

  	  int key = cv::waitKey(1) & 255;
		if (key == 27)
			break;
  	  }


  return 0;

}

#endif
