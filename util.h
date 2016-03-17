#ifndef ADCAM_UTIL
#define ADCAM_UTIL

#include <opencv/cv.h>
#include <sstream>
#include <algorithm>
#include <vector>
#include <string>
#include <numeric>
#include <sys/stat.h>

namespace std {

template<class T> cv::Point_<T> sqrt(const cv::Point_<T>& p) {
	return cv::Point_<T>(sqrt(p.x), sqrt(p.y));
}

}

namespace ac {

template<class T> T sqr(const T& x) {
	return x * x;
}
template<class T> cv::Point_<T> sqr(const cv::Point_<T>& p) {
	return cv::Point_<T>(p.x * p.x, p.y * p.y);
}
template<class T> T mean(const std::vector<T>& vec) {
	return std::accumulate(vec.begin(), vec.end(), T(0)) / vec.size();
}
template<class T> cv::Point_<T> operator/(const cv::Point_<T>& p, double divisor) {
	return cv::Point_<T>(p.x / divisor, p.y / divisor);
}
template<class T> T variance(const std::vector<T>& vec, T meanVal) {
	T sum(0);
	for (typename std::vector<T>::const_iterator it = vec.begin(); it != vec.end(); ++it)
		sum += sqr(*it - meanVal);
	return sum / vec.size();
}
template<class T> T variance(const std::vector<T>& vec) {
	return variance(vec, mean(vec));
}
template<class T> T stddev(const std::vector<T>& vec, T meanVal) {
	return std::sqrt(variance(vec, meanVal));
}
template<class T> T stddev(const std::vector<T>& vec) {
	return std::sqrt(variance(vec));
}
template<class T> T CoefOfVariance(const std::vector<T>& vec, T meanVal) {
	return 100 * stddev(vec, meanVal) / meanVal;
}

/**
 * Generate random number from range [low, high)
 * @param low - minimal value
 * @param high - maximal value + 1
 */
inline int RandInRange(int low, int high) {
	return low + rand() % (high - low);
}

/**
 * Normalize matrix to have 0 mean and unit standard deviation.
 */
void NormalizeMeanStdDev(cv::Mat_<double> mat);

template<class T> inline T ScalarMul(cv::Point_<T> a, cv::Point_<T> b) {
	return a.x * b.x + a.y * b.y;
}

template<class T> inline size_t CountNonempty(const T& a, const T& b, const T& c) {
	return (a.empty() ? 0 : 1) + (b.empty() ? 0 : 1) + (c.empty() ? 0 : 1);
}

inline cv::Point2f GetCenter(const cv::Rect& rect) {
	return cv::Point2f(rect.x + rect.width / 2.0, rect.y + rect.height / 2.0);
}
inline cv::Point2f GetCenter(const cv::Rect* rect) {
	return GetCenter(*rect);
}

inline cv::RotatedRect GetRotatedRect(const cv::Rect& rect) {
	return cv::RotatedRect(GetCenter(rect), cv::Size(rect.width, rect.height), 0);
}

class FaceFeatures {
	cv::Ptr<cv::Point2f> left_, right_, mouth_;
public:
	FaceFeatures(const cv::Point2f* left, const cv::Point2f* right, const cv::Point2f* mouth);
	const cv::Point2f& left() const {
		return *left_;
	}
	const cv::Point2f& right() const {
		return *right_;
	}
	const cv::Point2f& mouth() const {
		return *mouth_;
	}
	bool hasLeft() const {
		return left_ != NULL;
	}
	bool hasRight() const {
		return right_ != NULL;
	}
	bool hasMouth() const {
		return mouth_ != NULL;
	}
	int partsCount() const {
		return (left_ ? 1 : 0) + (right_ ? 1 : 0) + (mouth_ ? 1 : 0);
	}
};

class FaceInfo {
	const cv::Rect face_;
	cv::Ptr<cv::Rect> leftRect_, rightRect_, mouthRect_;
	cv::Ptr<FaceFeatures> feat_;
	bool inGlasses_;
public:
	FaceInfo(const cv::Rect& face, const cv::Point2f* left, const cv::Point2f* right,
			const cv::Point2f* mouth, const bool inGlasses);
	FaceInfo(const cv::Rect& face, const cv::Rect* leftRect, const cv::Rect* rightRect,
			const cv::Rect* mouthRect, const bool inGlasses);
	const cv::Rect& face() const {
		return face_;
	}
	const cv::Rect* leftRect() const {
		return leftRect_;
	}
	const cv::Rect* rightRect() const {
		return rightRect_;
	}
	const cv::Rect* mouthRect() const {
		return mouthRect_;
	}
	const FaceFeatures& feat() const {
		return *feat_;
	}
	bool inGlasses() const {
		return inGlasses_;
	}
};

typedef std::vector<cv::Rect> Rects;

enum Gender {
	MALE, FEMALE, UNKNOWN
};

inline std::string GetGenderStr(Gender gender) {
	switch (gender) {
	case MALE:
		return "male";
	case FEMALE:
		return "female";
	case UNKNOWN:
	default:
		return "unknown";
	}
}

const cv::Scalar kRed(0, 0, 255), kGreen(0, 255, 0), kBlue(255, 0, 0);

inline cv::Scalar GetGenderColor(Gender gender) {
	switch (gender) {
	case MALE:
		return kBlue;
	case FEMALE:
		return kRed;
	case UNKNOWN:
	default:
		return kGreen;
	}
}

inline int GetElapsedMillis(int64 start, int64 end = 0) {
	static const double freq = cv::getTickFrequency() / 1000;
	if (end == 0)
		end = cv::getTickCount();
	return round((end - start) / freq);
}

inline cv::Mat ExtractChannel(const cv::Mat& img, int channelInd) {
	cv::Mat res(img.size(), CV_MAKETYPE(CV_MAT_DEPTH(img.flags), 1));
	int fromTo[] = { channelInd, 0 };
	cv::mixChannels(&img, 1, &res, 1, fromTo, 1);
	return res;
}

inline cv::Mat GetBWFromBGR(const cv::Mat& img) {
	cv::Mat res(img.size(), CV_8UC1);
	cv::cvtColor(img, res, CV_BGR2GRAY);
	return res;
}

inline cv::Mat GetBWFromHSV(const cv::Mat& img) {
	return ExtractChannel(img, 2);
}

inline cv::Mat GetHSVFromBGR(const cv::Mat& img) {
	cv::Mat res(img.size(), CV_8UC3);
	cv::cvtColor(img, res, CV_BGR2HSV);
	return res;
}

cv::Mat GetFaceMarked(const cv::Mat& img, const FaceInfo& faceInfo);

inline bool FileExists(std::string filename) {
	struct stat buffer;
	return !stat(filename.c_str(), &buffer);
}

inline std::string ExtractExt(const std::string& filename) {
	int dotPos = filename.find_last_of('.');
	if (dotPos == -1)
		return "";
	int slashPos = filename.find_last_of('/');
	if (slashPos != -1 && slashPos > dotPos)
		return "";
	return filename.substr(dotPos + 1);
}

/**
 * @return the name without extension
 */
inline std::string ExtractFilename(const std::string& nameWithExt) {
	return nameWithExt.substr(0, nameWithExt.rfind('.'));
}

template<typename T> inline std::string ToString(T number) {
	std::ostringstream ss;
	ss << number;
	return ss.str();
}

template<class T> float Percentage(T count, T maxCount) {
	return 100.0f * count / maxCount;
}

/**
 * Act like matlab's [Y,I] = SORT(X)
 * @param unsorted
 * @param sorted Sorted vector, allowed to be same as unsorted
 * @param indexMap An index map such that sorted[i] = unsorted[index_map[i]]
 */
template<class T>
void sort(const std::vector<T>& unsorted, std::vector<T>& sorted, std::vector<size_t>& indexMap);

/**
 * Act like matlab's Y = X[I], where I contains a vector of indices so that after, Y[j] = X[I[j]]
 * This implies that Y.size() == I.size()
 * X and Y are allowed to be the same reference
 */
template<class T>
void reorder(const std::vector<T>& unordered, const std::vector<size_t>& indexMap,
		std::vector<T>& ordered);

inline void rotatedRectPoints(cv::RotatedRect rect, cv::Point* faceRectPoints_i) {
	cv::Point2f faceRectPoints[4];
	rect.points(faceRectPoints);
	for (int i = 0; i < 4; i++) {
		faceRectPoints_i[i].x = (int) roundf(faceRectPoints[i].x);
		faceRectPoints_i[i].y = (int) roundf(faceRectPoints[i].y);
	}
}

inline void drawRotatedRect(cv::Mat &img, cv::RotatedRect rect, cv::Scalar color) {
	cv::Point faceRectPoints_i[4];
	rotatedRectPoints(rect, faceRectPoints_i);
	const cv::Point *polys[1] = { faceRectPoints_i };
	const int npts[1] = { 4 };
	cv::polylines(img, polys, npts, 1, true, color, 1, 8, 0);
}

inline void fillRotatedRect(cv::Mat &img, cv::RotatedRect rect, cv::Scalar color) {
	cv::Point faceRectPoints_i[4];
	rotatedRectPoints(rect, faceRectPoints_i);
	cv::fillConvexPoly(img, faceRectPoints_i, 4, color, 8, 0);
}

int rectToDelete(const cv::RotatedRect* rect1, const cv::RotatedRect* rect2);

inline std::vector<cv::Point2f> rotRectToPoly(cv::RotatedRect rect) {
	std::vector<cv::Point2f> poly;
	cv::Point2f points[4];
	rect.points(points);
	for (int i = 0; i < 4; i++) {
		poly.push_back(points[i]);
	}
	return poly;
}

inline std::vector<cv::Point2f> rectToPoly(cv::Rect rect) {
	std::vector<cv::Point2f> poly;
	poly.push_back(rect.tl());
	poly.push_back(cv::Point2f(rect.x + rect.width, rect.y));
	poly.push_back(rect.br());
	poly.push_back(cv::Point2f(rect.x, rect.y + rect.height));
	return poly;
}

std::vector<cv::Point2f> clipPoly(const std::vector<cv::Point2f> poly1,
		const std::vector<cv::Point2f> poly2);

float polyIntersection(const std::vector<cv::Point2f> poly1, const std::vector<cv::Point2f> poly2,
		int& deletePoly, float rate = 0.4);

inline float rotRectIntersection(const cv::RotatedRect rect1, const cv::RotatedRect rect2,
		int& deletePoly, float rate = 0.4) {
	return polyIntersection(rotRectToPoly(rect1), rotRectToPoly(rect2), deletePoly, rate);
}

inline float rectIntersection(const cv::Rect rect1, const cv::Rect rect2, int& deletePoly,
		float rate = 0.4) {
	return polyIntersection(rectToPoly(rect1), rectToPoly(rect2), deletePoly, rate);
}

}

#endif /* ADCAM_UTIL */
