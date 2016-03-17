
#include "file_storage.h"
#include "tclap/CmdLine.h"
#include <fstream>
#include <opencv2/opencv.hpp>
#include <errno.h>
#include <stdio.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/statvfs.h>

using namespace std;

bool ReadPCA(string precompFile, cv::PCA& pca) {
	ifstream file(precompFile.c_str(), ios::binary | ios::in);
	if (!file.is_open()) {
		cerr << "Couldn't read PCA from precomp file" << endl;
		return false;
	}
	ac::ReadPCA(file, pca);
	return true;
}

inline int TrueCount(const vector<bool>& vec) {
	int count = 0;
	for (vector<bool>::const_iterator it = vec.begin(); it != vec.end(); it++)
		if (*it)
			count++;
	return count;
}

cv::Mat SelectVectors(const vector<bool>& selected, const cv::Mat& vectors, int newNComps,
		bool selectCols) {
	if (newNComps == -1)
		newNComps = TrueCount(selected);
	cv::Mat selectedVectors(selectCols ? vectors.rows : newNComps,
			selectCols ? newNComps : vectors.cols, vectors.type());
	typedef cv::Mat (cv::Mat::*RowOrCol)(int) const;
	RowOrCol rowOrCol = selectCols ? &cv::Mat::col : &cv::Mat::row;
	int i = 0;
	for (int j = 0, max = selectCols ? vectors.cols : vectors.rows; j < max; j++)
		if (selected[j])
			(vectors.*rowOrCol)(j).copyTo((selectedVectors.*rowOrCol)(i++));
	return selectedVectors;
}

void ReducePCA(const vector<bool>& selected, cv::PCA& pca) {
	int selCount = TrueCount(selected);
	pca.eigenvalues = SelectVectors(selected, pca.eigenvalues, selCount, false);
	pca.eigenvectors = SelectVectors(selected, pca.eigenvectors, selCount, false);
}

inline std::string ExtractParentDir(const std::string& path) {
	size_t slashInd = path.find_last_of("/\\", path.size() - 2);
	if (slashInd == std::string::npos)
		return "";
	return path.substr(0, slashInd);
}

bool IsDirectory(const string& filename) {
	struct stat st;
	return stat(filename.c_str(), &st) == 0 && (st.st_mode & S_IFDIR) != 0;
}

#ifdef WIN32

#include <windows.h>

namespace ac {

longlong GetFreeBytes(const string& path) {
	if (!FileExists(path) && !CreateDir(path))
		return 0;
	unsigned __int64 lpFreeBytesAvailableToCaller, lpTotalNumberOfBytes, lpTotalNumberOfFreeBytes;
	GetDiskFreeSpaceEx(path.c_str(), (PULARGE_INTEGER) &lpFreeBytesAvailableToCaller,
			(PULARGE_INTEGER) &lpTotalNumberOfBytes, (PULARGE_INTEGER) &lpTotalNumberOfFreeBytes);
	return lpTotalNumberOfFreeBytes;
}

longlong GetFileSize(const string& filename) {
	WIN32_FILE_ATTRIBUTE_DATA fad;
	if (!GetFileAttributesEx(filename.c_str(), GetFileExInfoStandard, &fad)) {
		cerr << "ERROR: can't get size of '" << filename << "', error " << GetLastError() << endl;
		return -1;
	}
	LARGE_INTEGER size;
	size.HighPart = fad.nFileSizeHigh;
	size.LowPart = fad.nFileSizeLow;
	return size.QuadPart;
}

bool DoCreateDir(const string& path) {
	return CreateDirectory(path.c_str(), NULL);
}

}

#else

namespace ac {

bool DoCreateDir(const string& path) {
	return mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IRWXO) == 0;
}

}

#endif

bool CreateDir(const string& path, bool ignoreExists = true,
		bool createParents = true) {
	if (ignoreExists && IsDirectory(path))
		return true;
	if (createParents) {
		string parent;
		if (!(parent = ExtractParentDir(path)).empty()
				&& !CreateDir(parent, true, true))
			return false;
	}
	bool res = ac::DoCreateDir(path);
	if (!res)
		cerr << "ERROR: can't create directory '" << path << "', errno "
				<< errno << ": " << strerror(errno) << endl;
	return res;
}

void VisualizeComponents(string precompFile, string outDir, int rows) {

	if (!IsDirectory(outDir))
		CreateDir(outDir);
	cv::PCA pca;
	ReadPCA(precompFile, pca);
	int num = pca.eigenvectors.rows;
    for ( int i = 0; i < num; i++) {
    	cv::Mat pcacomp = (pca.eigenvectors.row(i)).reshape(0,rows);
    	stringstream ss;
    	ss << i;
    	string str = ss.str();
    	cv::normalize(pcacomp, pcacomp, 0, 255, cv::NORM_MINMAX);
    	cv::Mat outputimg;
    	pcacomp.convertTo( outputimg, CV_8U );
    	cv::imwrite(outDir + "/" + str + ".png", outputimg);
    }
    std::cout << "Vizualizing done" << endl;
}

void VisualizeBackprojection(string precompFile, string choiceFile, string imgFile) {
	cv::Mat original = cv::imread(imgFile, CV_LOAD_IMAGE_GRAYSCALE);
	if (original.empty()) {
		cerr << "Couldn't read image" << endl;
		return;
	}

	cv::PCA pca;
	if (!ReadPCA(precompFile, pca))
		return;
	if (!choiceFile.empty()) {
		ifstream file(choiceFile.c_str(), ios::binary | ios::in);
		if (!file.is_open()) {
			cerr << "Couldn't read choice file" << endl;
			return;
		}
		vector<bool> selected;
		ac::ReadVec(file, selected);
		ReducePCA(selected, pca);
	}

	cv::Mat reshaped = original.reshape(1, 1);
	cv::Mat backprojected = pca.backProject(pca.project(reshaped));

	cv::Mat divider(original.rows, 3, original.type(), cv::Scalar(255));
	backprojected = backprojected.reshape(1, original.rows);
	cv::Mat restored;
	backprojected.convertTo(restored, original.type());
	cv::Mat united;
	vector<cv::Mat> imgs = { original, divider, restored };
	cv::hconcat(imgs, united);

	char *windowName = (char*) "original, backprojected";
	imshow(windowName, united);
	cv::moveWindow(windowName, 0, 0);
	imwrite("out.png", united);
	cv::waitKey();
}

int main(int argc, char **argv) {
	try {
		TCLAP::CmdLine cmd("", ' ', "1.0");
		TCLAP::ValueArg<string> imgFileArg("i", "image", "Image file to test on", false, "", "path",
				cmd);
		TCLAP::ValueArg<string> outDirArg("d", "out_dir",
				"Output directory for principal components", false, "", "path",
				cmd);
		TCLAP::ValueArg<string> choiceFileArg("c", "choice", "Choice file with selected principal "
				"components. If not specified, then all components are used", false, "", "path",
				cmd);
		TCLAP::ValueArg<string> precompFileArg("p", "precomp", "Precomputation data file to take "
				"PCA data from", true, "", "path", cmd);
		cmd.parse(argc, argv);

		cv::Mat img = cv::imread(imgFileArg.getValue(), CV_LOAD_IMAGE_GRAYSCALE);
		int rows = img.rows;

		if (outDirArg.isSet())
			VisualizeComponents(precompFileArg.getValue(), outDirArg.getValue(), rows);
		else
			VisualizeBackprojection(precompFileArg.getValue(), choiceFileArg.getValue(),
					imgFileArg.getValue());

	} catch (TCLAP::ArgException& e) {
		cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
		return 0;
	}
}
