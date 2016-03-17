#ifndef FILE_STORAGE_H
#define FILE_STORAGE_H

#include <vector>
#include <iostream>
#include <opencv/cv.h>

namespace ac {

void WriteMat(std::ostream& out, const cv::Mat& mat);
void ReadMat(std::istream& in, cv::Mat& mat);
void ReadMat(std::istream& in, CvMat*& mat);

template<class T> inline void WriteVariable(std::ostream& out, const T& var) {
	out.write(reinterpret_cast<const char*>(&var), sizeof(T));
}

template<class T> inline void ReadVariable(std::istream& in, T& var) {
	in.read(reinterpret_cast<char*>(&var), sizeof(T));
}

inline void WriteBool(std::ostream& out, bool val) {
	out.write(reinterpret_cast<const char*>(&val), sizeof(bool));
}

inline bool ReadBool(std::istream& in) {
	bool val;
	in.read(reinterpret_cast<char*>(&val), sizeof(bool));
	return val;
}

inline void WritePCA(std::ostream& out, const cv::PCA& pca) {
	WriteMat(out, pca.mean);
	WriteMat(out, pca.eigenvalues);
	WriteMat(out, pca.eigenvectors);
}

inline void ReadPCA(std::istream& in, cv::PCA& pca) {
	ReadMat(in, pca.mean);
	ReadMat(in, pca.eigenvalues);
	ReadMat(in, pca.eigenvectors);
}

template<class T> void WriteVec(std::ostream& out, const std::vector<T>& vec) {
	WriteVariable(out, vec.size());
	for (size_t i = 0; i < vec.size(); i++)
		WriteVariable(out, vec[i]);
}

template<class T> void ReadVec(std::istream& in, std::vector<T>& vec) {
	size_t size;
	ReadVariable(in, size);
	vec.resize(size);
	for (size_t i = 0; i < size; i++) {
		T var;
		ReadVariable(in, var);
		vec[i] = var;
	}
}

}

#endif /* FILE_STORAGE_H */
