#if 0

#include <iostream>

// Include standard OpenCV headers
#include "cv.h"
#include "highgui.h"

using namespace std;

// All the new API is put into "cv" namespace
using namespace cv;

int main (int argc, char *argv[])
{
    // Open the default camera
    VideoCapture capture(1);

    // Check if the camera was opened
    if(!capture.isOpened())
    {
        cerr << "Could not create capture";
        return -1;
    }

    // Get the properties from the camera
    double width = capture.get(CV_CAP_PROP_FRAME_WIDTH);
    double height = capture.get(CV_CAP_PROP_FRAME_HEIGHT);

    cout << "Camera properties\n";
    cout << "width = " << width << endl <<"height = "<< height << endl;

    // Create a matrix to keep the retrieved frame
    Mat frame;

    // Create a window to show the image
    namedWindow ("Capture", CV_WINDOW_AUTOSIZE);

    // Create the video writer
    VideoWriter video("capture.avi", CV_FOURCC('P','I','M','1') , 30, cvSize((int)width,(int)height) );

    // Check if the video was opened
    if(!video.isOpened())
    {
        cerr << "Could not create video.";
        return -1;
    }

    cout << "Press any key to start recording." << endl;

    for ( ; ; ) {

    	capture >> frame;
    	imshow("Capture", frame);

        if (waitKey(1)!=-1)
        		break;
    }


    cout << "Press Esc to stop recording." << endl;

    // Get the next frame until the user presses the escape key
    while(true)
    {
        // Get frame from capture
        capture >> frame;

        // Check if the frame was retrieved
        if(!frame.data)
        {
            cerr << "Could not retrieve frame.";
            return -1;
        }

        // Save frame to video
        video << frame;

        // Show image
        imshow("Capture", frame);

        // Exit with escape key
        if(waitKey(1) == 27)
            break;
    }

    // Exit
    return 0;
}

#else

#endif

#if 0

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "file_storage.h"

#include <fstream>
#include <sstream>
#include <glob.h>

using namespace cv;
using namespace std;

// Reads the images and labels from a given CSV file, a valid file would
// look like this:
//
//      /path/to/person0/image0.jpg;0
//      /path/to/person0/image1.jpg;0
//      /path/to/person1/image0.jpg;1
//      /path/to/person1/image1.jpg;1
//      ...
//
void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels) {
    std::ifstream file(filename.c_str(), ifstream::in);
    if(!file)
        throw std::exception();
    std::string line, path, classlabel;
    // For each line in the given file:
    while (std::getline(file, line)) {
        // Get the current line:
        std::stringstream liness(line);
        // Split it at the semicolon:
        std::getline(liness, path, ';');
        std::getline(liness, classlabel);
        // And push back the data into the result vectors:
        images.push_back(imread(path, IMREAD_GRAYSCALE));
        labels.push_back(atoi(classlabel.c_str()));
    }
}

// Normalizes a given image into a value range between 0 and 255.
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

// Converts the images given in src into a row matrix.
Mat asRowMatrix(const vector<Mat>& src, int rtype, double alpha = 1, double beta = 0) {
    // Number of samples:
    size_t n = src.size();
    // Return empty matrix if no matrices given:
    if(n == 0)
        return Mat();
    // dimensionality of (reshaped) samples
    size_t d = src[0].total();
    // Create resulting data matrix:
    Mat data(n, d, rtype);
    // Now copy data:
    for(int i = 0; i < n; i++) {
        //
        if(src[i].empty()) {
            string error_message = format("Image number %d was empty, please check your input data.", i);
            CV_Error(CV_StsBadArg, error_message);
        }
        // Make sure data can be reshaped, throw a meaningful exception if not!
        if(src[i].total() != d) {
            string error_message = format("Wrong number of elements in matrix #%d! Expected %d was %d.", i, d, src[i].total());
            CV_Error(CV_StsBadArg, error_message);
        }
        // Get a hold of the current row:
        Mat xi = data.row(i);
        // Make reshape happy by cloning for non-continuous matrices:
        if(src[i].isContinuous()) {
            src[i].reshape(1, 1).convertTo(xi, rtype, alpha, beta);
        } else {
            src[i].clone().reshape(1, 1).convertTo(xi, rtype, alpha, beta);
        }
    }
    return data;
}

int main(int argc, const char *argv[]) {
    // Holds some images:
    vector<Mat> db;
    int glob(const char *pattern, int flags,
                    int (*errfunc) (const char *epath, int eerrno),
                    glob_t *pglob);

    // Load the greyscale images. The images in the example are
    // taken from the AT&T Facedatabase, which is publicly available
    // at:
    //
    //      http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html
    //
    // This is the path to where I stored the images, yours is different!
    //
    string prefix = "/home/gunmachine/workspace/pca_1";

    glob_t gl;
    size_t num = 0;
    if(glob("/home/gunmachine/workspace/pca_1/*.png", GLOB_NOSORT, NULL, &gl) == 0)
      num = gl.gl_pathc;
    globfree(&gl);

    for ( int i = 1; i < num+1; i++) {

    	stringstream ss;
    	ss << i;
    	string str = ss.str();
    	/*std::to_string(i)*/

    	Mat neweyemask = imread(prefix + "/" + str + ".png", CV_LOAD_IMAGE_GRAYSCALE );
    	db.push_back( neweyemask );

    }

    // The following would read the images from a given CSV file
    // instead, which would look like:
    //
    //      /path/to/person0/image0.jpg;0
    //      /path/to/person0/image1.jpg;0
    //      /path/to/person1/image0.jpg;1
    //      /path/to/person1/image1.jpg;1
    //      ...
    //
    // Uncomment this to load from a CSV file:
    //

    /*
    vector<int> labels;
    read_csv("/home/philipp/facerec/data/at.txt", db, labels);
    */

    // Build a matrix with the observations in row:
    Mat data = asRowMatrix(db, CV_32FC1);

    // Number of components to keep for the PCA:
    int num_components = 10;

    // Perform a PCA:
    PCA pca(data, Mat(), CV_PCA_DATA_AS_ROW);

    // And copy the PCA results:
    Mat mean = pca.mean.clone();
    Mat eigenvalues = pca.eigenvalues.clone();
    Mat eigenvectors = pca.eigenvectors.clone();

    // The mean face:
    imshow("avg", norm_0_255(mean.reshape(1, db[0].rows)));

    // The first three eigenfaces:
    imshow("pc1", norm_0_255(pca.eigenvectors.row(0)).reshape(1, db[0].rows));
    imshow("pc2", norm_0_255(pca.eigenvectors.row(1)).reshape(1, db[0].rows));
    imshow("pc3", norm_0_255(pca.eigenvectors.row(2)).reshape(1, db[0].rows));

    std::ofstream ofs ("/home/gunmachine/workspace/pca_eigenvectors/test.pca", std::ofstream::out|std::ios::binary);
    ac::WritePCA(ofs, pca);
    ofs.close();



    // Show the images:
    waitKey(0);

    // Success!
    return 0;
}

#endif
