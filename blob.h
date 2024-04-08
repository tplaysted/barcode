#include <iostream>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <fstream>

const int THRESHOLD = 127;
const int CAMERA_PORT = 0;
const double pi = 3.14159265358979323846;

using namespace std;
using namespace cv;

void writeCSV(string filename, Mat m)
{
    ofstream myfile;
    myfile.open(filename.c_str());
    myfile << cv::format(m, cv::Formatter::FMT_CSV) << std::endl;
    myfile.close();
}

VideoCapture open_external_cam() { // open a capture stream with preference given to external devices
    VideoCapture cap;
    for (int i=2; i>=0; i--) {
        cap.open(i);
        if (cap.isOpened()) {
            return cap;
        }
    }

    return cap;
}

Mat make_grayscale(Mat &image) { // convert an image to grayscale
    Mat gray_image; 

    if (image.channels() == 1) {
        gray_image = image;
    } else {
        cvtColor(image, gray_image, COLOR_BGR2GRAY);
    }
    
    return gray_image;
}

Mat full_contrast(Mat &image) { // scale pixel values to full brightness range
    Mat target;
    double min, max;
    minMaxLoc(image, &min, &max);  // Get the dimmest and brightest pixels for scaling
    image.convertTo(target, image.type(), 1.0, (-1.0 * min));
    target.convertTo(target, image.type(), 255 / (max - min + 1), 0); // scale image to full range

    return target;
}

Mat apply_thresholding(Mat &image) {
    Mat target;
    threshold(image, target, THRESHOLD, 255, THRESH_BINARY_INV);  // do thresholding
    return target;
}

Mat apply_adaptive_thresholding(Mat &image) { // a more sophisticated thresholding procedure
    Mat target;
    adaptiveThreshold(image, target, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 11, 2);  // do thresholding
    return target;
}

Mat apply_otsu_thresholding(Mat &image, int radius) { // otsu thresholding
    Mat target;
    Mat blur;
    GaussianBlur(image, blur, Size(0, 0), radius, radius);
    threshold(blur, target, THRESHOLD, 255, THRESH_BINARY_INV + THRESH_OTSU);  // do thresholding
    return target;
}

vector<Moments> get_moments(Mat &image) {  // get the 0th-3rd order moments for each component
    Mat labels, stats, centroids;
    connectedComponentsWithStats(image, labels, stats, centroids);  // this is where we do connectivity analysis

    int num_blobs = centroids.rows - 1;
    vector<Mat> blobs(num_blobs);

    for (int i=0; i<num_blobs; i++) {
        inRange(labels, i + 1, i + 1, blobs[i]);
    }

    vector<Moments> m(num_blobs);

    for (int i=0; i<num_blobs; i++) {
        m[i] = moments(blobs[i], true);
    }

    return m;
}

vector<int> get_centroid(Moments &m) { // get the x,y coordinates given moment data
    vector<int> centroid(2);

    centroid[0] = (int) (m.m10 / m.m00);
    centroid[1] = (int) (m.m01 / m.m00);

    return centroid;
}

double get_orientation(Moments &m) { // get the axis of minimum moments of inertia
    double n = 2 * (m.m00 * m.m11 - m.m10 * m.m01); 
    double d = (m.m00 * m.m20 - m.m10 * m.m10) - (m.m00 * m.m02 - m.m01 * m.m01); 

    return -0.5 * atan2(n, d); // atan2 retrieves the principal angle by default
}

void mark_line(Mat &image, Moments &m) { // draw the axis of orientation on a blob
    vector<int> c = get_centroid(m);
    double o = get_orientation(m);
    double short_axis = sqrt(m.m00 / 4.0);
    double long_axis = 2 * short_axis;

    Point2i short_1, short_2, long_1, long_2;

    short_1 = Point2i(c[0] - short_axis * sin(o), c[1] - short_axis * cos(o));
    short_2 = Point2i(c[0] + short_axis * sin(o), c[1] + short_axis * cos(o));
    long_1 = Point2i(c[0] - long_axis * cos(o), c[1] + long_axis * sin(o));
    long_2 = Point2i(c[0] + long_axis * cos(o), c[1] - long_axis * sin(o));

    line(image, short_1, short_2, Scalar(0, 255, 0), 2, LINE_AA);
    line(image, long_1, long_2, Scalar(0, 255, 0), 2, LINE_AA);
}

vector<int> get_tri_classifier_labels(vector<Moments> &m) { // classify blobs according to area via kmeans algorithm
    Mat data(1, m.size(), CV_32F);

    for (int i=0; i<m.size(); i++) {
        data.at<float>(0, i) = m[i].m00;
    }

    double min, max;
    minMaxLoc(data, &min, &max);

    for (int i=0; i<m.size(); i++) {
        data.at<float>(0, i) = data.at<float>(0, i) / max; // normalise area values
    }

    int K = std::min((int)m.size(), 3); // kmeans will break if we have less than K data points

    vector<int> labels;
    kmeans(data, K, labels, TermCriteria(TermCriteria::COUNT, 10, 1.0), 3, KmeansFlags::KMEANS_PP_CENTERS);

    return labels;
}

void draw_label(Mat &image, int &x, int &y, char &label) {
    String lstr = {label};
    putText(image, lstr, Point(x, y), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 4);
}

Mat capture_photo() { // Display camera output and await user input before capturing
    VideoCapture cap = open_external_cam(); // On my laptop "0" is the built-in camera. 
    if (!cap.isOpened()) {
        cerr << "Error opening the camera!" << endl;
    }

    Mat frame;
    
    for (;;)
    {
        // wait for a new frame from camera and store it into 'frame'
        cap.read(frame);
        // check if we succeeded
        if (frame.empty()) {
            cout << "ERROR! blank frame grabbed\n";
            break;
        }

        // show live and wait for a key with timeout long enough to show images
        imshow("Live", frame);
        if (waitKey(5) >= 0)
            break;
    }
    // destroyAllWindows();
    return frame;
}

Mat create_test_image(double angle_delta) {
    int x = 1000;
    int y = 1000;
    Mat im(y, x, CV_8UC3);

    // Draw reference ellipse
    ellipse(im, Point(x / 3, y / 2), Size(150, 100), 0, 0, 360, Scalar(255, 255, 255), -1);

    // Draw delta ellipse
    ellipse(im, Point(2 * x / 3, y / 2), Size(150, 100), angle_delta, 0, 360, Scalar(255, 255, 255), -1);

    GaussianBlur(im, im, Size(0, 0), 5, 5);
    return Scalar(255, 255, 255) - im;
}
