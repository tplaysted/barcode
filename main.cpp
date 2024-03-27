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

class Bar { // convenience class for extracting t-values
    int width = 0;
    int type = 0;

    public:
        Bar(int w, int t = 0) {
            width = w;
            type = t;
        }
};

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

Mat apply_thresholding(Mat &image, int radius) {
    Mat target;
    Mat blur;
    GaussianBlur(image, blur, Size(0, 0), radius, radius);
    threshold(blur, target, THRESHOLD, 255, THRESH_BINARY);  // do thresholding
    return target;
}

Mat apply_otsu_thresholding(Mat &image, int radius) { // otsu thresholding
    Mat target;
    Mat blur;
    GaussianBlur(image, blur, Size(0, 0), radius, radius);
    threshold(blur, target, THRESHOLD, 255, THRESH_BINARY + THRESH_OTSU);  // do thresholding
    return target;
}

Moments get_moments(Mat &image) {  // get the 0th-3rd order moments for white parts of image
    return moments(255 - image, true);
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

void mark_blob(Mat &image, Moments &m) { // draw the axis of orientation on a blob
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

vector<int> get_line_of_pixels(Mat &image) { // scans a line of pixels across the bar code
    Moments m = get_moments(image);
    vector<int> c = get_centroid(m);
    float o = get_orientation(m);
    int w = image.size().width; // scan across entire image

    int x1 = c[0] - w * cos(o); // define points to draw a line between
    int y1 = c[1] - w * sin(o);
    int x2 = c[0] + w * cos(o);
    int y2 = c[1] + w * sin(o);

    LineIterator it = LineIterator(image, Point(x1, y1), Point(x2, y2), 4);
    vector<int> line(it.count);

    for (int i=0; i < it.count; i++, it++) {
        switch ((int)image.at<uchar>(it.pos())) {
            case 0: line[i] = 0; break;
            default: line[i] = 1;
        }
    }

    return line;
}

vector<Bar> extract_bars_from_line(vector<int> &line) {
    vector<Bar> bars;

    int last_pix = line[0];
    int len = 1;

    for (auto &pix: line) {
        if (pix == last_pix) {
            len++;
        } else {
            bars.push_back(Bar(len, last_pix));
            len = 1;
        }

        last_pix = pix;
    }

    return bars;
}

int main() {
    Mat img = imread("IMG_20240227_0004.jpg");
    Mat gray = make_grayscale(img);
    Mat bin = apply_otsu_thresholding(gray, 1);

    Moments m = get_moments(bin);
    mark_blob(img, m);

    vector<int> line = get_line_of_pixels(bin);
    vector<Bar> bars = extract_bars_from_line(line);
    imshow("Image", img);
    imshow("Grayscale", gray);
    imshow("Binary", bin);
    waitKey(0);

    cout << "Hello, world!" << endl;
    return 0;
}