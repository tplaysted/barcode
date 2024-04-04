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
    public:
        int width = 0;
        int type = 0;

        Bar(int w, int t = 0) {
            width = w;
            type = t;
        }
};

class TVal {
    public:
        int t1 = 0;
        int t2 = 0;
        int t3 = 0;
        int t4 = 0;
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
    threshold(blur, target, THRESHOLD, 255, THRESH_BINARY_INV + THRESH_OTSU);  // do thresholding
    return target;
}

Moments get_moments(Mat &image) {  // get the 0th-3rd order moments for white parts of image
    return moments(image, true);
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

vector<Bar> extract_bars_from_line(vector<int> &line) { // get a sequence of bar objects which represents a run of 1's or 0's
    vector<Bar> bars;

    int last_pix = line[0];
    int len = 0;

    for (auto &pix: line) {
        if (pix == last_pix) {
            len++;
        } else {
            bars.push_back(Bar(len, last_pix));
            len = 1;
        }

        last_pix = pix;
    }

    bars.push_back(Bar(len, last_pix));

    return bars;
}

int convert_to_module(int ti, int t) { // takes absolute t-value and total bar width to find modular t value
    float ratio = (float)ti / t;
    return (int)round(7 * ratio);
}

vector<TVal> extract_t_values(vector<Bar> &bars) { // get module t values for EAN-13 decoding
    /* Assumptions: 
            - There is a 'quiet section' of zeros before the first guard bar
              and after the last guard bar
            - There are 12 characters (4 bars each) and 3 guard bars i.e. raw_bars has len 61
    */

    vector<TVal> tvals;

    for (int i=4; i<28; i+=4) { // left half traversal
        vector<Bar> unit(bars.begin() + i, bars.begin() + i + 4); // extract 4 bars at a time

        int total_width = 0; // get the total width of 4 bars
        for (auto &b: unit) {total_width += b.width;}

        TVal tval; // compute the t-values, in reverse since we are on LHS
        tval.t1 = convert_to_module(unit[3].width + unit[2].width, total_width); 
        tval.t2 = convert_to_module(unit[2].width + unit[1].width, total_width); 
        tval.t3 = convert_to_module(unit[1].width + unit[0].width, total_width); 
        tval.t4 = convert_to_module(unit[0].width, total_width); 

        tvals.push_back(tval);
    }

    for (int i=33; i<57; i+=4) { // right half traversal
        vector<Bar> unit(bars.begin() + i, bars.begin() + i + 4); // extract 4 bars at a time

        int total_width = 0; // get the total width of 4 bars
        for (auto &b: unit) {total_width += b.width;}

        TVal tval; // compute the t-values
        tval.t1 = convert_to_module(unit[0].width + unit[1].width, total_width); 
        tval.t2 = convert_to_module(unit[1].width + unit[1].width, total_width); 
        tval.t3 = convert_to_module(unit[2].width + unit[3].width, total_width); 
        tval.t4 = convert_to_module(unit[3].width, total_width); 

        tvals.push_back(tval);
    }

    return tvals;
}

int main() {
    // Mat img = imread("IMG_20240227_0003.jpg");
    Mat img = capture_photo();
    Mat gray = make_grayscale(img);
    Mat bin = apply_otsu_thresholding(gray, 1);

    Moments m = get_moments(bin);
    mark_blob(img, m);

    vector<int> line = get_line_of_pixels(bin);
    vector<Bar> bars = extract_bars_from_line(line);
    vector<TVal> tvals = extract_t_values(bars);

    imshow("Image", img);
    imshow("Grayscale", gray);
    imshow("Binary", bin);
    waitKey(0);

    cout << "Hello, world!" << endl;
    return 0;
}