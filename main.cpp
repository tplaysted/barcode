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

class Digit {
    public: 
        int val = -1;
        bool even = true;
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

Mat apply_otsu_thresholding(Mat &image) { // otsu thresholding
    Mat target;
    threshold(image, target, THRESHOLD, 255, THRESH_BINARY_INV + THRESH_OTSU);  // do thresholding
    return target;
}

Mat apply_otsu_thresholding_with_blur(Mat &image, int radius) { // otsu thresholding
    Mat target;
    Mat blur;
    GaussianBlur(image, blur, Size(0, 0), radius, radius);
    threshold(image, target, THRESHOLD, 255, THRESH_BINARY_INV + THRESH_OTSU);  // do thresholding
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

void mark_line(Mat &image, Moments &m) { // draw the axis of orientation on a blob
    vector<int> c = get_centroid(m);
    double o = get_orientation(m);
    int w = image.size().width; // scan across entire image

    Point2i long_1, long_2;

    long_1 = Point2i(c[0] - w * cos(o), c[1] + w * sin(o));
    long_2 = Point2i(c[0] + w * cos(o), c[1] - w * sin(o));

    line(image, long_1, long_2, Scalar(0, 0, 255), 2, LINE_AA);
}

vector<int> get_line_of_pixels(Mat &image) { // scans a line of pixels across the bar code
    Moments m = get_moments(image);
    vector<int> c = get_centroid(m);
    double o = get_orientation(m);
    int w = image.size().width; // scan across entire image

    Point2i long_1, long_2; // poits to draw between

    long_1 = Point2i(c[0] - w * cos(o), c[1] + w * sin(o));
    long_2 = Point2i(c[0] + w * cos(o), c[1] - w * sin(o));

    LineIterator it = LineIterator(image, long_1, long_2, 4);
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

int convert_to_module_seven(int ti, int t) { // takes absolute t-value and total bar width to find modular t value
    float ratio = (float)ti / t;
    int mod = (int)round(7 * ratio);

    if (mod > 5) {return 5;} // in case we over shoot
    if (mod < 1) {return 1;} // in case we under shoot

    return mod;
}

int convert_to_module_three(int ti, int t) { // takes absolute t-value and total bar width to find modular t value
    float ratio = (float)ti / t;
    int mod = (int)round(3 * ratio);

    return mod;
}

bool is_outer_guard_bar(vector<Bar> &bars) { // looks at 3 bars and checks whether it is an outer guard bar
    if(bars[0].type == 0) {return false;} // guard bar begins with white

    int total_width = 0; // get the total width of 3 bars
    for (auto &b: bars) {total_width += b.width;}

    int t1 = convert_to_module_three(bars[0].width + bars[1].width, total_width);
    int t2 = convert_to_module_three(bars[1].width + bars[2].width, total_width);

    return (t1 == 2) & (t2 == 2); // all t=2 corresponds to middle guard bar
}

vector<TVal> extract_t_values(vector<Bar> &bars) { // get module t values for EAN-13 decoding
    /* Assumptions: 
            - There is a 'quiet section' of zeros before the first guard bar
              and after the last guard bar
            - There are 12 characters (4 bars each) and 3 guard bars i.e. bars has len at least 61
    */
    // First we need to locate the left guard bar
    int quiet_zone_start = 0;
    for (int i=0; i<bars.size() - 3; i++) {
        vector<Bar> three(bars.begin() + i, bars.begin() + i + 3);
        if (is_outer_guard_bar(three)) {
            quiet_zone_start = i - 1;
            break;
        }
    }

    vector<TVal> tvals;

    for (int i=quiet_zone_start + 4; i<quiet_zone_start + 28; i+=4) { // left half traversal
        vector<Bar> unit(bars.begin() + i, bars.begin() + i + 4); // extract 4 bars at a time

        int total_width = 0; // get the total width of 4 bars
        for (auto &b: unit) {total_width += b.width;}

        TVal tval; // compute the t-values, in reverse since we are on LHS
        tval.t1 = convert_to_module_seven(unit[3].width + unit[2].width, total_width); 
        tval.t2 = convert_to_module_seven(unit[2].width + unit[1].width, total_width); 
        tval.t3 = convert_to_module_seven(unit[1].width + unit[0].width, total_width); 
        tval.t4 = convert_to_module_seven(unit[0].width, total_width); 

        tvals.push_back(tval);
    }

    for (int i=quiet_zone_start + 33; i<quiet_zone_start + 57; i+=4) { // right half traversal
        vector<Bar> unit(bars.begin() + i, bars.begin() + i + 4); // extract 4 bars at a time

        int total_width = 0; // get the total width of 4 bars
        for (auto &b: unit) {total_width += b.width;}

        TVal tval; // compute the t-values
        tval.t1 = convert_to_module_seven(unit[0].width + unit[1].width, total_width); 
        tval.t2 = convert_to_module_seven(unit[1].width + unit[2].width, total_width); 
        tval.t3 = convert_to_module_seven(unit[2].width + unit[3].width, total_width); 
        tval.t4 = convert_to_module_seven(unit[3].width, total_width); 

        tvals.push_back(tval);
    }

    return tvals;
}

Digit decode_t_val(TVal &tval) { // convert a set of t-values into an EAN-13 digit
    Digit d;

    switch (tval.t1) {
        case 2: {
            switch (tval.t2) {
                case 2: d.even = true; d.val = 6; break;
                case 3: d.even = false; d.val = 0; break;
                case 4: d.even = true; d.val = 4; break;
                case 5: d.even = false; d.val = 3; break;
                default: break;
            }
        } break;
        case 3: {
            switch (tval.t2) {
                case 2: d.even = false; d.val = 9; break;
                case 3: {
                    switch (tval.t4) {
                        case 2: d.even = true; d.val = 2; break;
                        case 3: d.even = true; d.val = 8; break;
                        default: break;
                    }
                } break;
                case 4: {
                    switch (tval.t4) {
                        case 2: d.even = false; d.val = 1; break;
                        case 1: d.even = false; d.val = 7; break;
                        default: break;
                    }
                } break;
                case 5: d.even = true; d.val = 5; break;
                default: break;
            }
        }  break;
        case 4: {
            switch (tval.t2) {
                case 2: d.even = true; d.val = 9; break;
                case 3: {
                    switch (tval.t4) {
                        case 2: d.even = false; d.val = 2; break;
                        case 1: d.even = false; d.val = 8; break;
                        default: break;
                    }
                } break;
                case 4: {
                    switch (tval.t4) {
                        case 1: d.even = true; d.val = 1; break;
                        case 2: d.even = true; d.val = 7; break;
                        default: break;
                    }
                } break;
                case 5: d.even = false; d.val = 5; break;
                default: break;
            }
        } break;
        case 5: {
            switch (tval.t2) {
                case 2: d.even = false; d.val = 6; break;
                case 3: d.even = true; d.val = 0; break;
                case 4: d.even = false; d.val = 4; break;
                case 5: d.even = true; d.val = 3; break;
                default: break;
            }
        } break;
        default: break;
    }

    return d;
}

vector<Digit> decode_t_vals(vector<TVal> &tvals) {
    vector<Digit> digits;
    for (auto &t: tvals) {digits.push_back(decode_t_val(t));}

    return digits;
}

void orient_digits(vector<Digit> &digits) { // checks if orientation is correct, and reverses digits if not
    // The leftmost digit alwas has odd parity in an EAN-13 encoding
    if (digits[0].even) {reverse(digits.begin(), digits.end());}
}

int get_country_code(vector<Digit> &digits) { // checks parity of first 7 digits to find the country code
    char p = 0b0;

    for (int i=0; i<6; i++) {
        if (digits[i].even) {
            p = p << 1;
        } else {
            p += 1;
            p = p << 1;
        }
    }

    switch (p) {
        case 0b1111110: return 0; break;
        case 0b1101000: return 1; break;
        case 0b1100100: return 2; break;
        case 0b1100010: return 3; break;
        case 0b1011000: return 4; break;
        case 0b1001100: return 5; break;
        case 0b1000110: return 6; break;
        case 0b1010100: return 7; break;
        case 0b1010010: return 8; break;
        case 0b1001010: return 9; break;
        default: return -1;
    }
}

vector<int> get_full_decoding(vector<Digit> &digits) {
    vector<int> decoding;
    decoding.push_back(get_country_code(digits));

    for (auto &d: digits) {decoding.push_back(d.val);}

    return decoding;
}

int get_checksum(vector<int> &digits) { // get the checksum of an EAN-13 digit string
    int odds = 0;
    int evens = 0;

    for (int i=1; i<13; i+=2) {odds += digits[i];}
    for (int i=0; i<12; i+=2) {evens += digits[i];}

    int check = (3 * odds + evens) % 10;

    if (check == 0) {
        return check;
    } else {
        return 10 - check;
    }
}


int main() {
    Mat img = imread("angle.png");
    // Mat img = capture_photo();
    Mat gray = make_grayscale(img);
    Mat bin = apply_otsu_thresholding(gray);

    Moments m = get_moments(bin); // get orientation of barcode
    mark_line(img, m);

    imshow("Binary", bin);
    imshow("Image", img);
    waitKey(0);

    vector<int> line = get_line_of_pixels(bin); // get line of pixels across middle of barcode [11000001111000000...]
    vector<Bar> bars = extract_bars_from_line(line); // [{2,1}, {5,0}, {...}]
    vector<TVal> tvals = extract_t_values(bars); // []
    vector<Digit> digits = decode_t_vals(tvals);  // doing the EAN-13 decoding
    orient_digits(digits); // first digit should be odd parity

    vector<int> decoding = get_full_decoding(digits); // figuring out the country code

    cout << "Decoded values:" << endl; // Print decoded values
    cout << decoding[0] << " ";
    for (int i=1; i<7; i++) {cout << decoding[i];} cout << " ";
    for (int i=7; i<13; i++) {cout << decoding[i];}
    cout << endl;

    cout << "Last digit is " << decoding[12] << " where the checksum is " << get_checksum(decoding) << endl;

    return 0;
}