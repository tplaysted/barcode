#include "barcode.h" //  all of the logic for decoding is in this header

int main() {
    Mat img = imread("IMG_20240227_0008.jpg");
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