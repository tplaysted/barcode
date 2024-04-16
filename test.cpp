#include "barcode.h"

int main() {
    vector<int> ref_code = {9,3,1,0,2,3,2,9,5,4,7,9,0};
    String img_dir = "Noisy images";

    vector<double> scores;

    for (int i=10; i<=100; i+=10) {
        String dir = img_dir + "/" + to_string(i) + "/" + to_string(i) + "_ ";
        double score = 0;

        for (int j=1; j<=3; j++) {
            Mat img = imread(dir + "(" + to_string(j) + ").png"); // grab the test image
            Mat gray = make_grayscale(img);
            Mat bin = apply_otsu_thresholding_with_blur(gray, 1); // A bit of blurring 

            Moments m = get_moments(bin); // get orientation of barcode

            vector<int> line = get_line_of_pixels(bin); // get line of pixels across middle of barcode [11000001111000000...]
            vector<Bar> bars = extract_bars_from_line(line); // [{2,1}, {5,0}, {...}]
            vector<TVal> tvals = extract_t_values(bars); // []
            vector<Digit> digits = decode_t_vals(tvals);  // doing the EAN-13 decoding
            orient_digits(digits); // first digit should be odd parity

            vector<int> decoding = get_full_decoding(digits); // figuring out the country code

            for (int k=0; k<13; k++) {
                if (decoding[k] == ref_code[k]) {
                    score += (float) 1 / 39; // checking accuracy
                }
            }
        }

        scores.push_back(score);
    }

    for (auto &score: scores) {
        cout << score << endl;
    }
}