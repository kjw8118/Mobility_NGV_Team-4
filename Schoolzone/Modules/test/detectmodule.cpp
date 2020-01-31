#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

typedef unsigned char uint8_t;
typedef unsigned short uint16_t;

extern "C" {
uint16_t* line_detect(uint8_t* img, uint16_t H, uint16_t W){
    
    uint16_t Line[2][H];
    
    /* Deactivated for module's input/output test
    uint16_t* Line_L = malloc(sizeof(uint16_t)*(H+1));
    uint16_t* Line_R = malloc(sizeof(uint16_t)*(H+1));
    uint8_t band =100, band_tmp1, band_tmp2;
    uint16_t np_L, np_R;
    uint16_t n_L, n_R;
    Line_L[0] = (W*3/8);
    Line_R[0] = (W*5/8);
    for(uint16_t j=0; j<H; j++){
        if(Line_L[j] - band <0) band_tmp1 = Line_L[j];
        if(Line_R[j] + band >W) band_tmp2 = W - Line_R[j];
        if(band_tmp1 >= band_tmp2) band = band_tmp2;
        else band = band_tmp1;
        
        np_L = 0;
        np_R = 0;
        n_L = 0;
        n_R = 0;
        uint16_t* List_L = malloc(sizeof(uint16_t)*2*band);
        uint16_t* List_R = malloc(sizeof(uint16_t)*2*band);
        for(uint16_t k=0;k<(2*band);k++){
            List_L[k] = Line_L[j] - band + k;
            List_R[k] = Line_R[j] - band + k;
            np_L = np_L + img[W*j+List_L[k]] * List_L[k];
            np_R = np_R + img[W*j+List_R[k]] * List_R[k];
            n_L = n_L + List_L[k];
            n_R = n_R + List_R[k];
        }
        if(n_L != 0) Line_L[j+1] = (np_L/n_L);
        else Line_L[j+1] = Line_L[j];
        if(n_R != 0) Line_R[j+1] = (np_R/n_R);
        else Line_R[j+1] = Line_R[j];
        Line[0][j] = Line_L[j+1];
        Line[1][j] = Line_R[j+1];
        
        free(List_L);
        free(List_R);
    }
    free(Line_L);
    free(Line_R);
    */
    
    Mat image(H,W,CV_8UC3, img);
    namedWindow("Window",CV_WINDOW_AUTOSIZE);
    imshow("Window", image);
    waitKey(0);
    
    return *Line;
}

void line_detect_free(uint8_t* img, uint16_t* Line){
    free(img);
    free(Line);
}
}
