#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "mnist.h"
#define relu(x) ((x) > 0 ? (x) : 0)

// const char *input = "Lenna100.jpg";
// const char *output = "filtered_lenna%d.ppm";

// clang-format off
int kernel[3][3] = {
    { -1, -1, -1 },
    { -1,  8, -1 },
    { -1, -1, -1 }
};
// clang-format on

// could use imglib.h but no
// void convolveB(int input[HEIGHT][WIDTH], int output[HEIGHT][WIDTH],
//               int kernel[3][3]) {
//  int kernelSize = 3; // Kernel size (3x3)
//  int kHalf = kernelSize / 2;

// Iterate over each pixel in the input image (excluding the border pixels)
//  for (int i = kHalf; i < HEIGHT - kHalf; i++) {
//    for (int j = kHalf; j < WIDTH - kHalf; j++) {
//      int sum = 0; // Sum for convolution result
//
//      // Perform convolution with the kernel
//      for (int m = -kHalf; m <= kHalf; m++) {
//        for (int n = -kHalf; n <= kHalf; n++) {
//          sum += input[i + m][j + n] * kernel[kHalf + m][kHalf + n];
//        }
//      }
//
//     // Store the result in the output image
//      output[i][j] = sum;
//   }
//  }
//}
// convolution on stack with return
int *convolve(int **input, int **kernel, int dimention) {
  int *output = malloc(dimention * dimention * sizeof(int)); // sqrt
  int kernelSize = 3;
  int kHalf = kernelSize / 2;

  // iterate over each pixel int eh input image (exculuding border pixels)
  for (int i = kHalf; i < dimention - kHalf; i++) {
    for (int j = kHalf; j < dimention - kHalf; j++) {
      int sum = 0;

      for (int m = -kHalf; m <= kHalf; m++) {
        for (int n = -kHalf; n <= kHalf; n++) {
          sum += input[i + m][j + n] * kernel[kHalf + m][kHalf + n];
        }
      }
      output[i * dimention + j] = sum;
    }
  }
  return output;
}
int main() { return 1; }
