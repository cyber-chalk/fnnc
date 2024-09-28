#include "./mnist2.h" // may want to write my own because why not
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#define relu(x) ((x) > 0 ? (x) : 0) // maybe make it a leaky reLU
#define IMAGE_SIZE 28 * 28

// head -10 t10k-images.idx3-ubyte | xxd -b -- interesting command

// clang-format off
int kernel[3][3] = {
    { -1, -1, -1 },
    { -1,  8, -1 },
    { -1, -1, -1 }
};
// clang-format on
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

int main() {
  const char *filename = "./t10k-images.idx3-ubyte";
  load_mnist(0);
  print_mnist_pixel(train_image, 1);
  printf("\n"); // for nvim terminal
  return 1;
}
