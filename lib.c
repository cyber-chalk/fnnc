#include "./mnist2.h" // may want to write my own because why not
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#define relu(x) ((x) > 0 ? (x) : 0) // maybe make it a leaky reLU
#define BATCH_SIZE 64               // change if not running on chromebook
// TODO: add softmax

//  head -10 ~/2024S2_SProj5_Mnistcnn/data/t10k-images.idx3-ubyte | xxd -b --
//  interesting command

// create reference on first layer (just on the activation) and then just modify
// that one for the rest of the network

// clang-format off
int kernel[3][3] = {
    { 1, 1, 1 },
    { 1, 8, 1 },
    { 1, 1, 1 } // delete later
};
// clang-format on
// convolution on stack
// come back to later
void convolve(double *input, int kernel[3][3], int width, int height) {
  double temp[width * height]; // Temporary array to store the results

  int kSize = 3;

  for (int i = 1; i < height - 1; i++) {
    for (int j = 1; j < width - 1; j++) {
      double sum = 0.0;

      for (int ki = 0; ki < kSize; ki++) {
        for (int kj = 0; kj < kSize; kj++) {
          int image_i = i + ki - 1;
          int image_j = j + kj - 1;
          int image_index = image_i * width + image_j;
          sum += input[image_index] * kernel[ki][kj];
        }
      }

      int output_index = i * width + j;
      temp[output_index] = sum;
    }
  }

  // Copy the temp array back into the input array
  for (int i = 0; i < width * height; i++) {
    input[i] = temp[i];
  }
}

int main() {
  load_mnist(0); // 0.48 mb on the stack
  // run convole on train_image
  // convolve(getSingle(train_image, 0), kernel, 28, 28);
  printSingle(getSingle(train_image, 0));
  printf("\n"); // for nvim terminal
  return 0;
}
