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

typedef enum _layerType { DENSE, CONV, POOL } LayerType;

typedef struct _layer {

  int width, height; // dimentions of input/image
  int nneurons;      // number of neurons
  double *weights;   // weights for each neuron
  double *biases;    // biases for each neuron
  int nweights;
  int nbiases;
  /* for backpropagation
    double *gradients;
    double *uweights;  // updated weights
    double * ubiases;
  */
  LayerType type;

  union {
    struct {
      // No additional specific params, since fully connected layers mainly use
      // nneurons
    } dense;

    struct {
      int kernel_size;
      int stride;
      int padding;
    } conv;

    struct {
      int pool_size;
      int stride;
    } pool;
  } params;
} Layer;

// clang-format off
int kernel[3][3] = {
    { 1, 1, 1 },
    { 1, 8, 1 },
    { 1, 1, 1 } // delete later
};
// clang-format on
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

  int batchSize = 10;             // Define the size of the batch
  double images[batchSize][SIZE]; // To store the batch of images
  int labels[batchSize];          // To store the batch of labels

  int seekto = 0; // Start from the first image in the dataset
  int test = 0;   // Load training data (test=0)

  // Load a batch of MNIST data
  load_mnist(test, seekto, batchSize, images, labels);
  seekto += batchSize;

  // Print the batch of images and their corresponding labels
  printf("Printing batch of images:\n");
  print_mnist_pixel(images, batchSize); // Print the images
  print_mnist_label(labels, batchSize); // Print the corresponding labels

  return 0;
};
