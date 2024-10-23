#include "./mnist2.h" // may want to write my own because why not
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define relu(x) ((x) > 0 ? (x) : 0) // maybe make it a leaky reLU
#define relu_derivative(x) (x > 0) ? 1 : 0
#define BATCH_SIZE 64 // change if not running on chromebook
#define NUMLAYERS 3
#define MOMENTUM 0.9
// try to use stack allocation wherever possible
// https://medium.com/@chenymj23/memory-whether-to-store-on-the-heap-or-the-stack-4ff33b2c1e5f

//  head -10 ~/2024S2_SProj5_Mnistcnn/data/t10k-images.idx3-ubyte | xxd -b --
//  interesting command

double expo(double y) {
  if (y > 80)
    y = 80; // limit prevents overflow in exp
  return exp(y);
}

double Softmax(double x, double *Niz, int Iter) {
  double Sum = 0;

  // sum of exponentials
  for (int i = 0; i < Iter; i++) {
    Sum += expo(Niz[i]);
  }

  if (Sum == 0)
    Sum = 0.001;

  return (expo(x)) / Sum;
}

typedef struct _Layer {
  double *weights, *biases;
  double *weightM, *biasM; // momentum

  int nnodes; // number of nodes
  struct _Layer *prevLayer;
} Layer;

typedef struct _Network {
  Layer hidden[3]; // array of layers
  // Layer output;
} Network;

void initLayer(Layer *layer, Layer *prev, int size) {
  int prevNum = (prev == NULL) ? 784 : prev->nnodes;
  int n = size * prevNum; // total number of weights, each input/previous
                          // neuron is connected to each neuron in this layer
  layer->weights = malloc(n * sizeof(double));
  layer->biases = calloc(size, sizeof(double));
  layer->weightM = calloc(n, sizeof(double));
  layer->biasM = calloc(size, sizeof(double));

  // (double)rand() / (((double)(RAND_MAX) + 1) / 2); // [0, 2)
  // https://www.baeldung.com/cs/ml-neural-network-weights
  // https://medium.com/@shauryagoel/kaiming-he-initialization-a8d9ed0b5899
  double stddev = sqrt(2.0 / prevNum); // stdev
  for (int i = 0; i < n; i++) {
    layer->weights[i] = ((double)rand() / RAND_MAX) * stddev -
                        (stddev / 2); // uniform distribution
  }
  layer->nnodes = size;
  layer->prevLayer = prev;
}

void forward(Layer *layer, double *input, double *output) {
  // n * w + n * w ... + b
  int inputSize = (layer->prevLayer == NULL) ? 784 : layer->prevLayer->nnodes;
  for (int i = 0; i < inputSize; i++) {
    output[i] = layer->biases[i];
    for (int j = 0; i < inputSize; i++) {
      output[i] += input[j] * layer->weights[i * inputSize + j]; // +j (i think)
    }
    output[i] = relu(output[i]);
  }
}

void back(Layer *layer, double *input, double *dInput, double *dOutput,
          double learningRate) {
  /* dinput/output: The gradient of the loss function with respect to the
   * input/output of the current layer (i.e., the error from the next layer or
   * the loss function). */
  int inputSize = (layer->prevLayer == NULL) ? 784 : layer->prevLayer->nnodes;
  if (dInput) {
    for (int j = 0; j < inputSize; j++) {
      dInput[j] = 0.0f; // maybe calloc
      for (int i = 0; i < layer->nnodes; i++) {
        dInput[j] += dOutput[i] * layer->weights[j * layer->nnodes + i];
      }
    }
  }

  // update weights and momentum foreach input-output connection
  for (int j = 0; j < inputSize; j++) {
    double in_j = input[j]; // Input node value
    for (int i = 0; i < layer->nnodes; i++) {
      double grad = dOutput[i] *
                    in_j; // Gradient for weight between input[j] and output[i]

      // weight momentum (momentum * prev momentum + learning rate *
      // current gradient)
      layer->weightM[j * layer->nnodes + i] =
          MOMENTUM * layer->weightM[j * layer->nnodes + i] +
          learningRate * grad;

      // Update weights using momentum
      layer->weights[j * layer->nnodes + i] -=
          layer->weightM[j * layer->nnodes + i];
    }
  }

  // update biases and their momentum
  for (int i = 0; i < layer->nnodes; i++) {
    // update bias momentum
    layer->biasM[i] = MOMENTUM * layer->biasM[i] + learningRate * dOutput[i];

    // update bias thru momentum
    layer->biases[i] -= layer->biasM[i];
  }
}

int main() {
  srand(time(NULL));
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

  // maybe put into function?
  Network net;

  initLayer(&net.hidden[0], NULL, 16); // first layer
  for (int i = 1; i < NUMLAYERS; i++)
    initLayer(&net.hidden[i], &net.hidden[i - 1], 16);

  return 0;
};
