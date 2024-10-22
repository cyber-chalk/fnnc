#include "./mnist2.h" // may want to write my own because why not
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#define relu(x) ((x) > 0 ? (x) : 0) // maybe make it a leaky reLU
#define BATCH_SIZE 64               // change if not running on chromebook
// TODO: add softmax 
// try to use stack allocation wherever possible https://medium.com/@chenymj23/memory-whether-to-store-on-the-heap-or-the-stack-4ff33b2c1e5f

//  head -10 ~/2024S2_SProj5_Mnistcnn/data/t10k-images.idx3-ubyte | xxd -b --
//  interesting command

typedef struct _Layer {
  double *weights, *biases;
  double *weightM, *biasM; // momentum

  int nnodes; // number of nodes
  struct _Layer *prevLayer;
} Layer;

typedef struct _Network {
  int numLayers;
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

void initNetwork(Network *net, int *nodes) {
  // probably move to main
 
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
  net.numLayers = 3;


  initLayer(&net.hidden[0], NULL, 16); // first layer
  for (int i = 1; i < net.numLayers; i++) 
    initLayer(&net.hidden[i], &net.hidden[i - 1], 16);

  return 0;
};
