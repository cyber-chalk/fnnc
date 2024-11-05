#include "./mnist2.h"
#include <assert.h>
#include "./mongoose.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define relu(x) ((x) > 0 ? (x) : 0) // maybe make it a leaky reLU
#define relu_derivative(x) (x > 0) ? 1 : 0
#define BATCH_SIZE 64 // change if not running on chromebook
#define NUMLAYERS 3   // not including input layer
#define MOMENTUM 0.9
#define epochNum 9
// #define learningRate 0.0005
double learningRate = 0.0005;
// try to use stack allocation wherever possible
// https://medium.com/@chenymj23/memory-whether-to-store-on-the-heap-or-the-stack-4ff33b2c1e5f

double expo(double y) {
  if (y > 80)
    y = 80; // limit prevents overflow in exp
  return exp(y);
}

double softmax(double x, double *Niz, int Iter) {
  double Sum = 0;

  // sum of exponentials
  for (int i = 0; i < Iter; i++) {
    Sum += exp(Niz[i]);
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
  Layer hidden[NUMLAYERS]; // array of layers
  // includes output, excludes input
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
  double stddev = sqrt((double)(2.0 / prevNum)); // stdev
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
  for (int i = 0; i < layer->nnodes; i++) {
    output[i] = layer->biases[i];
  }

  for (int i = 0; i < layer->nnodes; i++) {
    for (int j = 0; j < inputSize; j++) {
      output[i] += input[j] * layer->weights[i * inputSize + j]; // +j (i think)
    }
    output[i] = relu(output[i]);
  }
}

void back(Layer *layer, double *input, double *dInput, double *dOutput,
          double learningRate) {
  assert(layer != NULL && "Layer cannot be null");
  assert(input != NULL && "Input cannot be null");
  assert(dOutput != NULL && "dOutput cannot be null");
  assert(learningRate > 0 && "Learning rate must be positive");
  assert(learningRate < 1.0 && "Learning rate should be less than 1.0");
  /* dinput/output: The gradient of the loss function with respect to the
   * input/output of the current layer (i.e., the error from the next layer or
   * the loss function). */
  int inputSize = (layer->prevLayer == NULL) ? 784 : layer->prevLayer->nnodes;

  assert(layer->nnodes > 0 && "Number of nodes must be positive");
  assert(layer->weights != NULL && "Weights array cannot be null");
  assert(layer->biases != NULL && "Biases array cannot be null");
  assert(layer->weightM != NULL && "Weight momentum array cannot be null");
  assert(layer->biasM != NULL && "Bias momentum array cannot be null");

  if (dInput) {
    for (int j = 0; j < inputSize; j++) {
      // dInput[j] = 0.0f; // calloced
      for (int i = 0; i < layer->nnodes; i++) {
        dInput[j] += dOutput[i] * layer->weights[j * layer->nnodes + i];
        // assert(fabs(dInput[j]) < 1e6 && "Gradient too large - possible
        // exploding gradient");
      }
    }
  }

  // update weights and momentum foreach input/output connection
  for (int j = 0; j < inputSize; j++) {
    double in_j = input[j];
    for (int i = 0; i < layer->nnodes; i++) {
      double grad = dOutput[i] *
                    in_j; // gradient for weight between input[j] and output[i]

      // assert(fabs(grad) > 1e-15 && "Gradient too small - possible vanishing
      // gradient"); assert(fabs(grad) < 1e6 && "Gradient too large - possible
      // exploding gradient");

      // weight momentum (momentum * prev momentum + learning rate *
      // current gradient)
      layer->weightM[j * layer->nnodes + i] =
          MOMENTUM * layer->weightM[j * layer->nnodes + i] +
          learningRate * grad;

      // update weights thru momentum
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

/* returns an array on what the input most likley is */
double *train(Network *net, double *image, int label, double learningRate) {
  double *output = calloc(10, sizeof(double));
  double *outputs[NUMLAYERS + 1]; // output foreach layer
  // outputs[0] = image;
  // fix indexing
  for (int i = 1; i <= NUMLAYERS; i++) {
    outputs[i] = calloc(net->hidden[i - 1].nnodes, sizeof(double));
  }

  forward(&net->hidden[0], image, outputs[1]);
  for (int i = 1; i < NUMLAYERS; i++) {
    forward(&net->hidden[i], outputs[i],
            outputs[i + 1]); // forward prop
  }

  for (int i = 0; i < 10; i++) {
    outputs[NUMLAYERS][i] =
        softmax(outputs[NUMLAYERS][i], outputs[NUMLAYERS], 10);
    output[i] = outputs[NUMLAYERS][i] - (i == label);
  }

  // back propogates backwards, may cause segfault
  for (int i = NUMLAYERS - 1; i >= 1; i--) {
    for (int j = 0; j < net->hidden[i].nnodes; j++) {
      outputs[i][j] *= relu_derivative(outputs[i][j]); // Apply ReLU derivative
    }
    back(&net->hidden[i], outputs[i], outputs[i + 1], output, learningRate);
  }
  back(&net->hidden[0], image, NULL, output, learningRate);

  for (int i = 1; i <= NUMLAYERS; i++) {
    free(outputs[i]);
  }
  return output;
}

/* returns the most likley number */
int test(Network *net, double *image) {
  double *outputs[NUMLAYERS + 1]; // (will become array of heap mem)

  for (int i = 1; i <= NUMLAYERS; i++) {
    outputs[i] = calloc(net->hidden[i - 1].nnodes, sizeof(double));
  }

  forward(&net->hidden[0], image, outputs[1]);
  for (int i = 1; i < NUMLAYERS; i++) {
    forward(&net->hidden[i], outputs[i],
            outputs[i + 1]); // forward prop
  }

  for (int i = 0; i < 10; i++) {
    outputs[NUMLAYERS][i] =
        softmax(outputs[NUMLAYERS][i], outputs[NUMLAYERS], 10);
  }

  int highest = 0;
  for (int i = 0; i < 10; i++) {
    if (outputs[NUMLAYERS][i] > outputs[NUMLAYERS][highest])
      highest = i;
  }

  for (int i = 1; i <= NUMLAYERS; i++) {
    free(outputs[i]);
  }

  return highest;
}

void shuffle(double (*array)[SIZE], int labels[], size_t n) {
  double(*pointers[n])[SIZE];
  for (size_t i = 0; i < n; i++) {
    pointers[i] = &array[i];
  }

  // shuffle the pointers using Fisher-Yates algorithm
  for (size_t i = 0; i < n - 1; i++) {
    size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
    double(*temp)[SIZE] = pointers[j];
    pointers[j] = pointers[i];
    pointers[i] = temp;

    // labels
    int tempLabel = labels[j];
    labels[j] = labels[i];
    labels[i] = tempLabel;
  }

  // rebuild
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < SIZE; j++) {
      array[i][j] = (*pointers[i])[j];
    }
  }
}

void printNetworkSummary(Network *net) {
  printf("Network Structure Summary:\n");

  for (int i = 0; i < NUMLAYERS; i++) {
    Layer *layer = &net->hidden[i];
    printf("Layer %d:\n", i + 1);
    printf("  Nodes: %d\n", layer->nnodes);

    // Display a few sample biases
    printf("  Sample Biases: ");
    for (int j = 0; j < 3 && j < layer->nnodes; j++) { // limiting to 3 nodes
      printf("%lf ", layer->biases[j]);
    }
    printf("\n");

    // Display a few sample weights for the first node
    printf("  Sample Weights for Node 0: ");
    int ninputs = layer->prevLayer ? layer->prevLayer->nnodes : 0;
    for (int k = 0; k < 3 && k < ninputs; k++) { // limit to 3 weights
      printf("%.3f ", layer->weights[k]);
    }
    printf("\n");
  }
}

int main() {
  srand(time(NULL));
  // printf("Printing batch of images:\n");
  // print_mnist_pixel(images, batchSize);
  // print_mnist_label(labels, batchSize);
  Network net;
  // // Layer layers[NUMLAYERS];

  initLayer(&net.hidden[0], NULL, 16); // first layer
  for (int i = 1; i < NUMLAYERS; i++)
    initLayer(&net.hidden[i], &net.hidden[i - 1], 16); // um ?

  /*
  | ||
  || |_
    */
  double loss = 0;

  int batchSize = BATCH_SIZE;
  int totalImages = 60000;

  for (int epoch = 0; epoch < 1; epoch++) {
    int seekto = 0;
    printf("working\n");
    while (seekto < totalImages) {
      double images[batchSize][SIZE];
      int labels[batchSize];
      load_mnist(0, seekto, batchSize, images, labels);
      seekto += batchSize;

      for (int i = 0; i < batchSize; i++) {
        if (epoch != 1)
          shuffle(images, labels, batchSize);

        double *output = train(&net, images[i], labels[i], learningRate);
        /*for (int i = 0; i < 10; i++)*/
        /*  printf("%lf \n", output[i]);*/
        double safeOutput = fmax(output[labels[i]], 1e-10);
        loss += -logf(safeOutput);
        /*loss += -logf(output[labels[i]] + 1e-10); // cross entropy loss*/
        free(output);
      }
    }
  }
  printf("finished training, loss: %lf\n", loss);
  printNetworkSummary(&net);

  // testing:

  double testImages[15][SIZE];
  int testLabels[15];
  load_mnist(1, 0, 15, testImages, testLabels);
  int numCorrect = 0;
  printf("\n)");
  print_mnist_label(testLabels, 15);
  for (int i = 0; i < 15; i++) {
    if (i == 1) {
      for (int j = 0; j < SIZE; j++)
        printf("%lf", testImages[i][j]);
    }
    int res = test(&net, testImages[i]);
    printf("result: %d ", res);
    if (res == testLabels[i])
      numCorrect++;
  }

  printf("accuracy %d / %d\n", numCorrect, 15);
  double newarr[] = {1.0, 2.0, 3.0};
  printf("softmax test: %lf\n",
         softmax(1, newarr, sizeof(newarr) / sizeof(double)));

  for (int i = 0; i < NUMLAYERS; i++) {
    free(net.hidden[i].weights);
    free(net.hidden[i].biases);
    free(net.hidden[i].weightM);
    free(net.hidden[i].biasM);
  }

  return 0;
};
