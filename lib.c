#include "./mnist2.h"
#include "./mongoose.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define relu(x) ((x) > 0 ? (x) : 0)
#define relu_derivative(x) (x > 0) ? 1 : 0
#define BATCH_SIZE 64
#define NUMLAYERS 3 // not including output layer
#define MOMENTUM 0.9
#define epochNum 1
double learningRate = 0.0005;
char *jsonC;

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

// unused
void apply_relu_derivative(double *gradients, double *outputs, int numNodes) {
  for (int i = 0; i < numNodes; i++)
    gradients[i] *= (outputs[i] > 0) ? 1.0 : 0.0;
}

typedef struct _Layer {
  double *weights, *biases;
  double *weightM, *biasM; // momentum

  int nnodes; // number of nodes in next layer
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

void forward(Layer *layer, double *input, double *output, int applyRelu) {
  // n * w + n * w ... + b
  int inputSize = (layer->prevLayer == NULL) ? 784 : layer->prevLayer->nnodes;
  for (int i = 0; i < layer->nnodes; i++) {
    output[i] = layer->biases[i];
  }
  for (int j = 0; j < inputSize; j++) {
    for (int i = 0; i < layer->nnodes; i++) {
      output[i] += input[j] * layer->weights[j * layer->nnodes + i];
    }
  }
  for (int i = 0; i < layer->nnodes; i++) {
    output[i] = relu(output[i]);
  }
}

void back(Layer *layer, double *input, double *dOutput, double *dInput,
          double learningRate) {
  int inputSize = (layer->prevLayer == NULL) ? 784 : layer->prevLayer->nnodes;

  if (dInput != NULL) {
    for (int j = 0; j < inputSize; j++) {
      dInput[j] = 0.0;
      for (int i = 0; i < layer->nnodes; i++) {
        dInput[j] += dOutput[i] * layer->weights[j * layer->nnodes + i];
      }
    }
  }

  for (int j = 0; j < inputSize; j++) {
    double in_j = input[j];
    double *weightRow = &layer->weights[j * layer->nnodes];
    double *momentumRow = &layer->weightM[j * layer->nnodes];
    for (int i = 0; i < layer->nnodes; i++) {
      double grad = dOutput[i] * in_j;
      momentumRow[i] = MOMENTUM * momentumRow[i] + learningRate * grad;
      // Update weights
      if (dInput)
        dInput[j] += dOutput[i] * weightRow[i];
      weightRow[i] -= momentumRow[i];
    }
  }
  for (int i = 0; i < layer->nnodes; i++) {
    layer->biasM[i] = MOMENTUM * layer->biasM[i] + learningRate * dOutput[i];
    layer->biases[i] -= layer->biasM[i];
  }
}

/* returns an array on what the input most likley is */
double *train(Network *net, double *image, int label, double learningRate) {
  double *outputs[NUMLAYERS]; // output foreach layer (3)
  double *gradients[NUMLAYERS + 1];

  gradients[1] = calloc(16, sizeof(double));
  gradients[2] = calloc(16, sizeof(double));
  gradients[3] = calloc(10, sizeof(double));
  // ~ dont need outputs[0] because its the input for foward.1
  outputs[1] = calloc(net->hidden[0].nnodes, sizeof(double));
  outputs[2] = calloc(net->hidden[1].nnodes, sizeof(double));
  double *finalOutput = calloc(10, sizeof(double));
  // final output is from 2->3

  forward(&net->hidden[0], image, outputs[1], 1);
  forward(&net->hidden[1], outputs[1], outputs[2], 1);
  forward(&net->hidden[2], outputs[2], finalOutput, 0);

  for (int i = 0; i < 10; i++) {
    // finalOutput[i] = softmax(finalOutput[i], finalOutput, 10); this kinda
    // sucks
    gradients[3][i] = finalOutput[i] - (i == label);
  }
  // back propogates backward
  back(&net->hidden[2], outputs[2], gradients[3], gradients[2], learningRate);
  back(&net->hidden[1], outputs[1], gradients[2], gradients[1], learningRate);
  back(&net->hidden[0], image, gradients[1], NULL, learningRate);

  free(gradients[1]);
  free(gradients[2]);
  free(gradients[3]);
  free(outputs[1]);
  free(outputs[2]);
  return finalOutput;
}

/* returns the most likley number */
int test(Network *net, double *image) {
  double *outputs[NUMLAYERS + 1]; // (will become array of heap mem)

  for (int i = 1; i <= NUMLAYERS; i++) {
    outputs[i] = calloc(net->hidden[i - 1].nnodes, sizeof(double));
  }

  forward(&net->hidden[0], image, outputs[1], 0);
  for (int i = 1; i < NUMLAYERS; i++) {
    forward(&net->hidden[i], outputs[i], outputs[i + 1], 0); // forward prop
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
    for (int k = 0; k < 3; k++) { // limit to 3 weights
      printf("%.3f ", layer->weights[k]);
    }
    printf("\n");
  }
}
void ev_handler(struct mg_connection *c, int ev, void *ev_data) {
  if (ev == MG_EV_HTTP_MSG) {
    struct mg_http_message *hm = (struct mg_http_message *)ev_data;

    if (mg_match(hm->uri, mg_str("/api"), NULL)) {
      mg_http_reply(c, 200, "Content-Type: application/json\r\n", "%s", jsonC);
    }
    // Serve frontend HTML and assets from the "frontend/" directory
    else {
      struct mg_http_serve_opts opts = {.root_dir = "./frontend/"};
      mg_http_serve_dir(c, hm, &opts);
    }
  }
}
// formats data to json
char *yason(double images[15][SIZE], int labels[15], int rightArr[15],
            double loss, size_t tSize) {
  char *out = malloc(tSize);
  int offset = 0;
  offset += snprintf(out + offset, tSize - offset, "{\n  \"testImages\": [\n");
  for (int i = 0; i < 15; i++) {
    offset += snprintf(out + offset, tSize - offset, "    [");
    for (int j = 0; j < SIZE; j++) {
      offset += snprintf(out + offset, tSize - offset, "%lf", images[i][j]);
      if (j < SIZE - 1)
        offset += snprintf(out + offset, tSize - offset, ", ");
    }
    if (i < 14)
      offset += snprintf(out + offset, tSize - offset, "], \n");
    else
      offset += snprintf(out + offset, tSize - offset, "]\n");
  }
  offset += snprintf(out + offset, tSize - offset, "  ],\n  \"testLabels\": [");
  for (int i = 0; i < 15; i++) {
    offset += snprintf(out + offset, tSize - offset, "%d", labels[i]);
    if (i < 14)
      offset += snprintf(out + offset, tSize - offset, ", ");
  }
  offset += snprintf(out + offset, tSize - offset, "],\n  \"rightArr\": [");

  for (int i = 0; i < 15; i++) {
    offset += snprintf(out + offset, tSize - offset, "%d", rightArr[i]);
    if (i < 14)
      offset += snprintf(out + offset, tSize - offset, ", ");
  }
  offset += snprintf(out + offset, tSize - offset, "],\n \"loss\": %lf", loss);
  offset += snprintf(out + offset, tSize - offset, "\n}");
  out[tSize - 1] = '\0';
  return out;
}

int main() {
  srand(time(NULL));
  // printf("Printing batch of images:\n");
  // print_mnist_pixel(images, batchSize);
  // print_mnist_label(labels, batchSize);
  Network net;

  initLayer(&net.hidden[0], NULL, 16); // first layer
  initLayer(&net.hidden[1], &net.hidden[0], 16);
  initLayer(&net.hidden[2], &net.hidden[1], 10);
  /*
  | ||
  || |_
    */
  double loss = 0;

  int batchSize = BATCH_SIZE;
  int totalImages = 3000;

  for (int epoch = 0; epoch < epochNum; epoch++) {
    int seekto = 0;
    printf("working\n");
    while (seekto < totalImages) {
      double images[batchSize][SIZE];
      int labels[batchSize];
      load_mnist(0, seekto, batchSize, images, labels);
      seekto += batchSize;

      for (int i = 0; i < batchSize; i++) {
        if (epoch > 0)
          shuffle(images, labels, batchSize);

        double *output = train(&net, images[i], labels[i], learningRate);
        double safeOutput = fmax(output[labels[i]], 1e-10);
        loss += -logf(safeOutput);
        free(output);
      }
    }
  }
  printf("finished training, loss: %lf\n", loss);
  printNetworkSummary(&net);

  // testing:

  double testImages[15][SIZE];
  int testLabels[15];
  int rightArr[15];
  load_mnist(1, 0, 15, testImages, testLabels);
  int numCorrect = 0;
  printf("\n)");

  print_mnist_label(testLabels, 15);
  // ! this doesnt print because of the eventloop below (and I dont
  // want to thread it), however, the code still works
  for (int i = 0; i < 15; i++) {
    int res = test(&net, testImages[i]);
    printf("result: %d ", res);
    if (res == testLabels[i]) {
      numCorrect++;
      rightArr[i] = res;
    }
  }
  printf("%d\n", numCorrect);
  printf("%lf\n", learningRate);
  size_t jsonSize = 200000; // yuck
  // global
  jsonC = yason(testImages, testLabels, rightArr, loss, jsonSize);

  // webserver by mongoose
  struct mg_mgr mgr; // Declare event manager
  mg_mgr_init(&mgr); // Initialise event manager
  mg_http_listen(&mgr, "http://127.0.0.1:8000", ev_handler,
                 NULL); // Setup listener
  for (;;) {            // Run an infinite event loop
    mg_mgr_poll(&mgr, 1000);
  }
  free(jsonC);
  for (int i = 0; i < NUMLAYERS; i++) {
    free(net.hidden[i].weights);
    free(net.hidden[i].biases);
    free(net.hidden[i].weightM);
    free(net.hidden[i].biasM);
  }

  return 0;
};
