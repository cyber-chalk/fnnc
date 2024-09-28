#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define TRAIN_IMAGE "./data/train-images.idx3-ubyte"
#define TRAIN_LABEL "./data/train-labels.idx1-ubyte"
#define TEST_IMAGE "./data/t10k-images.idx3-ubyte"
#define TEST_LABEL "./data/t10k-labels.idx1-ubyte"

#define SIZE 784 // 28*28
#define NUM_TRAIN 60000
#define NUM_TEST 10000
#define LEN_INFO_IMAGE 4
#define LEN_INFO_LABEL 2

#define MAX_IMAGESIZE 1280
#define MAX_BRIGHTNESS 255
#define MAX_FILENAME 256
#define MAX_NUM_OF_IMAGES 1

unsigned char image[MAX_NUM_OF_IMAGES][MAX_IMAGESIZE][MAX_IMAGESIZE];
int width[MAX_NUM_OF_IMAGES], height[MAX_NUM_OF_IMAGES];

int info_image[LEN_INFO_IMAGE];
int info_label[LEN_INFO_LABEL];

unsigned char train_image_char[NUM_TRAIN][SIZE];
unsigned char test_image_char[NUM_TEST][SIZE];
unsigned char train_label_char[NUM_TRAIN][1];
unsigned char test_label_char[NUM_TEST][1];

double train_image[NUM_TRAIN][SIZE];
double test_image[NUM_TEST][SIZE];
int train_label[NUM_TRAIN];
int test_label[NUM_TEST];

void FlipLong(unsigned char *ptr) {
  unsigned char temp;
  // swap 1st and 4th bytes
  temp = ptr[0];
  ptr[0] = ptr[3];
  ptr[3] = temp;

  // swap 2nd and 3rd bytes
  temp = ptr[1];
  ptr[1] = ptr[2];
  ptr[2] = temp;
}
// avoid stack overflow (but should work)
// also may want to change void | unsigned char (cant because two arrays)
void readToArr(char *filename, int numData, int infoLen, int arrN,
               unsigned char dataChar[][arrN], int infoArr[]) {
  int opened_fd = open(filename, O_RDONLY);
  // basically just reads the size of the data into an array
  read(opened_fd, infoArr, infoLen * sizeof(int));
  // big edian to little
  for (int i = 0; i < infoLen; i++) {
    unsigned char *ptr = (unsigned char *)(infoArr + i);
    FlipLong(ptr);
    ptr += 4; // 4 = size of int
  }

  for (int i = 0; i < numData; i++) {
    read(opened_fd, dataChar[i], arrN * sizeof(unsigned char));
  }
  close(opened_fd);
}

void image_char2double(int num_data, unsigned char data_image_char[][SIZE],
                       double data_image[][SIZE]) {
  for (int i = 0; i < num_data; i++)
    for (int j = 0; j < SIZE; j++)
      data_image[i][j] = (double)data_image_char[i][j] / 255.0;
}

void label_char2int(int num_data, unsigned char data_label_char[][1],
                    int data_label[]) {
  for (int i = 0; i < num_data; i++)
    data_label[i] = (int)data_label_char[i][0];
}

// void convertToDouble(int num_data, unsigned char data_char[][SIZE],
//                      void *data_array, size_t typeSize, double divisor) {
// for (int i = 0; i < num_data; i++) {
//  for (int j = 0; j < SIZE; j++) {
//   if (typeSize == sizeof(double)) {
//    ((double *)data_array)[i * SIZE + j] =
//       (double)data_char[i][j] / divisor;
// } else if (typeSize == sizeof(int)) {
//  ((int *)data_array)[i * SIZE + j] = (int)data_char[i][j];
//}
//}
//}
//}
// clang-format off
/*
 * bool, 1 = test, 0 = train
 * */
void load_mnist(int test) {
    char *image_file = test ? TEST_IMAGE : TRAIN_IMAGE;
    char *label_file = test ? TEST_LABEL : TRAIN_LABEL;
    int num_data = test ? NUM_TEST : NUM_TRAIN;

    readToArr(image_file, num_data, LEN_INFO_IMAGE, SIZE, test ? test_image_char : train_image_char, info_image);
    image_char2double(num_data, test ? test_image_char : train_image_char, test ? test_image : train_image);
    readToArr(label_file, num_data, LEN_INFO_LABEL, 1, test ? test_label_char : train_label_char, info_label);
    label_char2int(num_data, test ? test_label_char : train_label_char, test ? test_label : train_label);
} // confusing and slightly hard to read
// clang-format on

void print_mnist_pixel(double data_image[][SIZE], int num_data) {
  for (int i = 0; i < num_data; i++) {
    printf("image %d/%d\n", i + 1, num_data);
    for (int j = 0; j < SIZE; j++) {
      printf("%1.1f ", data_image[i][j]);
      if ((j + 1) % 28 == 0)
        putchar('\n');
    }
    putchar('\n');
  }
}

void print_mnist_label(int data_label[], int num_data) {
  for (int i = 0; i < num_data; i++) {
    printf("label[%d]: %d\n", i, data_label[i]);
  }
}
