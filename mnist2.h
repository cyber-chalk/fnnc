#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>

#define TRAIN_IMAGE "./data/train-images.idx3-ubyte"
#define TRAIN_LABEL "./data/train-labels.idx1-ubyte"
#define TEST_IMAGE "./data/t10k-images.idx3-ubyte"
#define TEST_LABEL "./data/t10k-labels.idx1-ubyte"

#define SIZE 784        // 28*28
#define NUM_TRAIN 60000 // number of images (train)
#define NUM_TEST 10000  // number of images (test)
#define LEN_INFO_IMAGE 4
#define LEN_INFO_LABEL 2

int info_image[LEN_INFO_IMAGE];
int info_label[LEN_INFO_LABEL];

unsigned char train_image_char[NUM_TRAIN][SIZE];
unsigned char test_image_char[NUM_TEST][SIZE];
unsigned char train_label_char[NUM_TRAIN][1];
unsigned char test_label_char[NUM_TEST][1];
// array of images (array of pixels/doubles)
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
// more efficient to get array into unsigned char and then convert to double
// Since double is 8 bytes and unsigned char is 1 byte, its more efficient to
// have both

// filename, number of images, length of info, length of array, seek, array of
// data, info arr
void readToArr(char *filename, int numData, int infoLen, int arrN, int seekto,
               unsigned char dataChar[][arrN], int infoArr[]) {
  int opened_fd = open(filename, O_RDONLY);
  // basically just reads the size of the data into an array
  read(opened_fd, infoArr, infoLen * sizeof(int));
  // big edian to little
  for (int i = 0; i < infoLen; i++) {
    unsigned char *ptr = (unsigned char *)(infoArr + i);
    FlipLong(ptr);
    ptr += 4; // 4 = sizoef(int);
  }

  // Seek to the specified position (seekto) in the file to start reading
  lseek(opened_fd, seekto * arrN * sizeof(unsigned char), SEEK_CUR);

  for (int i = 0; i < numData; i++) {
    read(opened_fd, dataChar[i], arrN * sizeof(unsigned char));
  }
  close(opened_fd);
}
// number of images, array of pixels, array of doubles
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

/*
 * bool, 1 = test, 0 = train
 * */
void load_mnist(int test, int seekto, int bSize, double images[bSize][SIZE],
                int labels[bSize]) {

  unsigned char temp_image_char[bSize][SIZE];
  unsigned char temp_label_char[bSize][1];

  if (test == 1) {
    // Test data, load all at once
    readToArr(TEST_IMAGE, NUM_TEST, LEN_INFO_IMAGE, SIZE, seekto,
              temp_image_char, info_image);
    image_char2double(bSize, temp_image_char, images);

    readToArr(TEST_LABEL, NUM_TEST, LEN_INFO_LABEL, 1, seekto, temp_label_char,
              info_label);
    label_char2int(bSize, temp_label_char, labels);
  } else {
    // Training data, process in batches
    readToArr(TRAIN_IMAGE, bSize, LEN_INFO_IMAGE, SIZE, seekto, temp_image_char,
              info_image);
    image_char2double(bSize, temp_image_char, images);

    readToArr(TRAIN_LABEL, bSize, LEN_INFO_LABEL, 1, seekto, temp_label_char,
              info_label);
    label_char2int(bSize, temp_label_char, labels);
  }
}

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

double *getSingle(double data_image[][SIZE], int single) {
  return data_image[single];
}

void printSingle(double *data_image) {
  for (int i = 0; i < SIZE; i++) {
    printf("%1.1f ", data_image[i]);
    if ((i + 1) % 28 == 0)
      putchar('\n');
  }
}

void print_mnist_label(int data_label[], int num_data) {
  for (int i = 0; i < num_data; i++) {
    printf("label[%d]: %d\n", i, data_label[i]);
  }
}
