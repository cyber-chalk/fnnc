# 2024S2_SProj5_Mnistcnn
Student: 8161133. Project for Semester 2, 2024 of Structured Project (Digital Technologies).

Initial description (Week 9, Semester2): 'fully connected neural network in c . analyzing the mnist dataset of numbers https://www.kaggle.com/datasets/hojjatk/mnist-dataset'

how to run:
either use cmake or
gcc lib.c mongoose.c -o exec -lm
./exec
wait for network to finish
open localhost:8000


```mermaid
flowchart TD
    A@{ shape: circle, label: "Start" } --> B[Compile and Execute CMake]
    B --> C[Wait for Neural Network Training/Testing]
    C --> |Training/Testing Complete|D[Open Website]
    D --> E[Preview Testing Results on Website]
    E --> F[End]
```


```mermaid
sequenceDiagram
    actor User as User
    participant Data as Data
    participant Network as Network
    participant Train as Train Function
    participant Test as Test Function
    participant Forward as Forward Propagation
    participant Backward as Backward Propagation
    participant Softmax as Softmax
    participant Webserver as Webserver
   
   User->>+Network: initialise network
   Network-->>-User: structure of network in memory

   
    loop epoch
    loop batch
    User->>+Data: load batch of images
    Data-->>-User: batch of images 
    loop single image
    User->>+Train: Calls test() with image and label
    Train->>Train: initialises output arrays
    loop Moves Forward for each layer
    Train->>+Forward: calls foward()
    Forward-->>-Train: Returns an output for next layer/final output
    end
    loop Moves backwards for each layer
    Train->>+Backward: calls back()
    %%Backward-->>-Train: 
    Backward->>-Network: Affects turning of neural network
    end
    loop Softmax
    Train->>+Softmax: calls softmax()
    Softmax-->>-Train: applies softmax on outputs from foward()
    end
    Train-->>-User: Returns its opinion
    end 
    end
    end
    loop each test image (15)
    User->>+Test: test the network
    Test->>Test: Ititialises output arrays
    loop  Moves Forward for each layer
    Test->>+Forward: calls foward()
    Forward-->>-Test: returns an output for each a layer
    end
    loop for each number of final output layer
    Test->>+Softmax: calls softmax for each index in output array
    Softmax-->>-Test: returns affected array
    end
    Test-->>-User: returns highest choice from output array
    end
    User->>+Webserver: starts webserver
    Webserver->>-User: Webpage on localhost 8000
```

```mermaid
sequenceDiagram
    actor User as User
    participant Data as Data
    participant Network as Network
    participant Train as Train Function
    participant Test as Test Function
    participant Forward as Forward Propagation
    participant Backward as Backward Propagation
    participant Softmax as Softmax
    participant Webserver as Webserver
   
    User->>+Network: Initialize network
    Network-->>-User: Returns network structure in memory

  

    loop epoch
        loop batch
            User->>+Data: Load batch of images
            Data-->>-User: Returns batch of images
            loop single image
                User->>+Train: Calls train() with image and label
                Train->>Train: Initializes output arrays
                
                loop Forward through layers
                    Train->>+Forward: Calls forward()
                    Forward-->>-Train: Outputs for next layer or final output
                end

                loop Backward through layers
                    Train->>+Backward: Calls back()
                    Backward->>-Network: Updates neural network parameters
                end

                Train->>+Softmax: Calls softmax()
                Softmax-->>-Train: Applies softmax on final outputs
                
                Train-->>-User: Returns classification
            end
        end
    end

    loop each test image (15)
        User->>+Test: Tests the network
        Test->>Test: Initializes output arrays

        loop Forward through layers
            Test->>+Forward: Calls forward()
            Forward-->>-Test: Outputs for each layer
        end

        loop Softmax for each output
            Test->>+Softmax: Calls softmax()
            Softmax-->>-Test: Returns affected output
        end

        Test-->>-User: Returns highest choice from output array
    end

    User->>+Webserver: Starts web server
    Webserver-->>-User: Serves webpage on localhost:8000

```

```mermaid
classDiagram
    class Layer {
        - double* weights
        - double* biases
        - double* weightM
        - double* biasM
        - int nnodes
        - Layer* prevLayer
        + void initLayer(Layer* layer, Layer* prev, int size)
        + void forward(Layer* layer, double* input, double* output)
        + void back(Layer* layer, double* input, double* dInput, double* dOutput, double learningRate)
    }

    class Network {
        - Layer hidden[NUMLAYERS]
        + double* train(Network* net, double* image, int label, double learningRate)
        + int test(Network* net, double* image)
        + void printNetworkSummary(Network* net)
    }

    %% Utility functions without creating a struct or class
    class UtilityFunctions {
        <<global functions>>
        + double expo(double y)
        + double softmax(double x, double* Niz, int Iter)
        + void shuffle(double (*array)[SIZE], int labels[], size_t n)
        + void ev_handler(struct mg_connection* c, int ev, void* ev_data)
    }

    Layer "1" --> "0..*" Network : hidden layers
    Network "1" -- "*" Layer : contains layers
    UtilityFunctions .. Network : uses

    Layer : + initLayer()
    Layer : + forward()
    Layer : + back()
    Network : + train()
    Network : + test()
    Network : + printNetworkSummary()

```
