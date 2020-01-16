g++ -g -fPIC -shared inference.cpp ../common/utils.cpp /usr/local/tensorrt/samples/common/logger.cpp -o libinference.so -I../common -std=c++11 -I/usr/local/tensorrt/include -I/usr/local/tensorrt/samples/common -I/usr/local/cuda-10.0/include -L/usr/local/tensorrt/lib -lnvinfer -lnvonnxparser `pkg-config --cflags --libs opencv4` -L/usr/local/cuda/lib64 -lcudart -lboost_timer