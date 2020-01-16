#include <iostream>
#include <chrono>
#include <fstream>
#include <string>
#include <memory>
#include <thread>
#include <boost/timer/timer.hpp>
#include "utils.h"
#include "NvInfer.h"
#include "buffers.h"
#include "common.h"

using namespace std;

std::vector<std::vector<float>> prior_data;

int main()
{
	auto runtime = createInferRuntime(gLogger);
	//student detector
	ifstream engine_file("std_det.engine", std::ios::binary);
    engine_file.seekg(0, engine_file.end);
    auto filesize = engine_file.tellg();
    engine_file.seekg(0, engine_file.beg);

    vector<char> engine_data(filesize);
    engine_file.read(engine_data.data(), filesize);
    auto engine = runtime->deserializeCudaEngine(engine_data.data(), filesize, nullptr);
	std::shared_ptr<nvinfer1::ICudaEngine> mEngine{nullptr};
    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(engine, samplesCommon::InferDeleter());
    cout << "student detector loaded" << endl;

    //student action
	ifstream engine_file1("std_act.onnx_b16_fp16.engine", std::ios::binary);
    engine_file1.seekg(0, engine_file1.end);
    filesize = engine_file1.tellg();
    engine_file1.seekg(0, engine_file1.beg);

    vector<char> engine_data1(filesize);
    engine_file1.read(engine_data1.data(), filesize);
    auto engine1 = runtime->deserializeCudaEngine(engine_data1.data(), filesize, nullptr);
	std::shared_ptr<nvinfer1::ICudaEngine> mEngine1{nullptr};
    mEngine1 = std::shared_ptr<nvinfer1::ICudaEngine>(engine1, samplesCommon::InferDeleter());
    cout << "student action classifier loaded" << endl;

    //student emotion
	ifstream engine_file2("std_emo.onnx_b16_fp16.engine", std::ios::binary);
    engine_file2.seekg(0, engine_file2.end);
    filesize = engine_file2.tellg();
    engine_file2.seekg(0, engine_file2.beg);

    vector<char> engine_data2(filesize);
    engine_file2.read(engine_data2.data(), filesize);
    auto engine2 = runtime->deserializeCudaEngine(engine_data2.data(), filesize, nullptr);
	std::shared_ptr<nvinfer1::ICudaEngine> mEngine2{nullptr};
    mEngine2 = std::shared_ptr<nvinfer1::ICudaEngine>(engine2, samplesCommon::InferDeleter());
    cout << "student emotion classifier loaded" << endl;

    //student direction
	ifstream engine_file3("std_dir.onnx_b16_fp16.engine", std::ios::binary);
    engine_file3.seekg(0, engine_file3.end);
    filesize = engine_file3.tellg();
    engine_file3.seekg(0, engine_file3.beg);

    vector<char> engine_data3(filesize);
    engine_file3.read(engine_data3.data(), filesize);
    auto engine3 = runtime->deserializeCudaEngine(engine_data3.data(), filesize, nullptr);
	std::shared_ptr<nvinfer1::ICudaEngine> mEngine3{nullptr};
    mEngine3 = std::shared_ptr<nvinfer1::ICudaEngine>(engine3, samplesCommon::InferDeleter());
    cout << "student direction classifier loaded" << endl;

    while(1)
    {
    	this_thread::sleep_for(chrono::seconds(1));
    }
}