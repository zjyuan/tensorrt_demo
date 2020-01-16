#include <iostream>
#include <string>
#include <memory>
#include "utils.h"
#include "NvInfer.h"
#include "buffers.h"
#include "common.h"

using namespace std;
using namespace cv;

int main()
{
	auto runtime = createInferRuntime(gLogger);
    ifstream engine_file("std_act.onnx_b16_fp16.engine", std::ios::binary);
    engine_file.seekg(0, engine_file.end);
    auto filesize = engine_file.tellg();
    engine_file.seekg(0, engine_file.beg);

    vector<char> engine_data(filesize);
    engine_file.read(engine_data.data(), filesize);
    auto engine = runtime->deserializeCudaEngine(engine_data.data(), filesize, nullptr);
	std::shared_ptr<nvinfer1::ICudaEngine> mEngine{nullptr};
    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(engine, samplesCommon::InferDeleter());

    int nbindings = mEngine.get()->getNbBindings();
    for (int b = 0; b < nbindings; ++b)
    {
        nvinfer1::Dims dims = mEngine.get()->getBindingDimensions(b);
        if (mEngine.get()->bindingIsInput(b))
        {
            if (true)
            {
                gLogInfo << "Found input: " << mEngine.get()->getBindingName(b) << " shape=" << dims
                         << " dtype=" << (int) mEngine.get()->getBindingDataType(b) << std::endl;
            }
            // mInOut["input"] = mEngine.get()->getBindingName(b);
        }
        else
        {
            if (true)
            {
                gLogInfo << "Found output: " << mEngine.get()->getBindingName(b) << " shape=" << dims
                         << " dtype=" << (int) mEngine.get()->getBindingDataType(b) << std::endl;
            }
            // mInOut["output"] = mEngine.get()->getBindingName(b);
        }
    }

    auto context = mEngine->createExecutionContext();

    samplesCommon::BufferManager buffers(mEngine, 1);

    float* hostInputBuffer = static_cast<float*>(buffers.getHostBuffer("input"));

	auto m = imread("demo.jpg");
	ImgPreprocess(m, 112, 112, hostInputBuffer, make_float3(0.485, 0.456, 0.406), 255.0, make_float3(0.229, 0.224, 0.225), 1.0);

	cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // Asynchronously copy data from host input buffers to device input buffers
    buffers.copyInputToDeviceAsync(stream);

    // Asynchronously enqueue the inference work
    if (!context->enqueue(1, buffers.getDeviceBindings().data(), stream, nullptr))
    {
    	cout << "enqueue failed" << endl;
    }

    // Asynchronously copy data from device output buffers to host output buffers
    buffers.copyOutputToHostAsync(stream);

    // Wait for the work in the stream to complete
    cudaStreamSynchronize(stream);

    // Release stream
    cudaStreamDestroy(stream);

    float* output = static_cast<float*>(buffers.getHostBuffer("output"));
    std::vector<float> output_vec(output, output+4);
    for_each(output_vec.begin(), output_vec.end(), [](float& item){item=exp(item);});
    float sum = accumulate(output_vec.begin(), output_vec.end(), 0);
    for_each(output_vec.begin(), output_vec.end(), [=](float& item){item=item/sum;});
    copy(output_vec.begin(), output_vec.end(), ostream_iterator<float>(cout, " "));
    vector<string> labels{"听讲","看书","举手","站立"};
    auto index = max_element(output_vec.begin(), output_vec.end()) - output_vec.begin();
    cout << endl << labels[index] << endl;
}