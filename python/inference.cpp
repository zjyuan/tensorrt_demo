#include <iostream>
#include <string>
#include <memory>
#include "utils.h"
#include "NvInfer.h"
#include "buffers.h"
#include "common.h"

using namespace cv;
using namespace std;

std::vector<std::vector<float>> prior_data;

void do_classify(const char* image)
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

	auto m = imread(image);
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

static void load_prior_data()
{
	using namespace std;
	ifstream ifs("prior_data.txt");
	
	string c;
	getline(ifs, c);
	auto start = 0;
	auto index = c.find(",");
	vector<float> item;
	while(index != string::npos)
	{
		auto sub = c.substr(start, index-start);
		float data = stof(sub);
		item.push_back(data);
		if(item.size() == 4)
		{
			prior_data.push_back(item);
			item.clear();
		}
		start = index+1;
		index = c.find(",", start);
	}
}

void do_detect(const char* image)
{
	load_prior_data();

	auto runtime = createInferRuntime(gLogger);
    ifstream engine_file("std_det.engine", std::ios::binary);
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
        }
        else
        {
            if (true)
            {
                gLogInfo << "Found output: " << mEngine.get()->getBindingName(b) << " shape=" << dims
                         << " dtype=" << (int) mEngine.get()->getBindingDataType(b) << std::endl;
            }
        }
    }

    // auto_cpu_timer t;

    auto context = mEngine->createExecutionContext();

    samplesCommon::BufferManager buffers(mEngine, 1);

    float* hostInputBuffer = static_cast<float*>(buffers.getHostBuffer("input"));

    auto origin = imread(image);
    auto m = origin.clone();
    ImgPreprocess(m, 1920, 1080, hostInputBuffer, make_float3(123, 117, 104), 1.0, make_float3(1.0, 1.0, 1.0), 1.0);

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

    float* loc_data = static_cast<float*>(buffers.getHostBuffer("output1"));
    float* conf_data = static_cast<float*>(buffers.getHostBuffer("output2"));

    std::vector<NvDsInferParseObjectInfo> objectList = post_process(loc_data, conf_data);
    draw_bbox(m, objectList);
    imwrite("output.jpg", m);
    cout << "writing result to output.jpg" << endl;
}

extern "C" {

void classify(const char* image)
{
	do_classify(image);
}

void detect(const char* image)
{
	do_detect(image);
}

}