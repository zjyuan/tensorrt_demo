#include <iostream>
#include <string>
#include <memory>
#include <boost/timer/timer.hpp>
#include "utils.h"
#include "NvInfer.h"
#include "buffers.h"
#include "common.h"

using namespace std;
using namespace cv;
using namespace boost::timer;

std::vector<std::vector<float>> prior_data;

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

int main(){
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

    auto_cpu_timer t;

    auto context = mEngine->createExecutionContext();

    samplesCommon::BufferManager buffers(mEngine, 1);

    float* hostInputBuffer = static_cast<float*>(buffers.getHostBuffer("input"));

    auto origin = imread("demo.jpg");
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
}