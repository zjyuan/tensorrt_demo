#include <string>
#include <iostream>
#include <fstream>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <vector>
#include <future>
#include "utils.h"
#include "NvInfer.h"
#include "buffers.h"
#include "common.h"

using namespace std;
using namespace cv;

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

queue<Mat> q_body;
mutex q_body_mutex;
condition_variable q_body_cv;
atomic<bool> finished;

IRuntime* runtime;

void detect_thread(const string& video_path)
{
	//do detection
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

    VideoCapture cap(video_path);
    Mat m;
    while(1)
    {
    	cap >> m;
    	if(m.empty())
    		return;

    	float* hostInputBuffer = static_cast<float*>(buffers.getHostBuffer("input"));

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
	    unique_lock<mutex> l(q_body_mutex);
	    for(auto& item:objectList)
	    {
	    	if(item.classId != 0)
	    		continue;
	    	try{
	    		int right = clamp(item.left+item.width, 0, 1920);
	    		int top = clamp(item.top+item.height, 0, 1080);
		    	auto roi = m(Rect(Point(item.left, item.top), Point(right, top)));
		    	q_body.push(roi);
		    }
		    catch(...)
		    {
		    	cout << "caught excepion. (left,top,width,height)" << item.left << " "
		    	<< item.top << " " << item.width << " " << item.height << endl;
		    }
	    }
	    q_body_cv.notify_all();
    }

    finished = true;
}

void student_action_thread()
{
	cout << "starting student_action_thread" << endl;

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

	while(!finished)
	{
		unique_lock<mutex> l(q_body_mutex);
		q_body_cv.wait(l, [](){return q_body.empty() == false;});
		auto m = q_body.front();
		q_body.pop();

		//do classification
		float* hostInputBuffer = static_cast<float*>(buffers.getHostBuffer("input"));
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
	    float sum = accumulate(output_vec.begin(), output_vec.end(), 0.0);
	    for_each(output_vec.begin(), output_vec.end(), [=](float& item){item=item/sum;});
	    copy(output_vec.begin(), output_vec.end(), ostream_iterator<float>(cout, " "));
	    vector<string> labels{"听讲","看书","举手","站立"};
	    auto index = max_element(output_vec.begin(), output_vec.end()) - output_vec.begin();
	    cout << endl << labels[index] << endl;
	}
}

int main()
{
	load_prior_data();
	finished = false;

	runtime = createInferRuntime(gLogger);

	auto f1 = async(launch::async, detect_thread, "/media/luis/e/temp/5_20190606_0800.mp4");
	auto f2 = async(launch::async, student_action_thread);
	f1.get();
	f2.get();
}