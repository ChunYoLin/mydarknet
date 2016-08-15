#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<opencv2/gpu/gpu.hpp>
#include <iostream>
#include <cv.h>
#include <time.h>
using namespace cv;
using namespace std;
bool Brake_light(cv::Mat imgBGR)
{
	//clock_t t;
	//t = clock();
	//imshow("123",imgBGR);
	//cv::waitKey(1);
	cv::Mat imgLab;
	if(imgBGR.empty()){
		cout << "Image Not Found" << endl;
		return -1;
	}
	cv::gpu::GpuMat input_gpu(imgBGR);
	cv::gpu::GpuMat output_gpu;
	cv::gpu::cvtColor(input_gpu, output_gpu, CV_BGR2Lab);
	output_gpu.download(imgLab);
	Point3_<int> pixelData;
	int step = imgLab.step;
	int channels = imgLab.channels();
	int count = 0;	
	for(int i = 0; i < imgLab.rows; i++){
		for(int j = 0; j < imgLab.cols; j++){	
			pixelData.x = imgLab.data[step*i + channels*j + 0];
			pixelData.y = imgLab.data[step*i + channels*j + 1];
			pixelData.z = imgLab.data[step*i + channels*j + 2];
			//imgLab.data[step*i + channels*j + 0] = 255;
			//imgLab.data[step*i + channels*j + 1] = 255;
			//imgLab.data[step*i + channels*j + 2] = 255;
			//cout << "L:" << pixelData.x << " a:" << pixelData.y << " b:" << pixelData.z << endl; 
			if(pixelData.x > 50 && pixelData.x < 150 &&  pixelData.y > 130 && pixelData.y < 220 
					&& pixelData.z > 110 && pixelData.z < 180){
				count++;
			}
		}
	}
	if((float)count/(float)(imgLab.rows*imgLab.cols) > 0.05){
		return true;
		//cout << "Warning!" << endl;
	}
	else return false;
	//t = clock() - t;
	//cout << "fps:" << 10000 / (t / (double)CLOCKS_PER_SEC) << endl;
}
