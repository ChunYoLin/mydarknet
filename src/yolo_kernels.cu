#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "network.h"
#include "detection_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "thpool.h"
#include <sys/time.h>
#include <signal.h>
#include <unistd.h>
}

/* Change class number here */
#define CLS_NUM 20

#ifdef OPENCV
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
extern "C" IplImage* image_to_Ipl(image img, int w, int h, int depth, int c, int step);
extern "C" image ipl_to_image(IplImage* src);
extern "C" void convert_yolo_detections(float *predictions, int classes, int num, int square, int side, int w, int h, float thresh, float **probs, box *boxes, int only_objectness);
extern "C" void draw_yolo(image im, int num, float thresh, box *boxes, float **probs);

extern "C" char *voc_names[];
extern "C" image voc_labels[];
extern "C" void draw_text(image a, char Text[], CvPoint TextPos);
static float **probs;
static box *boxes;
static network net;
static image in   ;
static image in_s ;
static image in_m ;
static image in_op1;
static image in_op2;
static image det  ;
static image det_s;
static image det_m;
static image det_op1;
static image det_op2;
static image disp ;
static cv::VideoCapture cap;
static cv::VideoWriter cap_out;
static float fps = 0;
static float demo_thresh = 0;
static int w, h, depth, c, step= 0;
static int MODE = -1;
timer_t timer_fetch,timer_m,timer_op1,timer_op2;
typedef struct ObjDetArg{
    image ROI;
    int draw;
}ODA;
threadpool thpool_cpu = thpool_init(4);
threadpool thpool_gpu = thpool_init(1);
void *fetch_in_thread(void *Elastic){
    struct timeval tval_before, tval_after, tval_result;
    
    //int elastic = *((int*)Elastic);
    cv::Mat frame_m;   
    gettimeofday(&tval_before,NULL);   
    cap >> frame_m;
    gettimeofday(&tval_after,NULL); 
    IplImage frame = frame_m;
    //mandatory
    cv::Mat frame_cropM;
    cv::Point M_p1(886,560);
    cv::Point M_p2(1334,1008);
    cv::Rect ROI_M(M_p1,M_p2);
    frame_cropM = frame_m(ROI_M).clone();
    cv::rectangle(frame_m,ROI_M,cv::Scalar(0,0,255),2);
    IplImage frame_ROIM = frame_cropM;
    //optional1
    cv::Mat frame_cropop1;
    cv::Point op1_p1(438,560);
    cv::Point op1_p2(886,1008);
    cv::Rect ROI_op1(op1_p1,op1_p2);
    frame_cropop1 = frame_m(ROI_op1).clone();
    cv::rectangle(frame_m,ROI_op1,cv::Scalar(255,0,0),2);
    IplImage frame_ROIop1 = frame_cropop1;
    //optional2
    cv::Mat frame_cropop2;
    cv::Point op2_p1(1334,560);
    cv::Point op2_p2(1782,1008);
    cv::Rect ROI_op2(op2_p1,op2_p2);
    frame_cropop2 = frame_m(ROI_op2).clone();
    cv::rectangle(frame_m,ROI_op2,cv::Scalar(255,0,0),2);
    IplImage frame_ROIop2 = frame_cropop2;
    
    in = ipl_to_image(&frame);
    rgbgr_image(in);
    in_s = resize_image(in, net.w, net.h);
    in_m = ipl_to_image(&frame_ROIM);
    in_op1 = ipl_to_image(&frame_ROIop1);
    in_op2 = ipl_to_image(&frame_ROIop2);
    rgbgr_image(in_s);
    rgbgr_image(in_m);
    rgbgr_image(in_op1);
    rgbgr_image(in_op2);
    
    timersub(&tval_after, &tval_before, &tval_result);
    printf("%f\n",((long int)tval_result.tv_usec)/1000000.f);
    return 0;
}

void *detect_in_thread(void *arg)
{
    ODA tmp = *((ODA*)arg);
    float nms = .4;
    detection_layer l = net.layers[net.n-1];
    float *X = tmp.ROI.data;
    float *predictions = network_predict(net, X);
    //free_image(tmp.ROI);
    convert_yolo_detections(predictions, l.classes, l.n, l.sqrt, l.side, 1, 1, demo_thresh, probs, boxes, 0);
    if (nms > 0) do_nms(boxes, probs, l.side*l.side*l.n, l.classes, nms);
    draw_detections(det, l.side*l.side*l.n, demo_thresh, boxes, probs, voc_names, voc_labels, CLS_NUM,tmp.draw);
    //print FPS
    //printf("\033[2J");
    //printf("\033[1;1H");
    //printf("\nFPS:%.0f\n",fps);
    //printf("Objects:\n\n");

    return 0;
}

void timerHandler( int sig, siginfo_t *si, void *uc ){
    timer_t *tidp;
    tidp = (timer_t *)si->si_value.sival_ptr;
    ODA *tmp = (ODA*)malloc(sizeof(ODA));
    if ( *tidp == timer_fetch ){
	printf("add work fetch\n");
	thpool_add_work(thpool_cpu,fetch_in_thread,0);	    	
    }
    else if ( *tidp == timer_m ){
	printf("add work detect mandatory\n");
	tmp->ROI = in_m;
	tmp->draw = 1;
        thpool_add_work(thpool_gpu,detect_in_thread,tmp);
    }
    else if ( *tidp == timer_op1 ){
	printf("add work detect optional1\n");
	tmp->ROI = in_op1;
	tmp->draw = 2;
        thpool_add_work(thpool_gpu,detect_in_thread,tmp);
    }
    else if ( *tidp == timer_op2 ){
	printf("add work detect optional2\n");
	tmp->ROI = in_op2;
	tmp->draw = 3;
        thpool_add_work(thpool_gpu,detect_in_thread,tmp);
    }
}
int makeTimer( char *name, timer_t *timerID, int expireMS, int intervalMS ){
    struct sigevent te;
    struct itimerspec its;
    struct sigaction sa;
    int sigNo = SIGRTMIN;
    /* Set up signal handler. */
    sa.sa_flags = SA_SIGINFO;
    sa.sa_sigaction = timerHandler;
    sigemptyset(&sa.sa_mask);
    if (sigaction(sigNo, &sa, NULL) == -1){
        printf("error");
    }
    /* Set and enable alarm */
    te.sigev_notify = SIGEV_SIGNAL;
    te.sigev_signo = sigNo;
    te.sigev_value.sival_ptr = timerID;
    timer_create(CLOCK_REALTIME, &te, timerID);
    its.it_interval.tv_sec = 0;
    its.it_interval.tv_nsec = intervalMS * 1000000;
    its.it_value.tv_sec = 0;
    its.it_value.tv_nsec = expireMS * 1000000;
    timer_settime(*timerID, 0, &its, NULL);
    return(0);
}

void *TIMER(void *mode){
    int tmode = *((int*)mode);
    if(tmode == 0){
	makeTimer("Timer_fetch", &timer_fetch, 33, 33);
    	makeTimer("Timer_mandartory", &timer_m, 100, 100);
	//makeTimer("Timer_optional1", &timer_op1, 200, 200);
    	//makeTimer("Timer_optionla2", &timer_op2, 200, 200);
    }
    while(1);
}


extern "C" void demo_yolo(char *cfgfile, char *weightfile, float thresh, int cam_index, char *videofile)
{
    demo_thresh = thresh;
    printf("YOLO demo\n");
    net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);

    srand(2222222);

if(cam_index != -1)
{
    MODE = 0; 
    cv::VideoCapture cam(cam_index);
    cap = cam;
    if(!cap.isOpened()) error("Couldn't connect to webcam.\n");
}
else 
{
    MODE = 1;
    printf("Video File name is: %s\n", videofile);
    cv::VideoCapture videoCap(videofile);
    cap = videoCap;
    if(!cap.isOpened()) error("Couldn't read video file.\n");

    cv::Size S = cv::Size((int)videoCap.get(CV_CAP_PROP_FRAME_WIDTH), (int)videoCap.get(CV_CAP_PROP_FRAME_HEIGHT));
    //cv::VideoWriter outputVideo("out.avi", CV_FOURCC('D','I','V','X'), videoCap.get(CV_CAP_PROP_FPS), S, true);
    //if(!outputVideo.isOpened()) error("Couldn't write video file.\n");
    //cap_out = outputVideo;
}
 
    detection_layer l = net.layers[net.n-1];
    int j;

    boxes = (box *)calloc(l.side*l.side*l.n, sizeof(box));
    probs = (float **)calloc(l.side*l.side*l.n, sizeof(float *));
    for(j = 0; j < l.side*l.side*l.n; ++j) probs[j] = (float *)calloc(l.classes, sizeof(float *));
    pthread_t timer;
    ODA *arg = (ODA*)malloc(sizeof(ODA));
    fetch_in_thread(0);
    det = in;
    det_s = in_s;
    det_m = in_m;
    det_op1 = in_op1;
    det_op2 = in_op2;
    arg->ROI = det_m;
    arg->draw = 0;
    fetch_in_thread(arg);
    detect_in_thread(arg);
    disp = det;
    det = in;
    det_s = in_s;
    det_m = in_m;
    det_op1 = in_op1;
    det_op2 = in_op2;
    int flag = 0;
    fetch_in_thread(0);
    while(1){
        struct timeval tval_before, tval_after, tval_result;
	gettimeofday(&tval_before,NULL);
	//thpool_add_work(thpool_cpu,fetch_in_thread,0);
	int *mode = (int*)malloc(sizeof(int));
	*mode = 0;
	if(!flag)pthread_create(&timer,0,TIMER,mode);
	flag = 1;
	gettimeofday(&tval_after,NULL);
	show_image(disp, "YOLO");
	//free_image(disp);
        cvWaitKey(1);
	//thpool_wait(thpool_cpu);
        disp  = det;
        det   = in;
        det_s = in_s;
        det_m = in_m;
        det_op1 = in_op1;
        det_op2 = in_op2;
 
        timersub(&tval_after, &tval_before, &tval_result);
        float curr = 1000000.f/((long int)tval_result.tv_usec);
        fps = .9*fps + .1*curr;
    }
}
#else
extern "C" void demo_yolo(char *cfgfile, char *weightfile, float thresh, int cam_index){
    fprintf(stderr, "YOLO demo needs OpenCV for webcam images.\n");
}
#endif

