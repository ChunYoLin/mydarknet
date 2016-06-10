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
static cv::VideoCapture cap;
static cv::VideoWriter cap_out;
static float fps = 0;
static float demo_thresh = 0;
static int w, h, depth, c, step= 0;
static int MODE = -1;
timer_t timer_fetch,timer_m,timer_op1,timer_op2;
typedef struct ObjDetArg{
    int frameid;
    int draw;
}ODA;
typedef struct Ela_frame{
    int frameid;
    image wholeframe;
    int draw_m;
    int draw_op1;
    int draw_op2;
}Eframe;
//global variable declare
static int mode = 0;
static threadpool thpool_cpu = thpool_init(4);
static threadpool thpool_gpu = thpool_init(1);
static Eframe *frame_buffer = (Eframe*)malloc(sizeof(Eframe)*1000);
static int current_fetch_id = 0;
static int current_m_id = 0;
static int current_op1_id = 0;
static int current_op2_id = 0;
static int current_draw_id = 0;
static pthread_mutex_t mutex;
static pthread_cond_t cond;
void *fetch_in_thread(void *Elastic){
    
    struct timeval tval_before, tval_after, tval_result;
    cv::Mat frame_m; 
    gettimeofday(&tval_before,NULL);
    printf("yes\n");   
    cap >> frame_m;
    IplImage frame = frame_m;
    in = ipl_to_image(&frame);
    rgbgr_image(in);
    frame_buffer[current_fetch_id].wholeframe = in;
    frame_buffer[current_fetch_id].frameid = current_fetch_id;
    gettimeofday(&tval_after,NULL); 
    timersub(&tval_after, &tval_before, &tval_result);
    //printf("%f\n",((long int)tval_result.tv_usec)/1000000.f);
    printf("no\n");
    return 0;
}

void *detect_in_thread(void *arg)
{
    ODA tmp = *((ODA*)arg);
    image ROI;
    if(tmp.draw == 0)ROI = frame_buffer[tmp.frameid].wholeframe;
    else if(tmp.draw == 1)ROI = crop_image(frame_buffer[tmp.frameid].wholeframe,886,560,448,448);
    else if(tmp.draw == 2)ROI = crop_image(frame_buffer[tmp.frameid].wholeframe,438,560,448,448);
    else if(tmp.draw == 3)ROI = crop_image(frame_buffer[tmp.frameid].wholeframe,1334,560,448,448);
    float nms = .4;
    detection_layer l = net.layers[net.n-1];
    float *X = ROI.data;
    float *predictions = network_predict(net, X);
    free_image(ROI);
    convert_yolo_detections(predictions, l.classes, l.n, l.sqrt, l.side, 1, 1, demo_thresh, probs, boxes, 0);
    if (nms > 0) do_nms(boxes, probs, l.side*l.side*l.n, l.classes, nms);
    pthread_mutex_lock(&mutex);
    draw_detections(frame_buffer[tmp.frameid].wholeframe, l.side*l.side*l.n, demo_thresh, boxes, probs, voc_names, voc_labels, CLS_NUM,tmp.draw);
    if(tmp.draw == 1)frame_buffer[tmp.frameid].draw_m = 1;
    else if(tmp.draw == 2)frame_buffer[tmp.frameid].draw_op1 = 1;
    else if(tmp.draw == 3)frame_buffer[tmp.frameid].draw_op2 = 1;
    pthread_cond_signal(&cond);
    pthread_mutex_unlock(&mutex);
    //print FPS
    //printf("\033[2J");
    //printf("\033[1;1H");
    //printf("\nFPS:%.0f\n",fps);
    //printf("Objects:\n\n");

    return 0;
}
void *show_frame(void *arg){
    while(1){
    	if(mode == 0){
	    pthread_mutex_lock(&mutex);
	    while(!frame_buffer[current_draw_id].draw_m){
		pthread_cond_wait(&cond,&mutex);
	    }
	    //draw_box(frame_buffer[current_draw_id].wholeframe,886,560,1334,1008+1,0,0,255);
   	    //draw_box(frame_buffer[current_draw_id].wholeframe,438-1,560,886,1008+1,255,0,0);
    	    //draw_box(frame_buffer[current_draw_id].wholeframe,1334,560,1782+1,1008+1,255,0,0);
	    pthread_mutex_unlock(&mutex);
	    struct timeval tv;
    	    struct timezone tz;
    	    gettimeofday(&tv,&tz);
    	    printf("time: %d ",tv.tv_usec/1000);
	    printf("finish drawing frame %d \n",current_draw_id);
	    show_image(frame_buffer[current_draw_id].wholeframe,"YOLO");
	    cvWaitKey(1);
	    current_draw_id++;
	}
    }
}
void timerHandler( int sig, siginfo_t *si, void *uc ){
    timer_t *tidp;
    tidp = (timer_t *)si->si_value.sival_ptr;
    ODA *tmp = (ODA*)malloc(sizeof(ODA));
    struct timeval tv;
    struct timezone tz;
    gettimeofday(&tv,&tz);
    printf("time: %d ",tv.tv_usec/1000);
    pthread_mutex_lock(&mutex);
    if ( *tidp == timer_fetch ){
	printf("add work fetch frame %d\n",current_fetch_id);
	thpool_add_work(thpool_cpu,fetch_in_thread,0);
	current_fetch_id++;   	
    }
    else if ( *tidp == timer_m ){
	printf("add work detect mandatory frame %d\n",current_m_id);
	tmp->frameid = current_m_id;
	tmp->draw = 1;
        thpool_add_work(thpool_gpu,detect_in_thread,tmp);
	current_m_id++;
    }
    else if ( *tidp == timer_op1 ){
	printf("add work detect optional1 frame %d\n",current_op1_id);
	tmp->frameid = current_op1_id;
	tmp->draw = 2;
        thpool_add_work(thpool_gpu,detect_in_thread,tmp);
	current_op1_id++;
    }
    else if ( *tidp == timer_op2 ){
	printf("add work detect optional2 frame %d\n",current_op2_id);
	tmp->frameid = current_op2_id;
	tmp->draw = 3;
        thpool_add_work(thpool_gpu,detect_in_thread,tmp);
	current_op2_id++;
    }
    pthread_mutex_unlock(&mutex);
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
    timer_gettime(*timerID,&its);
    return(0);
}

void *TIMER(void *arg){
    if(mode == 0){
	makeTimer("Timer_fetch", &timer_fetch, 33, 30);
    	makeTimer("Timer_mandartory", &timer_m, 100, 100);
	//makeTimer("Timer_optional1", &timer_op1, 200, 200);
    	//makeTimer("Timer_optionla2", &timer_op2, 200, 200);
    }
    while(1);
}


extern "C" void demo_yolo(char *cfgfile, char *weightfile, float thresh, int cam_index, char *videofile){
    demo_thresh = thresh;
    printf("YOLO demo\n");
    net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    srand(2222222);

if(cam_index != -1){
    MODE = 0; 
    cv::VideoCapture cam(cam_index);
    cap = cam;
    if(!cap.isOpened()) error("Couldn't connect to webcam.\n");
}
else{
    MODE = 1;
    printf("Video File name is: %s\n", videofile);
    cv::VideoCapture videoCap(videofile);
    cap = videoCap;
    if(!cap.isOpened()) error("Couldn't read video file.\n");
}
    pthread_cond_init(&cond,NULL);
    pthread_mutex_init(&mutex,NULL);
    detection_layer l = net.layers[net.n-1];
    int j;
    boxes = (box *)calloc(l.side*l.side*l.n, sizeof(box));
    probs = (float **)calloc(l.side*l.side*l.n, sizeof(float *));
    for(j = 0; j < l.side*l.side*l.n; ++j) probs[j] = (float *)calloc(l.classes, sizeof(float *));
    pthread_t timer,drawer;
    ODA *arg = (ODA*)malloc(sizeof(ODA));
    int flag = 0;
    fetch_in_thread(0);
    while(1){
        struct timeval tval_before, tval_after, tval_result;
	gettimeofday(&tval_before,NULL);
	if(!flag)pthread_create(&timer,0,TIMER,0);
        if(!flag)pthread_create(&drawer,0,show_frame,0);
	flag = 1;
	gettimeofday(&tval_after,NULL);
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

