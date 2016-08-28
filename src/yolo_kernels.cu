#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#include "gpu_lane_detection.cpp"
#include "brake_light_gpu.cpp"
extern "C" {
#include "image.h"
#include "network.h"
#include "detection_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
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
#define BUFFERSIZE 400
#define step_m 1
#define step_op1 1
#define step_op2 1
#define ngetc(c) (read (0, (c), 1))
#define RESULT_SIZE 55
int result[RESULT_SIZE] = {0};
int current = 0;
static float **probs;
static box *boxes;
static box *boxesM[BUFFERSIZE];
static box *boxesOP1[BUFFERSIZE];
static box *boxesOP2[BUFFERSIZE];
static network net;
static image in   ;

static cv::VideoCapture cap;
static cv::VideoWriter cap_out;
static float fps = 0;
static float demo_thresh = 0;

timer_t timer_fetch,timer_m,timer_op1,timer_op2;
typedef struct ObjDetArg{
    int frameid;
    int draw;
}ODA;
typedef struct Ela_frame{
    pthread_mutex_t rwmutex;
    int frameid;
    image wholeframe;
	box_adjusted box_detected[10];
	int box_detected_num;
	box boxesM;
	box boxesOP1;
	box boxesOP2;
}Eframe;
typedef struct Ela_frame_state{
    volatile int fetch;
    volatile int draw_m;
    volatile int draw_op1;
    volatile int draw_op2;
	volatile int detect_M;
	volatile int detect_OP1;
	volatile int detect_OP2;
}Eframe_s;

//global variable declare
volatile int mode = 0;
threadpool thpool_cpu = thpool_init(1);
threadpool thpool_gpu = thpool_init(1);
static Eframe frame_buffer[BUFFERSIZE] = {0};
Mat frame_buffer_m[BUFFERSIZE];
Eframe_s frame_buffer_s[2000] = {0};
volatile int current_fetch_id = 0;
volatile int current_m_id = 0;
volatile int current_op1_id = 0;
volatile int current_op2_id = 0;
volatile int current_draw_id = 0;
static int fetchjobnum = 0;
static int detect_mjobnum = 0;
static int detect_op1jobnum = 0;
static int detect_op2jobnum = 0;
float *X;
float *predictions;
pthread_mutex_t mutex;
pthread_mutex_t mutex_mode;
pthread_mutex_t mutex_current_fetch;
pthread_cond_t cond;
static CvCapture *capture;
//IplImage *frame;
struct timeval tval_before, tval_after, tval_result,tv;
struct timezone tz;
int s = 0;
int keepM = 0;
int keepOP1 = 0;
int keepOP2 = 0;
box *boxesM_keep;
box *boxesOP1_keep;
box *boxesOP2_keep;
void *fetch_in_thread(void *arg){
	detection_layer l = net.layers[net.n-1];
	if(current_fetch_id > 10){
		// M
		if(frame_buffer_s[current_fetch_id-10].detect_M){
			draw_detections(frame_buffer[(current_fetch_id-10)%BUFFERSIZE].wholeframe, l.side*l.side*l.n, demo_thresh, boxesM[(current_fetch_id-10)%BUFFERSIZE], probs, voc_names, voc_labels, CLS_NUM, 1);
			boxesM_keep = boxesM[(current_fetch_id-10)%BUFFERSIZE];
			keepM = 2;
		}	
		else {
			if(keepM<=0)boxesM_keep = boxes;
			draw_detections(frame_buffer[(current_fetch_id-10)%BUFFERSIZE].wholeframe, l.side*l.side*l.n, demo_thresh, boxesM_keep, probs, voc_names, voc_labels, CLS_NUM, 1);
			keepM--;
		}
		// OP1
		if(frame_buffer_s[current_fetch_id-10].detect_OP1){
			draw_detections(frame_buffer[(current_fetch_id-10)%BUFFERSIZE].wholeframe, l.side*l.side*l.n, demo_thresh, boxesOP1[(current_fetch_id-10)%BUFFERSIZE], probs, voc_names, voc_labels, CLS_NUM, 1);
			boxesOP1_keep = boxesOP1[(current_fetch_id-10)%BUFFERSIZE];
			keepOP1 = 2;
		}
		else {
			if(keepOP1<=0)boxesOP1_keep = boxes;
			draw_detections(frame_buffer[(current_fetch_id-10)%BUFFERSIZE].wholeframe, l.side*l.side*l.n, demo_thresh, boxesOP1_keep, probs, voc_names, voc_labels, CLS_NUM, 1);
			keepOP1--;
		}
		// OP2
		if(frame_buffer_s[current_fetch_id-10].detect_OP2){
			draw_detections(frame_buffer[(current_fetch_id-10)%BUFFERSIZE].wholeframe, l.side*l.side*l.n, demo_thresh, boxesOP2[(current_fetch_id-10)%BUFFERSIZE], probs, voc_names, voc_labels, CLS_NUM, 1);
			boxesOP2_keep = boxesOP2[(current_fetch_id-10)%BUFFERSIZE];
			keepOP2 = 2;
		}
		else {
			if(keepOP2<=0)boxesOP2_keep = boxes;
			draw_detections(frame_buffer[(current_fetch_id-10)%BUFFERSIZE].wholeframe, l.side*l.side*l.n, demo_thresh, boxesM_keep, probs, voc_names, voc_labels, CLS_NUM, 1);
			keepOP2--;
		}
		draw_box(frame_buffer[(current_fetch_id-10)%BUFFERSIZE].wholeframe,886,560,1334,1008+1,0,0,255);
		draw_box(frame_buffer[(current_fetch_id-10)%BUFFERSIZE].wholeframe,438-1,560,886,1008+1,255,0,0);
		draw_box(frame_buffer[(current_fetch_id-10)%BUFFERSIZE].wholeframe,1334,560,1782+1,1008+1,255,0,0);
		//show_image(frame_buffer[(current_fetch_id-10)%BUFFERSIZE].wholeframe,"YOLO");			
		cvWaitKey(1);
	}
	int jobid = *((int*) arg);
	//assert(jobid == current_fetch_id);
	gettimeofday(&tv,NULL);
    printf("time: %ld ",tv.tv_usec/1000);
    printf("start fetch frame %d \n",current_fetch_id); 
	cv::Mat frame_m;
	cap >> frame_m;
	gettimeofday(&tv,NULL);
	//printf("time: %ld ",tv.tv_usec/1000);
	//printf("start detect frame %d's lane\n",current_fetch_id);
	//int output = gpu_lane_detection(frame_m,result,RESULT_SIZE,current);
	gettimeofday(&tv,NULL); 
    //printf("time: %ld ",tv.tv_usec/1000);
	//printf("finish detect frame %d's lane \n",current_fetch_id);
    IplImage frame = frame_m;
	//frame = cvQueryFrame(capture);
	in = ipl_to_image(&frame);
    rgbgr_image(in);
    free_image(frame_buffer[current_fetch_id%BUFFERSIZE].wholeframe);
    frame_buffer[current_fetch_id%BUFFERSIZE].wholeframe = in;
    frame_buffer_m[current_fetch_id%BUFFERSIZE] = frame_m;
    frame_buffer[current_fetch_id%BUFFERSIZE].frameid = current_fetch_id;
    gettimeofday(&tv,NULL); 
    printf("time: %ld ",tv.tv_usec/1000);
    printf("finish fetch frame %d \n",current_fetch_id);
	
    frame_buffer_s[current_fetch_id].fetch = 1;	
	current_fetch_id++;
	return 0;
}

void *detect_in_thread(void *arg)
{
    ODA tmp = *((ODA*)arg);
    image ROI;
    Eframe *detectframe;	
	Mat detectframe_m;
	while(!frame_buffer_s[tmp.frameid].fetch);
    detectframe = &frame_buffer[tmp.frameid%BUFFERSIZE];
	detectframe_m = frame_buffer_m[tmp.frameid%BUFFERSIZE];
    if(tmp.draw == 0){
		ROI = detectframe->wholeframe;
    }
    else if(tmp.draw == 1){
		ROI = crop_image(detectframe->wholeframe,886,560,448,448);
		gettimeofday(&tv,&tz);
		printf("time: %ld ",tv.tv_usec/1000);
		printf("start detect frame %d's mandatory \n",tmp.frameid);
    }
    else if(tmp.draw == 2){
		ROI = crop_image(detectframe->wholeframe,438,560,448,448);
		gettimeofday(&tv,&tz);
		printf("time: %ld ",tv.tv_usec/1000);
    	printf("start detect frame %d's optional1 \n",tmp.frameid);
    }
    else if(tmp.draw == 3){
		ROI = crop_image(detectframe->wholeframe,1334,560,448,448);
		gettimeofday(&tv,&tz);
		printf("time: %ld ",tv.tv_usec/1000);
		printf("start detect frame %d's optional2 \n",tmp.frameid);
    }
    float nms = .4;
    detection_layer l = net.layers[net.n-1];
    X = ROI.data;
    predictions = network_predict(net, X);
    free_image(ROI);
	memset(detectframe->box_detected, 0, sizeof(detectframe->box_detected));
	/*
	detectframe->box_detected_num = cal_boxdetected_info(detectframe->wholeframe, l.side*l.side*l.n, demo_thresh, boxes, probs, voc_names, voc_labels, CLS_NUM, tmp.draw, detectframe->box_detected);
	for(int i = 0;i < detectframe->box_detected_num;i++){	
		int x = detectframe->box_detected[i].left;
		int y = detectframe->box_detected[i].top;
		int h = detectframe->box_detected[i].bot - detectframe->box_detected[i].top + 1;
		int w = detectframe->box_detected[i].right - detectframe->box_detected[i].left + 1;
		Rect region_of_interest = Rect(x, y, w, h);
		printf("x:%d y:%d w:%d h:%d\n",x,y,w,h);
		cv::Mat car = detectframe_m(region_of_interest);
		bool warning = Brake_light(car);
		printf("warning : %d\n",warning);
	}*/
	gettimeofday(&tv,&tz);
    printf("time: %ld ",tv.tv_usec/1000);
    if(tmp.draw == 1){
		convert_yolo_detections(predictions, l.classes, l.n, l.sqrt, l.side, 1, 1, demo_thresh, probs, boxesM[tmp.frameid%BUFFERSIZE], 0);
		if (nms > 0) do_nms(boxesM[tmp.frameid%BUFFERSIZE], probs, l.side*l.side*l.n, l.classes, nms);
		frame_buffer_s[tmp.frameid].detect_M = 1;
		frame_buffer_s[tmp.frameid].draw_m = 1;
		printf("finish detect frame %d's mandatory \n",tmp.frameid);
    }
    else if(tmp.draw == 2){
		convert_yolo_detections(predictions, l.classes, l.n, l.sqrt, l.side, 1, 1, demo_thresh, probs, boxesOP1[tmp.frameid], 0);
		if (nms > 0) do_nms(boxesOP1[tmp.frameid], probs, l.side*l.side*l.n, l.classes, nms);
		frame_buffer_s[tmp.frameid].detect_OP1 = 1;
		frame_buffer_s[tmp.frameid].draw_op1 = 1;
		printf("finish detect frame %d's optional1 \n",tmp.frameid);
    }
    else if(tmp.draw == 3){
		convert_yolo_detections(predictions, l.classes, l.n, l.sqrt, l.side, 1, 1, demo_thresh, probs, boxesOP2[tmp.frameid], 0);
		if (nms > 0) do_nms(boxesOP2[tmp.frameid], probs, l.side*l.side*l.n, l.classes, nms);
		frame_buffer_s[tmp.frameid].detect_OP2 = 1;
		frame_buffer_s[tmp.frameid].draw_op2 = 1;
    	printf("finish detect frame %d's optional2 \n",tmp.frameid);
    }

    return 0;
}
/*
void *show_frame(void *arg){
	int drawid = 0;
	box *boxesM_keep;
	int keep = 0;
    while(1){
		int tmpmode = mode;
		detection_layer l = net.layers[net.n-1];
        if(tmpmode == 0){
	    	while(!frame_buffer_s[drawid].fetch){
				printf("waiting fetch frame %d\n",drawid);
			}
			
			if(frame_buffer_s[drawid].detect_M){
				draw_detections(frame_buffer[drawid%BUFFERSIZE].wholeframe, l.side*l.side*l.n, demo_thresh, boxesM[drawid], probs, voc_names, voc_labels, CLS_NUM, 1);
				boxesM_keep = boxesM[drawid];
				keep = 2;
			}
			
			else {
				if(keep<=0)boxesM_keep = boxes;
				draw_detections(frame_buffer[drawid%BUFFERSIZE].wholeframe, l.side*l.side*l.n, demo_thresh, boxesM_keep, probs, voc_names, voc_labels, CLS_NUM, 1);
				keep--;
			}
			gettimeofday(&tv,&tz);
    		printf("time: %ld ",tv.tv_usec/1000);
	   		printf("start drawing frame %d \n",drawid);
			draw_box(frame_buffer[drawid%BUFFERSIZE].wholeframe,886,560,1334,1008+1,0,0,255);
   	    	draw_box(frame_buffer[drawid%BUFFERSIZE].wholeframe,438-1,560,886,1008+1,255,0,0);
    		draw_box(frame_buffer[drawid%BUFFERSIZE].wholeframe,1334,560,1782+1,1008+1,255,0,0);
    		show_image(frame_buffer[drawid%BUFFERSIZE].wholeframe,"YOLO");			
	    	cvWaitKey(1);
			gettimeofday(&tv,&tz);
    		printf("time: %ld ",tv.tv_usec/1000);
	   		printf("finish drawing frame %d \n",drawid);

	    	drawid++;
		}
    }
}*/

void timerHandler( int sig, siginfo_t *si, void *uc ){
    timer_t *tidp;
    tidp = (timer_t *)si->si_value.sival_ptr;
    ODA *tmp = (ODA*)malloc(sizeof(ODA));
    if ( *tidp == timer_fetch ){
		gettimeofday(&tv,&tz);
    	printf("time: %d ",tv.tv_usec/1000);
		printf("add work fetch frame %d\n",fetchjobnum);
		int cmp = fetchjobnum;
		thpool_add_work(thpool_cpu,fetch_in_thread,&cmp);
		fetchjobnum++;
    }
    else if ( *tidp == timer_m ){
        tmp->frameid = fetchjobnum;
		tmp->draw = 1;
		gettimeofday(&tv,&tz);
    	printf("time: %ld ",tv.tv_usec/1000);
		printf("add work detect frame %d's mandatory \n",tmp->frameid);
        thpool_add_work(thpool_gpu,detect_in_thread,tmp);
    }
    else if ( *tidp == timer_op1 ){
		tmp->frameid = fetchjobnum;
		tmp->draw = 2;
		gettimeofday(&tv,&tz);
    	printf("time: %ld ",tv.tv_usec/1000);
		printf("add work detect frame %d's optional1 \n",tmp->frameid);
        thpool_add_work(thpool_gpu,detect_in_thread,tmp);
    }
    else if ( *tidp == timer_op2 ){
		tmp->frameid = current_fetch_id;
		tmp->draw = 3;
		gettimeofday(&tv,&tz);
    	printf("time: %ld ",tv.tv_usec/1000);
		printf("add work detect frame %d's optional2 \n",tmp->frameid);
        thpool_add_work(thpool_gpu,detect_in_thread,tmp);
    }
}
int makeTimer( timer_t *timerID, int expireMS, int intervalMS ){
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
    gettimeofday(&tv,&tz);
    printf("time: %ld ",tv.tv_usec/1000);
    printf("start all timer\n");
    if(mode == 0){
		makeTimer(&timer_fetch, 33, 33);
    	makeTimer(&timer_m, 100, 100);
		makeTimer(&timer_op1, 100, 100);
    	makeTimer(&timer_op2, 100, 100);
    }
    else if(mode == 1){
    	makeTimer(&timer_fetch, 33, 33);
		makeTimer(&timer_m, 100, 100);
    }
    while(1);
}
void *MODE_CONTROLLER(void *arg){
    char c;
    while(1){
        ngetc(&c);
		if(c != '\n'){
			printf("c == %c\n",c);
			pthread_mutex_lock(&mutex_mode);
			if(c == '0'){
			    printf("mode0\n");
			    mode = 0;
			}
			else if(c == '1'){
			    printf("mode1\n");
			    mode = 1;
			}
			else if(c == '2'){
				printf("mode2\n");
				mode = 2;
			}
			pthread_mutex_unlock(&mutex_mode);
			pthread_cond_signal(&cond);
			printf("signal\n");
        }
    }
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
    //MODE = 0; 
    cv::VideoCapture cam(cam_index);
    cap = cam;
    if(!cap.isOpened()) error("Couldn't connect to webcam.\n");
}
else{
    //MODE = 1;
    printf("Video File name is: %s\n", videofile);
    capture = cvCreateFileCapture(videofile);
    cv::VideoCapture videoCap(videofile);
    
    cap = videoCap;
    if(!cap.isOpened()) error("Couldn't read video file.\n");
}
	//initial every mutex
    pthread_mutex_init(&mutex,NULL);
    pthread_mutex_init(&mutex_current_fetch,NULL);
	pthread_cond_init(&cond,NULL);


    detection_layer l = net.layers[net.n-1];
    boxes = (box *)calloc(l.side*l.side*l.n, sizeof(box));
    
    probs = (float **)calloc(l.side*l.side*l.n, sizeof(float *));
    for(int j = 0; j < BUFFERSIZE;j++){
		boxesM[j] = (box *)calloc(l.side*l.side*l.n, sizeof(box));
		boxesOP1[j] = (box *)calloc(l.side*l.side*l.n, sizeof(box));
		boxesOP2[j] = (box *)calloc(l.side*l.side*l.n, sizeof(box));
	}
	for(int j = 0; j < l.side*l.side*l.n; ++j) probs[j] = (float *)calloc(l.classes, sizeof(float *));
    pthread_t timer,drawer,modecontroller;
    ODA *arg = (ODA*)malloc(sizeof(ODA));
    pthread_create(&modecontroller,0,MODE_CONTROLLER,0);
    //pthread_create(&drawer,0,show_frame,0);
	while(1){
		pthread_cond_wait(&cond,&mutex);
        pthread_create(&timer,0,TIMER,0);
    }
    //while(1);
}
#else
extern "C" void demo_yolo(char *cfgfile, char *weightfile, float thresh, int cam_index){
    fprintf(stderr, "YOLO demo needs OpenCV for webcam images.\n");
}
#endif

