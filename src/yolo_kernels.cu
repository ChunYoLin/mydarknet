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
#define BUFFERSIZE 300
#define step_m 1
#define step_op1 1
#define step_op2 1
#define ngetc(c) (read (0, (c), 1))
#define RESULT_SIZE 55
int result[RESULT_SIZE] = {0};
int current = 0;
static float **probs;
static box *boxes;
static network net;
static image in   ;

static cv::VideoCapture cap;
static cv::VideoWriter cap_out;
static float fps = 0;
static float demo_thresh = 0;
//static int w, h, depth, c, step= 0;
//static int MODE = -1;
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
	//IplImage *wholeframe_ipl;
}Eframe;
typedef struct Ela_frame_state{
    pthread_mutex_t fetch_mutex;
	pthread_mutex_t draw_mmutex;
	pthread_mutex_t draw_op1mutex;
	pthread_mutex_t draw_op2mutex;
	pthread_cond_t fetch_cond;
	pthread_cond_t draw_mcond;
	pthread_cond_t draw_op1cond;
	pthread_cond_t draw_op2cond;
    volatile int fetch;
    volatile int draw_m;
    volatile int draw_op1;
    volatile int draw_op2;
}Eframe_s;
//global variable declare
volatile int mode = 0;
threadpool thpool_cpu = thpool_init(1);
threadpool thpool_gpu = thpool_init(1);
//Eframe *frame_buffer = (Eframe*)malloc(sizeof(Eframe)*BUFFERSIZE);
Eframe frame_buffer[BUFFERSIZE] = {0};
Mat frame_buffer_m[BUFFERSIZE];
Eframe_s frame_buffer_s[2000] = {0};
//static const Eframe zeroEframe;
volatile int current_fetch_id = 0;
volatile int current_m_id = 0;
volatile int current_op1_id = 0;
volatile int current_op2_id = 0;
volatile int current_draw_id = 0;
float *X;
float *predictions;
pthread_mutex_t mutex;
pthread_mutex_t mutex_mode;
pthread_mutex_t mutex_current_fetch;
pthread_mutex_t mutex_current_m;
pthread_mutex_t mutex_current_op1;
pthread_mutex_t mutex_current_op2;
pthread_mutex_t mutex_current_draw;
pthread_mutex_t mutex_fetchjob;
pthread_mutex_t mutex_detect_mjob;
pthread_mutex_t mutex_detect_op1job;
pthread_mutex_t mutex_detect_op2job;
pthread_cond_t cond;
static CvCapture *capture;
//IplImage *frame;
struct timeval tval_before, tval_after, tval_result,tv;
struct timezone tz;
int s = 0;
void *fetch_in_thread(void *Elastic){
    gettimeofday(&tv,NULL);
    pthread_mutex_lock(&mutex_current_fetch);
    //printf("time: %ld ",tv.tv_usec/1000);
    //printf("start fetch frame %d \n",current_fetch_id);
    pthread_mutex_unlock(&mutex_current_fetch);
    cv::Mat frame_m;
    //frame_buffer[current_fetch_id%BUFFERSIZE].wholeframe_mat = frame_m;
	cap >> frame_m;
	gettimeofday(&tv,NULL);
	//printf("time: %ld ",tv.tv_usec/1000);
	//printf("start detect frame %d's lane\n",current_fetch_id);

	int output = gpu_lane_detection(frame_m,result,RESULT_SIZE,current);
	gettimeofday(&tv,NULL); 
    //printf("time: %ld ",tv.tv_usec/1000);
	//printf("finish detect frame %d's lane \n",current_fetch_id);
    IplImage frame = frame_m;
	//frame = cvQueryFrame(capture);
    //frame_buffer[current_fetch_id%BUFFERSIZE].wholeframe_ipl = &frame;
	//cvSaveImage("foo2.png",&frame_buffer[current_fetch_id%BUFFERSIZE].wholeframe_ipl);
	in = ipl_to_image(&frame);
    rgbgr_image(in);
    free_image(frame_buffer[current_fetch_id%BUFFERSIZE].wholeframe);
    frame_buffer[current_fetch_id%BUFFERSIZE].wholeframe = in;
    frame_buffer_m[current_fetch_id%BUFFERSIZE] = frame_m;
    frame_buffer[current_fetch_id%BUFFERSIZE].frameid = current_fetch_id;
    gettimeofday(&tv,NULL); 
    //printf("time: %ld ",tv.tv_usec/1000);
    //printf("finish fetch frame %d \n",current_fetch_id);
	
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
	gettimeofday(&tv,&tz);
    //printf("time: %ld ",tv.tv_usec/1000);
    if(tmp.draw == 0){
		ROI = detectframe->wholeframe;
    }
    else if(tmp.draw == 1){
		ROI = crop_image(detectframe->wholeframe,886,560,448,448);
		//printf("start detect frame %d's mandatory \n",tmp.frameid);
    }
    else if(tmp.draw == 2){
		ROI = crop_image(detectframe->wholeframe,438,560,448,448);
    	//printf("start detect frame %d's optional1 \n",tmp.frameid);
    }
    else if(tmp.draw == 3){
		ROI = crop_image(detectframe->wholeframe,1334,560,448,448);
		//printf("start detect frame %d's optional2 \n",tmp.frameid);
    }
    float nms = .4;
    detection_layer l = net.layers[net.n-1];
    X = ROI.data;
    predictions = network_predict(net, X);
    free_image(ROI);
    convert_yolo_detections(predictions, l.classes, l.n, l.sqrt, l.side, 1, 1, demo_thresh, probs, boxes, 0);
    if (nms > 0) do_nms(boxes, probs, l.side*l.side*l.n, l.classes, nms);
	memset(detectframe->box_detected, 0, sizeof(detectframe->box_detected));
    //draw_detections(detectframe->wholeframe, l.side*l.side*l.n, demo_thresh, boxes, probs, voc_names, voc_labels, CLS_NUM,tmp.draw);
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
	}
	gettimeofday(&tv,&tz);
    //printf("time: %ld ",tv.tv_usec/1000);
    if(tmp.draw == 1){
		pthread_mutex_lock(&frame_buffer_s[tmp.frameid].draw_mmutex);
		frame_buffer_s[tmp.frameid].draw_m = 1;
		pthread_mutex_unlock(&frame_buffer_s[tmp.frameid].draw_mmutex);
		//printf("finish detect frame %d's mandatory \n",tmp.frameid);
		pthread_mutex_lock(&mutex_current_m);
		current_m_id+=step_m;
		pthread_mutex_unlock(&mutex_current_m);
    }
    else if(tmp.draw == 2){
		pthread_mutex_lock(&frame_buffer_s[tmp.frameid].draw_op1mutex);
		frame_buffer_s[tmp.frameid].draw_op1 = 1;
		pthread_mutex_unlock(&frame_buffer_s[tmp.frameid].draw_op1mutex);
		//printf("finish detect frame %d's optional1 \n",tmp.frameid);
		pthread_mutex_lock(&mutex_current_op1);
		current_op1_id+=step_op1;
		pthread_mutex_unlock(&mutex_current_op1);
    }
    else if(tmp.draw == 3){
		pthread_mutex_lock(&frame_buffer_s[tmp.frameid].draw_op2mutex);
		frame_buffer_s[tmp.frameid].draw_op2 = 1;
		pthread_mutex_unlock(&frame_buffer_s[tmp.frameid].draw_op2mutex);
    	//printf("finish detect frame %d's optional2 \n",tmp.frameid);
		pthread_mutex_lock(&mutex_current_op2);
		current_op2_id+=step_op2;
		pthread_mutex_unlock(&mutex_current_op2);
    }
    //pthread_cond_signal(&cond);
    //print FPS
    //printf("\033[2J");
    //printf("\033[1;1H");
    //printf("\nFPS:%.0f\n",fps);
    //printf("Objects:\n\n");

    return 0;
}

void *show_frame(void *arg){
    while(1){
		pthread_mutex_lock(&mutex_current_draw);
		int tmpdrawid = current_draw_id;
		pthread_mutex_unlock(&mutex_current_draw);
		pthread_mutex_lock(&mutex_mode);
		int tmpmode = mode;
    	pthread_mutex_unlock(&mutex_mode); 
        if(tmpmode == 0){
	    	while(!frame_buffer_s[tmpdrawid].fetch){			
				if(tmpdrawid % step_m == 0){
					//while(!frame_buffer_s[tmpdrawid].draw_m){
						//printf("stuck in frame %d draw_m\n",tmpdrawid);
					//	;
					//}
				}
				if(tmpdrawid % step_op1 == 0){
					//while(!frame_buffer_s[tmpdrawid].draw_op1){
						//printf("stuck in frame %d draw_op1\n",tmpdrawid);
						//;
					//}
				}
				if(tmpdrawid % step_op2 == 0){
					//while(!frame_buffer_s[tmpdrawid].draw_op2){
						//printf("stuck in frame %d draw_op2\n",tmpdrawid);
						//;
					//}
				}
			}
	    	draw_box(frame_buffer[tmpdrawid%BUFFERSIZE].wholeframe,886,560,1334,1008+1,0,0,255);
   	    	draw_box(frame_buffer[tmpdrawid%BUFFERSIZE].wholeframe,438-1,560,886,1008+1,255,0,0);
    		draw_box(frame_buffer[tmpdrawid%BUFFERSIZE].wholeframe,1334,560,1782+1,1008+1,255,0,0);
    		gettimeofday(&tv,&tz);
    		//printf("time: %ld ",tv.tv_usec/1000);
	   		//printf("finish drawing frame %d \n",current_draw_id);
	    	show_image(frame_buffer[tmpdrawid%BUFFERSIZE].wholeframe,"YOLO");			
	    	cvWaitKey(1);
			pthread_mutex_lock(&mutex_current_draw);
	    	current_draw_id++;
			pthread_mutex_unlock(&mutex_current_draw);
		}
    }
}
void timerHandler( int sig, siginfo_t *si, void *uc ){
    timer_t *tidp;
    tidp = (timer_t *)si->si_value.sival_ptr;
    ODA *tmp = (ODA*)malloc(sizeof(ODA));
    static int fetchjobnum = 0;
    static int detect_mjobnum = 0;
    static int detect_op1jobnum = 0;
    static int detect_op2jobnum = 0;
    if ( *tidp == timer_fetch ){
		pthread_mutex_lock(&mutex_fetchjob);
		gettimeofday(&tv,&tz);
    	//printf("time: %d ",tv.tv_usec/1000);
		//printf("add work fetch frame %d\n",fetchjobnum);
		thpool_add_work(thpool_cpu,fetch_in_thread,0);
		fetchjobnum++;
		pthread_mutex_unlock(&mutex_fetchjob);
    }
    else if ( *tidp == timer_m ){
		pthread_mutex_lock(&mutex_current_fetch);
        tmp->frameid = current_fetch_id;
        pthread_mutex_unlock(&mutex_current_fetch);
		//pthread_mutex_lock(&mutex_detect_mjob);
		//tmp->frameid = detect_mjobnum;
		tmp->draw = 1;
		gettimeofday(&tv,&tz);
    	//printf("time: %ld ",tv.tv_usec/1000);
		//printf("add work detect frame %d's mandatory \n",tmp->frameid);
        thpool_add_work(thpool_gpu,detect_in_thread,tmp);
		//detect_mjobnum+=step_m;
		//pthread_mutex_unlock(&mutex_detect_mjob);
    }
    else if ( *tidp == timer_op1 ){
		pthread_mutex_lock(&mutex_current_fetch);
		tmp->frameid = current_fetch_id;
		pthread_mutex_unlock(&mutex_current_fetch);
		//pthread_mutex_lock(&mutex_detect_op1job);
		//tmp->frameid = detect_op1jobnum;
		tmp->draw = 2;
		gettimeofday(&tv,&tz);
    	//printf("time: %ld ",tv.tv_usec/1000);
		//printf("add work detect frame %d's optional1 \n",tmp->frameid);
        thpool_add_work(thpool_gpu,detect_in_thread,tmp);
		//detect_op1jobnum+=step_op1;
		//pthread_mutex_unlock(&mutex_detect_op1job);
    }
    else if ( *tidp == timer_op2 ){
		pthread_mutex_lock(&mutex_current_fetch);
		tmp->frameid = current_fetch_id;
		pthread_mutex_unlock(&mutex_current_fetch);
		//pthread_mutex_lock(&mutex_detect_op2job);
		//tmp->frameid = detect_op2jobnum;
		tmp->draw = 3;
		gettimeofday(&tv,&tz);
    	//printf("time: %ld ",tv.tv_usec/1000);
		//printf("add work detect frame %d's optional2 \n",tmp->frameid);
        thpool_add_work(thpool_gpu,detect_in_thread,tmp);
		//detect_op2jobnum+=step_op2;
		//pthread_mutex_unlock(&mutex_detect_op2job);
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
    //printf("time: %ld ",tv.tv_usec/1000);
    //printf("start all timer\n");
    if(mode == 0){
		makeTimer(&timer_fetch, 33, 33);
    	makeTimer(&timer_m, 100, 100);
		//makeTimer(&timer_op1, 200, 200);
    	//makeTimer(&timer_op2, 200, 200);
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
    for(int i = 0;i < BUFFERSIZE;i++)pthread_mutex_init(&frame_buffer[i].rwmutex,NULL);
    for(int i = 0;i < 2000;i++){
		pthread_mutex_init(&frame_buffer_s[i].fetch_mutex,NULL);
		pthread_mutex_init(&frame_buffer_s[i].draw_mmutex,NULL);
		pthread_mutex_init(&frame_buffer_s[i].draw_op1mutex,NULL);
		pthread_mutex_init(&frame_buffer_s[i].draw_op2mutex,NULL);
		pthread_cond_init(&frame_buffer_s[i].fetch_cond,NULL);
		pthread_cond_init(&frame_buffer_s[i].draw_mcond,NULL);
		pthread_cond_init(&frame_buffer_s[i].draw_op1cond,NULL);
		pthread_cond_init(&frame_buffer_s[i].draw_op2cond,NULL);
	}
    pthread_mutex_init(&mutex,NULL);
    pthread_mutex_init(&mutex_mode,NULL);
    pthread_mutex_init(&mutex_current_fetch,NULL);
    pthread_mutex_init(&mutex_current_m,NULL);
    pthread_mutex_init(&mutex_current_op1,NULL);
    pthread_mutex_init(&mutex_current_op2,NULL);
	pthread_mutex_init(&mutex_current_draw,NULL);
    pthread_mutex_init(&mutex_fetchjob,NULL);
    pthread_mutex_init(&mutex_detect_mjob,NULL);
    pthread_mutex_init(&mutex_detect_op1job,NULL);
    pthread_mutex_init(&mutex_detect_op2job,NULL);
    pthread_cond_init(&cond,NULL);

    detection_layer l = net.layers[net.n-1];
    boxes = (box *)calloc(l.side*l.side*l.n, sizeof(box));
    probs = (float **)calloc(l.side*l.side*l.n, sizeof(float *));
    for(int j = 0; j < l.side*l.side*l.n; ++j) probs[j] = (float *)calloc(l.classes, sizeof(float *));
    pthread_t timer,drawer,modecontroller;
    ODA *arg = (ODA*)malloc(sizeof(ODA));
    pthread_create(&modecontroller,0,MODE_CONTROLLER,0);
    while(1){
		pthread_cond_wait(&cond,&mutex);
        pthread_create(&timer,0,TIMER,0);
        pthread_create(&drawer,0,show_frame,0);
    }
    //while(1);
}
#else
extern "C" void demo_yolo(char *cfgfile, char *weightfile, float thresh, int cam_index){
    fprintf(stderr, "YOLO demo needs OpenCV for webcam images.\n");
}
#endif

