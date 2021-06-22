#include <QCoreApplication>
#include <opencv2/highgui/highgui.hpp>
#include<opencv2/opencv.hpp>
//#include<stdio.h>
#include <iostream>
//#include "opencv/cv.h"
//#include "opencv/highgui.h"
#include <algorithm>
#include <math.h>

using namespace cv;
using namespace std;


#define motion_tracking_open
#define  HSV_open
#define depth_calib

#define PI 3.14159265

const int row_num = 480;

int c = 0, meanX, meanZ;
Mat imgArr, imgArr2, imgHeight, HsvArr, Chan[3];
int maxArr[row_num][7];
int IntTresh = 50;
int IntDiff, ind, sumNext = 0;

int G_start = 80;
double scale = 20;
//double scale = 3.5;
const int MAX_CORNERS = 500;

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);


    CvCapture* capLeft = cvCreateCameraCapture(0);
    assert(capLeft);
    //    double Exposure = cvGetCaptureProperty(capLeft, CV_CAP_PROP_AUTO_EXPOSURE);
    //    cvSetCaptureProperty(capLeft, CV_CAP_PROP_AUTO_EXPOSURE, false);
    //    double Exposure2 = cvGetCaptureProperty(capLeft, CV_CAP_PROP_AUTO_EXPOSURE);

    IplImage*img = cvQueryFrame( capLeft );
    IplImage* imgLzr = cvCloneImage(img);

    int wdt = img->width;
    int hgt = img->height;

    imgArr = cvarrToMat(imgLzr);
    imgArr2 = imgArr;

    //    imgHeight = new cv::Mat();
    //    imgHeight->clone(img);
    imgHeight = cvarrToMat(cvCloneImage(img));

    for (int i=0;i<hgt;i++)
    {
        maxArr[i][2] = 0;

        for (int j=0;j<wdt-1;j++)
        {
            //Vec3b p= imgArr2.at<Vec3b>(i,j);
            Vec3b p1= imgArr2.at<Vec3b>(i,j+1);

            //int sum = p[0]+p[1]+p[2];
            int sum1 = p1[0]+p1[1]+p1[2];

            if (sum1>maxArr[i][2])
            {
                maxArr[i][0]= i;
                maxArr[i][1]= j+1;
                maxArr[i][2]= sum1;
                maxArr[i][3] = 30 + 0.136*maxArr[i][1];
            }
        }

        Vec3b scan_clr = 0;
        //scan_clr[1] = 255;
        imgArr2.at<Vec3b>(i,maxArr[i][1]) = scan_clr;
    }

#ifdef depth_calib
    meanX = 0;
    meanZ = 0;
    for (int p = 0;p<row_num;p++)
    {
        meanX += maxArr[p][1];
        meanZ += maxArr[p][3];
    }
    meanX /= row_num;
    meanZ /= row_num;
#endif

    cvNamedWindow( "Video Stream", CV_WINDOW_AUTOSIZE );
    cvNamedWindow("HeightMap",1);

    while(1) {

#ifdef motion_tracking_open
        IplImage* imgC = cvCloneImage(img);
        CvSize img_sz = cvGetSize( imgC );
        IplImage* imgA = cvCreateImage( img_sz, imgC->depth,1 );
        cvCvtColor(imgC, imgA, CV_BGR2GRAY);
#endif

        img = cvQueryFrame( capLeft );
        waitKey(1);

#ifdef motion_tracking_open
        IplImage* imgB = cvCreateImage( img_sz, imgC->depth,1 );
        cvCvtColor(img, imgB, CV_BGR2GRAY);

        int win_size = 10;

        IplImage* eig_image = cvCreateImage( img_sz, IPL_DEPTH_32F, 1 );
        IplImage* tmp_image = cvCreateImage( img_sz, IPL_DEPTH_32F, 1 );
        int corner_count = MAX_CORNERS;
        CvPoint2D32f* cornersA = new CvPoint2D32f[ MAX_CORNERS ];

        cvGoodFeaturesToTrack(	imgA, eig_image, tmp_image,	cornersA, &corner_count, 0.01, 5.0, 0, 3, 0, 0.04 );

        cvFindCornerSubPix(	imgA, cornersA,	corner_count, cvSize(win_size,win_size), cvSize(-1,-1),	cvTermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,20,0.03) );

        char features_found[ MAX_CORNERS ];
        float feature_errors[ MAX_CORNERS ];
        CvSize pyr_sz = cvSize( imgA->width+8, imgB->height/3 );
        IplImage* pyrA = cvCreateImage( pyr_sz, IPL_DEPTH_32F, 1 );
        IplImage* pyrB = cvCreateImage( pyr_sz, IPL_DEPTH_32F, 1 );
        CvPoint2D32f* cornersB = new CvPoint2D32f[ MAX_CORNERS ];

        cvCalcOpticalFlowPyrLK(	imgA, imgB,	pyrA, pyrB,	cornersA, cornersB,	corner_count, cvSize(win_size,win_size), 5, features_found,
                                feature_errors,	cvTermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, .3 ), 0	);

        double xdiff;
        double ydiff;

        double Q1ang[MAX_CORNERS] = {0};
        double Q2ang[MAX_CORNERS] = {0};
        double Q3ang[MAX_CORNERS] = {0};
        double Q4ang[MAX_CORNERS] = {0};

        double Q1dis[MAX_CORNERS] = {0};
        double Q2dis[MAX_CORNERS] = {0};
        double Q3dis[MAX_CORNERS] = {0};
        double Q4dis[MAX_CORNERS] = {0};
        int NumQ1=0, NumQ2=0, NumQ3=0, NumQ4=0;


        for( int i=0; i<corner_count; i++ ) {
            if( features_found[i]==0|| feature_errors[i]>550 ) {
                //printf(" Error is %f ",feature_errors[i]);
                continue;
            }
            //printf(" Got it ");
            CvPoint p0 = cvPoint( cvRound( cornersA[i].x ),	cvRound( cornersA[i].y ));
            CvPoint p1 = cvPoint( cvRound( cornersB[i].x ),	cvRound( cornersB[i].y ));

            if ((p0.x<wdt/2)&&(p0.y<hgt/2)) {       // P0 in 1. quadrant
                if ((p1.x<wdt/2)&&(p1.y<hgt/2)) {   // P1 also in 1. quadrant
                    cvLine( imgC, p0, p1, CV_RGB(0,0,255),1 );
                    xdiff = p1.x - p0.x;
                    ydiff = p1.y - p0.y;
                    Q1dis[NumQ1] =  (int)sqrt(xdiff*xdiff + ydiff*ydiff);
                    Q1ang[NumQ1] = (int)(atan2(ydiff,xdiff)*180 / PI);
                    if (Q1ang[NumQ1]<0) Q1ang[NumQ1] += 360;
                    NumQ1++;
                }
            }

            if ((p0.x>wdt/2)&&(p0.y<hgt/2)) {       // P0 in 2. quadrant
                if ((p1.x>wdt/2)&&(p1.y<hgt/2)) {   // P1 also in 2. quadrant
                    cvLine( imgC, p0, p1, CV_RGB(0,0,255),1 );
                    xdiff = p1.x - p0.x;
                    ydiff = p1.y - p0.y;
                    Q2dis[NumQ2] =  (int)sqrt(xdiff*xdiff + ydiff*ydiff);
                    Q2ang[NumQ2] = (int)(atan2(ydiff,xdiff)*180 / PI);
                    if (Q2ang[NumQ2]<0) Q2ang[NumQ2] += 360;
                    NumQ2++;
                }
            }

            if ((p0.x<wdt/2)&&(p0.y>hgt/2)) {       // P0 in 3. quadrant
                if ((p1.x<wdt/2)&&(p1.y>hgt/2)) {   // P1 also in 3. quadrant
                    cvLine( imgC, p0, p1, CV_RGB(0,0,255),1 );
                    xdiff = p1.x - p0.x;
                    ydiff = p1.y - p0.y;
                    Q3dis[NumQ3] =  (int)sqrt(xdiff*xdiff + ydiff*ydiff);
                    Q3ang[NumQ3] = (int)(atan2(ydiff,xdiff)*180 / PI);
                    if (Q3ang[NumQ3]<0) Q3ang[NumQ3] += 360;
                    NumQ3++;
                }
            }

            if ((p0.x>wdt/2)&&(p0.y>hgt/2)) {       // P0 in 1. quadrant
                if ((p1.x>wdt/2)&&(p1.y>hgt/2)) {   // P1 also in 1. quadrant
                    cvLine( imgC, p0, p1, CV_RGB(0,0,255),1 );
                    xdiff = p1.x - p0.x;
                    ydiff = p1.y - p0.y;
                    Q4dis[NumQ4] =  (int)sqrt(xdiff*xdiff + ydiff*ydiff);
                    Q4ang[NumQ4] = (int)(atan2(ydiff,xdiff)*180 / PI);
                    if (Q4ang[NumQ4]<0) Q4ang[NumQ4] += 360;
                    NumQ4++;
                }
            }
        }

        float MDisQ1, MDisQ2, MDisQ3, MDisQ4;
        float MAngQ1, MAngQ2, MAngQ3, MAngQ4;

        float AccDis = 0, AccAng = 0;
        for (int k=0;k<NumQ1;k++) { AccDis = AccDis + Q1dis[k]; AccAng = AccAng + Q1ang[k]; }
        MDisQ1 = AccDis/NumQ1;
        MAngQ1 = AccAng/NumQ1;

        AccDis = 0, AccAng = 0;
        for (int k=0;k<NumQ2;k++) { AccDis = AccDis + Q2dis[k]; AccAng = AccAng + Q2ang[k]; }
        MDisQ2 = AccDis/NumQ2;
        MAngQ2 = AccAng/NumQ2;

        AccDis = 0, AccAng = 0;
        for (int k=0;k<NumQ3;k++) { AccDis = AccDis + Q3dis[k]; AccAng = AccAng + Q3ang[k]; }
        MDisQ3 = AccDis/NumQ3;
        MAngQ3 = AccAng/NumQ3;

        AccDis = 0, AccAng = 0;
        for (int k=0;k<NumQ4;k++) { AccDis = AccDis + Q4dis[k]; AccAng = AccAng + Q4ang[k]; }
        MDisQ4 = AccDis/NumQ4;
        MAngQ4 = AccAng/NumQ4;

        float maxDist1 = MDisQ1*1.5;
        float minDist1 = MDisQ1*0.5;
        float maxAng1  = MAngQ1*1.5;
        float minAng1  = MAngQ1*0.5;

        float CrMDisQ1;
        float CrMAngQ1;
        int CrNumQ1 = 0;
        AccDis = 0, AccAng = 0;
        for (int k = 0;k<NumQ1;k++) {
            if ((Q1dis[k]>minDist1)&&(Q1dis[k]<maxDist1)
                    &&(Q1ang[k]>minAng1)&&(Q1ang[k]<maxAng1)) {
                AccDis = AccDis + Q1dis[k];
                AccAng += Q1ang[k];
                CrNumQ1++;
            }
        }
        CrMDisQ1 = AccDis/CrNumQ1;
        CrMAngQ1 = AccAng/CrNumQ1;


        float maxDist2 = MDisQ2*1.5;
        float minDist2 = MDisQ2*0.5;
        float maxAng2  = MAngQ2*1.5;
        float minAng2  = MAngQ2*0.5;

        float CrMDisQ2;
        float CrMAngQ2;
        int CrNumQ2 = 0;
        AccDis = 0, AccAng = 0;
        for (int k = 0;k<NumQ2;k++) {
            if ((Q2dis[k]>minDist2)&&(Q2dis[k]<maxDist2)
                    &&(Q2ang[k]>minAng2)&&(Q2ang[k]<maxAng2)) {
                AccDis = AccDis + Q2dis[k];
                AccAng += Q2ang[k];
                CrNumQ2++;
            }
        }
        CrMDisQ2 = AccDis/CrNumQ2;
        CrMAngQ2 = AccAng/CrNumQ2;


        float maxDist3 = MDisQ3*1.8;
        float minDist3 = MDisQ3*0.2;
        float maxAng3  = MAngQ3*1.8;
        float minAng3  = MAngQ3*0.2;

        float CrMDisQ3;
        float CrMAngQ3;
        int CrNumQ3 = 0;
        AccDis = 0, AccAng = 0;
        for (int k = 0;k<NumQ3;k++) {
            if ((Q3dis[k]>minDist3)&&(Q3dis[k]<maxDist3)
                    &&(Q3ang[k]>minAng3)&&(Q3ang[k]<maxAng3)) {
                AccDis = AccDis + Q3dis[k];
                AccAng += Q3ang[k];
                CrNumQ3++;
            }
        }
        CrMDisQ3 = AccDis/CrNumQ3;
        CrMAngQ3 = AccAng/CrNumQ3;


        float maxDist4 = MDisQ4*1.8;
        float minDist4 = MDisQ4*0.2;
        float maxAng4  = MAngQ4*1.8;
        float minAng4  = MAngQ4*0.2;

        float CrMDisQ4;
        float CrMAngQ4;
        int CrNumQ4 = 0;
        AccDis = 0, AccAng = 0;
        for (int k = 0;k<NumQ4;k++) {
            if ((Q4dis[k]>minDist4)&&(Q4dis[k]<maxDist4)
                    &&(Q4ang[k]>minAng4)&&(Q4ang[k]<maxAng4)) {
                AccDis = AccDis + Q4dis[k];
                AccAng += Q4ang[k];
                CrNumQ4++;
            }
        }
        CrMDisQ4 = AccDis/CrNumQ4;
        CrMAngQ4 = AccAng/CrNumQ4;


        CvPoint Org1 = cvPoint(160,120);
        cvLine( imgC, cvPoint(160,117),cvPoint(160,123), CV_RGB(255,255,255),1 );
        cvLine( imgC, cvPoint(158,120),cvPoint(162,120), CV_RGB(255,255,255),1 );

        CvPoint Org2 = cvPoint(480,120);
        cvLine( imgC, cvPoint(480,117),cvPoint(480,123), CV_RGB(255,255,255),1 );
        cvLine( imgC, cvPoint(478,120),cvPoint(482,120), CV_RGB(255,255,255),1 );

        CvPoint Org3 = cvPoint(160,360);
        cvLine( imgC, cvPoint(160,357),cvPoint(160,363), CV_RGB(255,255,255),1 );
        cvLine( imgC, cvPoint(158,360),cvPoint(162,360), CV_RGB(255,255,255),1 );

        CvPoint Org4 = cvPoint(480,360);
        cvLine( imgC, cvPoint(480,357),cvPoint(480,363), CV_RGB(255,255,255),1 );
        cvLine( imgC, cvPoint(478,360),cvPoint(482,360), CV_RGB(255,255,255),1 );

        CvPoint V1 = cvPoint(Org1.x - CrMDisQ1 * cos(CrMAngQ1*PI/180),
                             Org1.y - CrMDisQ1 * sin(CrMAngQ1*PI/180));
        if (CrNumQ1) cvLine( imgC, Org1, V1, CV_RGB(255,255,255),1 );

        CvPoint V2 = cvPoint(Org2.x - CrMDisQ2 * cos(CrMAngQ2*PI/180),
                             Org2.y - CrMDisQ2 * sin(CrMAngQ2*PI/180));
        if (CrNumQ2) cvLine( imgC, Org2, V2, CV_RGB(255,255,255),1 );

        CvPoint V3 = cvPoint(Org3.x - CrMDisQ3 * cos(CrMAngQ3*PI/180),
                             Org3.y - CrMDisQ3 * sin(CrMAngQ3*PI/180));
        if (CrNumQ3) cvLine( imgC, Org3, V3, CV_RGB(255,255,255),1 );

        CvPoint V4 = cvPoint(Org4.x - CrMDisQ4 * cos(CrMAngQ4*PI/180),
                             Org4.y - CrMDisQ4 * sin(CrMAngQ4*PI/180));
        if (CrNumQ4) cvLine( imgC, Org4, V4, CV_RGB(255,255,255),1 );



        cvShowImage("MotionDetection",imgC);
        cvMoveWindow("MotionDetection",1050,15);
#endif

#ifdef HSV_open
        IplImage* imgHSV = cvCloneImage(img);
        cvCvtColor(imgHSV, imgHSV, CV_BGR2HSV);
        cvShowImage("HSV",imgHSV);
        cvMoveWindow("HSV",1000,525);

        IplImage* imgThreshed = cvCreateImage(cvGetSize(img), 8, 1);
        cvInRangeS(imgHSV, cvScalar(10, 10, 10), cvScalar(60, 255, 255), imgThreshed);

        //cvShowImage("Treshed", imgThreshed);
        //cvMoveWindow("Treshed",1050,15);
        cvReleaseImage(&imgThreshed);
#endif

        IplImage* imgLzr = cvCloneImage(img);

        imgArr = cvarrToMat(imgLzr);
        //imgArr2 = imgArr;

        imgArr2 = Mat::zeros(img->height,img->width,CV_8UC3);
        imgHeight = Mat::zeros(img->height,img->width,CV_8UC3);


        for (int i=0;i<hgt;i++)
        {
            maxArr[i][2] = 0;

            for (int j=0;j<wdt-5;j++)
            {
                //Vec3b p= imgArr.at<Vec3b>(i,j);
                Vec3b p1= imgArr.at<Vec3b>(i,j+1);

                //int sum = p[0]+p[1]+p[2];
                int sum1 = p1[0]+p1[1]+p1[2];


                if (sum1>maxArr[i][2])
                {
                    maxArr[i][0]= i;
                    maxArr[i][1]= j+1;
                    maxArr[i][2]= sum1;
                    maxArr[i][3] = 30 + 0.136*maxArr[i][1];

                    maxArr[i][4] = p1[0];
                    maxArr[i][5] = p1[1];
                    maxArr[i][6] = p1[2];
                }

            }


            if (maxArr[i][2]>400)
            {
                Vec3b scan_clr = Vec3b(255,255, 255);

                imgArr2.at<Vec3b>(i,maxArr[i][1]) = scan_clr;

                ind = maxArr[i][1];
                IntDiff = 0;
                int lzr_range = 5;
                while ((IntDiff < IntTresh)&&(ind <= wdt-2)&&(lzr_range>=0))
                {
                    Vec3b p= imgArr.at<Vec3b>(i,ind+1);
                    sumNext = p[0]+p[1]+p[2];
                    IntDiff = maxArr[i][2] - sumNext;
                    if (IntDiff < IntTresh) imgArr2.at<Vec3b>(i,ind + 1) = scan_clr;
                    ind++;
                    if (ind >= wdt-2) 	break;
                    lzr_range--;
                }

                ind = maxArr[i][1];
                IntDiff = 0;
                lzr_range = 5;
                while ((IntDiff < IntTresh)&&(ind >= 2)&&(lzr_range>=0))
                {
                    Vec3b p= imgArr.at<Vec3b>(i,ind-1);
                    sumNext = p[0]+p[1]+p[2];
                    IntDiff = maxArr[i][2] - sumNext;
                    if (IntDiff < IntTresh) imgArr2.at<Vec3b>(i,ind - 1) = scan_clr;
                    ind--;
                    if (ind <= 2) 	break;
                    lzr_range--;
                }


                //for (int p= 0;p<wdt-1;p++)
                //{
                //	if ((imgArr2).at<Vec3b>(i,p)[0] != 0)
                //	{
                //		int height = (30 + 0.136*p)*scale;
                //		imgHeight.at<Vec3b>(10+height,G_start+i) = scan_clr;
                //	}
                //}

                for (int p= 10;p<630;p++)
                {
                    if ((imgArr2).at<Vec3b>(i,p)[0] != 0)
                    {
                        //int ofset = 0;
                        //while ((imgArr2).at<Vec3b>(i,p+ofset++)[0] != 0);
                        //int height = scale*(500 - (p + ofset/2))/30;
                        int height = scale*(630 - p)/30;
                        imgHeight.at<Vec3b>(10+height,G_start+i) = scan_clr;
                        //break;
                    }
                }

            }
        }

        //cvShowImage( "Video Stream", img);

        cvMoveWindow("HeightMap",240,15);
        imshow("HeightMap", imgHeight);

        cvMoveWindow("Video Stream",300,525);
        imshow("Video Stream", imgArr);
        //cvShowImage("LazerMap", &(IplImage)imgArr2);


#ifdef depth_calib
        meanX = 0;
        meanZ = 0;
        for (int p = 0;p<row_num;p++)
        {
            meanX += maxArr[p][1];
            meanZ += maxArr[p][3];
        }
        meanX /= row_num;
        meanZ /= row_num;
#endif

        cvReleaseImage(&imgLzr);

#ifdef motion_tracking_open
        cvReleaseImage(&imgA);
        cvReleaseImage(&imgB);
        cvReleaseImage(&imgC);
        cvReleaseImage(&eig_image);
        cvReleaseImage(&tmp_image);
        cvReleaseImage(&pyrA);
        cvReleaseImage(&pyrB);
#endif

#ifdef HSV_open
        cvReleaseImage(&imgHSV);
#endif

        char c = cvWaitKey(1);
        if( c == 27 ) break;

    }


    cvReleaseCapture( &capLeft );
    //cvDestroyWindow( "Video Stream" );

    return a.exec();
}
