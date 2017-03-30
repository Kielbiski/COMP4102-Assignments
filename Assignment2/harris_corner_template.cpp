/*
    Harris Corner Template by Rossen Atanassov
    Last modified: Feb 9, 2006
*/

#include <stdio.h>
#include <string>
#include "opencv/cv.h"
#include "opencv/highgui.h"
#include <math.h>

#define APETURE_SIZE 3
#define BLOCK_SIZE 5
#define K 0.04
#define MAX_CORNERS 100000
#define EIGENVALUE_THRESHOLD 320000

char *wdSource = "Conrners";		// name of the app window
char *wdResult = "Result";		    // name of the window that is used for debuging
char *tbThreshName = "Threshold";	// name of the tab slider
char *tbDistanceName = "Distance";	// name of the distance tab slider

int count = 0,			/* number of corners found */
	Q_size = 5;			/* the size of N that is used in the neighborhood Q */



// Load the source image. HighGUI use.
IplImage *image = 0, *harris_responce = 0, *gray=0;
IplImage *eig_image=0, *temp_image=0;

// An array of points representing the found corners
CvPoint3D32f *corners;

// Functions used to manipulate the final image
void drawCorners (IplImage *, CvPoint3D32f *, int);     /* Displays the found corners as crosses or cirles*/


void generateCorners( ) {
	int rows;
    int cols;
    double smallestEigenvalue;
	int i, j;

	CvSize src_size = cvGetSize (gray);

	// create temporary images to store the two image derivatives
	CvMat *dx = cvCreateMat (src_size.height, src_size.width, CV_16SC1);
    CvMat *dy = cvCreateMat (src_size.height, src_size.width, CV_16SC1);

    cvSobel (gray, dx, 1, 0, APETURE_SIZE);
    cvSobel (gray, dy, 0, 1, APETURE_SIZE);

	// covert type to CV_64FC1 so that cvmGet can access
	CvMat *dx1 = cvCreateMat(src_size.height, src_size.width, CV_64FC1);
	CvMat *dy1 = cvCreateMat(src_size.height, src_size.width, CV_64FC1);

	cvConvertScale( dx, dx1 );
	cvConvertScale( dy, dy1 );

	// Setup the buffers
    harris_responce = cvCloneImage (image);

    // This array will store all corners found in the image
    corners = (CvPoint3D32f*)cvAlloc (MAX_CORNERS * sizeof (corners));
	
	/* ---------- STUDENT MUST CALL THEIR VERSION OF CORNER DETECTOR HERE ----------- //
        At this point you have to call a function that you have written in order to
        detect all corners in the image based on the following:
        1. Use a threshold for smallest eigenvalue called EIGENVALUE_THRESHOLD.
        2. The precomputed derivatives stored inside DX and DY.
		3. Save the corners and also save the number of corners in variable count. 
		3. Output an image with the pixels which are corners marked in some way (a small circle).
		4. Save this image in a file, and return the file and the source code. 
        Hints:
        - You can use cvmGet(image, x, y) in order to get the value of a pixel.
        - You can use cvmSet(image, x, y, value) to set the value of a pixel.
        - For more info look at the "CXCORE Reference Manual" and "CV Reference Manual" 
          of your OpenCV documentation.
	
	// ------------------------------------------------------------------------------ */
	// MUST BE COMPLETED BY THE STUDENT


	printf("Corner count: %d", count);

    // Draw all corners
	drawCorners (harris_responce, corners, count);
    
    // Display the images
	cvShowImage (wdSource, image);
	cvShowImage (wdResult, harris_responce);

	cvSaveImage("Corner-out.jpg", harris_responce);

	// Release the buffers
	cvReleaseMat( &dx );
    cvReleaseMat( &dy );
	cvReleaseImage (&harris_responce);
}

//----------------------- Image painting functions -----------------------//


/* Displays the found corners as points */
void drawCorners (IplImage *img, CvPoint3D32f *corners, int count) {
	/* Displays the found corners as points */
    int i;

    CvScalar color = cvScalar(222,0,0,0);

    for(i = 0; i < count; i++){
		CvPoint pt = cvPoint(corners[i].y, corners[i].x);
		cvCircle(img, pt, 1, color, 1, 8, 0);
	}

	return;
}



//----------------------- Application main function -----------------------//

int main( int argc, char** argv ) {

	// Check for the input image
	char *filename = (argc == 2) ? argv[1] : (char*)"checker.jpg";

	// Load the image from the file
	if ((image = cvLoadImage (filename,1)) == 0) return -1;

    // Convert to grayscale
    gray= cvCreateImage(cvSize(image->width,image->height), IPL_DEPTH_8U, 1);
    cvCvtColor(image, gray, CV_BGR2GRAY);

    // Create windows.
    cvNamedWindow(wdSource, 1);
    cvNamedWindow(wdResult, 1);

	// Get all corners
    generateCorners();


    // Wait for a key stroke
    cvWaitKey(0);
    cvReleaseImage(&image);
    cvReleaseImage(&harris_responce);
 
    cvDestroyWindow(wdSource);
    cvDestroyWindow(wdResult);

    return 0;
}
