/*
Simple example of using OpenCV descriptor-based matching to find a target image
in another image or webcam feed. Supports SIFT, SURF, or ORB algorithms; the
former two require OpenCV to be built with the optional contributions modules, as
they are patent encumbered. ORB is standard with OpenCV, and is free for all use.

Author: John Grime, The University of Oklahoma.

Example compilation:

g++ \
-I/usr/local/include/opencv4 \
-lopencv_core -lopencv_highgui -lopencv_imgproc \
-lopencv_imgcodecs -lopencv_videoio -lopencv_calib3d \
-lopencv_features2d -lopencv_xfeatures2d \
-std=c++11 -Wall -Wextra -pedantic -O2 \
example.cpp
*/

#include <iostream>
#include <cctype>
#include <algorithm>

#include "opencv2/core.hpp"

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp"

#include "Util.hpp"

using std::cout;
using std::endl;

#ifdef HAVE_OPENCV_XFEATURES2D

#include "opencv2/xfeatures2d.hpp"

//
// Two little wrapper structs to keep things neat
//

struct KeypointsAndDescriptors
{
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    template<typename detector_t>
    void DetectAndCompute( const cv::Mat& img, const cv::Ptr<detector_t>& detector )
    {
        keypoints.clear();
        detector->detectAndCompute( img, cv::noArray(), keypoints, descriptors );
    }
};

struct KNNMatcher
{
    using DMatchVector = std::vector<cv::DMatch>;

    std::vector<DMatchVector> all_matches;
    DMatchVector good_matches;

    template<typename matcher_t>
    void Match(
        const KeypointsAndDescriptors& kpd1,
        const KeypointsAndDescriptors& kpd2,
        const cv::Ptr<matcher_t>& matcher,
        float Lowe_ratio_thresh = 0.7 )
    {
        all_matches.clear(); // clear or knnMatch() won't (re)calculate matches
        matcher->knnMatch( kpd1.descriptors, kpd2.descriptors, all_matches, 2 );

        good_matches.clear(); // clear, as we're appending matches below.
        for( const auto& m : all_matches )
        {
            if (m[0].distance < Lowe_ratio_thresh * m[1].distance)
            {
                good_matches.push_back(m[0]);
            }
        }
    }    
};

void printUsage( const char* progname )
{
    cout << endl;
    cout << "Usage : " << progname << " find=path [in=path[:scale[:webcamIndex]]] [using=x] [superpose=x] [min=N] [every=N] [gray=yes|no]" << endl;
    cout << endl;
    cout << "Where:" << endl;
    cout << endl;
    cout << "  find  : path to image to detect" << endl;
    cout << "  in    : OPTIONAL path to image in which to search (default: 'webcam', i.e. use webcam feed)" << endl;
    cout << "  using : OPTIONAL algorithm to use, one of 'SURF', 'SIFT', or 'ORB' (default: SIFT)" << endl;
    cout << "  superpose : OPTIONAL path to image to superpose onto matched region" << endl;
    cout << "  min   : OPTIONAL minimum N matching features before bounding box drawn (default: 4)" << endl;
    cout << "  every : OPTIONAL run processing every N frames (default: 1)" << endl;
    cout << "  gray  : OPTIONAL use grayscale images (default: yes)" << endl;
    cout << endl;
    cout << "Notes:" << endl;
    cout << endl;
    cout << "The SURF and ORB algorithms can be accompanied with algorithm-specific data;" << endl;
    cout << "  - for SURF, this is the Hessian tolerance e.g. 'using=SURF:400' (default value: 400')" << endl;
    cout << "  - for ORB, this is the number of features e.g. 'using=ORB:500' (default value: 500')" << endl;
    cout << endl;
    cout << "The 'in' parameter can be decorated with a scale value for the data, e.g.: in=webcam:0.5," << endl;
    cout << "in=mypic.png:1.5. The default scale value is 1.0 (i.e., no scaling will be performed)." << endl;
    cout << "If webcam use is specified, a further webcam index can be provided as a third parameter," << endl;
    cout << "e.g. in=webcam:1.0:0 (default: 0)." << endl;
    cout << endl;

    exit(-1);
}

//
// Off we go ...
//

int main( int argc, char* argv[] )
{
    cv::VideoCapture cap;
    cv::Mat img_ref, img_super, img, img_tmp, transform;

    KeypointsAndDescriptors kpd_ref, kpd;
    KNNMatcher knn;

    cv::Ptr<cv::Feature2D> detector;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    std::vector<char> drawMatchesMask;

    // Put default values in the parameter map
    std::map<std::string,std::vector<std::string>> params {
        { "find",      {""} },
        { "in",        {"webcam"} },
        { "using",      {"SIFT"} },
        { "superpose", {""} },
        { "min",       {"4"} },
        { "every",     {"1"} },
        { "gray",      {"yes"} },
    };

    // Simple lambda to load an image & convert to grayscale if needed
    auto LoadImage = [](cv::Mat& img, const std::string& filepath, bool grayscale = true) {
        if ( (img=cv::imread(filepath)).empty() )
        {
            cout << "Could not load image '" << filepath << "'" << endl;
            exit(-1);
        }
        if (grayscale) cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    };

    bool use_grayscale = true;
    int minMatchesForBoundingBox = 0, processEvery = 1;
    double resize = 1.0;

    //
    // Parse command line arguments
    //

    if (argc<2) printUsage( argv[0] );

    Util::ParseArgs( argc, argv, params );

    cout << "Parameters:" << endl;
    for (const auto& it : params )
    {
        cout << "  " << it.first << " : ";
        for (const auto& v : it.second) {cout << v << " ";}
        cout << endl;
    }

    if (params["gray"][0]!="yes") use_grayscale = false;

    if (!Util::ToNumberIfExists(params["min"],0,minMatchesForBoundingBox))
    {
        cout << "Bad minimum feature matches value '" << params["min"][0] << "'!" << endl;
        exit( -1 );
    }

    if (!Util::ToNumberIfExists(params["every"],0,processEvery))
    {
        cout << "Bad process every value '" << params["every"][0] << "'!" << endl;
        exit( -1 );
    }

    if (!Util::ToNumberIfExists(params["in"],1,resize))
    {
        cout << "Bad resize value '" << params["in"][1] << "'!" << endl;
        exit( -1 );
    }

    //
    // Load reference image, superpose image. If latter defined, also resize to match
    // reference image.
    //

    LoadImage( img_ref, params["find"][0], use_grayscale );
    cout << "Find image dims: " << img_ref.cols << " x " << img_ref.rows << endl;

    if (params["superpose"][0]!="")
    {
        LoadImage( img_super, params["superpose"][0], use_grayscale );
        cv::resize( img_super, img_super, cv::Size(img_ref.cols,img_ref.rows) );
        cout << "Superpose image dims: " << img_super.cols << " x " << img_super.rows << endl;
    }

    //
    // Create detector and appropriate matcher; SIFT, SURF, or ORB.
    //

    {
        const auto& algo_info = params["using"];
        
        auto algo = algo_info[0];

        std::transform( algo.begin(), algo.end(), algo.begin(), [](int x){ return std::tolower(x); } );

        if (algo=="sift")
        {
            detector = cv::xfeatures2d::SIFT::create();
            matcher = cv::FlannBasedMatcher::create();
        }
        else if (algo=="surf")
        {
            int minHessian = 400;

            if (!Util::ToNumberIfExists(algo_info,1,minHessian))
            {
                cout << "Unable to convert SURF minHessian token '" << algo_info[1] << "' into an integer" << endl;
                exit(-1);
            }

            detector = cv::xfeatures2d::SURF::create( minHessian );
            matcher = cv::FlannBasedMatcher::create();
        }
        else if(algo=="orb")
        {
            // Default nFeatures is 500, but this tends not to work so well.
            // OpenCV docs indicate NORM_HAMMING should be used with ORB.
            // If WTA_K is 3 or 4 in ORB constructor (default: 2), use NORM_HAMMING2

            int nFeatures = 500;

            if (!Util::ToNumberIfExists(algo_info,1,nFeatures))
            {
                cout << "Unable to convert ORB nFeatures token '" << algo_info[1] << "' into an integer" << endl;
                exit(-1);
            }

            detector = cv::ORB::create( nFeatures );
            matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
        }
        else
        {
            cout << "Unknown recogniser type " << algo << endl;
            exit( -1 );
        }
    }

    //
    // Get reference keypoints/descriptors.
    //

    {
        kpd_ref.DetectAndCompute( img_ref, detector );
        if (kpd_ref.keypoints.size()<3)
        {
            cout << "Need at least 3 keypoints from reference image; got " << kpd_ref.keypoints.size() << endl;
            exit( -1 );
        }
	cout << kpd_ref.keypoints.size() << " keypoints found in input image." << endl;
    }

    //
    // Process data, either from input image or looping over webcam frames
    //

    int fpsCounter = 0, frameNo = 0;
    bool useWebcam = (params["in"][0] == "webcam");

    if (useWebcam)
    {
        int webcamIndex = 0;
        if (!Util::ToNumberIfExists(params["in"],2,webcamIndex))
        {
            cout << "Bad webcam index '" << params["in"][2] << "'!" << endl;
            exit( -1 );
        }

        cap.open(webcamIndex);
        if (!cap.isOpened())
        {
        	cout << "Unable to open webcam!" << endl;
        	exit(-1);
        }
    }

    //
    // Create an output window
    //

    cv::namedWindow("Good Matches",1);

    //
    // Process data, either from input image or looping over webcam frames
    //

    Util::StatsSet stats;
    std::vector<cv::Point2f> srcPoints, dstPoints;

    const int detect_idx = stats.AddName( "detect" );
    const int knn_idx = stats.AddName( "knn" );
    const int homography_idx = stats.AddName( "homography" );
    const int draw_idx = stats.AddName( "draw" );
    const int resize_idx = stats.AddName( "resize" );

    auto start_ticks = cv::getTickCount();
    for(;;)
    {
        bool haveTransform = false;

        frameNo++;
        fpsCounter++;

        if (useWebcam)
        {
            cap >> img; // get a new frame from webcam
            if (use_grayscale) cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
        }
        else
        {
            LoadImage( img, params["in"][0], use_grayscale );
        }

        if (resize!=1.0)
        {
            auto t1 = cv::getTickCount();
            cv::resize( img, img, cv::Size(), resize, resize );
            stats.AddSampleByIndex( resize_idx, cv::getTickCount()-t1 );
        }

        if ((!useWebcam) || (frameNo%processEvery == 0))
        {
            auto t1 = cv::getTickCount();
            kpd.DetectAndCompute( img, detector );
            stats.AddSampleByIndex( detect_idx, cv::getTickCount()-t1 );

            //
            // We may not have any keypoints if the camera is covered! Need at least
            // 4 points (with 3 non-colinear) to get proper homography transform.
            //
            
            if (kpd.keypoints.size() > 4)
            {
                //
                // KNN matching
                //

                t1 = cv::getTickCount();
                knn.Match( kpd_ref, kpd, matcher );
                stats.AddSampleByIndex( knn_idx, cv::getTickCount()-t1 );

                bool sufficientGoodMatches = ((int)knn.good_matches.size()>minMatchesForBoundingBox);

                if (sufficientGoodMatches)
                {
                    //
                    // Find homography and transform for image of interest.
                    // Replace with something else to avoid camera calib module?
                    //

                    t1 = cv::getTickCount();
                    srcPoints.clear();
                    dstPoints.clear();
                    for (const auto& m : knn.good_matches)
                    {
                        srcPoints.push_back( kpd_ref.keypoints[m.queryIdx].pt );
                        dstPoints.push_back( kpd.keypoints[m.trainIdx].pt );
                    }
                    // def. reproj. value is 3.0 per OpenCV 4.1.1; smaller = slower?
                    transform = cv::findHomography( srcPoints, dstPoints, cv::RANSAC );
                    haveTransform = (!transform.empty());
                    stats.AddSampleByIndex( homography_idx, cv::getTickCount()-t1 );
                }
            }
        }

        //
        // Output to screen
        //

        {
            auto t1 = cv::getTickCount();

            //
            // Annotate output image, if sufficient good matching points found
            // and homography transform matrix is valid.
            //

            float cols1 = img_ref.cols;
            float rows1 = img_ref.rows;

            float cols2 = img.cols;
            float rows2 = img.rows;

            if (haveTransform)
            {
                //
                // Transform superposition image; consider smaller output mat, zero
                // translation components of transform matrix, then explicit translate
                // to save memory / CPU time in add()?
                //

                if (!img_super.empty())
                {
                    cv::warpPerspective( img_super, img_tmp, transform, cv::Size(img.cols,img.rows) );
                    cv::add( img, img_tmp, img );
                }

                //
                // Draw bounding box
                //

                srcPoints = { {0,0}, {0,rows1-1}, {cols1-1,rows1-1}, {cols1-1,0}, {0,0} };
                dstPoints.resize( srcPoints.size() );

                cv::perspectiveTransform( srcPoints, dstPoints, transform );

                for (size_t i=0, max_i=dstPoints.size()-1; i<max_i; i++ )
                {
                    cv::line( img, dstPoints[i], dstPoints[i+1], 255, 2 );
                }

                //
                // Draw mapping of keypoints from reference onto current image
                //

                cv::drawMatches(
                    img_ref, kpd_ref.keypoints,
                    img, kpd.keypoints,
                    knn.good_matches,
                    img_tmp,
                    cv::Scalar::all(-1), cv::Scalar::all(-1),
                    drawMatchesMask,
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
            }
            else
            {
                img_tmp = cv::Mat::zeros(cv::Size(cols1+cols2,std::max(rows1,rows2)), img.type());
                img_ref.copyTo( img_tmp(cv::Rect(0,0,cols1,rows1)) );
                img.copyTo( img_tmp(cv::Rect(cols1,0,cols2,rows2)) );
            }

            cv::imshow("Good Matches", img_tmp);

            stats.AddSampleByIndex( draw_idx, cv::getTickCount()-t1 );
        }

        //
        // Print some stats if needed.
        // "potential fps" is how fast the code could run if only the image
        // processing + display time is taken into account (i.e. ignores IO
        // bottlenecks like reading from camera etc).
        //

        auto end_ticks = cv::getTickCount();
        auto ticks_per_s = cv::getTickFrequency();
        auto elapsed_s = (double)(end_ticks-start_ticks) / ticks_per_s;

        if ((!useWebcam) || (elapsed_s>1))
        {
            double tmp = 0;

            printf( "%.1f fps : ", (double)fpsCounter/elapsed_s );
            for (const auto& it : stats.key_to_idx)
            {
                auto mean = stats.stats_vec[it.second].mean;
                printf( "%s %.2g ms : ", it.first.c_str(), (mean/ticks_per_s) / 1e-3 );
                tmp += mean;
            }
            printf( "%d good matches in %dx%d frame (potential %.2g fps)\n",
                (int)knn.good_matches.size(), img.cols,img.rows, 1.0/(tmp/ticks_per_s) );

            if (haveTransform)
            {
                auto r1 = transform.ptr<double>(0);
                auto r2 = transform.ptr<double>(1);
                auto r3 = transform.ptr<double>(2);

                printf( "| %+8.2f %+8.2f %+8.2f |\n", r1[0], r1[1], r1[2] );
                printf( "| %+8.2f %+8.2f %+8.2f |\n", r2[0], r2[1], r2[2] );
                printf( "| %+8.2f %+8.2f %+8.2f |\n", r3[0], r3[1], r3[2] );
            }

            start_ticks = end_ticks;
            fpsCounter = 0;

            stats.Clear();
        }

        if (useWebcam)
        {
            if(cv::waitKey(10) >= 0) break;
        }
        else
        {
            cv::waitKey();
            break;
        }
    }
}

#else

int main()
{
    cout << "This code requires OpenCV contribution modules to run." << endl;
    return 0;
}

#endif
