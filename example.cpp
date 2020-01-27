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

#include "Recognize.hpp"
#include "Util.hpp"

using ArgsMap = std::map<std::string,std::vector<std::string>>;

//
// If svec[idx] exists, try to convert into a number. Returns
// false where svec[idx] exists AND conversion failed, else true.
//

template<typename T>
bool ToNumberIfExists(
    const std::vector<std::string>& svec,
    size_t idx,
    T& val )
{
    // ignore conversion attempt if no element at idx; that's okay!
    return (idx>=svec.size()) || Util::String::ToNumber(svec[idx],val);
}

//
// Split up command line argments, create map of key => [val]
//

void ParseArgs(
    int argc,
    char* argv[],
    ArgsMap& args,
    const char* keyval_sep = "=", const char* val_sep = ":" )
{
    std::vector<std::string> toks, vals;

    for( int i=1; i<argc; i++ ) {
        if (Util::String::Tokenize(argv[i],toks,keyval_sep)<2) continue;
        Util::String::Tokenize(toks[1],vals,val_sep);
        args[ toks[0] ] = vals;
    }
}

//
// Load image file into a cv matrix, converting to grayscale if needed.
//

void LoadImage(
    cv::Mat& img,
    const std::string& filepath,
    bool grayscale = true)
{
    img = cv::imread(filepath);
    if (img.empty()) {
        printf("Could not load image '%s'\n", filepath.c_str());
        exit(-1);
    }
    if (grayscale) cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
}

//
// Show a brief user guide.
//

void printUsage( const char* progname )
{
    printf("\n");
    printf("Usage : %s find=path [in=path[:scale[:webcamIndex]]] [using=x] [min=N] [every=N] [gray=yes|no] [superpose=yes|no]\n", progname);
    printf("\n");
    printf("Where:\n");
    printf("\n");
    printf("  find  : path to image to detect\n");
    printf("  in    : OPTIONAL path to image in which to search (default: 'webcam', i.e. use webcam feed)\n");
    printf("  using : OPTIONAL algorithm to use, one of 'SURF', 'SIFT', 'ORB', or 'AKAZE' (default: ORB)\n");
    printf("  min   : OPTIONAL minimum N matching features before bounding box drawn (default: 4)\n");
    printf("  every : OPTIONAL run processing every N frames (default: 1)\n");
    printf("  gray  : OPTIONAL use grayscale images (default: yes)\n");
    printf("  superpose : OPTIONAL flag to superpose reference image onto scene (default: no)\n");
    printf("\n");
    printf("Notes:\n");
    printf("\n");
    printf("The SURF and ORB algorithms can be accompanied with algorithm-specific data;\n");
    printf("  - for SURF, this is the Hessian tolerance e.g. 'using=SURF:400' (default value: 400')\n");
    printf("  - for ORB, this is the number of features e.g. 'using=ORB:500' (default value: 500')\n");
    printf("\n");
    printf("The 'in' parameter can be decorated with a scale value for the data, e.g.: in=webcam:0.5,\n");
    printf("in=mypic.png:1.5. The default scale value is 1.0 (i.e., no scaling will be performed).\n");
    printf("If webcam use is specified, a further webcam index can be provided as a third parameter,\n");
    printf("e.g. in=webcam:1.0:0 (default: 0).\n");
    printf("\n");

    exit(-1);
}

//
// Off we go ...
//

int main( int argc, char* argv[] )
{
    auto TicksToSecs = Recognize::Time::TicksToSeconds;
    auto Now = Recognize::Time::Now;

    cv::VideoCapture webcam;
    cv::Mat img, img_tmp;

    Recognize::Recognizer rec;
    std::vector<cv::Mat> ref_imgs;


    // Put default values in the parameter map
    std::map<std::string,std::vector<std::string>> params {
        { "find",      {""} },
        { "in",        {"webcam"} },
        { "using",      {"ORB"} },
        { "min",       {"5"} },
        { "every",     {"1"} },
        { "gray",      {"yes"} },
        { "superpose", {"no"} },
    };

    bool grayscale = true, superpose = false;
    int minMatches = 0, processEvery = 1;
    double resize = 1.0;

    //
    // Parse command line arguments
    //

    if (argc<2) printUsage( argv[0] );

    ParseArgs( argc, argv, params );

    printf("Parameters:\n");
    for (const auto& it : params ) {
        printf("  %s : ", it.first.c_str());
        for (const auto& v : it.second) printf("%s ", v.c_str());
        printf("\n");
    }

    if (params["gray"][0] != "yes") grayscale = false;
    if (params["superpose"][0] == "yes") superpose = true;

    if (!ToNumberIfExists(params["min"],0,minMatches)) {
        printf("Bad minimum feature matches value '%s'!\n", params["min"][0].c_str());
        exit( -1 );
    }

    if (!ToNumberIfExists(params["every"],0,processEvery)) {
        printf("Bad process every value '%s'!\n", params["every"][0].c_str());
        exit( -1 );
    }

    if (!ToNumberIfExists(params["in"],1,resize)) {
        printf("Bad resize value '%s'!\n", params["in"][1].c_str());
        exit( -1 );
    }

    //
    // Create detector and appropriate matcher; SIFT, SURF, or ORB.
    //

    {
        using Type = Recognize::DetectorMatcherPair::Type;
        const auto& algo_info = params["using"];
        
        auto algo = algo_info[0];

        std::transform( algo.begin(), algo.end(), algo.begin(), [](int x){ return std::tolower(x); } );

        if(algo=="orb") {
            // Default nFeatures is 500, but this tends not to work so well.
            // OpenCV docs indicate NORM_HAMMING should be used with ORB.
            // If WTA_K is 3 or 4 in ORB constructor (default: 2), use NORM_HAMMING2

            int nFeatures = 500;

            if (!ToNumberIfExists(algo_info,1,nFeatures)) {
                printf("ORB nFeatures '%s' isn't an integer\n", algo_info[1].c_str());
                exit(-1);
            }

            rec.Prepare(Type::ORB, nFeatures);
        }
        else if(algo=="akaze") {
            rec.Prepare(Type::AKAZE);
        }
        else if (algo=="sift") {
            rec.Prepare(Type::SIFT);
        }
        else if (algo=="surf") {
            int minHessian = 400;

            if (!ToNumberIfExists(algo_info,1,minHessian)) {
                printf("SURF minHessian '%s' isn't an integer\n", algo_info[1].c_str());
                exit(-1);
            }

            rec.Prepare(Type::SURF, minHessian);
        }
        else {
            printf("Unknown recogniser type %s\n", algo.c_str());
            exit( -1 );
        }
    }

    //
    // Load reference image(s).
    //

    {
        cv::Mat img;
        for (const auto& f : params["find"]) {
            LoadImage( img, f, grayscale );
            printf("Find image dims: %d x %d\n", img.cols, img.rows);

            if (!rec.AddReferenceImage(img)) {
                printf("Unable to add reference image\n");
                exit( -1 );
            }

            ref_imgs.push_back(img);
        }
    }

    //
    // Check we can actually open the webcam, if specified.
    //

    bool useWebcam = (params["in"][0] == "webcam");

    if (useWebcam) {
        int webcamIndex = 0;
        if (!ToNumberIfExists(params["in"],2,webcamIndex)) {
            printf("Bad webcam index '%s'!\n", params["in"][2].c_str());
            exit( -1 );
        }

        webcam.open(webcamIndex);
        if (!webcam.isOpened()) {
        	printf("Unable to open webcam!\n");
        	exit(-1);
        }
    }

    //
    // Process data, either from input image or looping over webcam frames.
    //

    int frameNo = 0, fpsCounter = 0;
    std::vector<cv::Point2f> srcPts, dstPts;

    Util::StatsSet timings;

    auto process_idx = timings.AddName("Process");
    auto detect_idx = timings.AddName("Detect");
    auto match_idx = timings.AddName("Match");
    auto homog_idx = timings.AddName("Homography");

    cv::namedWindow("Good Matches",1);

    auto start_ticks = Now();
    for(;;) {
        int nFound = 0;

        fpsCounter++;
        frameNo++;

        if (useWebcam) {
            webcam >> img; // get a new frame from webcam
            if (grayscale) cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
        } else {
            LoadImage( img, params["in"][0], grayscale );
        }

        if (resize!=1.0) {
            cv::resize( img, img, cv::Size(), resize, resize );
        }

        if ((!useWebcam) || (frameNo%processEvery == 0)) {
            nFound = rec.ProcessImage(img, minMatches);
        }

        //
        // Draw info onto image where any references found
        //

        if (nFound>0) {
            for (size_t i=0, max_i=rec.references.size(); i<max_i; i++) {

                const auto& ref = rec.references[i];

                if (ref.ignore || !ref.present) continue;

                const auto& img_ref = ref_imgs[i];

                // Superpose reference image onto input frame?
                if (superpose) {
                    cv::warpPerspective(img_ref, img_tmp, ref.transform, cv::Size(img.cols,img.rows));
                    cv::add(img, img_tmp, img);
                }

                // Draw bounding box?
                float x = img_ref.cols;
                float y = img_ref.rows;
                srcPts = { {0,0}, {0,y-1}, {x-1,y-1}, {x-1,0}, {0,0} };
                dstPts.resize( srcPts.size() );

                cv::perspectiveTransform( srcPts, dstPts, ref.transform );

                for (size_t j=0; j<dstPts.size()-1; j++ ) {
                    cv::line( img, dstPts[j], dstPts[j+1], 255, 2 );
                }
            }

            timings.AddSampleByIndex(process_idx, TicksToSecs(rec.perf.process));
            timings.AddSampleByIndex(detect_idx, TicksToSecs(rec.perf.detect));
            timings.AddSampleByIndex(match_idx, TicksToSecs(rec.perf.match));
            timings.AddSampleByIndex(homog_idx, TicksToSecs(rec.perf.homography));
        }

        cv::imshow("Good Matches", img);

        //
        // Periodically print some output
        //

        {
            double empty[] = { 0,0,0 };
            auto current_ticks = Now();
            auto elapsed_s = TicksToSecs(current_ticks-start_ticks);
            if ( (!useWebcam) || (elapsed_s>1) ) {

                printf( "%dx%d : %.1f fps : ", img.cols, img.rows, (double)fpsCounter/elapsed_s );
                for (const auto& it : timings.key_to_idx) {
                    const auto s = timings.stats_vec[it.second];
                    printf( "%s %.2g (%.2g) ", it.first.c_str(), s.mean/1e-3, s.StdDev()/1e-3 );
                }
                auto process = timings.stats_vec[process_idx].mean;
                if (process>0) printf(" ; notional max fps is %.2f", 1.0/process);

                printf(" [ ");
                for (const auto& ref: rec.references) printf("%d ", (int)ref.matches.size());
                printf("]");

                printf("\n");

                if (nFound>0) {
                    for (int row=0; row<3; row++ ) {
                        for (const auto& ref: rec.references) {
                            auto N = (int)ref.matches.size();
                            auto t = ref.transform;
                            auto r = (N<minMatches) ? empty : t.ptr<double>(row);
                            printf( "| %+8.2f %+8.2f %+8.2f | ", r[0], r[1], r[2] );
                        }
                        printf("\n");
                    }
                }

                timings.Clear();
                fpsCounter = 0;
                start_ticks = Now();
            }
        }

        //
        // If using a webcam and get a keypress in the next 10ms, quit.
        // For static images, just wait for a keypress and then quit.
        //

        if (useWebcam) {
            if (cv::waitKey(10) >= 0) break;
        }
        else {
            cv::waitKey();
            break;
        }
    }
}
