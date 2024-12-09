#include <CL/opencl.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <fstream>

static cl::Device device;
static cl::Context context;
static cl::CommandQueue queue;

#define SINGLE_CHANNEL_TYPE CL_INTENSITY

template<cl_channel_order co, cl_channel_type dt>
struct ocl_image {
    cl::Image2D image_mem;
    unsigned int w;
    unsigned int h;
    cl::ImageFormat image_format;
};

struct ocl_buffer { 
    cl::Buffer mem;
    unsigned int w;
    unsigned int h;
    cl::ImageFormat image_format; // the type of image data this temp buffer reflects
};

const int pyramid_lvls = 3;


ocl_image<SINGLE_CHANNEL_TYPE, CL_UNSIGNED_INT8> ocl_load_image(cl::Context context, std::string image_path) 
{
    cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (image.empty()) 
    {
        std::cerr << "Failed to load image at " << image_path << std::endl;
    }

    unsigned int w = image.cols;
    unsigned int h = image.rows;

    cl::ImageFormat image_format(SINGLE_CHANNEL_TYPE, CL_UNSIGNED_INT8);

    cl::Image2D image_mem(
                context, 
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                image_format, 
                w, 
                h, 
                0, 
                image.data
            );

    ocl_image<SINGLE_CHANNEL_TYPE, CL_UNSIGNED_INT8> img;
    img.w = w;
    img.h = h;
    img.image_format = image_format;
    img.image_mem = image_mem;

    return img;
}

cv::Mat save_ocl_image(ocl_image<SINGLE_CHANNEL_TYPE, CL_UNSIGNED_INT8> img, cl::CommandQueue queue, std::string out_str)
{
    unsigned char *img_ub  = (unsigned char *)malloc( (img.w) * img.h ) ;

    cl::array<cl::size_type, 3> origin = {0, 0, 0};  // Start at the top-left corner of the image
    cl::array<cl::size_type, 3> region = {img.w, img.h, 1};  // Region size is the image width and height

    queue.enqueueReadImage(img.image_mem, CL_TRUE, origin, region, 0, 0, (void*)img_ub);

    cv::Mat cv_img(img.h, img.w, CV_8UC1, img_ub);
    cv::imwrite(out_str, cv_img);

    free (img_ub);

    return cv_img;
}

cv::Mat save_ocl_image(ocl_image<SINGLE_CHANNEL_TYPE, CL_SIGNED_INT16> img, cl::CommandQueue queue, std::string out_str)
{
    short *img_short = (short*)malloc(img.w * img.h * sizeof(short));
    unsigned char *img_ub = (unsigned char*)malloc(img.w * img.h);

    cl::array<cl::size_type, 3> origin = {0, 0, 0};  // Start at the top-left corner of the image
    cl::array<cl::size_type, 3> region = {img.w, img.h, 1};  // Region size is the image width and height

    // Read the image data from the OpenCL buffer
    queue.enqueueReadImage(img.image_mem, CL_TRUE, origin, region, img.w * sizeof(short), 0, (void*)img_short);

    // Find the maximum positive value in the image
    short max_val = 0;
    for (unsigned int i = 0; i < img.w * img.h; i++) {
        if (img_short[i] > max_val) {
            max_val = img_short[i];
        }
    }

    // Avoid division by zero
    if (max_val == 0) max_val = 1;

    // Normalize values to 0-255 and keep negative values as 0
    for (unsigned int i = 0; i < img.w * img.h; i++) {
        short val = img_short[i];
        img_ub[i] = (val > 0) ? static_cast<unsigned char>((val * 255) / max_val) : 0;
    }

    // Create a cv::Mat and save the image
    cv::Mat cv_img(img.h, img.w, CV_8UC1, img_ub);
    cv::imwrite(out_str, cv_img);

    // Free allocated memory
    free(img_short);
    free(img_ub);

    return cv_img;
}

cv::Mat save_ocl_image(ocl_image<CL_RGBA, CL_SIGNED_INT32> img, cl::CommandQueue queue, std::string out_str)
{
    // Allocate memory for the input data and the normalized output data
    int *img_int = (int *)malloc(img.w * img.h * 4 * sizeof(int)); // 4 channels for RGBA
    unsigned char *img_ub = (unsigned char *)malloc(img.w * img.h * 4); // 4 channels for RGBA

    cl::array<cl::size_type, 3> origin = {0, 0, 0};  // Start at the top-left corner of the image
    cl::array<cl::size_type, 3> region = {img.w, img.h, 1};  // Full image dimensions

    // Read the image data from the OpenCL image memory
    queue.enqueueReadImage(img.image_mem, CL_TRUE, origin, region, img.w * 4 * sizeof(int), 0, (void *)img_int);

    // Find the maximum positive value across all channels
    int max_val = 0;
    for (unsigned int i = 0; i < img.w * img.h * 4; i++) {
        if (img_int[i] > max_val) {
            max_val = img_int[i];
        }
    }

    // Avoid division by zero
    if (max_val == 0) max_val = 1;

    // Normalize values for each channel
    for (unsigned int i = 0; i < img.w * img.h * 4; i++) {
        int val = img_int[i];
        img_ub[i] = (val > 0) ? static_cast<unsigned char>((val * 255) / max_val) : 0;
    }

    // Create a cv::Mat with 4 channels (CV_8UC4) for RGBA data
    cv::Mat cv_img(img.h, img.w, CV_8UC4, img_ub);

    // Write the image to file
    cv::imwrite(out_str, cv_img);

    // Free allocated memory
    free(img_int);
    free(img_ub);

    return cv_img;
}

void save_image_float2(ocl_buffer buff, cl::CommandQueue queue, std::string out_str)
{
    int width  = buff.w;
    int height = buff.h;

    // Each pixel has 2 float values (float2)
    float *motion_data = (float *)malloc(width * height * 2 * sizeof(float)); 

    queue.enqueueReadBuffer(buff.mem, CL_TRUE, 0, width * height * 2 * sizeof(float), motion_data);

    cv::Mat cv_img = cv::Mat::zeros(height, width, CV_8UC3);

    int sample_step = 10;
    float scale = 10.0f;

    for (int y = 0; y < height; y += sample_step) {
        for (int x = 0; x < width; x += sample_step) {
            int idx = (y * width + x) * 2;

            float vx = motion_data[idx];
            float vy = motion_data[idx + 1];

            // printf("width: %d, height: %d, x,y (%d, %d) dx,dy (%.2f, %.2f)\n", width, height, x, y, vx, vy);

            cv::Point start(x, y);
            cv::Point end(x + static_cast<int>(vx), y + static_cast<int>(vy));

            cv::arrowedLine(cv_img, start, end, cv::Scalar(0, 255, 0), 1, cv::LINE_AA, 0, 0.2);
        }
    }

    cv::imwrite(out_str, cv_img);
    free(motion_data);

    return;
}


int init_opencl() 
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.size() == 0) 
    {
        std::cout << " No OpenCL platforms found!\n";
        return 1;
    }

    cl::Platform platform = platforms[0];

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (devices.size() == 0) 
    {
        std::cerr << "No devices found!\n";
        return 1;
    }

    device  = devices[0];
    context = cl::Context(device);
    queue   = cl::CommandQueue(context, device);

    return 0;
}


cl::Program build_cl_program_from_file(std::string file_path)
{
    // load kernel file
    std::ifstream kernelFile(file_path);
    if (!kernelFile.is_open()) 
    {
        std::cerr << "Failed to open kernel file filters.cl" << std::endl;
    }

    // load code from kernel file
    std::string kernelCode((std::istreambuf_iterator<char>(kernelFile)),
                            std::istreambuf_iterator<char>());
    kernelFile.close();

    // create cl program with source code 
    cl::Program::Sources sources;
    sources.push_back({kernelCode.c_str(), kernelCode.length()});
    cl::Program program(context, sources);
    
    // build the program
    if (program.build({device}) != CL_SUCCESS) 
    {
        std::cerr << "Error building kernel: "
                  << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device)
                  << std::endl;
    }

    return program;
}


static inline size_t DivUp(size_t dividend, size_t divisor)
{
    return (dividend % divisor == 0) ? (dividend / divisor) : (dividend / divisor + 1);
}



cl::Kernel downfilter_kernel_x;
cl::Kernel downfilter_kernel_y;
cl::Kernel filter_3x1;
cl::Kernel filter_1x3;
cl::Kernel filter_G;
cl::Kernel lkflow_kernel;
cl::Kernel update_motion_kernel;
cl::Kernel convert_kernel;
cl::Kernel print_kernel;

ocl_buffer flow_lvl[3];

template<int lvls, cl_channel_order channel_order, cl_channel_type data_type>
class ocl_pyramid 
{
    public:
        ocl_image<channel_order, data_type> img_lvl[lvls];
        ocl_buffer scratch_buf;
        ocl_image<channel_order, data_type> scratch_img;

        ocl_pyramid();
        int init(int w, int h);
        int fill(ocl_image<SINGLE_CHANNEL_TYPE, CL_UNSIGNED_INT8>, cl::Kernel downfilter_x, cl::Kernel downfilter_y);
        int pyr_fill(ocl_pyramid<3, SINGLE_CHANNEL_TYPE, CL_UNSIGNED_INT8> pyramid, cl::Kernel, cl::Kernel, cl_int4, cl_int4);
        int convFill(ocl_pyramid<3, SINGLE_CHANNEL_TYPE, CL_UNSIGNED_INT8>, cl::Kernel);
        int G_Fill(
            ocl_pyramid<3, SINGLE_CHANNEL_TYPE, CL_SIGNED_INT16> &,
            ocl_pyramid<3, SINGLE_CHANNEL_TYPE, CL_SIGNED_INT16> &,
            cl::Kernel);
        int flowFill(
            ocl_pyramid<3, SINGLE_CHANNEL_TYPE, CL_UNSIGNED_INT8> &I,
            ocl_pyramid<3, SINGLE_CHANNEL_TYPE, CL_UNSIGNED_INT8> &J,
            ocl_pyramid<3, SINGLE_CHANNEL_TYPE, CL_SIGNED_INT16> &Ix,
            ocl_pyramid<3, SINGLE_CHANNEL_TYPE, CL_SIGNED_INT16> &Iy,
            ocl_pyramid<3, CL_RGBA, CL_SIGNED_INT32> &G,
            cl::Kernel);
};

template<int lvls, cl_channel_order channel_order, cl_channel_type data_type>
ocl_pyramid<lvls,channel_order,data_type>::ocl_pyramid() { }

template<int lvls, cl_channel_order channel_order, cl_channel_type data_type>
int ocl_pyramid<lvls,channel_order,data_type>::init(int w, int h) 
{
    for (int i = 0; i < lvls; i++) 
    {
        img_lvl[i].w = w >> i;
        img_lvl[i].h = h >> i;
        img_lvl[i].image_format = cl::ImageFormat(channel_order, data_type);
        img_lvl[i].image_mem    = cl::Image2D(
                                            context, 
                                            CL_MEM_READ_WRITE, 
                                            img_lvl[i].image_format, 
                                            img_lvl[i].w, 
                                            img_lvl[i].h,
                                            0
                                        );
    }

    // initialize a scratch img
    scratch_img.w = img_lvl[0].w;
    scratch_img.h = img_lvl[0].h;
    scratch_img.image_format    = img_lvl[0].image_format;
    scratch_img.image_mem       = cl::Image2D(
                                            context, 
                                            CL_MEM_READ_WRITE, 
                                            img_lvl[0].image_format, 
                                            img_lvl[0].w, 
                                            img_lvl[0].h,
                                            0
                                        );

    // initialize a scratch img
    scratch_buf.w               = img_lvl[0].w;
    scratch_buf.h               = img_lvl[0].h;
    scratch_buf.image_format    = img_lvl[0].image_format;
    int sz;
    if (data_type == CL_UNSIGNED_INT8) {
        sz = sizeof(char);
    } else if (data_type == CL_SIGNED_INT16) {
        sz = sizeof(short);
    } else if( data_type == CL_SIGNED_INT32 && channel_order == CL_RGBA ) {
        sz = sizeof(cl_int) * 4 ;
    } else if( data_type == CL_FLOAT && channel_order == CL_RGBA ) {
        sz = sizeof(cl_float) * 4 ;
    }
    int size = scratch_buf.h * scratch_buf.w * sz;
    scratch_buf.mem = cl::Buffer(context, CL_MEM_READ_WRITE, size);

    return 0;
}

template<int lvls, cl_channel_order channel_order, cl_channel_type data_type>
int ocl_pyramid<lvls,channel_order,data_type>::fill(ocl_image<SINGLE_CHANNEL_TYPE, CL_UNSIGNED_INT8> src_img, cl::Kernel downfilter_x, cl::Kernel downfilter_y)
{
    cl::array<cl::size_type, 3> src_origin  = {0, 0, 0};
    cl::array<cl::size_type, 3> dst_origin  = {0, 0, 0};
    cl::array<cl::size_type, 3> region      = {src_img.w, src_img.h, 1};

    queue.enqueueCopyImage(src_img.image_mem, img_lvl[0].image_mem, src_origin, dst_origin, region);

    for (int i = 1; i < lvls; i++)
    {
        cl::NDRange local( 32, 4 );
        cl::NDRange global( 32 * DivUp( img_lvl[i-1].w, 32 ), 4 * DivUp( img_lvl[i-1].h, 4) );

        int arg_cnt = 0;
        downfilter_kernel_x.setArg(arg_cnt++, img_lvl[i-1].image_mem);
        downfilter_kernel_x.setArg(arg_cnt++, scratch_buf.mem);
        downfilter_kernel_x.setArg(arg_cnt++, img_lvl[i-1].w);
        downfilter_kernel_x.setArg(arg_cnt++, img_lvl[i-1].h);

        queue.enqueueNDRangeKernel(downfilter_kernel_x, cl::NullRange, global, cl::NullRange);

        {
            cl::array<cl::size_type, 3> origin = {0, 0, 0};
            cl::array<cl::size_type, 3> region = {img_lvl[i-1].w, img_lvl[i-1].h, 1};

            queue.enqueueCopyBufferToImage(
                scratch_buf.mem,
                scratch_img.image_mem,
                0,
                origin,
                region
            );
        }

        arg_cnt = 0;
        downfilter_kernel_y.setArg(arg_cnt++, scratch_img.image_mem);
        downfilter_kernel_y.setArg(arg_cnt++, scratch_buf.mem);
        downfilter_kernel_y.setArg(arg_cnt++, img_lvl[i].w);
        downfilter_kernel_y.setArg(arg_cnt++, img_lvl[i].h);

        global = cl::NDRange( 32 * DivUp( img_lvl[i].w, 32 ), 4 * DivUp( img_lvl[i].h, 4) );

        queue.enqueueNDRangeKernel(downfilter_kernel_y, cl::NullRange, global, cl::NullRange);

        {
            cl::array<cl::size_type, 3> origin = { 0, 0, 0} ;
            cl::array<cl::size_type, 3> region = { img_lvl[i].w, img_lvl[i].h, 1 };

            queue.enqueueCopyBufferToImage(
                scratch_buf.mem,
                img_lvl[i].image_mem,
                0,
                origin,
                region
            );
        }
    }

    return 0;
}

template<int lvls, cl_channel_order channel_order, cl_channel_type data_type>
int ocl_pyramid<lvls,channel_order,data_type>::pyr_fill(ocl_pyramid<3, SINGLE_CHANNEL_TYPE, CL_UNSIGNED_INT8> pyr, cl::Kernel kernel_x, cl::Kernel kernel_y, cl_int4 Wx, cl_int4 Wy)
{
    for (int i = 0; i < lvls; i++) {

        cl::NDRange global( 32 * DivUp( pyr.img_lvl[i].w, 32 ), 4 * DivUp( pyr.img_lvl[i].h, 4) );

        int arg_cnt = 0;
        kernel_x.setArg(arg_cnt++, pyr.img_lvl[i].image_mem);
        kernel_x.setArg(arg_cnt++, scratch_buf.mem);
        kernel_x.setArg(arg_cnt++, pyr.img_lvl[i].w);
        kernel_x.setArg(arg_cnt++, pyr.img_lvl[i].h);
        kernel_x.setArg(arg_cnt++, Wx.s[0]);
        kernel_x.setArg(arg_cnt++, Wx.s[1]);
        kernel_x.setArg(arg_cnt++, Wx.s[2]);

        queue.enqueueNDRangeKernel(kernel_x, cl::NullRange, global, cl::NullRange);

        {
            cl::array<cl::size_type, 3> origin = { 0, 0, 0 };
            cl::array<cl::size_type, 3> region = { pyr.img_lvl[i].w, pyr.img_lvl[i].h, 1 };

            queue.enqueueCopyBufferToImage(
                scratch_buf.mem,
                scratch_img.image_mem,
                0,
                origin,
                region
            );
        }

        arg_cnt = 0;
        kernel_y.setArg(arg_cnt++, scratch_img.image_mem);
        kernel_y.setArg(arg_cnt++, scratch_buf.mem);
        kernel_y.setArg(arg_cnt++, pyr.img_lvl[i].w);
        kernel_y.setArg(arg_cnt++, pyr.img_lvl[i].h);
        kernel_y.setArg(arg_cnt++, Wy.s[0]);
        kernel_y.setArg(arg_cnt++, Wy.s[1]);
        kernel_y.setArg(arg_cnt++, Wy.s[2]);

        queue.enqueueNDRangeKernel(kernel_y, cl::NullRange, global, cl::NullRange);

        {
            cl::array<cl::size_type, 3> origin = { 0, 0, 0 };
            cl::array<cl::size_type, 3> region = { pyr.img_lvl[i].w, pyr.img_lvl[i].h, 1 };

            queue.enqueueCopyBufferToImage(
                scratch_buf.mem,
                img_lvl[i].image_mem,
                0,
                origin,
                region
            );
        }

        save_ocl_image(img_lvl[i], queue, "pyrfill"+std::to_string(i)+".png");
    }

    return 0;
}
 

template<int lvls, cl_channel_order channel_order, cl_channel_type data_type>
int ocl_pyramid<lvls,channel_order,data_type>::convFill(ocl_pyramid<3, SINGLE_CHANNEL_TYPE, CL_UNSIGNED_INT8> pyr, cl::Kernel convert_kernel)
{
    for (int i = 0; i < lvls; i++) {

        cl::NDRange local( 32, 4 );
        cl::NDRange global( 32 * DivUp( pyr.img_lvl[i].w, 32 ), 4 * DivUp( pyr.img_lvl[i].h, 4) );

        int arg_cnt = 0;
        convert_kernel.setArg(arg_cnt++, pyr.img_lvl[i].image_mem);
        convert_kernel.setArg(arg_cnt++, scratch_buf.mem);
        convert_kernel.setArg(arg_cnt++, pyr.img_lvl[i].w);
        convert_kernel.setArg(arg_cnt++, pyr.img_lvl[i].h);

        queue.enqueueNDRangeKernel(convert_kernel, cl::NullRange, global, local);

        {
            cl::array<cl::size_type, 3> origin = { 0, 0, 0 };
            cl::array<cl::size_type, 3> region = { pyr.img_lvl[i].w, pyr.img_lvl[i].h, 1 };

            queue.enqueueCopyBufferToImage(
                scratch_buf.mem,
                img_lvl[i].image_mem,
                0,
                origin,
                region
            );
        }
    }

    return 0;
}


template<int lvls, cl_channel_order channel_order, cl_channel_type data_type>
cl_int ocl_pyramid<lvls,channel_order,data_type>::G_Fill( 
            ocl_pyramid<3, SINGLE_CHANNEL_TYPE, CL_SIGNED_INT16> &Ix,
            ocl_pyramid<3, SINGLE_CHANNEL_TYPE, CL_SIGNED_INT16> &Iy,
            cl::Kernel kernel_G)
{
    for (int i = 0; i < lvls; i++) {

        cl::NDRange local (32, 4);
        cl::NDRange global( 32 * DivUp( img_lvl[i].w, 32 ), 4 * DivUp( img_lvl[i].h, 4) );

        int arg_cnt = 0;
        kernel_G.setArg(arg_cnt++, Ix.img_lvl[i].image_mem);
        kernel_G.setArg(arg_cnt++, Iy.img_lvl[i].image_mem);
        kernel_G.setArg(arg_cnt++, scratch_buf.mem);
        kernel_G.setArg(arg_cnt++, Iy.img_lvl[i].w);
        kernel_G.setArg(arg_cnt++, Iy.img_lvl[i].h);

        queue.enqueueNDRangeKernel(kernel_G, cl::NullRange, global, local);

        {
            cl::array<cl::size_type, 3> origin = { 0, 0, 0 };
            cl::array<cl::size_type, 3> region = { img_lvl[i].w, img_lvl[i].h, 1 };

            queue.enqueueCopyBufferToImage(
                scratch_buf.mem,
                img_lvl[i].image_mem,
                0,
                origin,
                region
            );
        }
    
        save_ocl_image(img_lvl[i], queue, "G"+std::to_string(i)+".png");
    }

    return 0;
}


float calc_flow( 
    ocl_pyramid<3, SINGLE_CHANNEL_TYPE,    CL_UNSIGNED_INT8> &I,
    ocl_pyramid<3, SINGLE_CHANNEL_TYPE,    CL_UNSIGNED_INT8> &J,
    ocl_pyramid<3, SINGLE_CHANNEL_TYPE,    CL_SIGNED_INT16> &Ix,
    ocl_pyramid<3, SINGLE_CHANNEL_TYPE,    CL_SIGNED_INT16> &Iy,
    ocl_pyramid<3, CL_RGBA, CL_SIGNED_INT32> &G,
    ocl_pyramid<3, CL_RGBA, CL_FLOAT> &J_float,
    ocl_buffer flowLvl[3],
    cl::Kernel lkflow_kernel )
{
    int lvls = 3;
    float t_flow = 0.0f;

    for (int i = lvls - 1; i >= 0; i--)
    {
        int use_guess = (i < lvls - 1) ? 1 : 0;

        cl::NDRange local( 16, 8 );
        cl::NDRange global( 16 * DivUp( flow_lvl[i].w, 16 ), 8 * DivUp( flow_lvl[i].h, 8) );

        int arg_cnt = 0;
        lkflow_kernel.setArg(arg_cnt++, I.img_lvl[i].image_mem);
        lkflow_kernel.setArg(arg_cnt++, Ix.img_lvl[i].image_mem);
        lkflow_kernel.setArg(arg_cnt++, Iy.img_lvl[i].image_mem);
        lkflow_kernel.setArg(arg_cnt++, G.img_lvl[i].image_mem);
        lkflow_kernel.setArg(arg_cnt++, J_float.img_lvl[i].image_mem);

        if (use_guess)
        {
            lkflow_kernel.setArg(arg_cnt++, flow_lvl[i+1].mem);
            lkflow_kernel.setArg(arg_cnt++, flow_lvl[i+1].w);
        } 
        else 
        {
            lkflow_kernel.setArg(arg_cnt++, flow_lvl[0].mem);
            lkflow_kernel.setArg(arg_cnt++, flow_lvl[0].w);
        }

        lkflow_kernel.setArg(arg_cnt++, flow_lvl[i].mem);
        lkflow_kernel.setArg(arg_cnt++, flow_lvl[i].w);
        lkflow_kernel.setArg(arg_cnt++, flow_lvl[i].h);
        lkflow_kernel.setArg(arg_cnt++, use_guess);

        queue.enqueueNDRangeKernel(lkflow_kernel, cl::NullRange, global, local);    

        save_image_float2(flow_lvl[i], queue, "flow"+std::to_string(i)+".png");
    }

    return t_flow;
}



int main(int argc, char *argv[]) 
{ 
    if (argc < 2) 
    {
        std::cerr << "Usage: " << argv[0] << " <image_path>\n";
        return -1;
    }

    int err = init_opencl();
    if (err)
    {
        std::cerr << "OpenCL not initialized!\n";
        return 1;
    }

    cl::Program filter_program = build_cl_program_from_file("filters.cl");    
    cl::Program lkflow_program = build_cl_program_from_file("lkflow.cl");  
    cl::Program motion_program = build_cl_program_from_file("motion.cl");  

    filter_1x3          = cl::Kernel(filter_program, "filter_1x3_g");
    filter_3x1          = cl::Kernel(filter_program, "filter_3x1_g");
    downfilter_kernel_x = cl::Kernel(filter_program, "downfilter_x_g");
    downfilter_kernel_y = cl::Kernel(filter_program, "downfilter_y_g");
    convert_kernel      = cl::Kernel(filter_program, "convertToRGBAFloat");
    print_kernel        = cl::Kernel(filter_program, "same_img");
    filter_G            = cl::Kernel(filter_program, "filter_G");
    lkflow_kernel       = cl::Kernel(lkflow_program, "lkflow");
    update_motion_kernel = cl::Kernel(motion_program, "motion");

    std::string image_path(argv[1]);

    ocl_image<SINGLE_CHANNEL_TYPE, CL_UNSIGNED_INT8> img1;
    ocl_pyramid<3, SINGLE_CHANNEL_TYPE, CL_UNSIGNED_INT8> *I;
    ocl_pyramid<3, SINGLE_CHANNEL_TYPE, CL_UNSIGNED_INT8> *J;
    ocl_pyramid<3, SINGLE_CHANNEL_TYPE, CL_SIGNED_INT16> *Ix;
    ocl_pyramid<3, SINGLE_CHANNEL_TYPE, CL_SIGNED_INT16> *Iy;
    ocl_pyramid<3, CL_RGBA, CL_SIGNED_INT32> *G;
    ocl_pyramid<3, CL_RGBA, CL_FLOAT> *J_float;

    img1 = ocl_load_image(context, image_path);
    ocl_image img2 = ocl_load_image(context, "frame11.png");

    I       = new ocl_pyramid<3, SINGLE_CHANNEL_TYPE, CL_UNSIGNED_INT8>();
    J       = new ocl_pyramid<3, SINGLE_CHANNEL_TYPE, CL_UNSIGNED_INT8>();
    Ix      = new ocl_pyramid<3, SINGLE_CHANNEL_TYPE, CL_SIGNED_INT16>();
    Iy      = new ocl_pyramid<3, SINGLE_CHANNEL_TYPE, CL_SIGNED_INT16>();
    G       = new ocl_pyramid<3, CL_RGBA, CL_SIGNED_INT32>();
    J_float = new ocl_pyramid<3, CL_RGBA, CL_FLOAT>();

    I->init(img1.w, img1.h);
    J->init(img1.w, img1.h);
    Ix->init(img1.w, img1.h);
    Iy->init(img1.w, img1.h);
    G->init(img1.w, img1.h);
    J_float->init(img1.w, img1.h);

    for (int i=0; i < 3; i++) 
    {
        flow_lvl[i].w = img1.w >> i;
        flow_lvl[i].h = img1.h >> i;
        flow_lvl[i].image_format = cl::ImageFormat(CL_RG, CL_FLOAT);
        int size = flow_lvl[i].w * flow_lvl[i].h* sizeof(cl_float2) ;
        flow_lvl[i].mem = cl::Buffer(context, CL_MEM_READ_WRITE, size);
    }

    // do the flow

    I->fill(img1, downfilter_kernel_x, downfilter_kernel_y);
    J->fill(img2, downfilter_kernel_x, downfilter_kernel_y);

    cl_int4 dx_Wx = { -1,  0,  1, 0 };
    cl_int4 dx_Wy = {  3, 10,  3, 0 };
    Ix->pyr_fill(*I, filter_3x1, filter_1x3, dx_Wx, dx_Wy);

    cl_int4 dy_Wx = {  3, 10, 3, 0 };
    cl_int4 dy_Wy = { -1,  0, 1, 0 }; 
    Iy->pyr_fill(*I, filter_3x1, filter_1x3, dy_Wx, dy_Wy);

    G->G_Fill(*Ix, *Iy, filter_G);
    J_float->convFill(*J, convert_kernel);

    float t_flow = 0.0f;
    t_flow = calc_flow( *I, *J, *Ix, *Iy, *G, *J_float, flow_lvl, lkflow_kernel );



    return 0;
}
