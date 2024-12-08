#include <CL/opencl.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <fstream>

static cl::Device device;
static cl::Context context;
static cl::CommandQueue queue;

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


ocl_image<CL_R, CL_UNSIGNED_INT8> ocl_load_image(cl::Context context, std::string image_path) 
{
    cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (image.empty()) 
    {
        std::cerr << "Failed to load image at " << image_path << std::endl;
    }

    unsigned int w = image.cols;
    unsigned int h = image.rows;

    cl::ImageFormat image_format(CL_R, CL_UNSIGNED_INT8);

    cl::Image2D image_mem(
                context, 
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                image_format, 
                w, 
                h, 
                0, 
                image.data
            );

    ocl_image<CL_R, CL_UNSIGNED_INT8> img;
    img.w = w;
    img.h = h;
    img.image_format = image_format;
    img.image_mem = image_mem;

    return img;
}

cv::Mat save_ocl_image(ocl_image<CL_R, CL_UNSIGNED_INT8> img, cl::CommandQueue queue, std::string out_str)
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

cv::Mat save_ocl_image(ocl_image<CL_R, CL_SIGNED_INT16> img, cl::CommandQueue queue, std::string out_str)
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

template<int lvls, cl_channel_order channel_order, cl_channel_type data_type>
class ocl_pyramid 
{
    public:
        ocl_image<channel_order, data_type> img_lvl[lvls];
        ocl_buffer scratch_buf;
        ocl_image<channel_order, data_type> scratch_img;

        ocl_pyramid();
        int init(int w, int h);
        int fill(ocl_image<CL_R, CL_UNSIGNED_INT8>, cl::Kernel downfilter_x, cl::Kernel downfilter_y);

        int pyr_fill(ocl_pyramid<3, CL_R, CL_UNSIGNED_INT8> pyramid, cl::Kernel, cl::Kernel, cl_int4, cl_int4);
        // int convFill( ocl_pyramid<3, SINGLE_CHANNEL_TYPE, CL_UNSIGNED_INT8>, cl_kernel );
        // int G_Fill(
        //     ocl_pyramid<3, SINGLE_CHANNEL_TYPE, CL_SIGNED_INT16> &,
        //     ocl_pyramid<3, SINGLE_CHANNEL_TYPE, CL_SIGNED_INT16> &,
        //     cl_kernel  );
        // int flowFill(
        //     ocl_pyramid<3, SINGLE_CHANNEL_TYPE,      CL_UNSIGNED_INT8> &I,
        //     ocl_pyramid<3, SINGLE_CHANNEL_TYPE,      CL_UNSIGNED_INT8> &J,
        //     ocl_pyramid<3, SINGLE_CHANNEL_TYPE, CL_SIGNED_INT16> &Ix,
        //     ocl_pyramid<3, SINGLE_CHANNEL_TYPE, CL_SIGNED_INT16> &Iy,
        //     ocl_pyramid<3, CL_RGBA,      CL_SIGNED_INT32> &G,
        //     cl_kernel );
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
    }
    if (data_type == CL_SIGNED_INT16) {
        sz = sizeof(short);
    }
    int size = scratch_buf.h * scratch_buf.w * sz;
    scratch_buf.mem = cl::Buffer(context, CL_MEM_READ_WRITE, size);

    return 0;
}

template<int lvls, cl_channel_order channel_order, cl_channel_type data_type>
int ocl_pyramid<lvls,channel_order,data_type>::fill(ocl_image<CL_R, CL_UNSIGNED_INT8> src_img, cl::Kernel downfilter_x, cl::Kernel downfilter_y)
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
int ocl_pyramid<lvls,channel_order,data_type>::pyr_fill(ocl_pyramid<3, CL_R, CL_UNSIGNED_INT8> pyr, cl::Kernel kernel_x, cl::Kernel kernel_y, cl_int4 Wx, cl_int4 Wy)
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
    }

    return 0;
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

    filter_1x3          = cl::Kernel(filter_program, "filter_1x3_g");
    filter_3x1          = cl::Kernel(filter_program, "filter_3x1_g");
    downfilter_kernel_x = cl::Kernel(filter_program, "downfilter_x_g");
    downfilter_kernel_y = cl::Kernel(filter_program, "downfilter_y_g");
    print_kernel        = cl::Kernel(filter_program, "same_img");

    std::string image_path(argv[1]);

    ocl_image<CL_R, CL_UNSIGNED_INT8> img1;
    ocl_pyramid<3, CL_R, CL_UNSIGNED_INT8> *I;
    ocl_pyramid<3, CL_R, CL_SIGNED_INT16> *Ix;
    ocl_pyramid<3, CL_R, CL_SIGNED_INT16> *Iy;


    img1 = ocl_load_image(context, image_path);

    I   = new ocl_pyramid<3, CL_R, CL_UNSIGNED_INT8>();
    Ix  = new ocl_pyramid<3, CL_R, CL_SIGNED_INT16>();
    Iy  = new ocl_pyramid<3, CL_R, CL_SIGNED_INT16>();

    I->init(img1.w, img1.h);
    Ix->init(img1.w, img1.h);
    Iy->init(img1.w, img1.h);

    I->fill(img1, downfilter_kernel_x, downfilter_kernel_y);


    cl_int4 dx_Wx = { -1,  0,  1, 0 };
    cl_int4 dx_Wy = {  3, 10,  3, 0 };
    Ix->pyr_fill(*I, filter_3x1, filter_1x3, dx_Wx, dx_Wy);

    // cl_int4 dy_Wx = {  3, 10, 3, 0 };
    // cl_int4 dy_Wy = { -1,  0, 1, 0 }; 
    // Iy.pyr_fill(I, filter_3x1, filter_1x3, dx_Wx, dx_Wy);



    return 0;
}
