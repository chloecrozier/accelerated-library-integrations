/*
 * To demonstrate the functionality of cuFFT, we apply a 2D discrete Fourier transform 
 * to a large image. We then output the spectrum's real and imaginary components. 
 * Additionally we demonstrate (though some what naively) a method for bluring and 
 * sharpening the image by modifying it in the frequency domain and then applying
 * an inverse transform.
 */
#include <cuda_runtime.h>
#include <cufft.h>

#include <cmath>
#include <iostream>
#include <algorithm>
#include <string>
#include <complex>
#include <vector>

// We vendor a small header only library, tinyDNG to 
// load and write tiff format images. This is in no way
// a requirement of cuFFT.
#define TINY_DNG_LOADER_IMPLEMENTATION
#define TINY_DNG_LOADER_NO_STB_IMAGE_INCLUDE
#define TINY_DNG_WRITER_IMPLEMENTATION
#include "../third_party/tinydng/tiny_dng_loader.h"
#include "../third_party/tinydng/tiny_dng_writer.h"

// cuFFT API error checking borrowed from the CUDALibrarySamples repo
#ifndef CUFFT_CALL
#define CUFFT_CALL( call )                                                                                             \
    {                                                                                                                  \
        auto status = static_cast<cufftResult>( call );                                                                \
        if ( status != CUFFT_SUCCESS )                                                                                 \
            fprintf( stderr,                                                                                           \
                     "ERROR: CUFFT call \"%s\" in line %d of file %s failed "                                          \
                     "with "                                                                                           \
                     "code (%d).\n",                                                                                   \
                     #call,                                                                                            \
                     __LINE__,                                                                                         \
                     __FILE__,                                                                                         \
                     status );                                                                                         \
    }
#endif  // CUFFT_CALL


// CUDA runtime error checking borrowed from the CUDALibrarySamples repo
#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL( call )                                                                                           \
    {                                                                                                                  \
        auto status = static_cast<cudaError_t>( call );                                                                \
        if ( status != cudaSuccess )                                                                                   \
            fprintf( stderr,                                                                                           \
                     "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "                                        \
                     "with "                                                                                           \
                     "%s (%d).\n",                                                                                     \
                     #call,                                                                                            \
                     __LINE__,                                                                                         \
                     __FILE__,                                                                                         \
                     cudaGetErrorString( status ),                                                                     \
                     status );                                                                                         \
    }
#endif  // CUDA_RT_CALL

// This small kernel naively applies a cutoff to a spectrum. To do this, it 
// calculates a radius for each pixel, it's distance to the zero frequency.
// For lowpass filters, if the radius is outside a cutoff it becomes zero,
// and for highpass, if the radius is within the cutoff then it's value
// becomes zero.
__global__
void filterKernel(const cufftComplex* input, cufftComplex* output,
                  int width, int height, int freqWidth, float cutoff, bool low_pass) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int stride = blockDim.x * gridDim.x;
  for (auto i = tid; i < freqWidth * height; i+= stride) {
    int x = static_cast<int>(i % freqWidth);
    int y = static_cast<int>(i / freqWidth);

    // DFT's naturally place high frequencies in the center and low ones towards the 
    // corners. We use this trick to "pretend" that the low frequencies are all 
    // at the origin for the purpose of calculating a radius.
    int ky = (y <= height / 2) ? y : y - height;

    // Radius spans as a percentage from zero to one.
    float nx = static_cast<float>(x) / (0.5f * static_cast<float>(width));
    float ny = static_cast<float>(ky) / (0.5f * static_cast<float>(height));
    float radius = std::sqrt(nx * nx + ny * ny);

    cufftComplex zero;
    zero.x = 0.0f;
    zero.y = 0.0f;

    if (low_pass) {
      output[i] = radius <= cutoff ? input[i] : zero;
    } else {
      output[i] = radius <= cutoff ? zero : input[i];
    }
  }
}

// Helper utility for writing out tiff images using tinyDNG
bool write_image(std::vector<unsigned char> img, int height, int width, std::string path) {
  tinydngwriter::DNGImage img_out;
  img_out.SetBigEndian(false);
  img_out.SetImageWidth(width);
  img_out.SetImageLength(height);
  img_out.SetRowsPerStrip(height);
  img_out.SetSamplesPerPixel(1);
  unsigned short bps[1] = {8};
  unsigned short sample_format[1] = {tinydngwriter::SAMPLEFORMAT_UINT};
  img_out.SetBitsPerSample(1, bps);
  img_out.SetSampleFormat(1, sample_format);
  img_out.SetPhotometric(tinydngwriter::PHOTOMETRIC_BLACK_IS_ZERO);
  img_out.SetPlanarConfig(tinydngwriter::PLANARCONFIG_CONTIG);
  img_out.SetCompression(tinydngwriter::COMPRESSION_NONE);
  img_out.SetXResolution(1.0);
  img_out.SetYResolution(1.0);
  img_out.SetResolutionUnit(tinydngwriter::RESUNIT_NONE);
  img_out.SetImageData(img.data(), img.size());

  tinydngwriter::DNGWriter writer(false);
  writer.AddImage(&img_out);
  std::string write_err;
  return writer.WriteToFile(path.c_str(), &write_err);
}


int main() {
  // Load the image file using tinyDNG
  // ignoring errors and warnings for this demo
  std::string input_filename = "nvidia_hq.tiff";
  std::vector<tinydng::FieldInfo> customFields;
  std::vector<tinydng::DNGImage> images;
  std::string warn, err;
  bool ret = tinydng::LoadDNG(input_filename.c_str(), customFields, &images, NULL, NULL);
  if (!ret) {
    return 1;
  }

  // tinyDNG batch loads images, but we just need one
  const tinydng::DNGImage& src = images.front();
  
  // Iteratively convert the pixels from uints to floats
  // to align with what cuFFT expects
  std::vector<float> h_input;
  h_input.reserve(src.width * src.height);
  for (auto pix : src.data) {
    h_input.push_back(static_cast<float>(pix));
  }

  const int width = src.width;
  const int height = src.height;
  const int in_count = width * height;
  // Because our input image only has a real part, we end up
  // with a symmetric spectrum, so we only need to represent half.
  const int freq_width = (width / 2 + 1);
  const int freq_count = freq_width * height;

  float* d_input = nullptr;

  // While cuFFT can apply the FFT in place, we declare extra memory 
  // so we can save the spectrum.
  cufftComplex* d_spectrum = nullptr;
  cufftComplex* d_lowpass_spectrum = nullptr;
  cufftComplex* d_highpass_spectrum = nullptr;

  float* d_lowpass_output = nullptr;
  float* d_highpass_output = nullptr;

  CUDA_RT_CALL(cudaMalloc(reinterpret_cast<void**>(&d_input), in_count * sizeof(float)));
  CUDA_RT_CALL(cudaMalloc(reinterpret_cast<void**>(&d_spectrum), freq_count * sizeof(cufftComplex)));
  CUDA_RT_CALL(cudaMalloc(reinterpret_cast<void**>(&d_lowpass_spectrum), freq_count * sizeof(cufftComplex)));
  CUDA_RT_CALL(cudaMalloc(reinterpret_cast<void**>(&d_highpass_spectrum), freq_count * sizeof(cufftComplex)));
  CUDA_RT_CALL(cudaMalloc(reinterpret_cast<void**>(&d_lowpass_output), in_count * sizeof(float)));
  CUDA_RT_CALL(cudaMalloc(reinterpret_cast<void**>(&d_highpass_output), in_count * sizeof(float)));

  // cuFFT works in two steps, first a plan is created given
  // the size of the input. This generates kernels and allocates
  // memory.
  cufftHandle plan_r2c;
  CUFFT_CALL(cufftPlan2d(&plan_r2c, height, width, CUFFT_R2C));

  // Then the plan is executed on the given data. This step computes the forward transform.
  CUDA_RT_CALL(cudaMemcpy(d_input, h_input.data(), in_count * sizeof(float), cudaMemcpyHostToDevice));
  CUFFT_CALL(cufftExecR2C(plan_r2c, d_input, d_spectrum));

  // Create two new filtered spectrums.
  filterKernel<<<1, 128, 0>>>(d_spectrum, d_lowpass_spectrum, width, height, freq_width, 0.05f, true);
  filterKernel<<<1, 128, 0>>>(d_spectrum, d_highpass_spectrum, width, height, freq_width, 0.1f, false);

  CUDA_RT_CALL(cudaGetLastError());

  // Plan and apply the inverse transform.
  cufftHandle plan_c2r;
  CUFFT_CALL(cufftPlan2d(&plan_c2r, height, width, CUFFT_C2R));
  CUFFT_CALL(cufftExecC2R(plan_c2r, d_lowpass_spectrum, d_lowpass_output));
  CUFFT_CALL(cufftExecC2R(plan_c2r, d_highpass_spectrum, d_highpass_output));

  std::vector<cufftComplex> h_spectrum(freq_count);
  CUDA_RT_CALL(cudaMemcpy(h_spectrum.data(), d_spectrum, freq_count * sizeof(cufftComplex), cudaMemcpyDeviceToHost));

  std::vector<float> h_lowpass(in_count);
  CUDA_RT_CALL(cudaMemcpy(h_lowpass.data(), d_lowpass_output, in_count * sizeof(float), cudaMemcpyDeviceToHost));

  std::vector<float> h_highpass(in_count);
  CUDA_RT_CALL(cudaMemcpy(h_highpass.data(), d_highpass_output, in_count * sizeof(float), cudaMemcpyDeviceToHost));

  // We cast the filtered images back to uint8s and scale back as well (the scaling happens naturally during the 
  // Fourier transform).
  std::vector<unsigned char> lowpass_image;
  lowpass_image.reserve(h_lowpass.size());
  for (auto& val : h_lowpass) {
    lowpass_image.push_back(static_cast<unsigned char>(std::clamp(val / (width * height), 0.0f, 255.0f) + 0.5f));
  }
  write_image(lowpass_image, height, width, "lowpass.tif");

  std::vector<unsigned char> highpass_image;
  highpass_image.reserve(h_highpass.size());
  for (auto& val : h_highpass) {
    highpass_image.push_back(static_cast<unsigned char>(std::clamp(val / (width * height), 0.0f, 255.0f) + 0.5f));
  }
  write_image(highpass_image, height, width, "highpass.tif");

  // Create a nice display of the spectrum as a power spectrum which combines the real and imaginary parts.
  // We additionally take the log to make the display clearer.
  std::vector<float> power_spectrum(in_count);
  for (int y = 0; y < height; ++y) {
    const int ky = y - height / 2;
    for (int x = 0; x < width; ++x) {
      const int kx = x - width / 2;
      const int src_x = std::abs(kx);
      const int src_ky = (kx < 0) ? -ky : ky;
      const int src_y = (src_ky % height + height) % height;
      const cufftComplex value = h_spectrum[src_y * freq_width + src_x];
      power_spectrum[y * width + x] = std::log1pf(value.x * value.x + value.y * value.y);
    }
  }

  // Re-scale the power spectrum to fit within our 8-bit image range.
  std::vector<unsigned char> power_image;
  power_image.reserve(power_spectrum.size());
  auto power_minmax = std::minmax_element(power_spectrum.begin(), power_spectrum.end());
  const float power_min = *power_minmax.first;
  const float power_max = *power_minmax.second;
  const float power_range = power_max - power_min;
  for (auto& power : power_spectrum) {
    power = (power - power_min) / power_range;
    power_image.push_back(static_cast<unsigned char>(power * 255.0f + 0.5f));
  }
  write_image(power_image, height, width, "power.tif");
}
