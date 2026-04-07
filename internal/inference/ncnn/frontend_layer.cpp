//go:build ncnn

#include "ncnn/c_api.h"
#include "ncnn/layer.h"
#include "ncnn/net.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <vector>

namespace {

constexpr int kSplitAudioLen = 144000;
constexpr int kSplitFrames = 511;
constexpr int kSplitBins = 96;
constexpr int kSplitChannels = 2;

constexpr int kSpec1FFT = 2048;
constexpr int kSpec1Hop = 278;
constexpr int kSpec1Bins = kSpec1FFT / 2 + 1;

constexpr int kSpec2FFT = 1024;
constexpr int kSpec2Hop = 280;
constexpr int kSpec2Bins = kSpec2FFT / 2 + 1;

constexpr double kSpec1PowExp = 0.22952409088611603;
constexpr double kSpec2PowExp = 0.1905273050069809;

constexpr float kNormEps = 9.999999974752427e-07f;
constexpr float kNormSub = 0.5f;
constexpr float kNormMul = 2.0f;

constexpr float kChannelScale0 = 0.19752595f;
constexpr float kChannelScale1 = 3.152703f;
constexpr float kChannelBias0 = -0.65386057f;
constexpr float kChannelBias1 = -0.34679556f;

std::vector<float> g_mel_fb1;
std::vector<float> g_mel_fb2;
std::vector<float> g_hann1;
std::vector<float> g_hann2;
std::vector<std::complex<double> > g_twiddle1;
std::vector<std::complex<double> > g_twiddle2;
std::mutex g_filterbank_mutex;
bool g_filterbanks_ready = false;

std::vector<float> periodic_hann(int n)
{
    std::vector<float> w(n);
    const double scale = 2.0 * M_PI / static_cast<double>(n);
    for (int i = 0; i < n; i++)
    {
        w[i] = static_cast<float>(0.5 * (1.0 - std::cos(scale * i)));
    }
    return w;
}

std::vector<std::complex<double> > compute_twiddle(int n)
{
    std::vector<std::complex<double> > t(n / 2);
    const double angle = -2.0 * M_PI / static_cast<double>(n);
    for (int k = 0; k < n / 2; k++)
    {
        t[k] = std::complex<double>(std::cos(angle * k), std::sin(angle * k));
    }
    return t;
}

std::vector<float> raw_bytes_to_float32(const unsigned char* bytes, int len)
{
    std::vector<float> out(len / 4);
    for (size_t i = 0; i < out.size(); i++)
    {
        float value = 0.f;
        std::memcpy(&value, bytes + (i * 4), sizeof(float));
        out[i] = value;
    }
    return out;
}

void fft_in_place(std::vector<std::complex<double> >& x, const std::vector<std::complex<double> >& twiddle)
{
    const int n = static_cast<int>(x.size());

    int j = 0;
    for (int i = 1; i < n; i++)
    {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1)
        {
            j ^= bit;
        }
        j ^= bit;
        if (i < j)
        {
            std::swap(x[i], x[j]);
        }
    }

    for (int length = 2; length <= n; length <<= 1)
    {
        const int half = length / 2;
        const int step = n / length;
        for (int i = 0; i < n; i += length)
        {
            for (int k = 0; k < half; k++)
            {
                const std::complex<double> u = x[i + k];
                const std::complex<double> v = x[i + k + half] * twiddle[k * step];
                x[i + k] = u + v;
                x[i + k + half] = u - v;
            }
        }
    }
}

void normalize_audio(const float* samples, int sample_count, std::vector<float>& out)
{
    out.resize(sample_count);
    if (sample_count == 0)
    {
        return;
    }

    float min_value = samples[0];
    for (int i = 1; i < sample_count; i++)
    {
        if (samples[i] < min_value)
        {
            min_value = samples[i];
        }
    }

    float max_sub = 0.f;
    for (int i = 0; i < sample_count; i++)
    {
        const float value = samples[i] - min_value;
        if (value > max_sub)
        {
            max_sub = value;
        }
    }

    const float denom = max_sub + kNormEps;
    for (int i = 0; i < sample_count; i++)
    {
        const float x = (samples[i] - min_value) / denom;
        out[i] = (x - kNormSub) * kNormMul;
    }
}

void stft_mel_spec(
    const std::vector<float>& signal,
    int fft_size,
    int hop,
    int freq_bins,
    const std::vector<float>& hann,
    const std::vector<float>& fb,
    double pow_exp,
    const std::vector<std::complex<double> >& twiddle,
    float* dst,
    int num_threads)
{
    const int signal_len = static_cast<int>(signal.size());

#if defined(_OPENMP)
#pragma omp parallel for num_threads(num_threads)
#endif
    for (int frame = 0; frame < kSplitFrames; frame++)
    {
        std::vector<std::complex<double> > buf(fft_size);
        std::vector<float> real_bins(freq_bins);
        std::vector<float> mel(kSplitBins, 0.f);

        const int start = frame * hop;
        for (int i = 0; i < fft_size; i++)
        {
            const int signal_index = start + i;
            if (signal_index < signal_len)
            {
                buf[i] = std::complex<double>(static_cast<double>(signal[signal_index]) * hann[i], 0.0);
            }
            else
            {
                buf[i] = 0.0;
            }
        }

        fft_in_place(buf, twiddle);

        for (int k = 0; k < freq_bins; k++)
        {
            real_bins[k] = static_cast<float>(std::real(buf[k]));
        }

        for (int k = 0; k < freq_bins; k++)
        {
            const float v = real_bins[k];
            if (v == 0.f)
            {
                continue;
            }

            const int base = k * kSplitBins;
            for (int j = 0; j < kSplitBins; j++)
            {
                mel[j] += v * fb[base + j];
            }
        }

        const int base = frame * kSplitBins;
        for (int j = 0; j < kSplitBins; j++)
        {
            const float power = mel[j] * mel[j];
            dst[base + j] = static_cast<float>(std::pow(power, pow_exp));
        }
    }
}

int birdnet_frontend_compute_impl(const float* samples, int sample_count, float* out, int out_count, int num_threads)
{
    if (!g_filterbanks_ready)
    {
        return -10;
    }
    if (sample_count != kSplitAudioLen)
    {
        return -11;
    }

    const int channel_size = kSplitFrames * kSplitBins;
    if (out_count != kSplitChannels * channel_size)
    {
        return -12;
    }

    std::vector<float> norm;
    normalize_audio(samples, sample_count, norm);

    std::vector<float> spec1(channel_size);
    std::vector<float> spec2(channel_size);

    stft_mel_spec(norm, kSpec1FFT, kSpec1Hop, kSpec1Bins, g_hann1, g_mel_fb1, kSpec1PowExp, g_twiddle1, spec1.data(), num_threads);
    stft_mel_spec(norm, kSpec2FFT, kSpec2Hop, kSpec2Bins, g_hann2, g_mel_fb2, kSpec2PowExp, g_twiddle2, spec2.data(), num_threads);

    for (int frame = 0; frame < kSplitFrames; frame++)
    {
        const int frame_base = frame * kSplitBins;
        for (int bin = 0; bin < kSplitBins; bin++)
        {
            const int src_index = frame_base + (kSplitBins - 1 - bin);
            const int dst_index = bin * kSplitFrames + frame;
            out[dst_index] = spec1[src_index] * kChannelScale0 + kChannelBias0;
            out[channel_size + dst_index] = spec2[src_index] * kChannelScale1 + kChannelBias1;
        }
    }

    return 0;
}

class BirdNETFrontend : public ncnn::Layer
{
public:
    BirdNETFrontend()
    {
        one_blob_only = true;
        support_inplace = false;
        support_vulkan = false;
        support_packing = false;
        support_bf16_storage = false;
        support_fp16_storage = false;
        support_vulkan_packing = false;
        support_any_packing = false;
        support_vulkan_any_packing = false;
    }

    virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const
    {
        if (bottom_blob.dims != 1 || bottom_blob.w != kSplitAudioLen)
        {
            return -1;
        }

        top_blob.create(kSplitFrames, kSplitBins, kSplitChannels, 4u, opt.blob_allocator);
        if (top_blob.empty())
        {
            return -100;
        }

        std::vector<float> output(kSplitChannels * kSplitFrames * kSplitBins);
        const int status = birdnet_frontend_compute_impl((const float*)bottom_blob, bottom_blob.w, output.data(), static_cast<int>(output.size()), opt.num_threads);
        if (status != 0)
        {
            return status;
        }

        const int channel_size = kSplitFrames * kSplitBins;
        float* channel0 = top_blob.channel(0);
        float* channel1 = top_blob.channel(1);
        std::memcpy(channel0, output.data(), channel_size * sizeof(float));
        std::memcpy(channel1, output.data() + channel_size, channel_size * sizeof(float));
        return 0;
    }
};

DEFINE_LAYER_CREATOR(BirdNETFrontend)

}  // namespace

extern "C" void birdnet_ncnn_frontend_set_filterbanks(const unsigned char* spec1, int spec1_len, const unsigned char* spec2, int spec2_len)
{
    std::lock_guard<std::mutex> lock(g_filterbank_mutex);

    g_mel_fb1 = raw_bytes_to_float32(spec1, spec1_len);
    g_mel_fb2 = raw_bytes_to_float32(spec2, spec2_len);
    g_hann1 = periodic_hann(kSpec1FFT);
    g_hann2 = periodic_hann(kSpec2FFT);
    g_twiddle1 = compute_twiddle(kSpec1FFT);
    g_twiddle2 = compute_twiddle(kSpec2FFT);

    g_filterbanks_ready =
        static_cast<int>(g_mel_fb1.size()) == (kSpec1Bins * kSplitBins) &&
        static_cast<int>(g_mel_fb2.size()) == (kSpec2Bins * kSplitBins);
}

extern "C" int birdnet_ncnn_frontend_compute(const float* samples, int sample_count, float* out, int out_count)
{
    return birdnet_frontend_compute_impl(samples, sample_count, out, out_count, 1);
}

extern "C" void birdnet_ncnn_register_custom_layers(ncnn_net_t net)
{
    ((ncnn::Net*)net->pthis)->register_custom_layer("BirdNETFrontend", BirdNETFrontend_layer_creator);
}
