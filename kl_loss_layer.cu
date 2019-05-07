#include <vector>
#include <string>
#include "caffe/layers/kl_loss_layer.hpp"

namespace caffe {
template <typename Dtype>
__global__ void KlForward(const int n, const Dtype* in, const Dtype* alpha, Dtype* out){
// f(x) = e^(-alpha) * (x-1/2) + alpha/2    if |x| > 1
//      = e^(-alpha) * x^2 * 1/2 + alpha/2  if |x| <= 1
  CUDA_KERNEL_LOOP(index, n) {
    Dtype x = in[index];
    Dtype abs_x = abs(x);
    Dtype a = alpha[index];
    if (abs_x > 1) {
      out[index] = exp(-a) * (abs_x - 0.5) + a * 0.5;
    }
    else {
      out[index] = exp(-a) * x * x * 0.5 + a * 0.5; 
    }
  }
}


template <typename Dtype>
void KlLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  int count = bottom[0]->count();
  caffe_gpu_sub(count, bottom[0]->gpu_data(), bottom[2]->gpu_data(), diff_.mutable_gpu_data());
  KlForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
    count, diff_.gpu_data(), bottom[1]->gpu_data(), error_.mutable_gpu_data());
  CUDA_POST_KERNEL_CHECK;

  Dtype loss;
  caffe_gpu_dot(count, ones_.gpu_data(), error_.gpu_data(), &loss);
  top[0]->mutable_cpu_data()[0] = loss / count;
}

template <typename Dtype>
__global__ void KlBackward(const int n, const Dtype* in1, const Dtype* in2,
    const Dtype* in3, const Dtype* in4, Dtype* out1, Dtype* out2) {
// f'(xe) = e^(-alpha) * (xe - xg)  if |xg - xe| <= 1
//        = -e^(-alpha)      if |xg - xe| > 1 and xg > xe
//        = e^(-alpha)       if |xg - xe| > 1 and xg < xe
//
// f'(alpha) = -(xg - xe)^2 * 0.5 * e^(-alpha) + 0.5   if |xg - xe| <= 1 
//           = -(abs(xg-xe) - 0.5) * e^(-alpha) + 0.5
        
  CUDA_KERNEL_LOOP(index, n) {
    Dtype d = in1[index];//xe - xg
    Dtype xe = in2[index];
    Dtype xg = in3[index];
    Dtype alpha = in4[index];
    Dtype abs_d = abs(d);
    Dtype ea = exp(-alpha);
    if (abs_d <= 1) {
      out1[index] = ea * d;
      out2[index] = -d*d * 0.5 * ea + 0.5;
    }
    else {
      if (xg > xe) { 
        out1[index] = -ea;
      }
      else {
        out1[index] = ea;
      }
      out2[index] = -(abs_d - 0.5) * ea + 0.5;
    }
  }
}

template <typename Dtype>
void KlLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[2]){
    LOG(FATAL) << this->type() << "Layer cannot backpropage to gt input!";
  }
  if (propagate_down[0] && propagate_down[1]){
    int count = diff_.count();
    Dtype* bottom_diff1 = bottom[0]->mutable_gpu_diff();
    Dtype* bottom_diff2 = bottom[1]->mutable_gpu_diff();
    KlBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, diff_.gpu_data(), bottom[0]->gpu_data(),bottom[2]->gpu_data(),
      bottom[1]->gpu_data(), bottom_diff1, bottom_diff2);
    CUDA_POST_KERNEL_CHECK;
    const  Dtype loss_weight = top[0]->cpu_diff()[0] / count;
    caffe_gpu_scal(count, loss_weight , bottom_diff1);
    caffe_gpu_scal(count, loss_weight , bottom_diff2);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(KlLossLayer);

} // namespace caffe
