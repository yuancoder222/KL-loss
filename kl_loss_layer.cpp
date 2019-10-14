#include <vector>
#include "caffe/layers/kl_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void KlLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  has_weights_=(bottom.size()==4);
}

template <typename Dtype>
void KlLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK(bottom[0]->shape()==bottom[1]->shape());
  CHECK(bottom[0]->shape()==bottom[2]->shape());
  if (has_weights_)
    CHECK(bottom[0]->shape()==bottom[3]->shape());
  diff_.ReshapeLike(*bottom[0]);
  error_.ReshapeLike(*bottom[0]);
  ones_.ReshapeLike(*bottom[0]);
  for (int i=0; i<bottom[0]->count(); ++i) {
    ones_.mutable_cpu_data()[i] = Dtype(1);
  }
}

template <typename Dtype>
void KlLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void KlLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED; 
}

#ifdef CPU_ONLY
STUB_GPU(KlLossLayer);
#endif

INSTANTIATE_CLASS(KlLossLayer);
REGISTER_LAYER_CLASS(KlLoss);

} //namespace caffe
