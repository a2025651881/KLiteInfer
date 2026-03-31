#include "op/add.h"
#include "kernels/kernels_interface.h"

namespace op {

base::Status VecAddLayer::check() const {
  RETURN_IF_FAILED(check_tensor(get_input(0), device_type(), data_type()));
  RETURN_IF_FAILED(check_tensor(get_input(1), device_type(), data_type()));
  RETURN_IF_FAILED(check_tensor(get_output(0), device_type(), data_type()));
  return base::error::Success();
}

base::Status VecAddLayer::forward() {
  auto& input1 = get_input(0);
  auto& input2 = get_input(1);
  auto& output = get_output(0);

  auto add_kernel = kernel::get_add_kernel(device_type());
  CHECK(add_kernel != nullptr);
  add_kernel(input1, input2, output, nullptr);
  return base::error::Success();
}

VecAddLayer::VecAddLayer(base::DeviceType device_type)
    : LayerParam(device_type, LayerType::kLayerAdd, false, "VecAddLayer") {
  reset_input_size(2);
  reset_output_size(1);
}

}  // namespace op