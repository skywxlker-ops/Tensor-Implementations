#include "autograd/Node.h"
#include "core/Tensor.h"

namespace OwnTensor {

variable_list Node::operator()(variable_list&& inputs) {
    variable_list processed_inputs = std::move(inputs);
    
    // Execute pre-hooks (only if they exist)
    if (num_pre_hooks() > 0) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& hook : pre_hooks_) {
            processed_inputs = hook(processed_inputs);
        }
    }
    
    // Apply the backward function
    variable_list outputs = apply(std::move(processed_inputs));
    
    // Execute post-hooks (only if they exist)
    if (num_post_hooks() > 0) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& hook : post_hooks_) {
            hook(processed_inputs, outputs);
        }
    }
    
    return outputs;
}

} // namespace OwnTensor
