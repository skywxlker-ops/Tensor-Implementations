#include "nn/NN.h"

using namespace OwnTensor::nn;

class MyModel : public Module {
public:
    MyModel() {
        auto seq = new Sequential({});
        auto lin = new Linear(10, 10);
        
        // This should FAIL to compile if register_module is protected
        // seq->register_module(lin); 
        
        // This is what they probably did instead:
        register_module(lin);
        
        // And they also did:
        register_module(seq);
    }
};

int main() {
    return 0;
}
