#include "autograd/Engine.h"
#include "autograd/Functions.h"
#include "core/AutogradMeta.h"
#include "core/TensorImpl.h"
#include "ops/TensorOps.h"
#include "utils/ThreadPool.h"
#include <algorithm>
#include <vector>
#include <queue>
#include <unordered_map>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <iostream>
#include <typeinfo>

namespace OwnTensor {
namespace autograd {

// =============================================================================
// Settings
// =============================================================================
namespace {
    ExecutionMode g_execution_mode = ExecutionMode::SEQUENTIAL;
    std::mutex g_mode_mutex;
    size_t g_parallel_threshold = 8;

    // Zero-Allocation ThreadLocal Buffers
    struct ThreadLocalBuffers {
        std::vector<Node*> queue;
        std::vector<int> dependencies;
        std::vector<variable_list> node_inputs;
        std::vector<bool> has_grad;

        void clear() {
            queue.clear();
            dependencies.clear();
            node_inputs.clear();
            has_grad.clear();
        }

        void prepare(size_t size) {
            queue.reserve(size);
            dependencies.resize(size, 0);
            node_inputs.resize(size);
            has_grad.resize(size, false);
        }
    };
    static thread_local ThreadLocalBuffers tl_buffers;
}

ExecutionMode get_execution_mode() {
    std::lock_guard<std::mutex> lock(g_mode_mutex);
    return g_execution_mode;
}
void set_execution_mode(ExecutionMode mode) {
    std::lock_guard<std::mutex> lock(g_mode_mutex);
    g_execution_mode = mode;
}
void set_parallel_threshold(size_t threshold) {
    g_parallel_threshold = threshold;
}

// =============================================================================
// ULTRA-FAST SEQUENTIAL PATH (Zero Bureaucracy)
// =============================================================================

static int find_node_id(const std::vector<Node*>& q, Node* n) {
    // For small graphs, linear search is faster than unordered_map
    for (size_t i = 0; i < q.size(); ++i) if (q[i] == n) return (int)i;
    return -1;
}

void fast_backward_sequential(Node* root_node, Tensor root_grad, uint32_t root_slot) {
    tl_buffers.clear();
    auto& q = tl_buffers.queue;
    
    // DEBUG INSTRUMENTATION
    // std::cout << "\n=== FAST BACKWARD START ===" << std::endl;
    // std::cout << "Root: " << typeid(*root_node).name() << std::endl;

    // 1. Discovery (BFS)
    q.push_back(root_node);
    size_t head = 0;
    while (head < q.size()) {
        Node* curr = q[head++];
        // std::cout << "BFS Visiting " << (head-1) << ": " << typeid(*curr).name() << " Edges: " << curr->next_edges().size() << std::endl; 
        for (const auto& edge : curr->next_edges()) {
            if (edge.is_valid()) {
                Node* next = edge.function.get();
                if (find_node_id(q, next) == -1) {
                    q.push_back(next);
                    // std::cout << "  -> Discovered: " << typeid(*next).name() << std::endl;
                }
            } else {
                // std::cout << "  -> Invalid Edge" << std::endl;
            }
        }
    }
    // std::cout << "Graph size: " << q.size() << std::endl;
    // for(size_t i=0; i<q.size(); ++i) {
    //     std::cout << " Node " << i << ": " << typeid(*q[i]).name() << std::endl;
    // }

    // 2. Prepare states (using tl_buffers to avoid heap allocs)
    tl_buffers.prepare(q.size());
    auto& deps = tl_buffers.dependencies;
    auto& inputs = tl_buffers.node_inputs;
    auto& has_g = tl_buffers.has_grad;

    // 3. Dependency counting
    for (size_t i = 0; i < q.size(); ++i) {
        for (const auto& edge : q[i]->next_edges()) {
            if (edge.is_valid()) {
                int id = find_node_id(q, edge.function.get());
                deps[id]++;
            }
        }
    }
    
    // std::cout << "Dependencies:" << std::endl;
    // for(size_t i=0; i<q.size(); ++i) {
    //     std::cout << " Node " << i << ": " << deps[i] << std::endl;
    // }

    // 4. Seed Root
    inputs[0].resize(root_slot + 1);
    inputs[0][root_slot] = std::move(root_grad);
    has_g[0] = true;

    // 5. Execution (using a simple vector walk instead of a queue)
    std::vector<size_t> ready;
    ready.reserve(q.size());
    ready.push_back(0);

    size_t ready_head = 0;
    while (ready_head < ready.size()) {
        size_t idx = ready[ready_head++];
        Node* node = q[idx];
        
        // std::cout << "Executing Node " << idx << " (" << typeid(*node).name() << ")" << std::endl;

        variable_list node_outputs;
        if (has_g[idx]) {
            node_outputs = (*node)(std::move(inputs[idx]));
            // std::cout << "  Output size: " << node_outputs.size() << std::endl;
        } else {
            // std::cout << "  SKIPPING EXEC (No Grad?)" << std::endl;
        }
        node->release_saved_variables();

        const auto& edges = node->next_edges();
        for (size_t i = 0; i < edges.size(); ++i) {
            if (!edges[i].is_valid()) continue;
            
            int next_idx = find_node_id(q, edges[i].function.get());
            uint32_t slot = edges[i].input_nr;

            if (i < node_outputs.size() && node_outputs[i].is_valid()) {
                if (inputs[next_idx].size() <= slot) inputs[next_idx].resize(slot + 1);
                
                if (!inputs[next_idx][slot].is_valid()) {
                    inputs[next_idx][slot] = std::move(node_outputs[i]);
                } else {
                    // Use in-place add to avoid allocation
                    operator+=(inputs[next_idx][slot], node_outputs[i]);
                }
                has_g[next_idx] = true;
                // std::cout << "  Propagating to Node " << next_idx << " Slot " << slot << std::endl;
            } else {
                 // std::cout << "  Wait: output " << i << " invalid? Or size mismatch?" << std::endl;
            }

            // std::cout << "  Decrementing dep for Node " << next_idx << " (" << deps[next_idx] << " -> " << (deps[next_idx]-1) << ")" << std::endl;
            if (--deps[next_idx] == 0) {
                ready.push_back(next_idx);
                // std::cout << "  -> Ready: Node " << next_idx << std::endl;
            }
        }
    }
    // std::cout << "=== FAST BACKWARD END ===\n" << std::endl;
    
    // Clear TL buffers to release large tensor references
    tl_buffers.clear();
}

// =============================================================================
// PARALLEL ENGINE (Optimized with BackwardContext)
// =============================================================================

struct NodeTask {
    std::vector<std::vector<Tensor>> input_grads;
    std::atomic<int> dependencies{0};
    std::atomic<bool> scheduled{false};
    uint32_t max_output_nr = 0;
    std::mutex task_mutex;

    void add_grad(uint32_t slot, Tensor&& grad) {
        std::lock_guard<std::mutex> lock(task_mutex);
        if (slot >= input_grads.size()) input_grads.resize(slot + 1);
        input_grads[slot].push_back(std::move(grad));
        if (slot > max_output_nr) max_output_nr = slot;
    }
};

struct BackwardContext {
    std::mutex state_mutex;
    std::condition_variable cv;
    std::atomic<int> active_tasks{0};
    std::unordered_map<Node*, int> node_to_id;
    std::vector<std::unique_ptr<NodeTask>> tasks;
    NodeTask* get_task(Node* node) {
        auto it = node_to_id.find(node);
        return it != node_to_id.end() ? tasks[it->second].get() : nullptr;
    }
};

static thread_local std::shared_ptr<BackwardContext> g_current_context = nullptr;

utils::ThreadPool& get_engine_pool() {
    static utils::ThreadPool pool(std::thread::hardware_concurrency());
    return pool;
}

static variable_list aggregate_parallel(NodeTask* task) {
    variable_list inputs;
    inputs.resize(task->max_output_nr + 1);
    bool any = false;
    for (size_t i = 0; i < task->input_grads.size(); ++i) {
        auto& grads = task->input_grads[i];
        if (!grads.empty()) {
            Tensor& res = grads[0];
            for (size_t j = 1; j < grads.size(); ++j) {
                operator+=(res, grads[j]);
            }
            inputs[i] = std::move(res);
            grads.clear();
            any = true;
        }
    }
    return any ? inputs : variable_list();
}

struct TaskFunctor {
    std::shared_ptr<BackwardContext> ctx;
    Node* node;
    void operator()() {
        g_current_context = ctx;
        try {
            Node* curr = node;
            while (curr) {
                NodeTask* task = ctx->get_task(curr);
                variable_list inputs;
                {
                    std::lock_guard<std::mutex> lock(task->task_mutex);
                    inputs = aggregate_parallel(task);
                }
                
                variable_list outputs = (*curr)(std::move(inputs));
                curr->release_saved_variables();
                
                Node* next_inline = nullptr;
                uint64_t max_topo = 0;
                for (size_t i = 0; i < curr->next_edges().size(); ++i) {
                    const auto& edge = curr->next_edges()[i];
                    if (!edge.is_valid()) continue;
                    
                    Node* next_node = edge.function.get();
                    NodeTask* next_task = ctx->get_task(next_node);
                    
                    if (i < outputs.size() && outputs[i].is_valid()) {
                        next_task->add_grad(edge.input_nr, std::move(outputs[i]));
                    }
                    
                    if (next_task->dependencies.fetch_sub(1) == 1) {
                        bool expected = false;
                        if (next_task->scheduled.compare_exchange_strong(expected, true)) {
                            uint64_t topo = next_node->topological_nr();
                            if (!next_inline || topo > max_topo) {
                                if (next_inline) schedule(ctx, next_inline);
                                next_inline = next_node;
                                max_topo = topo;
                            } else { schedule(ctx, next_node); }
                        }
                    }
                }
                curr = next_inline;
            }
        } catch (...) {}
        if (--ctx->active_tasks == 0) {
            std::lock_guard<std::mutex> lk(ctx->state_mutex);
            ctx->cv.notify_all();
        }
        g_current_context = nullptr;
    }
    static void schedule(std::shared_ptr<BackwardContext> ctx, Node* n) {
        ctx->active_tasks++;
        get_engine_pool().enqueue_detach(TaskFunctor{ctx, n});
    }
};

// =============================================================================
// Global Entry Point
// =============================================================================

void backward(const Tensor& root, const Tensor* grad_output) {
    // std::cout << "DEBUG: Engine::backward called on root node. ReqGrad=" << root.requires_grad() << " HasFn=" << (root.grad_fn() ? "Yes" : "No") << std::endl;
    if (!root.requires_grad()) throw std::runtime_error("backward: no grad");
    auto root_fn = root.grad_fn();
    
    Tensor root_grad = grad_output ? *grad_output : (root.numel() == 1 ? Tensor::ones(root.shape(), root.opts()) : throw std::runtime_error("backward: non-scalar requires grad_output"));

    if (!root_fn) {
        // std::cout << "DEBUG: Engine::backward - No grad_fn, accumulating in root." << std::endl;
        if (root.unsafeGetTensorImpl()->has_autograd_meta()) {
            auto* meta = static_cast<AutogradMeta*>(root.unsafeGetTensorImpl()->autograd_meta());
            if (meta->has_grad()) meta->set_grad(operator+(meta->mutable_grad(root.unsafeGetTensorImpl()), root_grad));
            else meta->set_grad(root_grad);
        }
        return;
    }

    ExecutionMode mode = get_execution_mode();

    // QUICK PATH: If Sequential requested, skip all discovery-related heap allocations
    if (mode == ExecutionMode::SEQUENTIAL) {
        fast_backward_sequential(root_fn.get(), std::move(root_grad), root.output_nr());
        return;
    }

    // DISCOVERY FOR PARALLEL (Once)
    auto ctx = std::make_shared<BackwardContext>();
    std::vector<Node*> queue;
    queue.push_back(root_fn.get());
    ctx->node_to_id[root_fn.get()] = 0;
    ctx->tasks.push_back(std::make_unique<NodeTask>());
    
    size_t head = 0;
    while(head < queue.size()) {
        Node* node = queue[head++];
        for (const auto& edge : node->next_edges()) {
            if (edge.is_valid()) {
                Node* next = edge.function.get();
                if (ctx->node_to_id.find(next) == ctx->node_to_id.end()) {
                    ctx->node_to_id[next] = (int)ctx->tasks.size();
                    ctx->tasks.push_back(std::make_unique<NodeTask>());
                    queue.push_back(next);
                }
                ctx->get_task(next)->dependencies.fetch_add(1);
            }
        }
    }

    // FALLBACK: If PARALLEL was chosen but graph is actually tiny
    if (ctx->tasks.size() < g_parallel_threshold) {
        fast_backward_sequential(root_fn.get(), std::move(root_grad), root.output_nr());
        return;
    }

    // EXECUTE PARALLEL
    auto* rt = ctx->tasks[0].get();
    rt->scheduled = true;
    rt->add_grad(root.output_nr(), std::move(root_grad));
    TaskFunctor::schedule(ctx, root_fn.get());
    
    std::unique_lock<std::mutex> lk(ctx->state_mutex);
    ctx->cv.wait(lk, [&] { return ctx->active_tasks.load() == 0; });
}

void queue_call_back(std::function<void()> callback) {
    if (get_execution_mode() == ExecutionMode::SEQUENTIAL) {
         callback();
    } else {
        auto ctx = g_current_context;
        if (ctx) {
            ctx->active_tasks++;
            get_engine_pool().enqueue_detach([ctx, cb = std::move(callback)]() {
                g_current_context = ctx; try { cb(); } catch (...) {}
                if (--ctx->active_tasks == 0) { std::lock_guard<std::mutex> lk(ctx->state_mutex); ctx->cv.notify_all(); }
                g_current_context = nullptr;
            });
        } else get_engine_pool().enqueue_detach(std::move(callback));
    }
}

std::vector<std::shared_ptr<Node>> topological_sort(const Tensor& root) {
    std::vector<std::shared_ptr<Node>> res;
    std::unordered_set<Node*> visited;
    auto root_fn = root.grad_fn();
    if (!root_fn) return res;
    struct Frame { std::shared_ptr<Node> node; size_t edge_idx = 0; };
    std::vector<Frame> stack; stack.push_back({root_fn, 0}); visited.insert(root_fn.get());
    while (!stack.empty()) {
        auto& frame = stack.back(); bool pushed = false;
        const auto& edges = frame.node->next_edges();
        while (frame.edge_idx < edges.size()) {
            const auto& e = edges[frame.edge_idx++];
            if (e.is_valid() && visited.find(e.function.get()) == visited.end()) {
                visited.insert(e.function.get()); stack.push_back({e.function, 0});
                pushed = true; break;
            }
        }
        if (!pushed) { res.push_back(frame.node); stack.pop_back(); }
    }
    std::reverse(res.begin(), res.end());
    return res;
}

} // namespace autograd
} // namespace OwnTensor