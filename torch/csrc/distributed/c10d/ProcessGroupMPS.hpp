#pragma once

#ifdef USE_C10D_MPS

#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>

namespace c10d::jaccl {
class JACCLTransport;
} // namespace c10d::jaccl

namespace c10d {

constexpr const char* MPS_BACKEND_NAME = "mps";

// TCP socket ring transport using the c10d Store for rendezvous.
// Each rank publishes its listen address, connects to (rank+1)%size,
// and accepts from (rank-1+size)%size.
class TCPRingTransport {
 public:
  TCPRingTransport(
      const c10::intrusive_ptr<Store>& store,
      int rank,
      int size);
  ~TCPRingTransport();

  void sendToRight(const void* data, size_t length);
  void recvFromLeft(void* data, size_t length);

  void sendTo(int dstRank, const void* data, size_t length);
  void recvFrom(int srcRank, void* data, size_t length);

  void barrier();

 private:
  void setupConnections();
  void sendAll(int fd, const void* data, size_t length);
  void recvAll(int fd, void* data, size_t length);
  int connectToRank(int rank);
  int acceptFromRank(int rank);

  c10::intrusive_ptr<Store> store_;
  int rank_;
  int size_;

  int listenFd_{-1};
  int sendFd_{-1}; // to right neighbor
  int recvFd_{-1}; // from left neighbor

  uint64_t p2pSendSeq_{0};
  uint64_t p2pRecvSeq_{0};
  uint64_t barrierCount_{0};
};

class TORCH_API ProcessGroupMPS : public Backend {
 public:
  class WorkMPS : public Work {
   public:
    WorkMPS(
        OpType opType,
        const char* profilingTitle = nullptr,
        const std::optional<std::vector<at::Tensor>>& inputTensors =
            std::nullopt);
    ~WorkMPS() override = default;

    bool isCompleted() override;
    bool isSuccess() const override;
    bool wait(std::chrono::milliseconds timeout = kNoTimeout) override;
    c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;

    void finishWork();
    void finishWorkError(const std::exception_ptr& eptr);

   protected:
    friend class ProcessGroupMPS;
    c10::intrusive_ptr<at::ivalue::Future> future_;
    std::vector<at::Tensor> outputTensors_;
  };

  struct TORCH_API Options : public Backend::Options {
    explicit Options(
        std::chrono::milliseconds timeout = kBackendDefaultTimeout);

    static c10::intrusive_ptr<Options> create(
        std::chrono::milliseconds timeout = kBackendDefaultTimeout) {
      return c10::make_intrusive<Options>(timeout);
    }
  };

  ProcessGroupMPS(
      const c10::intrusive_ptr<Store>& store,
      int rank,
      int size,
      c10::intrusive_ptr<Options> options = Options::create());

  ~ProcessGroupMPS() override;

  const std::string getBackendName() const override {
    return std::string(MPS_BACKEND_NAME);
  }

  c10::intrusive_ptr<Backend::Options> getBackendOptions() override {
    return c10::static_intrusive_pointer_cast<Backend::Options>(options_);
  }

  c10::intrusive_ptr<Options> getOptions() {
    return options_;
  }

  c10::intrusive_ptr<Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override;

  c10::intrusive_ptr<Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const BroadcastOptions& opts = BroadcastOptions()) override;

  c10::intrusive_ptr<Work> barrier(
      const BarrierOptions& opts = BarrierOptions()) override;

  c10::intrusive_ptr<Work> send(
      std::vector<at::Tensor>& tensors,
      int dstRank,
      int tag) override;

  c10::intrusive_ptr<Work> recv(
      std::vector<at::Tensor>& tensors,
      int srcRank,
      int tag) override;

 private:
  at::Tensor syncAndCopyToCPU(const at::Tensor& tensor);
  void copyToMPS(const at::Tensor& cpuTensor, at::Tensor& mpsTensor);

  void ringAllreduce(at::Tensor& data, const ReduceOp& op);

  void enqueue(std::function<void()> fn);
  void runLoop();

  c10::intrusive_ptr<Store> store_;
  c10::intrusive_ptr<Options> options_;
  std::unique_ptr<TCPRingTransport> transport_;
  std::unique_ptr<jaccl::JACCLTransport> jacclTransport_;
  bool useJACCL_{false};

  std::thread workerThread_;
  bool stop_{false};
  std::deque<std::function<void()>> workQueue_;
  std::mutex workMutex_;
  std::condition_variable workCV_;
};

} // namespace c10d

#endif // USE_C10D_MPS
