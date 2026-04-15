#ifdef USE_C10D_MPS

#include <torch/csrc/distributed/c10d/ProcessGroupMPS.hpp>
#include <torch/csrc/distributed/c10d/JACCLTransport.h>

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <ATen/mps/MPSStream.h>
#include <c10/util/irange.h>

namespace c10d {

// --- TCPRingTransport ---

TCPRingTransport::TCPRingTransport(
    const c10::intrusive_ptr<Store>& store,
    int rank,
    int size)
    : store_(store), rank_(rank), size_(size) {
  setupConnections();
}

TCPRingTransport::~TCPRingTransport() {
  if (sendFd_ >= 0)
    close(sendFd_);
  if (recvFd_ >= 0)
    close(recvFd_);
  if (listenFd_ >= 0)
    close(listenFd_);
}

void TCPRingTransport::sendAll(int fd, const void* data, size_t length) {
  auto ptr = static_cast<const uint8_t*>(data);
  size_t sent = 0;
  while (sent < length) {
    auto n = ::send(fd, ptr + sent, length - sent, 0);
    TORCH_CHECK(n > 0, "TCPRingTransport::sendAll failed: ", strerror(errno));
    sent += static_cast<size_t>(n);
  }
}

void TCPRingTransport::recvAll(int fd, void* data, size_t length) {
  auto ptr = static_cast<uint8_t*>(data);
  size_t received = 0;
  while (received < length) {
    auto n = ::recv(fd, ptr + received, length - received, 0);
    TORCH_CHECK(n > 0, "TCPRingTransport::recvAll failed: ", strerror(errno));
    received += static_cast<size_t>(n);
  }
}

void TCPRingTransport::setupConnections() {
  // Create listening socket
  listenFd_ = socket(AF_INET, SOCK_STREAM, 0);
  TORCH_CHECK(listenFd_ >= 0, "Failed to create listen socket");

  int opt = 1;
  setsockopt(listenFd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  struct sockaddr_in addr {};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port = 0; // ephemeral port

  TORCH_CHECK(
      bind(listenFd_, (struct sockaddr*)&addr, sizeof(addr)) == 0,
      "Failed to bind listen socket: ",
      strerror(errno));
  TORCH_CHECK(
      listen(listenFd_, size_) == 0,
      "Failed to listen: ",
      strerror(errno));

  // Get the assigned port
  socklen_t addrLen = sizeof(addr);
  getsockname(listenFd_, (struct sockaddr*)&addr, &addrLen);
  uint16_t port = ntohs(addr.sin_port);

  // Publish our address to the store
  // Format: 4 bytes IP + 2 bytes port
  // getsockname after binding to INADDR_ANY returns 0.0.0.0. To find our
  // routable IP, connect a UDP socket to MASTER_ADDR — the OS routing table
  // picks the right source address without sending any packets.
  uint32_t ip = addr.sin_addr.s_addr;
  if (ip == htonl(INADDR_ANY)) {
    const char* masterAddr = std::getenv("MASTER_ADDR");
    if (masterAddr) {
      int probe = socket(AF_INET, SOCK_DGRAM, 0);
      if (probe >= 0) {
        struct sockaddr_in target {};
        target.sin_family = AF_INET;
        target.sin_port = htons(9);  // discard port, no packet actually sent
        inet_pton(AF_INET, masterAddr, &target.sin_addr);
        if (connect(probe, (struct sockaddr*)&target, sizeof(target)) == 0) {
          struct sockaddr_in local {};
          socklen_t len = sizeof(local);
          getsockname(probe, (struct sockaddr*)&local, &len);
          ip = local.sin_addr.s_addr;
        }
        close(probe);
      }
    }
  }
  std::vector<uint8_t> addrData(6);
  memcpy(addrData.data(), &ip, 4);
  memcpy(addrData.data() + 4, &port, 2);
  store_->set("mps_ring/" + std::to_string(rank_), addrData);

  // Connect to right neighbor
  int rightRank = (rank_ + 1) % size_;
  if (rightRank != rank_) {
    sendFd_ = connectToRank(rightRank);
  }

  // Accept from left neighbor
  int leftRank = (rank_ - 1 + size_) % size_;
  if (leftRank != rank_) {
    recvFd_ = acceptFromRank(leftRank);
  }
}

int TCPRingTransport::connectToRank(int rank) {
  auto data = store_->get("mps_ring/" + std::to_string(rank));
  TORCH_CHECK(data.size() == 6, "Invalid address data from store");

  uint32_t ip;
  uint16_t port;
  memcpy(&ip, data.data(), 4);
  memcpy(&port, data.data() + 4, 2);

  int fd = socket(AF_INET, SOCK_STREAM, 0);
  TORCH_CHECK(fd >= 0, "Failed to create socket");

  int opt = 1;
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));

  struct sockaddr_in addr {};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = ip;
  addr.sin_port = htons(port);

  TORCH_CHECK(
      connect(fd, (struct sockaddr*)&addr, sizeof(addr)) == 0,
      "Failed to connect to rank ",
      rank,
      ": ",
      strerror(errno));

  return fd;
}

int TCPRingTransport::acceptFromRank(int /*rank*/) {
  struct sockaddr_in addr {};
  socklen_t addrLen = sizeof(addr);
  int fd = accept(listenFd_, (struct sockaddr*)&addr, &addrLen);
  TORCH_CHECK(fd >= 0, "Failed to accept connection: ", strerror(errno));

  int opt = 1;
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt));

  return fd;
}

void TCPRingTransport::sendToRight(const void* data, size_t length) {
  TORCH_CHECK(sendFd_ >= 0, "No connection to right neighbor");
  sendAll(sendFd_, data, length);
}

void TCPRingTransport::recvFromLeft(void* data, size_t length) {
  TORCH_CHECK(recvFd_ >= 0, "No connection from left neighbor");
  recvAll(recvFd_, data, length);
}

void TCPRingTransport::sendTo(int dstRank, const void* data, size_t length) {
  auto key = "mps_p2p/" + std::to_string(rank_) + "/" +
      std::to_string(dstRank) + "/" + std::to_string(p2pSendSeq_++);
  store_->set(
      key,
      std::vector<uint8_t>(
          static_cast<const uint8_t*>(data),
          static_cast<const uint8_t*>(data) + length));
}

void TCPRingTransport::recvFrom(int srcRank, void* data, size_t length) {
  auto key = "mps_p2p/" + std::to_string(srcRank) + "/" +
      std::to_string(rank_) + "/" + std::to_string(p2pRecvSeq_++);
  auto result = store_->get(key);
  TORCH_CHECK(
      result.size() == length,
      "TCPRingTransport::recvFrom: size mismatch, expected ",
      length,
      " got ",
      result.size());
  memcpy(data, result.data(), length);
}

void TCPRingTransport::barrier() {
  auto key = "mps_barrier/" + std::to_string(barrierCount_);
  store_->add(key, 1);

  // Wait until all ranks have incremented
  auto expected = std::vector<uint8_t>(sizeof(int64_t), 0);
  int64_t target = size_;
  memcpy(expected.data(), &target, sizeof(int64_t));

  store_->wait({key});
  // Spin until the counter reaches size_
  while (true) {
    auto val = store_->get(key);
    int64_t count = 0;
    memcpy(&count, val.data(), std::min(val.size(), sizeof(int64_t)));
    if (count >= size_)
      break;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  barrierCount_++;
}

// --- WorkMPS ---

ProcessGroupMPS::WorkMPS::WorkMPS(
    OpType opType,
    const char* profilingTitle,
    const std::optional<std::vector<at::Tensor>>& inputTensors)
    : Work(-1, opType, profilingTitle, inputTensors) {
  future_ = c10::make_intrusive<at::ivalue::Future>(
      c10::ListType::create(c10::TensorType::get()));
}

bool ProcessGroupMPS::WorkMPS::isCompleted() {
  return Work::isCompleted();
}

bool ProcessGroupMPS::WorkMPS::isSuccess() const {
  return Work::isSuccess();
}

bool ProcessGroupMPS::WorkMPS::wait(std::chrono::milliseconds timeout) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (!completed_) {
    if (timeout == kNoTimeout) {
      cv_.wait(lock, [this] { return completed_; });
    } else {
      cv_.wait_for(lock, timeout, [this] { return completed_; });
    }
  }
  if (exception_) {
    std::rethrow_exception(exception_);
  }
  return true;
}

c10::intrusive_ptr<c10::ivalue::Future>
ProcessGroupMPS::WorkMPS::getFuture() {
  return future_;
}

void ProcessGroupMPS::WorkMPS::finishWork() {
  future_->markCompleted(c10::IValue(outputTensors_));
  finish();
}

void ProcessGroupMPS::WorkMPS::finishWorkError(
    const std::exception_ptr& eptr) {
  future_->setError(eptr);
  finishAndThrow(eptr);
}

// --- ProcessGroupMPS::Options ---

ProcessGroupMPS::Options::Options(std::chrono::milliseconds timeout)
    : Backend::Options(MPS_BACKEND_NAME, timeout) {}

// --- ProcessGroupMPS ---

ProcessGroupMPS::ProcessGroupMPS(
    const c10::intrusive_ptr<Store>& store,
    int rank,
    int size,
    c10::intrusive_ptr<Options> options)
    : Backend(rank, size),
      store_(store),
      options_(std::move(options)) {
#if HAVE_JACCL
  if (jaccl::isAvailable()) {
    try {
      // Auto-detect RDMA devices. On macOS each Thunderbolt port exposes its
      // own rdma_en* device; only the port with a connected peer allows
      // ibv_alloc_pd. Probe each device and pick the first usable one. An
      // explicit JACCL_DEVICE override skips the probe.
      int numDevices = 0;
      auto devices = jaccl::ibv().getDeviceList(&numDevices);
      std::string firstDevice;
      const char* deviceOverride = std::getenv("JACCL_DEVICE");
      for (int i = 0; i < numDevices; i++) {
        std::string name = jaccl::ibv().getDeviceName(devices[i]);
        if (deviceOverride && name != deviceOverride) {
          continue;
        }
        auto ctx = jaccl::ibv().openDevice(devices[i]);
        if (!ctx) {
          continue;
        }
        auto pd = jaccl::ibv().allocPd(ctx);
        if (pd) {
          jaccl::ibv().deallocPd(pd);
          jaccl::ibv().closeDevice(ctx);
          firstDevice = name;
          break;
        }
        jaccl::ibv().closeDevice(ctx);
      }
      jaccl::ibv().freeDeviceList(devices);
      if (!firstDevice.empty()) {

        // Build device name list: use first device for all peers, empty for self
        std::vector<std::string> deviceNames(size);
        for (int i = 0; i < size; i++) {
          deviceNames[i] = (i == rank) ? "" : firstDevice;
        }

        // Exchange coordinator address via store. Use MASTER_ADDR as the host
        // (rank 0's hostname may not resolve from peers, or may resolve to a
        // different interface than the one MASTER_ADDR points at). Derive the
        // port from MASTER_PORT so both ranks agree without DNS.
        std::string coordAddr;
        if (rank == 0) {
          const char* masterAddr = std::getenv("MASTER_ADDR");
          const char* masterPortEnv = std::getenv("MASTER_PORT");
          int basePort = masterPortEnv ? std::atoi(masterPortEnv) : 29500;
          std::string host = masterAddr ? masterAddr : "127.0.0.1";
          coordAddr = host + ":" + std::to_string(basePort + 1);
          store_->set(
              "jaccl_coord",
              std::vector<uint8_t>(coordAddr.begin(), coordAddr.end()));
        } else {
          auto data = store_->get("jaccl_coord");
          coordAddr = std::string(data.begin(), data.end());
        }

        jacclTransport_ = std::make_unique<jaccl::JACCLTransport>(
            rank, size, coordAddr.c_str(), deviceNames);
        useJACCL_ = true;
        TORCH_WARN("ProcessGroupMPS: using JACCL RDMA transport");
      }
    } catch (const std::exception& e) {
      TORCH_WARN(
          "JACCL RDMA initialization failed, falling back to TCP: ", e.what());
      useJACCL_ = false;
      jacclTransport_.reset();
    }
  }
#endif
  if (!useJACCL_) {
    transport_ = std::make_unique<TCPRingTransport>(store_, rank, size);
  }
  workerThread_ = std::thread(&ProcessGroupMPS::runLoop, this);
}

ProcessGroupMPS::~ProcessGroupMPS() {
  {
    std::lock_guard<std::mutex> lock(workMutex_);
    stop_ = true;
  }
  workCV_.notify_one();
  if (workerThread_.joinable()) {
    workerThread_.join();
  }
}

void ProcessGroupMPS::runLoop() {
  while (true) {
    std::function<void()> fn;
    {
      std::unique_lock<std::mutex> lock(workMutex_);
      workCV_.wait(lock, [this] { return stop_ || !workQueue_.empty(); });
      if (stop_ && workQueue_.empty())
        return;
      fn = std::move(workQueue_.front());
      workQueue_.pop_front();
    }
    fn();
  }
}

void ProcessGroupMPS::enqueue(std::function<void()> fn) {
  {
    std::lock_guard<std::mutex> lock(workMutex_);
    workQueue_.push_back(std::move(fn));
  }
  workCV_.notify_one();
}

at::Tensor ProcessGroupMPS::syncAndCopyToCPU(const at::Tensor& tensor) {
  // Flush the MPS command buffer and wait for completion
  at::mps::getDefaultMPSStream()->synchronize(
      at::mps::SyncType::COMMIT_AND_WAIT);
  return tensor.to(at::kCPU);
}

void ProcessGroupMPS::copyToMPS(
    const at::Tensor& cpuTensor,
    at::Tensor& mpsTensor) {
  mpsTensor.copy_(cpuTensor);
}

static void applyReduceOp(
    at::Tensor& accumulator,
    const at::Tensor& incoming,
    const ReduceOp& op) {
  switch (op) {
    case ReduceOp::SUM:
      accumulator.add_(incoming);
      break;
    case ReduceOp::PRODUCT:
      accumulator.mul_(incoming);
      break;
    case ReduceOp::MIN:
      at::min_out(
          const_cast<at::Tensor&>(accumulator), accumulator, incoming);
      break;
    case ReduceOp::MAX:
      at::max_out(
          const_cast<at::Tensor&>(accumulator), accumulator, incoming);
      break;
    default:
      TORCH_CHECK(false, "Unsupported reduce op: ", static_cast<int>(op));
  }
}

void ProcessGroupMPS::ringAllreduce(
    at::Tensor& data,
    const ReduceOp& op) {
  if (size_ == 1) {
    return;
  }

  int64_t numElements = data.numel();
  int64_t chunkSize = (numElements + size_ - 1) / size_;
  auto flat = data.contiguous().view(-1);

  // Temporary buffer for receiving data
  auto recvBuf = at::empty({chunkSize}, flat.options());

  // Phase 1: Reduce-scatter
  // After this phase, each rank holds the fully-reduced chunk[rank].
  for (int step = 0; step < size_ - 1; step++) {
    int sendIdx = ((rank_ - step) % size_ + size_) % size_;
    int recvIdx = ((rank_ - step - 1) % size_ + size_) % size_;

    int64_t sendStart = sendIdx * chunkSize;
    int64_t sendEnd = std::min(sendStart + chunkSize, numElements);
    int64_t recvStart = recvIdx * chunkSize;
    int64_t recvEnd = std::min(recvStart + chunkSize, numElements);

    if (sendStart >= numElements || recvStart >= numElements) {
      continue;
    }

    auto sendSlice = flat.slice(0, sendStart, sendEnd);
    auto recvSlice = flat.slice(0, recvStart, recvEnd);
    int64_t recvLen = recvEnd - recvStart;

    transport_->sendToRight(
        sendSlice.data_ptr(),
        static_cast<size_t>(sendSlice.nbytes()));
    transport_->recvFromLeft(
        recvBuf.data_ptr(),
        static_cast<size_t>(recvLen * flat.element_size()));

    applyReduceOp(recvSlice, recvBuf.slice(0, 0, recvLen), op);
  }

  // Phase 2: Allgather
  // Each rank sends its reduced chunk around the ring.
  for (int step = 0; step < size_ - 1; step++) {
    int sendIdx = ((rank_ - step + 1) % size_ + size_) % size_;
    int recvIdx = ((rank_ - step) % size_ + size_) % size_;

    int64_t sendStart = sendIdx * chunkSize;
    int64_t sendEnd = std::min(sendStart + chunkSize, numElements);
    int64_t recvStart = recvIdx * chunkSize;
    int64_t recvEnd = std::min(recvStart + chunkSize, numElements);

    if (sendStart >= numElements || recvStart >= numElements) {
      continue;
    }

    auto sendSlice = flat.slice(0, sendStart, sendEnd);
    int64_t recvLen = recvEnd - recvStart;

    transport_->sendToRight(
        sendSlice.data_ptr(),
        static_cast<size_t>(sendSlice.nbytes()));
    transport_->recvFromLeft(
        recvBuf.data_ptr(),
        static_cast<size_t>(recvLen * flat.element_size()));

    flat.slice(0, recvStart, recvEnd).copy_(recvBuf.slice(0, 0, recvLen));
  }

  // Copy back if data was not contiguous
  if (!data.is_contiguous()) {
    data.copy_(flat.view(data.sizes()));
  }
}

c10::intrusive_ptr<Work> ProcessGroupMPS::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  TORCH_CHECK(
      tensors.size() == 1,
      "ProcessGroupMPS::allreduce: only single tensor supported");

  auto& tensor = tensors[0];
  auto work = c10::make_intrusive<WorkMPS>(
      OpType::ALLREDUCE,
      "mps:allreduce",
      std::optional<std::vector<at::Tensor>>({tensor}));
  work->outputTensors_ = {tensor};

  auto fn = [this, tensor, reduceOp = opts.reduceOp, work]() mutable {
    try {
      auto cpuTensor = syncAndCopyToCPU(tensor);
#if HAVE_JACCL
      if (useJACCL_) {
        jacclTransport_->allReduce(
            cpuTensor.data_ptr(),
            static_cast<size_t>(cpuTensor.nbytes()),
            cpuTensor.element_size(),
            cpuTensor.scalar_type(),
            reduceOp);
      } else
#endif
      {
        ringAllreduce(cpuTensor, reduceOp);
      }
      copyToMPS(cpuTensor, tensor);
      work->finishWork();
    } catch (...) {
      work->finishWorkError(std::current_exception());
    }
  };

  enqueue(std::move(fn));
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupMPS::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts) {
  TORCH_CHECK(
      tensors.size() == 1,
      "ProcessGroupMPS::broadcast: only single tensor supported");

  auto& tensor = tensors[0];
  auto work = c10::make_intrusive<WorkMPS>(
      OpType::BROADCAST,
      "mps:broadcast",
      std::optional<std::vector<at::Tensor>>({tensor}));
  work->outputTensors_ = {tensor};

  auto fn = [this, tensor, rootRank = opts.rootRank, work]() mutable {
    try {
      auto cpuTensor = syncAndCopyToCPU(tensor);

#if HAVE_JACCL
      if (useJACCL_) {
        jacclTransport_->broadcast(
            cpuTensor.data_ptr(),
            static_cast<size_t>(cpuTensor.nbytes()),
            rootRank);
      } else
#endif
      {
        if (rank_ == rootRank) {
          if (size_ > 1) {
            transport_->sendToRight(
                cpuTensor.data_ptr(),
                static_cast<size_t>(cpuTensor.nbytes()));
          }
        } else {
          transport_->recvFromLeft(
              cpuTensor.data_ptr(),
              static_cast<size_t>(cpuTensor.nbytes()));
          int lastRank = (rootRank + size_ - 1) % size_;
          if (rank_ != lastRank) {
            transport_->sendToRight(
                cpuTensor.data_ptr(),
                static_cast<size_t>(cpuTensor.nbytes()));
          }
        }
      }

      copyToMPS(cpuTensor, tensor);
      work->finishWork();
    } catch (...) {
      work->finishWorkError(std::current_exception());
    }
  };

  enqueue(std::move(fn));
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupMPS::barrier(
    const BarrierOptions& /*opts*/) {
  auto work = c10::make_intrusive<WorkMPS>(OpType::BARRIER, "mps:barrier");

  auto fn = [this, work]() {
    try {
#if HAVE_JACCL
      if (useJACCL_) {
        jacclTransport_->barrier();
      } else
#endif
      {
        transport_->barrier();
      }
      work->finishWork();
    } catch (...) {
      work->finishWorkError(std::current_exception());
    }
  };

  enqueue(std::move(fn));
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupMPS::send(
    std::vector<at::Tensor>& tensors,
    int dstRank,
    int /*tag*/) {
  TORCH_CHECK(
      tensors.size() == 1,
      "ProcessGroupMPS::send: only single tensor supported");

  auto& tensor = tensors[0];
  auto work = c10::make_intrusive<WorkMPS>(OpType::SEND, "mps:send");

  auto fn = [this, tensor, dstRank, work]() mutable {
    try {
      auto cpuTensor = syncAndCopyToCPU(tensor);
      int64_t nbytes = cpuTensor.nbytes();
#if HAVE_JACCL
      if (useJACCL_) {
        jacclTransport_->send(
            cpuTensor.data_ptr(),
            static_cast<size_t>(nbytes),
            dstRank);
      } else
#endif
      {
        transport_->sendTo(dstRank, &nbytes, sizeof(nbytes));
        transport_->sendTo(
            dstRank,
            cpuTensor.data_ptr(),
            static_cast<size_t>(nbytes));
      }
      work->finishWork();
    } catch (...) {
      work->finishWorkError(std::current_exception());
    }
  };

  enqueue(std::move(fn));
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupMPS::recv(
    std::vector<at::Tensor>& tensors,
    int srcRank,
    int /*tag*/) {
  TORCH_CHECK(
      tensors.size() == 1,
      "ProcessGroupMPS::recv: only single tensor supported");

  auto& tensor = tensors[0];
  auto work = c10::make_intrusive<WorkMPS>(OpType::RECV, "mps:recv");
  work->outputTensors_ = {tensor};

  auto fn = [this, tensor, srcRank, work]() mutable {
    try {
      auto cpuTensor = syncAndCopyToCPU(tensor);
      int64_t nbytes = cpuTensor.nbytes();
#if HAVE_JACCL
      if (useJACCL_) {
        jacclTransport_->recv(
            cpuTensor.data_ptr(),
            static_cast<size_t>(nbytes),
            srcRank);
      } else
#endif
      {
        int64_t recvNbytes = 0;
        transport_->recvFrom(srcRank, &recvNbytes, sizeof(recvNbytes));
        TORCH_CHECK(
            recvNbytes == nbytes,
            "ProcessGroupMPS::recv: size mismatch, expected ",
            nbytes,
            " got ",
            recvNbytes);
        transport_->recvFrom(
            srcRank,
            cpuTensor.data_ptr(),
            static_cast<size_t>(nbytes));
      }
      copyToMPS(cpuTensor, tensor);
      work->finishWork();
    } catch (...) {
      work->finishWorkError(std::current_exception());
    }
  };

  enqueue(std::move(fn));
  return work;
}

} // namespace c10d

#endif // USE_C10D_MPS
