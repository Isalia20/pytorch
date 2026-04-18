#pragma once

#ifdef USE_C10D_MPS

#if __has_include(<infiniband/verbs.h>)
#define HAVE_JACCL 1

#include <infiniband/verbs.h>

#include <netdb.h>
#include <sys/socket.h>

#include <span>
#include <unordered_map>
#include <vector>

#include <ATen/core/ScalarType.h>
#include <c10/macros/Macros.h>
#include <torch/csrc/distributed/c10d/Types.hpp>

namespace c10d::jaccl {

constexpr const char* JACCL_TAG = "[jaccl]";
constexpr int SEND_WR = 1;
constexpr int RECV_WR = 2;
constexpr int MAX_SEND_WR = 32;
constexpr int MAX_RECV_WR = 32;
constexpr int BUFFER_SIZES = 8;
constexpr int NUM_BUFFERS = 2;
constexpr int FRAME_SIZE = 4096;
constexpr int MESH_MAX_PEERS = 8;

inline std::pair<int, int64_t> bufferSizeFromMessage(int64_t msg) {
  for (int k = BUFFER_SIZES - 1; k > 0; k--) {
    if (msg >= FRAME_SIZE * (1 << k)) {
      return {k, FRAME_SIZE * (1 << k)};
    }
  }
  return {0, FRAME_SIZE};
}

// --- TCP utilities for side channel ---

struct Address {
  sockaddr_storage addr;
  socklen_t len;
  const sockaddr* get() const {
    return reinterpret_cast<const sockaddr*>(&addr);
  }
};

Address parseAddress(const std::string& ip_port);

class TCPSocket {
 public:
  explicit TCPSocket();
  TCPSocket(TCPSocket&& s);
  TCPSocket& operator=(TCPSocket&& s);
  ~TCPSocket();

  TCPSocket(const TCPSocket&) = delete;
  TCPSocket& operator=(const TCPSocket&) = delete;

  void listen(const Address& addr);
  TCPSocket accept();
  void send(const void* data, size_t len);
  void recv(void* data, size_t len);

  static TCPSocket connect(const Address& addr, int retries = 4, int waitMs = 500);

 private:
  explicit TCPSocket(int sock);
  int sock_;
};

// --- IBV wrapper (dynamically loaded) ---

struct IBVWrapper {
  IBVWrapper();
  bool isAvailable() const {
    return handle_ != nullptr;
  }

  ibv_device** (*getDeviceList)(int*);
  const char* (*getDeviceName)(ibv_device*);
  ibv_context* (*openDevice)(ibv_device*);
  void (*freeDeviceList)(ibv_device**);
  int (*closeDevice)(ibv_context*);

  ibv_pd* (*allocPd)(ibv_context*);
  ibv_qp* (*createQp)(ibv_pd*, ibv_qp_init_attr*);
  ibv_cq* (*createCq)(ibv_context*, int, void*, ibv_comp_channel*, int);
  int (*destroyCq)(ibv_cq*);
  int (*destroyQp)(ibv_qp*);
  int (*deallocPd)(ibv_pd*);

  int (*queryPort)(ibv_context*, uint8_t, ibv_port_attr*);
  int (*queryGid)(ibv_context*, uint8_t, int, ibv_gid*);
  int (*modifyQp)(ibv_qp*, ibv_qp_attr*, int);
  ibv_mr* (*regMr)(ibv_pd*, void*, size_t, int);
  int (*deregMr)(ibv_mr*);

 private:
  void* handle_;
};

IBVWrapper& ibv();
bool isAvailable();

// --- RDMA primitives ---

struct Destination {
  int localId;
  int queuePairNumber;
  int packetSequenceNumber;
  ibv_gid globalIdentifier;
};

class SharedBuffer {
 public:
  explicit SharedBuffer(size_t numBytes);
  SharedBuffer(SharedBuffer&& b);
  ~SharedBuffer();

  SharedBuffer(const SharedBuffer&) = delete;
  SharedBuffer& operator=(const SharedBuffer&) = delete;

  void registerToProtectionDomain(ibv_pd* pd);

  size_t size() const {
    return numBytes_;
  }
  uint32_t localKey(ibv_pd* pd) const {
    return memoryRegions_.at(pd)->lkey;
  }
  ibv_sge toScatterGatherEntry(ibv_pd* pd) const;

  template <typename T>
  T* data() {
    return static_cast<T*>(data_);
  }
  template <typename T>
  T* begin() {
    return static_cast<T*>(data_);
  }
  template <typename T>
  T* end() {
    return static_cast<T*>(data_) + numBytes_ / sizeof(T);
  }

 private:
  void* data_;
  size_t numBytes_;
  std::unordered_map<ibv_pd*, ibv_mr*> memoryRegions_;
};

struct Connection {
  ibv_context* ctx;
  ibv_pd* protectionDomain;
  ibv_cq* completionQueue;
  ibv_qp* queuePair;
  Destination src;
  // Index into our port's GID table that holds the GID we advertise and use
  // as source. Apple's rdma driver populates the table with one entry per IP
  // the netdev actually owns plus one EUI-64 entry the OS does NOT bind —
  // picking the wrong one makes IPv6 ND hang. Resolved in info() by matching
  // GID entries against getifaddrs() output for the underlying interface.
  uint8_t sgidIndex = 0;
  // Cached port active_mtu from queryPort. Used as path_mtu for RTR — setting
  // path_mtu > active_mtu is illegal and gets the driver to reject the RTR
  // transition. 0 means "not queried yet".
  ibv_mtu activeMtu = static_cast<ibv_mtu>(0);
  // Underlying network interface name (e.g. "en2"), derived from the RDMA
  // device name by stripping the "rdma_" prefix. Needed in info() to look up
  // which IPs the OS owns on this port so we pick a bound GID.
  std::string ifaceName;

  explicit Connection(ibv_context* ctx, std::string ifaceName = {});
  Connection(Connection&& c);
  ~Connection();

  Connection(const Connection&) = delete;
  Connection& operator=(const Connection&) = delete;

  void allocateProtectionDomain();
  void createCompletionQueue(int numEntries);
  void createQueuePair();

  const Destination& info();
  void queuePairInit();
  void queuePairRtr(const Destination& dst);
  void queuePairRts();

  void postSend(const SharedBuffer& buff, uint64_t wrId);
  void postRecv(const SharedBuffer& buff, uint64_t wrId);
  int poll(int numCompletions, ibv_wc* wc);
};

std::vector<Connection> createConnections(
    const std::vector<std::string>& deviceNames);

inline int pollConnections(
    std::span<Connection> connections,
    int numCompletions,
    ibv_wc* wc) {
  int completions = 0;
  for (auto& c : connections) {
    // Skip the self-slot (no RDMA context) rather than breaking — otherwise
    // ranks whose self-index is not the last would never poll later peers.
    if (c.ctx == nullptr) {
      continue;
    }
    if (completions >= numCompletions) {
      return completions;
    }
    int n = ibv_poll_cq(
        c.completionQueue, numCompletions - completions, wc + completions);
    completions += n;
  }
  return completions;
}

inline int pollConnections(
    std::span<Connection> c1,
    std::span<Connection> c2,
    int numCompletions,
    ibv_wc* wc) {
  int completions = pollConnections(c1, numCompletions, wc);
  completions +=
      pollConnections(c2, numCompletions - completions, wc + completions);
  return completions;
}

// --- Side channel for QP info exchange ---

class SideChannel {
 public:
  SideChannel(int rank, int size, const char* addr);
  SideChannel(SideChannel&& sc);

  SideChannel(const SideChannel&) = delete;
  SideChannel& operator=(const SideChannel&) = delete;

  template <typename T>
  std::vector<T> allGather(const T& v);

 private:
  int rank_;
  int size_;
  std::vector<TCPSocket> sockets_;
};

// --- Mesh allreduce implementation ---

class MeshImpl {
 public:
  MeshImpl(
      int rank,
      int size,
      std::vector<Connection>& conns,
      std::vector<SharedBuffer>& buffers)
      : rank_(rank),
        size_(size),
        connections_(conns),
        buffers_(buffers) {}

  MeshImpl() : rank_(0), size_(1) {}

  // Pipelined fully-connected mesh allreduce
  template <typename T, typename ReduceOp>
  void allReduce(const T* inPtr, T* outPtr, int64_t size, ReduceOp reduceOp);

  void allGather(const char* inPtr, char* outPtr, int64_t nBytes);
  void send(const char* inPtr, int64_t nBytes, int dst);
  void recv(char* outPtr, int64_t nBytes, int src);

 private:
  void sendTo(int sz, int rank, int buff);
  void recvFrom(int sz, int rank, int buff);
  SharedBuffer& sendBuffer(int sz, int buff);
  SharedBuffer& recvBuffer(int sz, int buff, int rank);
  void postSendAll(int sz, int buff);
  void postRecvAll(int sz, int buff);

  int rank_;
  int size_;
  std::span<Connection> connections_;
  std::span<SharedBuffer> buffers_;
};

// --- High-level JACCL transport for ProcessGroupMPS ---

class JACCLTransport {
 public:
  JACCLTransport(int rank, int size, const char* coordinatorAddr,
                 const std::vector<std::string>& deviceNames);
  ~JACCLTransport();

  void allReduce(void* data, size_t numBytes, size_t elementSize,
                 at::ScalarType dtype, ReduceOp op);
  void broadcast(void* data, size_t numBytes, int rootRank);
  void send(const void* data, size_t numBytes, int dstRank);
  void recv(void* data, size_t numBytes, int srcRank);
  void barrier();

 private:
  int rank_;
  int size_;
  SideChannel sideChannel_;
  std::vector<Connection> connections_;
  std::vector<SharedBuffer> buffers_;
  MeshImpl mesh_;
};

} // namespace c10d::jaccl

// --- Template implementations ---

namespace c10d::jaccl {

template <typename T>
std::vector<T> SideChannel::allGather(const T& v) {
  std::vector<T> result(size_);
  if (rank_ == 0) {
    result[rank_] = v;
    for (int i = 1; i < size_; i++) {
      sockets_[i - 1].recv(&result[i], sizeof(T));
    }
    for (int i = 1; i < size_; i++) {
      sockets_[i - 1].send(result.data(), size_ * sizeof(T));
    }
  } else {
    sockets_[0].send(&v, sizeof(T));
    sockets_[0].recv(result.data(), size_ * sizeof(T));
  }
  return result;
}

template <typename T, typename ReduceOp>
void MeshImpl::allReduce(
    const T* inPtr,
    T* outPtr,
    int64_t size,
    ReduceOp reduceOp) {
  if (inPtr != outPtr) {
    std::memcpy(outPtr, inPtr, size * sizeof(T));
  }

  T* data = outPtr;
  auto [sz, bufferSize] = bufferSizeFromMessage(size * sizeof(T));
  int64_t N = bufferSize / sizeof(T);
  constexpr int PIPELINE = 2;
  constexpr int WC_NUM = PIPELINE * MESH_MAX_PEERS * 2;
  int64_t total = size;
  int numPeers = size_ - 1;

  int inFlight = 0;
  int64_t readOffset = 0;
  int completedSendCount[PIPELINE] = {0};
  int completedRecvBegin[MESH_MAX_PEERS] = {0};
  int completedRecvEnd[MESH_MAX_PEERS] = {0};

  // Prefill the pipeline
  int buff = 0;
  while (readOffset < total && buff < PIPELINE) {
    postRecvAll(sz, buff);
    std::copy(
        data + readOffset,
        data + std::min(readOffset + N, total),
        sendBuffer(sz, buff).begin<T>());
    postSendAll(sz, buff);
    buff++;
    inFlight += 2 * numPeers;
    readOffset += N;
  }

  while (inFlight > 0) {
    ibv_wc wc[WC_NUM];
    int n = pollConnections(connections_, WC_NUM, wc);
    for (int i = 0; i < n; i++) {
      int workType = wc[i].wr_id >> 16;
      int b = (wc[i].wr_id >> 8) & 0xff;
      int rank = wc[i].wr_id & 0xff;
      inFlight--;

      if (workType == SEND_WR && readOffset < total) {
        completedSendCount[b]++;
        if (completedSendCount[b] == numPeers) {
          std::copy(
              data + readOffset,
              data + std::min(readOffset + N, total),
              sendBuffer(sz, b).begin<T>());
          postSendAll(sz, b);
          completedSendCount[b] = 0;
          inFlight += numPeers;
          readOffset += N;
        }
      } else if (workType == RECV_WR) {
        completedRecvEnd[rank]++;
      }
    }

    for (int r = 0; r < size_; r++) {
      int s = completedRecvBegin[r];
      int e = completedRecvEnd[r];
      int64_t w = static_cast<int64_t>(s) * N;
      while (w < readOffset && e - s > 0) {
        int b = s % PIPELINE;
        reduceOp(
            recvBuffer(sz, b, r).template begin<T>(),
            data + w,
            std::min(N, total - w));
        w += N;
        s++;
        if (w + (PIPELINE - 1) * N < total) {
          recvFrom(sz, r, b);
          inFlight++;
        }
      }
      completedRecvBegin[r] = s;
    }
  }
}

} // namespace c10d::jaccl

#else // !__has_include(<infiniband/verbs.h>)
#define HAVE_JACCL 0

namespace c10d::jaccl {
inline bool isAvailable() {
  return false;
}
// Stub so unique_ptr<JACCLTransport> compiles with a complete type
class JACCLTransport {
 public:
  ~JACCLTransport() = default;
};
} // namespace c10d::jaccl

#endif // __has_include(<infiniband/verbs.h>)

#endif // USE_C10D_MPS
