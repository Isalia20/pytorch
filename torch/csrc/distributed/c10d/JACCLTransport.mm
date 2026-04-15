#ifdef USE_C10D_MPS

#include <torch/csrc/distributed/c10d/JACCLTransport.h>

#if HAVE_JACCL

#include <torch/csrc/distributed/c10d/Types.hpp>

#include <array>
#include <dlfcn.h>
#include <iostream>
#include <netdb.h>
#include <unistd.h>

#include <cstring>
#include <iostream>
#include <sstream>
#include <thread>

#define LOAD_IBV_SYMBOL(symbol, variable)                          \
  {                                                                \
    variable = (decltype(variable))dlsym(handle_, #symbol);        \
    char* error = dlerror();                                       \
    if (error != nullptr) {                                        \
      std::cerr << JACCL_TAG << " " << error << std::endl;         \
      handle_ = nullptr;                                           \
      return;                                                      \
    }                                                              \
  }

namespace {

void* pageAlignedAlloc(size_t numBytes) {
  static size_t pageSize = sysconf(_SC_PAGESIZE);
  void* buf;
  if (posix_memalign(&buf, pageSize, numBytes)) {
    return nullptr;
  }
  return buf;
}

} // namespace

namespace c10d::jaccl {

// --- TCP utilities ---

Address parseAddress(const std::string& ipPort) {
  auto colon = ipPort.find(":");
  TORCH_CHECK(
      colon != std::string::npos,
      JACCL_TAG, " Can't parse address ", ipPort);

  std::string ip(ipPort.begin(), ipPort.begin() + colon);
  std::string port(ipPort.begin() + colon + 1, ipPort.end());

  struct addrinfo hints, *res;
  std::memset(&hints, 0, sizeof(hints));
  // TCPSocket and the rest of this file use AF_INET — keep resolution consistent
  // so bind() doesn't get a sockaddr_in6 from mDNS on macOS.
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;

  int status = getaddrinfo(ip.c_str(), port.c_str(), &hints, &res);
  TORCH_CHECK(
      status == 0,
      JACCL_TAG, " Can't resolve address ", ipPort);

  Address result;
  memcpy(&result.addr, res->ai_addr, res->ai_addrlen);
  result.len = res->ai_addrlen;
  freeaddrinfo(res);
  return result;
}

TCPSocket::TCPSocket() {
  sock_ = socket(AF_INET, SOCK_STREAM, 0);
  TORCH_CHECK(sock_ >= 0, JACCL_TAG, " Couldn't create socket");
}

TCPSocket::TCPSocket(int sock) : sock_(sock) {}

TCPSocket::TCPSocket(TCPSocket&& s) : sock_(s.sock_) {
  s.sock_ = -1;
}

TCPSocket& TCPSocket::operator=(TCPSocket&& s) {
  if (this != &s) {
    if (sock_ >= 0) {
      shutdown(sock_, 2);
      close(sock_);
    }
    sock_ = s.sock_;
    s.sock_ = -1;
  }
  return *this;
}

TCPSocket::~TCPSocket() {
  if (sock_ >= 0) {
    shutdown(sock_, 2);
    close(sock_);
  }
}

void TCPSocket::listen(const Address& addr) {
  int enable = 1;
  setsockopt(sock_, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(int));
  setsockopt(sock_, SOL_SOCKET, SO_REUSEPORT, &enable, sizeof(int));

  TORCH_CHECK(
      bind(sock_, addr.get(), addr.len) == 0,
      JACCL_TAG, " Couldn't bind socket: ", strerror(errno));
  TORCH_CHECK(
      ::listen(sock_, 16) == 0,
      JACCL_TAG, " Couldn't listen: ", strerror(errno));
}

TCPSocket TCPSocket::accept() {
  int peer = ::accept(sock_, nullptr, nullptr);
  TORCH_CHECK(peer >= 0, JACCL_TAG, " Accept failed: ", strerror(errno));
  return TCPSocket(peer);
}

void TCPSocket::send(const void* data, size_t len) {
  auto ptr = static_cast<const char*>(data);
  while (len > 0) {
    auto n = ::send(sock_, ptr, len, 0);
    TORCH_CHECK(n > 0, JACCL_TAG, " Send failed: ", strerror(errno));
    len -= n;
    ptr += n;
  }
}

void TCPSocket::recv(void* data, size_t len) {
  auto ptr = static_cast<char*>(data);
  while (len > 0) {
    auto n = ::recv(sock_, ptr, len, 0);
    TORCH_CHECK(n > 0, JACCL_TAG, " Recv failed: ", strerror(errno));
    len -= n;
    ptr += n;
  }
}

TCPSocket TCPSocket::connect(
    const Address& addr,
    int retries,
    int waitMs) {
  int sock = -1;
  int success = -1;
  int wait = waitMs;

  for (int attempt = 0; attempt < retries; attempt++) {
    sock = socket(AF_INET, SOCK_STREAM, 0);
    TORCH_CHECK(sock >= 0, JACCL_TAG, " Couldn't create socket");

    success = ::connect(sock, addr.get(), addr.len);
    if (success == 0)
      break;

    close(sock);
    sock = -1;
    if (wait > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(wait));
    }
    wait <<= 1;
  }

  TORCH_CHECK(
      success == 0,
      JACCL_TAG, " Couldn't connect after ", retries, " retries: ",
      strerror(errno));
  return TCPSocket(sock);
}

// --- IBVWrapper ---

IBVWrapper::IBVWrapper() {
  handle_ = dlopen("librdma.dylib", RTLD_NOW | RTLD_GLOBAL);
  if (handle_ == nullptr)
    return;

  LOAD_IBV_SYMBOL(ibv_get_device_list, getDeviceList);
  LOAD_IBV_SYMBOL(ibv_get_device_name, getDeviceName);
  LOAD_IBV_SYMBOL(ibv_open_device, openDevice);
  LOAD_IBV_SYMBOL(ibv_free_device_list, freeDeviceList);
  LOAD_IBV_SYMBOL(ibv_close_device, closeDevice);

  LOAD_IBV_SYMBOL(ibv_alloc_pd, allocPd);
  LOAD_IBV_SYMBOL(ibv_create_qp, createQp);
  LOAD_IBV_SYMBOL(ibv_create_cq, createCq);
  LOAD_IBV_SYMBOL(ibv_destroy_cq, destroyCq);
  LOAD_IBV_SYMBOL(ibv_destroy_qp, destroyQp);
  LOAD_IBV_SYMBOL(ibv_dealloc_pd, deallocPd);

  LOAD_IBV_SYMBOL(ibv_query_port, queryPort);
  LOAD_IBV_SYMBOL(ibv_query_gid, queryGid);
  LOAD_IBV_SYMBOL(ibv_modify_qp, modifyQp);
  LOAD_IBV_SYMBOL(ibv_reg_mr, regMr);
  LOAD_IBV_SYMBOL(ibv_dereg_mr, deregMr);
}

IBVWrapper& ibv() {
  static IBVWrapper wrapper;
  return wrapper;
}

bool isAvailable() {
  return ibv().isAvailable();
}

// --- SharedBuffer ---

SharedBuffer::SharedBuffer(size_t numBytes)
    : data_(pageAlignedAlloc(numBytes)), numBytes_(numBytes) {}

SharedBuffer::SharedBuffer(SharedBuffer&& b) : data_(nullptr), numBytes_(0) {
  std::swap(data_, b.data_);
  std::swap(numBytes_, b.numBytes_);
  std::swap(memoryRegions_, b.memoryRegions_);
}

SharedBuffer::~SharedBuffer() {
  for (auto& [pd, mr] : memoryRegions_) {
    ibv().deregMr(mr);
  }
  if (data_ != nullptr) {
    std::free(data_);
  }
}

void SharedBuffer::registerToProtectionDomain(ibv_pd* pd) {
  auto [it, inserted] = memoryRegions_.insert({pd, nullptr});
  TORCH_CHECK(inserted, JACCL_TAG, " Buffer already registered to this PD");

  it->second = ibv().regMr(
      pd, data_, numBytes_,
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
          IBV_ACCESS_REMOTE_WRITE);
  TORCH_CHECK(it->second, JACCL_TAG, " Register memory region failed");
}

ibv_sge SharedBuffer::toScatterGatherEntry(ibv_pd* pd) const {
  ibv_sge entry;
  entry.addr = reinterpret_cast<uintptr_t>(data_);
  entry.length = size();
  entry.lkey = localKey(pd);
  return entry;
}

// --- Connection ---

Connection::Connection(ibv_context* ctx_)
    : ctx(ctx_),
      protectionDomain(nullptr),
      completionQueue(nullptr),
      queuePair(nullptr) {
  src.localId = -1;
}

Connection::Connection(Connection&& c) : Connection(nullptr) {
  std::swap(ctx, c.ctx);
  std::swap(protectionDomain, c.protectionDomain);
  std::swap(completionQueue, c.completionQueue);
  std::swap(queuePair, c.queuePair);
  std::swap(src, c.src);
}

Connection::~Connection() {
  if (queuePair)
    ibv().destroyQp(queuePair);
  if (completionQueue)
    ibv().destroyCq(completionQueue);
  if (protectionDomain)
    ibv().deallocPd(protectionDomain);
  if (ctx)
    ibv().closeDevice(ctx);
}

void Connection::allocateProtectionDomain() {
  protectionDomain = ibv().allocPd(ctx);
  TORCH_CHECK(protectionDomain, JACCL_TAG, " Couldn't allocate PD");
}

void Connection::createCompletionQueue(int numEntries) {
  completionQueue = ibv().createCq(ctx, numEntries, nullptr, nullptr, 0);
  TORCH_CHECK(completionQueue, JACCL_TAG, " Couldn't create CQ");
}

void Connection::createQueuePair() {
  ibv_qp_init_attr initAttr;
  initAttr.qp_context = ctx;
  initAttr.send_cq = completionQueue;
  initAttr.recv_cq = completionQueue;
  initAttr.srq = nullptr;
  initAttr.cap.max_send_wr = MAX_SEND_WR;
  initAttr.cap.max_recv_wr = MAX_RECV_WR;
  initAttr.cap.max_send_sge = 1;
  initAttr.cap.max_recv_sge = 1;
  initAttr.cap.max_inline_data = 0;
  initAttr.qp_type = IBV_QPT_UC;
  initAttr.sq_sig_all = 0;

  queuePair = ibv().createQp(protectionDomain, &initAttr);
  TORCH_CHECK(queuePair, JACCL_TAG, " Couldn't create QP");
}

const Destination& Connection::info() {
  if (queuePair == nullptr || src.localId >= 0)
    return src;

  ibv_port_attr portAttr;
  ibv().queryPort(ctx, 1, &portAttr);
  ibv_gid gid;
  ibv().queryGid(ctx, 1, 1, &gid);

  src.localId = portAttr.lid;
  src.queuePairNumber = queuePair->qp_num;
  src.packetSequenceNumber = 7;
  src.globalIdentifier = gid;
  return src;
}

void Connection::queuePairInit() {
  ibv_qp_attr attr = {};
  attr.qp_state = IBV_QPS_INIT;
  attr.port_num = 1;
  attr.pkey_index = 0;
  attr.qp_access_flags =
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;

  int mask =
      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
  TORCH_CHECK(
      ibv().modifyQp(queuePair, &attr, mask) == 0,
      JACCL_TAG, " QP transition to INIT failed");
}

void Connection::queuePairRtr(const Destination& dst) {
  ibv_qp_attr attr = {};
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTR;
  attr.path_mtu = IBV_MTU_1024;
  attr.rq_psn = dst.packetSequenceNumber;
  attr.dest_qp_num = dst.queuePairNumber;
  attr.ah_attr.dlid = dst.localId;
  attr.ah_attr.sl = 0;
  attr.ah_attr.src_path_bits = 0;
  attr.ah_attr.port_num = 1;
  attr.ah_attr.is_global = 0;

  if (dst.globalIdentifier.global.interface_id) {
    attr.ah_attr.is_global = 1;
    attr.ah_attr.grh.hop_limit = 1;
    attr.ah_attr.grh.dgid = dst.globalIdentifier;
    attr.ah_attr.grh.sgid_index = 1;
  }

  int mask = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
      IBV_QP_RQ_PSN;
  TORCH_CHECK(
      ibv().modifyQp(queuePair, &attr, mask) == 0,
      JACCL_TAG, " QP transition to RTR failed");
}

void Connection::queuePairRts() {
  ibv_qp_attr attr = {};
  attr.qp_state = IBV_QPS_RTS;
  attr.sq_psn = src.packetSequenceNumber;

  int mask = IBV_QP_STATE | IBV_QP_SQ_PSN;
  TORCH_CHECK(
      ibv().modifyQp(queuePair, &attr, mask) == 0,
      JACCL_TAG, " QP transition to RTS failed");
}

void Connection::postSend(const SharedBuffer& buff, uint64_t wrId) {
  ibv_send_wr wr, *badWr;
  auto entry = buff.toScatterGatherEntry(protectionDomain);
  wr.wr_id = wrId;
  wr.sg_list = &entry;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_SEND;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.next = nullptr;
  TORCH_CHECK(
      ibv_post_send(queuePair, &wr, &badWr) == 0,
      JACCL_TAG, " post_send failed");
}

void Connection::postRecv(const SharedBuffer& buff, uint64_t wrId) {
  ibv_recv_wr wr, *badWr;
  auto entry = buff.toScatterGatherEntry(protectionDomain);
  wr.wr_id = wrId;
  wr.sg_list = &entry;
  wr.num_sge = 1;
  wr.next = nullptr;
  TORCH_CHECK(
      ibv_post_recv(queuePair, &wr, &badWr) == 0,
      JACCL_TAG, " post_recv failed");
}

int Connection::poll(int numCompletions, ibv_wc* wc) {
  return ibv_poll_cq(completionQueue, numCompletions, wc);
}

std::vector<Connection> createConnections(
    const std::vector<std::string>& deviceNames) {
  std::vector<Connection> connections;
  int numDevices = 0;
  ibv_device** devices = ibv().getDeviceList(&numDevices);
  for (auto& name : deviceNames) {
    if (name.empty()) {
      connections.emplace_back(nullptr);
      continue;
    }
    for (int i = 0; i < numDevices; i++) {
      if (name == ibv().getDeviceName(devices[i])) {
        auto ctx = ibv().openDevice(devices[i]);
        TORCH_CHECK(ctx, JACCL_TAG, " Could not open device ", name);
        connections.emplace_back(ctx);
        break;
      }
    }
  }
  ibv().freeDeviceList(devices);
  return connections;
}

// --- SideChannel ---

SideChannel::SideChannel(int rank, int size, const char* addr)
    : rank_(rank), size_(size) {
  auto address = parseAddress(addr);

  if (rank_ == 0) {
    TCPSocket server;
    server.listen(address);
    for (int i = 0; i < size - 1; i++) {
      sockets_.push_back(server.accept());
    }
    // Sort sockets by rank
    std::vector<int> ranks(size - 1);
    for (int i = 0; i < size - 1; i++) {
      sockets_[i].recv(&ranks[i], sizeof(int));
      ranks[i]--;
    }
    for (int i = 0; i < size - 1; i++) {
      while (i != ranks[i]) {
        std::swap(sockets_[i], sockets_[ranks[i]]);
        std::swap(ranks[i], ranks[ranks[i]]);
      }
    }
  } else {
    sockets_.push_back(TCPSocket::connect(address));
    sockets_[0].send(&rank_, sizeof(int));
  }
}

SideChannel::SideChannel(SideChannel&& sc)
    : rank_(sc.rank_), size_(sc.size_), sockets_(std::move(sc.sockets_)) {}

// --- MeshImpl non-template methods ---

void MeshImpl::sendTo(int sz, int rank, int buff) {
  connections_[rank].postSend(
      sendBuffer(sz, buff), SEND_WR << 16 | buff << 8 | rank);
}

void MeshImpl::recvFrom(int sz, int rank, int buff) {
  connections_[rank].postRecv(
      recvBuffer(sz, buff, rank), RECV_WR << 16 | buff << 8 | rank);
}

SharedBuffer& MeshImpl::sendBuffer(int sz, int buff) {
  return buffers_[sz * NUM_BUFFERS * size_ + buff * size_ + rank_];
}

SharedBuffer& MeshImpl::recvBuffer(int sz, int buff, int rank) {
  return buffers_[sz * NUM_BUFFERS * size_ + buff * size_ + rank];
}

void MeshImpl::postSendAll(int sz, int buff) {
  auto& b = sendBuffer(sz, buff);
  int wrId = SEND_WR << 16 | buff << 8;
  for (int i = 0; i < size_; i++) {
    if (i == rank_)
      continue;
    connections_[i].postSend(b, wrId | i);
  }
}

void MeshImpl::postRecvAll(int sz, int buff) {
  int b = sz * NUM_BUFFERS * size_ + buff * size_;
  int wrId = RECV_WR << 16 | buff << 8;
  for (int i = 0; i < size_; i++) {
    if (i == rank_)
      continue;
    connections_[i].postRecv(buffers_[b + i], wrId | i);
  }
}

void MeshImpl::allGather(const char* inPtr, char* outPtr, int64_t nBytes) {
  std::memcpy(outPtr + rank_ * nBytes, inPtr, nBytes);

  auto [sz, N] = bufferSizeFromMessage(nBytes);
  constexpr int PIPELINE = 2;
  constexpr int WC_NUM = PIPELINE * MESH_MAX_PEERS * 2;
  int64_t total = nBytes;
  int numPeers = size_ - 1;

  int inFlight = 0;
  int readOffset = 0;
  int completedSendCount[PIPELINE] = {0};
  int writeOffset[MESH_MAX_PEERS] = {0};

  int buff = 0;
  while (readOffset < total && buff < PIPELINE) {
    postRecvAll(sz, buff);
    char* ourData = outPtr + rank_ * nBytes;
    std::copy(
        ourData + readOffset,
        ourData + std::min(static_cast<int64_t>(readOffset + N), total),
        sendBuffer(sz, buff).begin<char>());
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
          char* ourData = outPtr + rank_ * nBytes;
          std::copy(
              ourData + readOffset,
              ourData + std::min(static_cast<int64_t>(readOffset + N), total),
              sendBuffer(sz, b).begin<char>());
          postSendAll(sz, b);
          completedSendCount[b] = 0;
          inFlight += numPeers;
          readOffset += N;
        }
      } else if (workType == RECV_WR) {
        std::copy(
            recvBuffer(sz, b, rank).begin<char>(),
            recvBuffer(sz, b, rank).begin<char>() +
                std::min(static_cast<int64_t>(N), total - writeOffset[rank]),
            outPtr + rank * nBytes + writeOffset[rank]);
        writeOffset[rank] += N;
        if (writeOffset[rank] + N * (PIPELINE - 1) < total) {
          recvFrom(sz, rank, b);
          inFlight++;
        }
      }
    }
  }
}

void MeshImpl::send(const char* inPtr, int64_t nBytes, int dst) {
  constexpr int PIPELINE = 2;
  auto [sz, N] = bufferSizeFromMessage(nBytes);

  int inFlight = 0;
  int64_t readOffset = 0;

  int buff = 0;
  while (readOffset < nBytes && buff < PIPELINE) {
    std::copy(
        inPtr + readOffset,
        inPtr + std::min(readOffset + static_cast<int64_t>(N), nBytes),
        sendBuffer(sz, buff).begin<char>());
    sendTo(sz, dst, buff);
    buff++;
    readOffset += N;
    inFlight++;
  }

  while (inFlight > 0) {
    ibv_wc wc[PIPELINE];
    int n = connections_[dst].poll(PIPELINE, wc);
    for (int i = 0; i < n; i++) {
      inFlight--;
      int b = (wc[i].wr_id >> 8) & 0xff;
      if (readOffset < nBytes) {
        std::copy(
            inPtr + readOffset,
            inPtr + std::min(readOffset + static_cast<int64_t>(N), nBytes),
            sendBuffer(sz, b).begin<char>());
        sendTo(sz, dst, b);
        readOffset += N;
        inFlight++;
      }
    }
  }
}

void MeshImpl::recv(char* outPtr, int64_t nBytes, int src) {
  constexpr int PIPELINE = 2;
  auto [sz, N] = bufferSizeFromMessage(nBytes);

  int inFlight = 0;
  int64_t writeOffset = 0;

  int buff = 0;
  while (static_cast<int64_t>(N) * buff < nBytes && buff < PIPELINE) {
    recvFrom(sz, src, buff);
    inFlight++;
    buff++;
  }

  while (inFlight > 0) {
    ibv_wc wc[PIPELINE];
    int n = connections_[src].poll(PIPELINE, wc);
    for (int i = 0; i < n; i++) {
      int b = (wc[i].wr_id >> 8) & 0xff;
      inFlight--;
      std::copy(
          recvBuffer(sz, b, src).begin<char>(),
          recvBuffer(sz, b, src).begin<char>() +
              std::min(nBytes - writeOffset, static_cast<int64_t>(N)),
          outPtr + writeOffset);
      writeOffset += N;
      if (writeOffset + (PIPELINE - 1) * N < nBytes) {
        recvFrom(sz, src, b);
        inFlight++;
      }
    }
  }
}

// --- JACCLTransport ---

JACCLTransport::JACCLTransport(
    int rank,
    int size,
    const char* coordinatorAddr,
    const std::vector<std::string>& deviceNames)
    : rank_(rank),
      size_(size),
      sideChannel_(rank, size, coordinatorAddr),
      connections_(createConnections(deviceNames)) {
  std::cerr << "[jaccl-ctor] rank=" << rank_ << " size=" << size_
            << " sideChannel+connections done" << std::endl;
  TORCH_CHECK(
      size_ <= MESH_MAX_PEERS,
      JACCL_TAG, " Mesh supports up to ", MESH_MAX_PEERS, " peers");

  // Initialize all connections
  for (auto& conn : connections_) {
    if (conn.ctx == nullptr)
      continue;
    std::cerr << "[jaccl-ctor] allocPd/CQ/QP on ctx=" << (void*)conn.ctx << std::endl;
    conn.allocateProtectionDomain();
    conn.createCompletionQueue(MAX_SEND_WR + MAX_RECV_WR);
    conn.createQueuePair();
  }
  std::cerr << "[jaccl-ctor] PD/CQ/QP done for all peers" << std::endl;

  // Allocate buffers
  for (int k = 0; k < BUFFER_SIZES; k++) {
    for (int i = 0; i < NUM_BUFFERS; i++) {
      for (int j = 0; j < size_; j++) {
        buffers_.emplace_back(FRAME_SIZE * (1 << k));
      }
    }
  }

  // Register buffers to protection domains
  for (int k = 0; k < BUFFER_SIZES; k++) {
    for (int i = 0; i < NUM_BUFFERS; i++) {
      for (int j = 0; j < size_; j++) {
        if (j == rank_) {
          // Send buffer: register to all peers
          for (auto& conn : connections_) {
            if (conn.ctx != nullptr) {
              buffers_[k * NUM_BUFFERS * size_ + i * size_ + j]
                  .registerToProtectionDomain(conn.protectionDomain);
            }
          }
        } else {
          // Recv buffer: register to sender's PD
          buffers_[k * NUM_BUFFERS * size_ + i * size_ + j]
              .registerToProtectionDomain(connections_[j].protectionDomain);
        }
      }
    }
  }

  // Initialize queue pairs
  for (int peer = 0; peer < size_; peer++) {
    if (peer == rank_)
      continue;
    std::cerr << "[jaccl-ctor] queuePairInit peer=" << peer << std::endl;
    connections_[peer].queuePairInit();
  }

  // Exchange connection info via side channel. Use std::array (trivially
  // copyable) rather than std::vector — SideChannel::allGather<T> does a raw
  // memcpy of sizeof(T) bytes per rank, which corrupts a std::vector's
  // pointer/size/capacity control block.
  std::array<Destination, MESH_MAX_PEERS> info{};
  for (int i = 0; i < size_; i++) {
    info[i] = connections_[i].info();
  }
  std::cerr << "[jaccl-ctor] sideChannel allGather info..." << std::endl;
  auto allInfos = sideChannel_.allGather(info);
  std::cerr << "[jaccl-ctor] allGather returned" << std::endl;

  // Transition to RTS
  for (int peer = 0; peer < size_; peer++) {
    if (peer == rank_)
      continue;
    std::cerr << "[jaccl-ctor] RTR/RTS peer=" << peer << std::endl;
    auto peerInfo = allInfos[peer][rank_];
    connections_[peer].queuePairRtr(peerInfo);
    connections_[peer].queuePairRts();
  }

  // Barrier to ensure everyone is ready
  std::cerr << "[jaccl-ctor] barrier allGather..." << std::endl;
  sideChannel_.allGather<int>(0);
  std::cerr << "[jaccl-ctor] barrier done, constructing MeshImpl" << std::endl;

  mesh_ = MeshImpl(rank_, size_, connections_, buffers_);
  std::cerr << "[jaccl-ctor] done" << std::endl;
}

JACCLTransport::~JACCLTransport() = default;

// Reduction op templates
template <typename T>
struct SumOp {
  void operator()(const T* input, T* output, int64_t N) const {
    for (int64_t i = 0; i < N; i++)
      output[i] += input[i];
  }
};

template <typename T>
struct MaxOp {
  void operator()(const T* input, T* output, int64_t N) const {
    for (int64_t i = 0; i < N; i++)
      output[i] = std::max(output[i], input[i]);
  }
};

template <typename T>
struct MinOp {
  void operator()(const T* input, T* output, int64_t N) const {
    for (int64_t i = 0; i < N; i++)
      output[i] = std::min(output[i], input[i]);
  }
};

template <typename T>
struct ProdOp {
  void operator()(const T* input, T* output, int64_t N) const {
    for (int64_t i = 0; i < N; i++)
      output[i] *= input[i];
  }
};

void JACCLTransport::allReduce(
    void* data,
    size_t numBytes,
    size_t elementSize,
    at::ScalarType dtype,
    c10d::ReduceOp op) {
  switch (op) {
    case ReduceOp::SUM:
      switch (dtype) {
        case at::kFloat: mesh_.allReduce(static_cast<const float*>(data), static_cast<float*>(data), numBytes / sizeof(float), SumOp<float>{}); break;
        case at::kDouble: mesh_.allReduce(static_cast<const double*>(data), static_cast<double*>(data), numBytes / sizeof(double), SumOp<double>{}); break;
        case at::kInt: mesh_.allReduce(static_cast<const int32_t*>(data), static_cast<int32_t*>(data), numBytes / sizeof(int32_t), SumOp<int32_t>{}); break;
        case at::kLong: mesh_.allReduce(static_cast<const int64_t*>(data), static_cast<int64_t*>(data), numBytes / sizeof(int64_t), SumOp<int64_t>{}); break;
        default: TORCH_CHECK(false, JACCL_TAG, " Unsupported dtype"); break;
      }
      break;
    case ReduceOp::PRODUCT:
      switch (dtype) {
        case at::kFloat: mesh_.allReduce(static_cast<const float*>(data), static_cast<float*>(data), numBytes / sizeof(float), ProdOp<float>{}); break;
        case at::kDouble: mesh_.allReduce(static_cast<const double*>(data), static_cast<double*>(data), numBytes / sizeof(double), ProdOp<double>{}); break;
        case at::kInt: mesh_.allReduce(static_cast<const int32_t*>(data), static_cast<int32_t*>(data), numBytes / sizeof(int32_t), ProdOp<int32_t>{}); break;
        case at::kLong: mesh_.allReduce(static_cast<const int64_t*>(data), static_cast<int64_t*>(data), numBytes / sizeof(int64_t), ProdOp<int64_t>{}); break;
        default: TORCH_CHECK(false, JACCL_TAG, " Unsupported dtype"); break;
      }
      break;
    case ReduceOp::MIN:
      switch (dtype) {
        case at::kFloat: mesh_.allReduce(static_cast<const float*>(data), static_cast<float*>(data), numBytes / sizeof(float), MinOp<float>{}); break;
        case at::kDouble: mesh_.allReduce(static_cast<const double*>(data), static_cast<double*>(data), numBytes / sizeof(double), MinOp<double>{}); break;
        case at::kInt: mesh_.allReduce(static_cast<const int32_t*>(data), static_cast<int32_t*>(data), numBytes / sizeof(int32_t), MinOp<int32_t>{}); break;
        case at::kLong: mesh_.allReduce(static_cast<const int64_t*>(data), static_cast<int64_t*>(data), numBytes / sizeof(int64_t), MinOp<int64_t>{}); break;
        default: TORCH_CHECK(false, JACCL_TAG, " Unsupported dtype"); break;
      }
      break;
    case ReduceOp::MAX:
      switch (dtype) {
        case at::kFloat: mesh_.allReduce(static_cast<const float*>(data), static_cast<float*>(data), numBytes / sizeof(float), MaxOp<float>{}); break;
        case at::kDouble: mesh_.allReduce(static_cast<const double*>(data), static_cast<double*>(data), numBytes / sizeof(double), MaxOp<double>{}); break;
        case at::kInt: mesh_.allReduce(static_cast<const int32_t*>(data), static_cast<int32_t*>(data), numBytes / sizeof(int32_t), MaxOp<int32_t>{}); break;
        case at::kLong: mesh_.allReduce(static_cast<const int64_t*>(data), static_cast<int64_t*>(data), numBytes / sizeof(int64_t), MaxOp<int64_t>{}); break;
        default: TORCH_CHECK(false, JACCL_TAG, " Unsupported dtype"); break;
      }
      break;
    default:
      TORCH_CHECK(false, JACCL_TAG, " Unsupported reduce op");
  }
}

void JACCLTransport::broadcast(void* data, size_t numBytes, int rootRank) {
  if (rank_ == rootRank) {
    for (int peer = 0; peer < size_; peer++) {
      if (peer == rank_)
        continue;
      mesh_.send(static_cast<const char*>(data), numBytes, peer);
    }
  } else {
    mesh_.recv(static_cast<char*>(data), numBytes, rootRank);
  }
}

void JACCLTransport::send(const void* data, size_t numBytes, int dstRank) {
  mesh_.send(static_cast<const char*>(data), numBytes, dstRank);
}

void JACCLTransport::recv(void* data, size_t numBytes, int srcRank) {
  mesh_.recv(static_cast<char*>(data), numBytes, srcRank);
}

void JACCLTransport::barrier() {
  sideChannel_.allGather<int>(0);
}

} // namespace c10d::jaccl

#endif // HAVE_JACCL

#endif // USE_C10D_MPS
