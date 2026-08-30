// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <Python.h>

#include <cuda_runtime_api.h>
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

constexpr const char* kCapsuleName = "vllm.cuda_copy_scheduler";

enum class Priority : int { kSpeculative = 0, kUrgent = 1 };

enum class JobStatus : int {
  kQueued = 0,
  kIssuing = 1,
  kIssued = 2,
  kCompleted = 3,
  kCanceled = 4,
  kFailed = 5,
};

struct CopySegment {
  uintptr_t src;
  uintptr_t dst;
  size_t nbytes;
  cudaMemcpyKind kind;
};

struct CopyJob {
  uint64_t cookie;
  std::vector<CopySegment> segments;
  cudaEvent_t wait_event;
  cudaEvent_t done_event;
  std::string label;
};

struct JobRecord {
  CopyJob job;
  Priority priority;
  JobStatus status = JobStatus::kQueued;
  bool committed = false;
  std::string error;
  cudaEvent_t completion_event = nullptr;
};

struct Window {
  uint64_t handle;
  std::vector<std::shared_ptr<JobRecord>> jobs;
  size_t next_index = 0;
  cudaEvent_t start_event = nullptr;
  cudaEvent_t stop_event = nullptr;
  bool accepting = true;
  bool activated = false;
  std::string error;
};

struct InflightJob {
  std::shared_ptr<JobRecord> job;
  std::shared_ptr<Window> window;
};

struct WindowSnapshot {
  uint64_t handle;
  std::vector<uint64_t> queued;
  std::vector<uint64_t> issuing;
  std::vector<uint64_t> issued;
  std::vector<uint64_t> completed;
  std::vector<uint64_t> canceled;
  std::vector<uint64_t> failed;
  std::string error;
};

std::string cuda_error(cudaError_t error, const char* operation) {
  return std::string(operation) + ": " + cudaGetErrorString(error);
}

void append_cuda_error(std::string& error, cudaError_t status,
                       const char* operation) {
  if (status == cudaSuccess) {
    return;
  }
  if (!error.empty()) {
    error += "; ";
  }
  error += cuda_error(status, operation);
}

class NvtxRange {
 public:
  explicit NvtxRange(const std::string& label) : active_(!label.empty()) {
    if (active_) {
      nvtxRangePushA(label.c_str());
    }
  }

  ~NvtxRange() {
    if (active_) {
      nvtxRangePop();
    }
  }

 private:
  bool active_;
};

class CudaCopyScheduler {
 public:
  CudaCopyScheduler(int device, uintptr_t stream_handle, size_t max_inflight,
                    int poll_interval_us)
      : device_(device),
        stream_(reinterpret_cast<cudaStream_t>(stream_handle)),
        max_inflight_(max_inflight) {
    if (stream_ == nullptr) {
      throw std::invalid_argument("stream_handle must be non-zero");
    }
    if (max_inflight_ == 0) {
      throw std::invalid_argument("max_inflight must be positive");
    }
    if (poll_interval_us <= 0) {
      throw std::invalid_argument("poll_interval_us must be positive");
    }
    // Retained for Python ABI compatibility; completion is event-driven.
    try {
      completion_worker_ =
          std::thread(&CudaCopyScheduler::run_completion, this);
      issue_worker_ = std::thread(&CudaCopyScheduler::run_issue, this);
      std::string startup_error;
      {
        std::unique_lock lock(mutex_);
        condition_.wait(lock, [&] { return workers_starting_ == 0; });
        startup_error = scheduler_error_;
      }
      if (!startup_error.empty()) {
        if (issue_worker_.joinable()) {
          issue_worker_.join();
        }
        if (completion_worker_.joinable()) {
          completion_worker_.join();
        }
        throw std::runtime_error(startup_error);
      }
    } catch (...) {
      {
        std::lock_guard lock(mutex_);
        closed_ = true;
      }
      condition_.notify_all();
      if (issue_worker_.joinable()) {
        issue_worker_.join();
      }
      if (completion_worker_.joinable()) {
        completion_worker_.join();
      }
      throw;
    }
  }

  ~CudaCopyScheduler() { close(); }

  CudaCopyScheduler(const CudaCopyScheduler&) = delete;
  CudaCopyScheduler& operator=(const CudaCopyScheduler&) = delete;

  uint64_t prepare_window(std::vector<CopyJob> jobs) {
    auto window = std::make_shared<Window>();
    std::unordered_set<uint64_t> cookies;
    window->jobs.reserve(jobs.size());
    for (auto& job : jobs) {
      if (!cookies.insert(job.cookie).second) {
        throw std::invalid_argument(
            "copy job cookies must be unique within a window");
      }
      window->jobs.push_back(
          std::make_shared<JobRecord>(JobRecord{std::move(job),
                                                Priority::kSpeculative,
                                                JobStatus::kQueued,
                                                false,
                                                {}}));
    }

    std::lock_guard lock(mutex_);
    ensure_accepting_locked();
    window->handle = next_window_handle_++;
    windows_by_handle_.emplace(window->handle, window);
    if (window->jobs.empty()) {
      window->accepting = false;
    }
    return window->handle;
  }

  void activate_window(uint64_t handle, cudaEvent_t start_event,
                       cudaEvent_t stop_event) {
    std::lock_guard lock(mutex_);
    ensure_accepting_locked();
    auto window = find_window_locked(handle);
    if (window->activated) {
      throw std::runtime_error("copy window is already active");
    }
    if (window->stop_event != nullptr && stop_event != nullptr) {
      throw std::runtime_error("copy window already has a stop event");
    }
    if (stop_event != nullptr) {
      window->stop_event = stop_event;
    }
    window->start_event = start_event;
    window->activated = true;
    if (window->accepting) {
      windows_.push_back(window);
      condition_.notify_all();
    }
  }

  uint64_t submit_window(std::vector<CopyJob> jobs, cudaEvent_t start_event,
                         cudaEvent_t stop_event) {
    uint64_t handle = prepare_window(std::move(jobs));
    activate_window(handle, start_event, stop_event);
    return handle;
  }

  void set_stop_event(uint64_t handle, cudaEvent_t event) {
    if (event == nullptr) {
      throw std::invalid_argument("stop event must be non-zero");
    }
    std::lock_guard lock(mutex_);
    auto window = find_window_locked(handle);
    if (window->stop_event != nullptr) {
      throw std::runtime_error("copy window already has a stop event");
    }
    window->stop_event = event;
    condition_.notify_all();
  }

  void enqueue_urgent(CopyJob job) {
    auto record = std::make_shared<JobRecord>(JobRecord{
        std::move(job), Priority::kUrgent, JobStatus::kQueued, false, {}});
    std::lock_guard lock(mutex_);
    ensure_accepting_locked();
    if (!urgent_by_cookie_.emplace(record->job.cookie, record).second) {
      throw std::invalid_argument("urgent copy cookie is already pending");
    }
    try {
      urgent_.push_back(record);
    } catch (...) {
      urgent_by_cookie_.erase(record->job.cookie);
      throw;
    }
    condition_.notify_all();
  }

  bool query_urgent_issued(uint64_t cookie) {
    std::lock_guard lock(mutex_);
    auto position = find_urgent_locked(cookie);
    if (position->second->status == JobStatus::kQueued ||
        position->second->status == JobStatus::kIssuing) {
      return false;
    }
    consume_urgent_locked(position);
    return true;
  }

  void wait_urgent_issued(uint64_t cookie) {
    std::unique_lock lock(mutex_);
    auto record = find_urgent_locked(cookie)->second;
    condition_.wait(lock, [&] {
      return record->status != JobStatus::kQueued &&
             record->status != JobStatus::kIssuing;
    });
    auto position = find_urgent_locked(cookie);
    consume_urgent_locked(position);
  }

  void submit_urgent(CopyJob job) {
    uint64_t cookie = job.cookie;
    enqueue_urgent(std::move(job));
    wait_urgent_issued(cookie);
  }

 private:
  using UrgentMap = std::unordered_map<uint64_t, std::shared_ptr<JobRecord>>;

  UrgentMap::iterator find_urgent_locked(uint64_t cookie) {
    auto position = urgent_by_cookie_.find(cookie);
    if (position == urgent_by_cookie_.end()) {
      throw std::invalid_argument("unknown urgent copy cookie");
    }
    return position;
  }

  void consume_urgent_locked(UrgentMap::iterator position) {
    auto record = position->second;
    urgent_by_cookie_.erase(position);
    if (record->status == JobStatus::kFailed) {
      throw std::runtime_error(record->error);
    }
    if (record->status == JobStatus::kCanceled) {
      throw std::runtime_error("urgent copy was canceled during shutdown");
    }
  }

 public:
  void set_stop_and_cancel(uint64_t handle) {
    std::unique_lock lock(mutex_);
    auto window = find_window_locked(handle);
    cancel_window_locked(window);
    condition_.notify_all();
    condition_.wait(lock, [&] { return active_window_ != window.get(); });
  }

  WindowSnapshot snapshot(uint64_t handle) {
    std::lock_guard lock(mutex_);
    return snapshot_locked(find_window_locked(handle));
  }

  size_t pending_count(uint64_t handle) {
    std::lock_guard lock(mutex_);
    auto window = find_window_locked(handle);
    return std::count_if(
        window->jobs.begin(), window->jobs.end(),
        [](const auto& job) { return job->status == JobStatus::kQueued; });
  }

  void release_window(uint64_t handle) {
    std::lock_guard lock(mutex_);
    auto window = find_window_locked(handle);
    if (window->accepting || active_window_ == window.get()) {
      throw std::runtime_error(
          "copy window must stop admission before it can be released");
    }
    windows_by_handle_.erase(handle);
  }

  std::vector<WindowSnapshot> pause_and_drain() {
    std::unique_lock lock(mutex_);
    ensure_open_locked();
    ++pause_depth_;
    for (const auto& [_, window] : windows_by_handle_) {
      cancel_window_locked(window);
    }
    condition_.notify_all();
    condition_.wait(lock, [&] {
      return active_job_ == nullptr && urgent_.empty() && inflight_.empty();
    });
    std::vector<std::pair<uint64_t, WindowSnapshot>> ordered;
    ordered.reserve(windows_by_handle_.size());
    for (const auto& [handle, window] : windows_by_handle_) {
      ordered.emplace_back(handle, snapshot_locked(window));
    }
    std::sort(
        ordered.begin(), ordered.end(),
        [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
    std::vector<WindowSnapshot> result;
    result.reserve(ordered.size());
    for (auto& [_, snapshot] : ordered) {
      result.push_back(std::move(snapshot));
    }
    return result;
  }

  void resume() {
    std::lock_guard lock(mutex_);
    if (pause_depth_ == 0) {
      if (closed_) {
        return;
      }
      throw std::runtime_error("CUDA copy scheduler is not paused");
    }
    --pause_depth_;
    condition_.notify_all();
  }

  void close() noexcept {
    std::lock_guard close_lock(close_mutex_);
    bool notify = false;
    {
      std::lock_guard lock(mutex_);
      if (!closed_) {
        closed_ = true;
        for (const auto& [_, window] : windows_by_handle_) {
          cancel_window_locked(window);
        }
        for (auto& job : urgent_) {
          job->status = JobStatus::kCanceled;
          ++canceled_jobs_;
        }
        urgent_.clear();
        notify = true;
      }
    }
    if (notify) {
      condition_.notify_all();
    }
    if (issue_worker_.joinable()) {
      issue_worker_.join();
    }
    if (completion_worker_.joinable()) {
      completion_worker_.join();
    }
  }

  struct Stats {
    uint64_t issued_jobs;
    uint64_t completed_jobs;
    uint64_t canceled_jobs;
    uint64_t failed_jobs;
    uint64_t issued_bytes;
    size_t inflight_jobs;
    size_t queued_urgent_jobs;
  };

  Stats stats() {
    std::lock_guard lock(mutex_);
    return Stats{issued_jobs_,  completed_jobs_,  canceled_jobs_, failed_jobs_,
                 issued_bytes_, inflight_.size(), urgent_.size()};
  }

 private:
  void run_issue() noexcept {
    if (!initialize_worker()) {
      return;
    }

    while (true) {
      std::shared_ptr<JobRecord> job;
      std::shared_ptr<Window> window;
      {
        std::unique_lock lock(mutex_);
        if (closed_ && urgent_.empty() && active_job_ == nullptr) {
          return;
        }

        if (!urgent_.empty()) {
          job = urgent_.front();
          urgent_.pop_front();
        } else if (!closed_ && pause_depth_ == 0 &&
                   speculative_inflight_ < max_inflight_) {
          window = reserve_speculative_locked();
          if (window != nullptr) {
            job = window->jobs[window->next_index++];
            if (window->next_index == window->jobs.size()) {
              window->accepting = false;
              remove_queued_window_locked(window.get());
            }
          }
        }

        if (job == nullptr) {
          condition_.wait(lock);
          continue;
        }

        job->status = JobStatus::kIssuing;
        active_job_ = job.get();
        active_window_ = window.get();
      }

      issue(job, window);
    }
  }

  void run_completion() noexcept {
    if (!initialize_worker()) {
      return;
    }

    while (true) {
      std::shared_ptr<JobRecord> record;
      std::shared_ptr<Window> window;
      {
        std::unique_lock lock(mutex_);
        condition_.wait(lock, [&] {
          return !inflight_.empty() || (closed_ && active_job_ == nullptr);
        });
        if (inflight_.empty()) {
          if (closed_) {
            return;
          }
          continue;
        }
        record = inflight_.front().job;
        window = inflight_.front().window;
      }

      std::string error;
      cudaError_t status = cudaEventSynchronize(record->completion_event);
      if (status != cudaSuccess) {
        error = cuda_error(status, "cudaEventSynchronize(completion_event)");
      }
      append_cuda_error(error, cudaEventDestroy(record->completion_event),
                        "cudaEventDestroy(completion_event)");

      std::lock_guard lock(mutex_);
      if (inflight_.empty() || inflight_.front().job != record) {
        fail_scheduler_locked("copy completion order was corrupted");
        return;
      }
      record->completion_event = nullptr;
      if (record->priority == Priority::kSpeculative) {
        --speculative_inflight_;
      }
      if (error.empty()) {
        record->status = JobStatus::kCompleted;
        ++completed_jobs_;
      } else {
        record->status = JobStatus::kFailed;
        record->error = error;
        if (window != nullptr && window->error.empty()) {
          window->error = error;
          cancel_window_locked(window);
        }
        ++failed_jobs_;
      }
      inflight_.erase(inflight_.begin());
      condition_.notify_all();
    }
  }

  std::shared_ptr<Window> reserve_speculative_locked() {
    while (!windows_.empty()) {
      auto window = windows_.front();
      if (!window->accepting || window->next_index == window->jobs.size()) {
        windows_.pop_front();
        continue;
      }
      if (window->stop_event != nullptr) {
        cudaError_t status = cudaEventQuery(window->stop_event);
        if (status == cudaSuccess) {
          cancel_window_locked(window);
          continue;
        }
        if (status != cudaErrorNotReady) {
          window->error = cuda_error(status, "cudaEventQuery(stop_event)");
          cancel_window_locked(window);
          continue;
        }
      }
      return window;
    }
    return nullptr;
  }

  void issue(const std::shared_ptr<JobRecord>& record,
             const std::shared_ptr<Window>& window) noexcept {
    std::string error;
    uint64_t bytes = 0;
    cudaEvent_t completion_event = nullptr;
    {
      NvtxRange range(record->job.label);
      cudaError_t status =
          cudaEventCreateWithFlags(&completion_event, cudaEventDisableTiming);
      if (status != cudaSuccess) {
        error =
            cuda_error(status, "cudaEventCreateWithFlags(completion_event)");
      }
      if (error.empty() && window != nullptr &&
          record == window->jobs.front() && window->start_event != nullptr) {
        status = cudaStreamWaitEvent(stream_, window->start_event, 0);
        if (status != cudaSuccess) {
          error = cuda_error(status, "cudaStreamWaitEvent(window start)");
        }
      }
      if (error.empty() && record->job.wait_event != nullptr) {
        status = cudaStreamWaitEvent(stream_, record->job.wait_event, 0);
        if (status != cudaSuccess) {
          error = cuda_error(status, "cudaStreamWaitEvent");
        }
      }
      for (const auto& segment : record->job.segments) {
        if (!error.empty()) {
          break;
        }
        cudaError_t status =
            cudaMemcpyAsync(reinterpret_cast<void*>(segment.dst),
                            reinterpret_cast<const void*>(segment.src),
                            segment.nbytes, segment.kind, stream_);
        if (status != cudaSuccess) {
          error = cuda_error(status, "cudaMemcpyAsync");
          break;
        }
        bytes += segment.nbytes;
      }
      if (error.empty()) {
        status = cudaEventRecord(record->job.done_event, stream_);
        if (status != cudaSuccess) {
          error = cuda_error(status, "cudaEventRecord(done_event)");
        }
      }
      if (error.empty()) {
        status = cudaEventRecord(completion_event, stream_);
        if (status != cudaSuccess) {
          error = cuda_error(status, "cudaEventRecord(completion_event)");
        }
      }
    }

    if (!error.empty()) {
      append_cuda_error(error, cudaStreamSynchronize(stream_),
                        "cudaStreamSynchronize(after issue failure)");
      if (completion_event != nullptr) {
        append_cuda_error(error, cudaEventDestroy(completion_event),
                          "cudaEventDestroy(completion_event)");
        completion_event = nullptr;
      }
    }

    std::lock_guard lock(mutex_);
    active_job_ = nullptr;
    active_window_ = nullptr;
    if (error.empty()) {
      record->status = JobStatus::kIssued;
      record->committed = true;
      record->completion_event = completion_event;
      inflight_.push_back(InflightJob{record, window});
      if (record->priority == Priority::kSpeculative) {
        ++speculative_inflight_;
      }
      ++issued_jobs_;
      issued_bytes_ += bytes;
    } else {
      record->status = JobStatus::kFailed;
      record->error = error;
      ++failed_jobs_;
      if (window != nullptr) {
        if (window->error.empty()) {
          window->error = error;
        }
        cancel_window_locked(window);
      }
    }
    condition_.notify_all();
  }

  bool initialize_worker() noexcept {
    cudaError_t status = cudaSetDevice(device_);
    std::lock_guard lock(mutex_);
    if (status != cudaSuccess) {
      fail_scheduler_locked(cuda_error(status, "cudaSetDevice"));
    }
    --workers_starting_;
    condition_.notify_all();
    return status == cudaSuccess;
  }

  void fail_scheduler_locked(const std::string& error) noexcept {
    scheduler_error_ = error;
    closed_ = true;
    for (const auto& [_, window] : windows_by_handle_) {
      if (window->error.empty()) {
        window->error = error;
      }
      cancel_window_locked(window);
    }
    for (auto& job : urgent_) {
      job->status = JobStatus::kFailed;
      job->error = error;
      ++failed_jobs_;
    }
    urgent_.clear();
    condition_.notify_all();
  }

  void cancel_window_locked(const std::shared_ptr<Window>& window) noexcept {
    if (!window->accepting) {
      return;
    }
    window->accepting = false;
    for (size_t index = window->next_index; index < window->jobs.size();
         ++index) {
      auto& job = window->jobs[index];
      if (job->status == JobStatus::kQueued) {
        job->status = JobStatus::kCanceled;
        ++canceled_jobs_;
      }
    }
    remove_queued_window_locked(window.get());
  }

  WindowSnapshot snapshot_locked(const std::shared_ptr<Window>& window) const {
    WindowSnapshot result;
    result.handle = window->handle;
    result.error = window->error;
    for (const auto& job : window->jobs) {
      if (job->committed) {
        result.issued.push_back(job->job.cookie);
      }
      if (job->status == JobStatus::kQueued) {
        result.queued.push_back(job->job.cookie);
      } else if (job->status == JobStatus::kIssuing) {
        result.issuing.push_back(job->job.cookie);
      } else if (job->status == JobStatus::kCompleted) {
        result.completed.push_back(job->job.cookie);
      } else if (job->status == JobStatus::kCanceled) {
        result.canceled.push_back(job->job.cookie);
      } else if (job->status == JobStatus::kFailed) {
        result.failed.push_back(job->job.cookie);
      }
    }
    return result;
  }

  void remove_queued_window_locked(const Window* window) noexcept {
    auto position = std::find_if(
        windows_.begin(), windows_.end(),
        [window](const auto& candidate) { return candidate.get() == window; });
    if (position != windows_.end()) {
      windows_.erase(position);
    }
  }

  std::shared_ptr<Window> find_window_locked(uint64_t handle) {
    auto position = windows_by_handle_.find(handle);
    if (position == windows_by_handle_.end()) {
      throw std::invalid_argument("unknown copy window handle");
    }
    return position->second;
  }

  void ensure_open_locked() const {
    if (closed_) {
      if (!scheduler_error_.empty()) {
        throw std::runtime_error(scheduler_error_);
      }
      throw std::runtime_error("CUDA copy scheduler is closed");
    }
  }

  void ensure_accepting_locked() const {
    ensure_open_locked();
    if (pause_depth_ != 0) {
      throw std::runtime_error("CUDA copy scheduler is paused");
    }
  }

  int device_;
  cudaStream_t stream_;
  size_t max_inflight_;
  std::mutex close_mutex_;
  std::mutex mutex_;
  std::condition_variable condition_;
  std::thread issue_worker_;
  std::thread completion_worker_;
  std::deque<std::shared_ptr<Window>> windows_;
  std::unordered_map<uint64_t, std::shared_ptr<Window>> windows_by_handle_;
  std::deque<std::shared_ptr<JobRecord>> urgent_;
  UrgentMap urgent_by_cookie_;
  std::vector<InflightJob> inflight_;
  JobRecord* active_job_ = nullptr;
  Window* active_window_ = nullptr;
  uint64_t next_window_handle_ = 1;
  size_t speculative_inflight_ = 0;
  size_t pause_depth_ = 0;
  size_t workers_starting_ = 2;
  bool closed_ = false;
  std::string scheduler_error_;
  uint64_t issued_jobs_ = 0;
  uint64_t completed_jobs_ = 0;
  uint64_t canceled_jobs_ = 0;
  uint64_t failed_jobs_ = 0;
  uint64_t issued_bytes_ = 0;
};

CudaCopyScheduler* get_scheduler(PyObject* capsule) {
  return static_cast<CudaCopyScheduler*>(
      PyCapsule_GetPointer(capsule, kCapsuleName));
}

bool parse_uint64(PyObject* object, uint64_t& value) {
  unsigned long long parsed = PyLong_AsUnsignedLongLong(object);
  if (PyErr_Occurred()) {
    return false;
  }
  value = static_cast<uint64_t>(parsed);
  return true;
}

bool parse_pointer(PyObject* object, uintptr_t& value) {
  unsigned long long parsed = PyLong_AsUnsignedLongLong(object);
  if (PyErr_Occurred()) {
    return false;
  }
  value = static_cast<uintptr_t>(parsed);
  return true;
}

bool parse_segment(PyObject* object, CopySegment& segment) {
  PyObject* sequence =
      PySequence_Fast(object, "copy segment must be a sequence");
  if (sequence == nullptr) {
    return false;
  }
  const Py_ssize_t size = PySequence_Size(sequence);
  if (size != 4) {
    Py_DECREF(sequence);
    PyErr_SetString(PyExc_ValueError,
                    "copy segment must contain src, dst, nbytes, and kind");
    return false;
  }
  PyObject* src = PySequence_GetItem(sequence, 0);
  PyObject* dst = PySequence_GetItem(sequence, 1);
  PyObject* nbytes = PySequence_GetItem(sequence, 2);
  PyObject* kind = PySequence_GetItem(sequence, 3);
  Py_DECREF(sequence);
  if (src == nullptr || dst == nullptr || nbytes == nullptr ||
      kind == nullptr) {
    Py_XDECREF(src);
    Py_XDECREF(dst);
    Py_XDECREF(nbytes);
    Py_XDECREF(kind);
    return false;
  }

  uintptr_t src_value;
  uintptr_t dst_value;
  uint64_t nbytes_value;
  long kind_value;
  bool valid = parse_pointer(src, src_value) && parse_pointer(dst, dst_value) &&
               parse_uint64(nbytes, nbytes_value);
  kind_value = valid ? PyLong_AsLong(kind) : -1;
  if (valid && PyErr_Occurred()) {
    valid = false;
  }
  Py_DECREF(src);
  Py_DECREF(dst);
  Py_DECREF(nbytes);
  Py_DECREF(kind);
  if (!valid) {
    return false;
  }
  if (nbytes_value == 0 || src_value == 0 || dst_value == 0) {
    PyErr_SetString(PyExc_ValueError,
                    "copy segment pointers and nbytes must be non-zero");
    return false;
  }
  if (kind_value < cudaMemcpyHostToHost || kind_value > cudaMemcpyDefault) {
    PyErr_SetString(PyExc_ValueError, "invalid cudaMemcpyKind");
    return false;
  }
  segment = CopySegment{src_value, dst_value, static_cast<size_t>(nbytes_value),
                        static_cast<cudaMemcpyKind>(kind_value)};
  return true;
}

bool parse_job(PyObject* object, CopyJob& job) {
  PyObject* sequence = PySequence_Fast(object, "copy job must be a sequence");
  if (sequence == nullptr) {
    return false;
  }
  if (PySequence_Size(sequence) != 5) {
    Py_DECREF(sequence);
    PyErr_SetString(PyExc_ValueError,
                    "copy job must contain cookie, segments, wait_event, "
                    "done_event, label");
    return false;
  }
  PyObject* cookie = PySequence_GetItem(sequence, 0);
  PyObject* segments = PySequence_GetItem(sequence, 1);
  PyObject* wait_event = PySequence_GetItem(sequence, 2);
  PyObject* done_event = PySequence_GetItem(sequence, 3);
  PyObject* label = PySequence_GetItem(sequence, 4);
  Py_DECREF(sequence);
  if (cookie == nullptr || segments == nullptr || wait_event == nullptr ||
      done_event == nullptr || label == nullptr) {
    Py_XDECREF(cookie);
    Py_XDECREF(segments);
    Py_XDECREF(wait_event);
    Py_XDECREF(done_event);
    Py_XDECREF(label);
    return false;
  }

  uint64_t cookie_value;
  uintptr_t wait_value;
  uintptr_t done_value;
  bool valid = parse_uint64(cookie, cookie_value) &&
               parse_pointer(wait_event, wait_value) &&
               parse_pointer(done_event, done_value);
  if (valid && done_value == 0) {
    PyErr_SetString(PyExc_ValueError, "copy job done_event must be non-zero");
    valid = false;
  }
  PyObject* encoded_label = nullptr;
  char* label_value = nullptr;
  Py_ssize_t label_size = 0;
  if (valid) {
    encoded_label = PyUnicode_AsUTF8String(label);
    valid = encoded_label != nullptr;
  }
  if (valid) {
    valid =
        PyBytes_AsStringAndSize(encoded_label, &label_value, &label_size) == 0;
  }

  std::vector<CopySegment> parsed_segments;
  if (valid) {
    PyObject* segment_sequence =
        PySequence_Fast(segments, "copy job segments must be a sequence");
    if (segment_sequence == nullptr) {
      valid = false;
    } else {
      const Py_ssize_t count = PySequence_Size(segment_sequence);
      if (count <= 0) {
        PyErr_SetString(PyExc_ValueError,
                        "copy job must contain at least one segment");
        valid = false;
      } else {
        parsed_segments.reserve(static_cast<size_t>(count));
        for (Py_ssize_t index = 0; index < count && valid; ++index) {
          PyObject* item = PySequence_GetItem(segment_sequence, index);
          if (item == nullptr) {
            valid = false;
            break;
          }
          CopySegment segment;
          valid = parse_segment(item, segment);
          Py_DECREF(item);
          if (valid) {
            parsed_segments.push_back(segment);
          }
        }
      }
      Py_DECREF(segment_sequence);
    }
  }

  if (valid) {
    job = CopyJob{cookie_value, std::move(parsed_segments),
                  reinterpret_cast<cudaEvent_t>(wait_value),
                  reinterpret_cast<cudaEvent_t>(done_value),
                  std::string(label_value, static_cast<size_t>(label_size))};
  }
  Py_XDECREF(encoded_label);
  Py_DECREF(cookie);
  Py_DECREF(segments);
  Py_DECREF(wait_event);
  Py_DECREF(done_event);
  Py_DECREF(label);
  return valid;
}

bool parse_jobs(PyObject* object, std::vector<CopyJob>& jobs) {
  PyObject* sequence = PySequence_Fast(object, "jobs must be a sequence");
  if (sequence == nullptr) {
    return false;
  }
  const Py_ssize_t count = PySequence_Size(sequence);
  jobs.reserve(static_cast<size_t>(count));
  for (Py_ssize_t index = 0; index < count; ++index) {
    PyObject* item = PySequence_GetItem(sequence, index);
    if (item == nullptr) {
      Py_DECREF(sequence);
      return false;
    }
    CopyJob job;
    bool valid = parse_job(item, job);
    Py_DECREF(item);
    if (!valid) {
      Py_DECREF(sequence);
      return false;
    }
    jobs.push_back(std::move(job));
  }
  Py_DECREF(sequence);
  return true;
}

PyObject* uint64_list(const std::vector<uint64_t>& values) {
  PyObject* result = PyList_New(static_cast<Py_ssize_t>(values.size()));
  if (result == nullptr) {
    return nullptr;
  }
  for (size_t index = 0; index < values.size(); ++index) {
    PyObject* value = PyLong_FromUnsignedLongLong(values[index]);
    if (value == nullptr) {
      Py_DECREF(result);
      return nullptr;
    }
    PyList_SetItem(result, static_cast<Py_ssize_t>(index), value);
  }
  return result;
}

PyObject* window_snapshot(const WindowSnapshot& snapshot) {
  PyObject* result = PyDict_New();
  PyObject* queued_list = uint64_list(snapshot.queued);
  PyObject* issuing_list = uint64_list(snapshot.issuing);
  PyObject* issued_list = uint64_list(snapshot.issued);
  PyObject* completed_list = uint64_list(snapshot.completed);
  PyObject* canceled_list = uint64_list(snapshot.canceled);
  PyObject* failed_list = uint64_list(snapshot.failed);
  PyObject* error = snapshot.error.empty()
                        ? Py_NewRef(Py_None)
                        : PyUnicode_FromString(snapshot.error.c_str());
  if (result == nullptr || queued_list == nullptr || issuing_list == nullptr ||
      issued_list == nullptr || completed_list == nullptr ||
      canceled_list == nullptr || failed_list == nullptr || error == nullptr) {
    Py_XDECREF(result);
    Py_XDECREF(queued_list);
    Py_XDECREF(issuing_list);
    Py_XDECREF(issued_list);
    Py_XDECREF(completed_list);
    Py_XDECREF(canceled_list);
    Py_XDECREF(failed_list);
    Py_XDECREF(error);
    return nullptr;
  }
  PyObject* handle = PyLong_FromUnsignedLongLong(snapshot.handle);
  if (handle == nullptr) {
    Py_DECREF(result);
    Py_DECREF(queued_list);
    Py_DECREF(issuing_list);
    Py_DECREF(issued_list);
    Py_DECREF(completed_list);
    Py_DECREF(canceled_list);
    Py_DECREF(failed_list);
    Py_DECREF(error);
    return nullptr;
  }
  PyDict_SetItemString(result, "handle", handle);
  PyDict_SetItemString(result, "queued", queued_list);
  PyDict_SetItemString(result, "issuing", issuing_list);
  PyDict_SetItemString(result, "issued", issued_list);
  PyDict_SetItemString(result, "completed", completed_list);
  PyDict_SetItemString(result, "canceled", canceled_list);
  PyDict_SetItemString(result, "failed", failed_list);
  PyDict_SetItemString(result, "error", error);
  Py_DECREF(issued_list);
  Py_DECREF(queued_list);
  Py_DECREF(issuing_list);
  Py_DECREF(completed_list);
  Py_DECREF(canceled_list);
  Py_DECREF(failed_list);
  Py_DECREF(error);
  Py_DECREF(handle);
  return result;
}

template <typename Function>
std::exception_ptr run_without_gil(Function&& function) {
  PyThreadState* thread_state = PyEval_SaveThread();
  std::exception_ptr error;
  try {
    function();
  } catch (...) {
    error = std::current_exception();
  }
  PyEval_RestoreThread(thread_state);
  return error;
}

void set_cpp_exception() {
  try {
    throw;
  } catch (const std::invalid_argument& error) {
    PyErr_SetString(PyExc_ValueError, error.what());
  } catch (const std::exception& error) {
    PyErr_SetString(PyExc_RuntimeError, error.what());
  }
}

void capsule_destructor(PyObject* capsule) {
  auto* scheduler = get_scheduler(capsule);
  if (scheduler != nullptr) {
    delete scheduler;
  } else {
    PyErr_Clear();
  }
}

PyObject* create_scheduler(PyObject*, PyObject* args) {
  int device;
  unsigned long long stream;
  Py_ssize_t max_inflight;
  int poll_interval_us;
  if (!PyArg_ParseTuple(args, "iKni", &device, &stream, &max_inflight,
                        &poll_interval_us)) {
    return nullptr;
  }
  try {
    auto scheduler = std::make_unique<CudaCopyScheduler>(
        device, static_cast<uintptr_t>(stream),
        static_cast<size_t>(max_inflight), poll_interval_us);
    PyObject* capsule =
        PyCapsule_New(scheduler.get(), kCapsuleName, capsule_destructor);
    if (capsule == nullptr) {
      return nullptr;
    }
    scheduler.release();
    return capsule;
  } catch (...) {
    set_cpp_exception();
    return nullptr;
  }
}

PyObject* submit_window(PyObject*, PyObject* args) {
  PyObject* capsule;
  PyObject* jobs_object;
  unsigned long long start_event;
  unsigned long long stop_event;
  if (!PyArg_ParseTuple(args, "OOKK", &capsule, &jobs_object, &start_event,
                        &stop_event)) {
    return nullptr;
  }
  auto* scheduler = get_scheduler(capsule);
  if (scheduler == nullptr) {
    return nullptr;
  }
  std::vector<CopyJob> jobs;
  if (!parse_jobs(jobs_object, jobs)) {
    return nullptr;
  }
  try {
    uint64_t handle = scheduler->submit_window(
        std::move(jobs),
        reinterpret_cast<cudaEvent_t>(static_cast<uintptr_t>(start_event)),
        reinterpret_cast<cudaEvent_t>(static_cast<uintptr_t>(stop_event)));
    return PyLong_FromUnsignedLongLong(handle);
  } catch (...) {
    set_cpp_exception();
    return nullptr;
  }
}

PyObject* prepare_window(PyObject*, PyObject* args) {
  PyObject* capsule;
  PyObject* jobs_object;
  if (!PyArg_ParseTuple(args, "OO", &capsule, &jobs_object)) {
    return nullptr;
  }
  auto* scheduler = get_scheduler(capsule);
  if (scheduler == nullptr) {
    return nullptr;
  }
  std::vector<CopyJob> jobs;
  if (!parse_jobs(jobs_object, jobs)) {
    return nullptr;
  }
  try {
    return PyLong_FromUnsignedLongLong(
        scheduler->prepare_window(std::move(jobs)));
  } catch (...) {
    set_cpp_exception();
    return nullptr;
  }
}

PyObject* activate_window(PyObject*, PyObject* args) {
  PyObject* capsule;
  unsigned long long handle;
  unsigned long long start_event;
  unsigned long long stop_event;
  if (!PyArg_ParseTuple(args, "OKKK", &capsule, &handle, &start_event,
                        &stop_event)) {
    return nullptr;
  }
  auto* scheduler = get_scheduler(capsule);
  if (scheduler == nullptr) {
    return nullptr;
  }
  try {
    scheduler->activate_window(
        handle,
        reinterpret_cast<cudaEvent_t>(static_cast<uintptr_t>(start_event)),
        reinterpret_cast<cudaEvent_t>(static_cast<uintptr_t>(stop_event)));
    Py_RETURN_NONE;
  } catch (...) {
    set_cpp_exception();
    return nullptr;
  }
}

PyObject* set_stop_event(PyObject*, PyObject* args) {
  PyObject* capsule;
  unsigned long long handle;
  unsigned long long event;
  if (!PyArg_ParseTuple(args, "OKK", &capsule, &handle, &event)) {
    return nullptr;
  }
  auto* scheduler = get_scheduler(capsule);
  if (scheduler == nullptr) {
    return nullptr;
  }
  try {
    scheduler->set_stop_event(
        handle, reinterpret_cast<cudaEvent_t>(static_cast<uintptr_t>(event)));
    Py_RETURN_NONE;
  } catch (...) {
    set_cpp_exception();
    return nullptr;
  }
}

PyObject* submit_urgent(PyObject*, PyObject* args) {
  PyObject* capsule;
  PyObject* job_object;
  if (!PyArg_ParseTuple(args, "OO", &capsule, &job_object)) {
    return nullptr;
  }
  auto* scheduler = get_scheduler(capsule);
  if (scheduler == nullptr) {
    return nullptr;
  }
  CopyJob job;
  if (!parse_job(job_object, job)) {
    return nullptr;
  }
  try {
    auto error =
        run_without_gil([&] { scheduler->submit_urgent(std::move(job)); });
    if (error != nullptr) {
      std::rethrow_exception(error);
    }
    Py_RETURN_NONE;
  } catch (...) {
    set_cpp_exception();
    return nullptr;
  }
}

PyObject* enqueue_urgent(PyObject*, PyObject* args) {
  PyObject* capsule;
  PyObject* job_object;
  if (!PyArg_ParseTuple(args, "OO", &capsule, &job_object)) {
    return nullptr;
  }
  auto* scheduler = get_scheduler(capsule);
  if (scheduler == nullptr) {
    return nullptr;
  }
  CopyJob job;
  if (!parse_job(job_object, job)) {
    return nullptr;
  }
  try {
    scheduler->enqueue_urgent(std::move(job));
    Py_RETURN_NONE;
  } catch (...) {
    set_cpp_exception();
    return nullptr;
  }
}

PyObject* query_urgent_issued(PyObject*, PyObject* args) {
  PyObject* capsule;
  unsigned long long cookie;
  if (!PyArg_ParseTuple(args, "OK", &capsule, &cookie)) {
    return nullptr;
  }
  auto* scheduler = get_scheduler(capsule);
  if (scheduler == nullptr) {
    return nullptr;
  }
  try {
    if (scheduler->query_urgent_issued(cookie)) {
      Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
  } catch (...) {
    set_cpp_exception();
    return nullptr;
  }
}

PyObject* wait_urgent_issued(PyObject*, PyObject* args) {
  PyObject* capsule;
  unsigned long long cookie;
  if (!PyArg_ParseTuple(args, "OK", &capsule, &cookie)) {
    return nullptr;
  }
  auto* scheduler = get_scheduler(capsule);
  if (scheduler == nullptr) {
    return nullptr;
  }
  try {
    auto error =
        run_without_gil([&] { scheduler->wait_urgent_issued(cookie); });
    if (error != nullptr) {
      std::rethrow_exception(error);
    }
    Py_RETURN_NONE;
  } catch (...) {
    set_cpp_exception();
    return nullptr;
  }
}

PyObject* cancel_and_snapshot(PyObject*, PyObject* args) {
  PyObject* capsule;
  unsigned long long handle;
  if (!PyArg_ParseTuple(args, "OK", &capsule, &handle)) {
    return nullptr;
  }
  auto* scheduler = get_scheduler(capsule);
  if (scheduler == nullptr) {
    return nullptr;
  }
  try {
    auto error =
        run_without_gil([&] { scheduler->set_stop_and_cancel(handle); });
    if (error != nullptr) {
      std::rethrow_exception(error);
    }
    return window_snapshot(scheduler->snapshot(handle));
  } catch (...) {
    set_cpp_exception();
    return nullptr;
  }
}

PyObject* snapshot(PyObject*, PyObject* args) {
  PyObject* capsule;
  unsigned long long handle;
  if (!PyArg_ParseTuple(args, "OK", &capsule, &handle)) {
    return nullptr;
  }
  auto* scheduler = get_scheduler(capsule);
  if (scheduler == nullptr) {
    return nullptr;
  }
  try {
    return window_snapshot(scheduler->snapshot(handle));
  } catch (...) {
    set_cpp_exception();
    return nullptr;
  }
}

PyObject* pending_count(PyObject*, PyObject* args) {
  PyObject* capsule;
  unsigned long long handle;
  if (!PyArg_ParseTuple(args, "OK", &capsule, &handle)) {
    return nullptr;
  }
  auto* scheduler = get_scheduler(capsule);
  if (scheduler == nullptr) {
    return nullptr;
  }
  try {
    return PyLong_FromSize_t(scheduler->pending_count(handle));
  } catch (...) {
    set_cpp_exception();
    return nullptr;
  }
}

PyObject* release_window(PyObject*, PyObject* args) {
  PyObject* capsule;
  unsigned long long handle;
  if (!PyArg_ParseTuple(args, "OK", &capsule, &handle)) {
    return nullptr;
  }
  auto* scheduler = get_scheduler(capsule);
  if (scheduler == nullptr) {
    return nullptr;
  }
  try {
    scheduler->release_window(handle);
    Py_RETURN_NONE;
  } catch (...) {
    set_cpp_exception();
    return nullptr;
  }
}

PyObject* pause_and_drain(PyObject*, PyObject* args) {
  PyObject* capsule;
  if (!PyArg_ParseTuple(args, "O", &capsule)) {
    return nullptr;
  }
  auto* scheduler = get_scheduler(capsule);
  if (scheduler == nullptr) {
    return nullptr;
  }
  try {
    std::vector<WindowSnapshot> windows;
    auto error =
        run_without_gil([&] { windows = scheduler->pause_and_drain(); });
    if (error != nullptr) {
      std::rethrow_exception(error);
    }
    PyObject* result = PyTuple_New(static_cast<Py_ssize_t>(windows.size()));
    if (result == nullptr) {
      return nullptr;
    }
    for (size_t index = 0; index < windows.size(); ++index) {
      PyObject* item = window_snapshot(windows[index]);
      if (item == nullptr) {
        Py_DECREF(result);
        return nullptr;
      }
      PyTuple_SetItem(result, static_cast<Py_ssize_t>(index), item);
    }
    return result;
  } catch (...) {
    set_cpp_exception();
    return nullptr;
  }
}

PyObject* resume(PyObject*, PyObject* args) {
  PyObject* capsule;
  if (!PyArg_ParseTuple(args, "O", &capsule)) {
    return nullptr;
  }
  auto* scheduler = get_scheduler(capsule);
  if (scheduler == nullptr) {
    return nullptr;
  }
  try {
    scheduler->resume();
    Py_RETURN_NONE;
  } catch (...) {
    set_cpp_exception();
    return nullptr;
  }
}

PyObject* close(PyObject*, PyObject* args) {
  PyObject* capsule;
  if (!PyArg_ParseTuple(args, "O", &capsule)) {
    return nullptr;
  }
  auto* scheduler = get_scheduler(capsule);
  if (scheduler == nullptr) {
    return nullptr;
  }
  run_without_gil([&] { scheduler->close(); });
  Py_RETURN_NONE;
}

PyObject* stats(PyObject*, PyObject* args) {
  PyObject* capsule;
  if (!PyArg_ParseTuple(args, "O", &capsule)) {
    return nullptr;
  }
  auto* scheduler = get_scheduler(capsule);
  if (scheduler == nullptr) {
    return nullptr;
  }
  auto stats = scheduler->stats();
  return Py_BuildValue(
      "{sK,sK,sK,sK,sK,sn,sn}", "issued_jobs", stats.issued_jobs,
      "completed_jobs", stats.completed_jobs, "canceled_jobs",
      stats.canceled_jobs, "failed_jobs", stats.failed_jobs, "issued_bytes",
      stats.issued_bytes, "inflight_jobs", stats.inflight_jobs,
      "queued_urgent_jobs", stats.queued_urgent_jobs);
}

PyObject* wait_event_on_stream(PyObject*, PyObject* args) {
  unsigned long long event;
  unsigned long long stream;
  if (!PyArg_ParseTuple(args, "KK", &event, &stream)) {
    return nullptr;
  }
  cudaError_t status = cudaStreamWaitEvent(
      reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(stream)),
      reinterpret_cast<cudaEvent_t>(static_cast<uintptr_t>(event)), 0);
  if (status != cudaSuccess) {
    PyErr_SetString(PyExc_RuntimeError,
                    cuda_error(status, "cudaStreamWaitEvent").c_str());
    return nullptr;
  }
  Py_RETURN_NONE;
}

PyMethodDef methods[] = {
    {"create", create_scheduler, METH_VARARGS, "Create a CUDA copy scheduler."},
    {"prepare_window", prepare_window, METH_VARARGS,
     "Prepare a speculative copy window without activating it."},
    {"activate_window", activate_window, METH_VARARGS,
     "Activate a prepared speculative copy window."},
    {"submit_window", submit_window, METH_VARARGS,
     "Submit an ordered speculative copy window."},
    {"set_stop_event", set_stop_event, METH_VARARGS,
     "Stop a window when an external CUDA event becomes ready."},
    {"submit_urgent", submit_urgent, METH_VARARGS,
     "Submit an urgent job and wait until it is committed to the stream."},
    {"enqueue_urgent", enqueue_urgent, METH_VARARGS,
     "Enqueue an urgent job without waiting for stream commitment."},
    {"query_urgent_issued", query_urgent_issued, METH_VARARGS,
     "Return whether an urgent job is committed to the stream."},
    {"wait_urgent_issued", wait_urgent_issued, METH_VARARGS,
     "Wait until an urgent job is committed to the stream."},
    {"cancel_and_snapshot", cancel_and_snapshot, METH_VARARGS,
     "Cancel unissued jobs and return an admission-fenced snapshot."},
    {"snapshot", snapshot, METH_VARARGS, "Return a copy window snapshot."},
    {"pending_count", pending_count, METH_VARARGS,
     "Return the number of unissued jobs in a copy window."},
    {"release_window", release_window, METH_VARARGS,
     "Release bookkeeping for a window whose admission has stopped."},
    {"pause_and_drain", pause_and_drain, METH_VARARGS,
     "Pause all admission and drain submitted jobs."},
    {"resume", resume, METH_VARARGS, "Resume speculative admission."},
    {"close", close, METH_VARARGS, "Close a CUDA copy scheduler."},
    {"stats", stats, METH_VARARGS, "Return scheduler counters."},
    {"wait_event_on_stream", wait_event_on_stream, METH_VARARGS,
     "Make an external CUDA stream wait for an external CUDA event."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "cuda_copy_scheduler_C",
    "Native CUDA copy scheduling primitives.",
    -1,
    methods,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};

}  // namespace

PyMODINIT_FUNC PyInit_cuda_copy_scheduler_C() {
  return PyModule_Create(&module);
}
