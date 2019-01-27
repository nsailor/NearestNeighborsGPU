#pragma once
#include <algorithm>
#include <chrono>
#include <string>
#include <vector>

class perf_timer {
 public:
  void start(std::string description) {
    current_stage = description;
    last_t = std::chrono::high_resolution_clock::now();
  }

  void end() {
    using namespace std::chrono;
    high_resolution_clock::time_point end_t = high_resolution_clock::now();
    long ms = duration_cast<milliseconds>(end_t - last_t).count();
    recorded_durations.push_back(std::make_pair(current_stage, ms));
  }

  void print_results() {
    using namespace std;
    cout << "Time (ms)\tTask" << endl;
    cout << "--------------------------------" << endl;
    for (auto recording : recorded_durations) {
      cout << recording.second << "\t\t" << recording.first << endl;
    }
    cout << "--------------------------------" << endl;
  }

  long total() {
    return std::accumulate(recorded_durations.begin(), recorded_durations.end(),
                           0, [&](long x, std::pair<std::string, long> rhs) {
                             return x + rhs.second;
                           });
  }

 private:
  std::vector<std::pair<std::string, long>> recorded_durations;
  std::string current_stage;
  std::chrono::high_resolution_clock::time_point last_t;
};