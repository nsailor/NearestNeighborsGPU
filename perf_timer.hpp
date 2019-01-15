#pragma once
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
    cout << "Performance" << endl;
    cout << "Time (ms)\tTask" << endl;
    for (auto recording : recorded_durations) {
      cout << recording.second << "\t" << recording.first << endl;
    }
  }

 private:
  std::vector<std::pair<std::string, long>> recorded_durations;
  std::string current_stage;
  std::chrono::high_resolution_clock::time_point last_t;
};