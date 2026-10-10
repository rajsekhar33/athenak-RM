// Test-only ring diagnostic state. No change to AthenaK's binary restart format.
#ifndef PGEN_TESTS_TRANSPORT_RING_STATE_HPP_
#define PGEN_TESTS_TRANSPORT_RING_STATE_HPP_
// Copyright(C) 2026 Athena code contributors
// Licensed under the 3-clause BSD License (the repository LICENSE).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>

namespace transport_ring_restart {

// Double precision matches the currently validated AthenaK builds. This helper
// intentionally does not imply support for single-precision restart files.
struct State {
  std::string fixture;
  std::int64_t cycle;
  double time, heat, laplacian, integral, initial_heat, minimum, maximum;

  void Validate() const {
    if (fixture.empty() || fixture.size() > 8192 ||
        fixture.find_first_of("\r\n") != std::string::npos) {
      throw std::runtime_error("invalid ring fixture identity");
    }
    for (double value : {time, heat, laplacian, integral, initial_heat,
                         minimum, maximum}) {
      if (!std::isfinite(value)) throw std::runtime_error("nonfinite ring state");
    }
    if (cycle < 0 || time < 0 || minimum > maximum) {
      throw std::runtime_error("invalid ring cycle/time/extrema");
    }
  }

  // Completed-step cycle/time, not the pre-increment Driver values. A repeated
  // endpoint is allowed only when its heat and Laplacian are unchanged.
  void Sample(std::int64_t next_cycle, double next_time, double next_heat,
              double next_lap, double next_min, double next_max) {
    Validate();
    State candidate{fixture, next_cycle, next_time, next_heat, next_lap, integral,
                    initial_heat, next_min, next_max};
    candidate.Validate();
    if (next_cycle == cycle && next_time == time) {
      if (next_heat != heat || next_lap != laplacian) {
        throw std::runtime_error("conflicting duplicate ring endpoint");
      }
    } else if (cycle == std::numeric_limits<std::int64_t>::max() ||
               next_cycle != cycle+1 || next_time <= time) {
      throw std::runtime_error("ring step coverage gap or reversed time");
    }
    candidate.integral += 0.5*(next_time-time)*(laplacian+next_lap);
    candidate.minimum = std::min(minimum, next_min);
    candidate.maximum = std::max(maximum, next_max);
    candidate.Validate();
    *this = candidate;  // Do not mutate the original state on rejected input.
  }

  std::string Encode() const {
    Validate();
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "ATHENAK_RING_DIAGNOSTIC 1\n" << std::quoted(fixture) << '\n' << cycle << '\n'
        << std::setprecision(std::numeric_limits<double>::max_digits10)
        << time << ' ' << heat << ' ' << laplacian << ' ' << integral << ' '
        << initial_heat << ' ' << minimum << ' ' << maximum << '\n';
    if (!out) throw std::runtime_error("ring state serialization failed");
    return out.str();
  }
};

inline State Decode(const std::string &text, const std::string &fixture,
                    std::int64_t cycle, double time) {
  std::istringstream in(text);
  in.imbue(std::locale::classic());
  std::string magic;
  int version;
  State state{};
  if (!(in >> magic >> version >> std::quoted(state.fixture) >> state.cycle >> state.time
        >> state.heat >> state.laplacian >> state.integral >> state.initial_heat
        >> state.minimum >> state.maximum) || magic != "ATHENAK_RING_DIAGNOSTIC" ||
      version != 1) {
    throw std::runtime_error("invalid or incomplete ring diagnostic sidecar");
  }
  in >> std::ws;
  if (!in.eof()) throw std::runtime_error("unexpected trailing ring state data");
  state.Validate();
  if (state.fixture != fixture || state.cycle != cycle || state.time != time) {
    throw std::runtime_error("ring sidecar/checkpoint identity mismatch");
  }
  return state;
}

}  // namespace transport_ring_restart
#endif  // PGEN_TESTS_TRANSPORT_RING_STATE_HPP_
