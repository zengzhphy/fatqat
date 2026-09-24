---
title: "Run a surface-code memory-Z experiment"
description: "Start from nine physical zero states on a noisy 17-qubit superconducting simulator and compare raw and corrected logical-Z error rates after 2, 3, and 5 QEC cycles."
icon: material-shield-check-outline
figure_alts:
  - "Seventeen-qubit surface-code patch with nine data qubits, four X checks, four Z checks, and the logical Z support"
  - "Z-check detection-event frequencies during five noisy QEC cycles and the final data readout"
  - "Raw and corrected logical Z error rates versus QEC cycles, with 128 shots per point and 95 percent Wilson intervals"
---

# Run a surface-code memory-Z experiment

Initialize the nine data qubits in $|0\rangle^{\otimes9}$, run repeated
syndrome extraction cycles, and measure whether their logical Z value survives.
This physical product state has Z-stabilizer and logical $Z_L$ eigenvalues
$+1$, but is not the fully encoded logical state $|0_L\rangle$.

The experiment includes repeated noisy measurements of both stabilizer
types and a final Z-basis measurement of all nine data
qubits. Each cycle includes a 600 ns ancilla measurement, during which the
data qubits idle, followed by an instantaneous, error-free ancilla reset.
The parity of the final data bits on the logical Z support gives one logical readout bit
per shot. A classical decoder uses the Z-check syndrome history, including
the final checks reconstructed from data readout, to predict whether that
logical bit should be flipped. We compare `logical_error_raw` and
`logical_error_corrected` using 128 shots each at 2, 3, and 5 QEC cycles.

This is a gate-level study using 9 data qubits and 8 ancillas.
The simulator's coupling graph and noise parameters are based on calibration
data from real superconducting hardware. This tutorial uses device-averaged
parameters to construct a local noise model.
The circuit and decoder below are self-contained and use FatQat's public interfaces.

## 1. Specify the code and initial state

Data qubits 0 through 8 form a row-major $3\times3$ square. Each ancilla measures the product of X or Z on its neighbouring data qubits. $H_X$ and $H_Z$ record those supports as binary matrices and the commutation relation requires $H_X H_Z^T=0\pmod2$. The chosen logical operators are $Z_L=Z_0Z_3Z_6$ and $X_L=X_0X_1X_2$.

`POSITIONS` assigns dimensionless coordinates to this code diagram. These coordinates
are used to draw the patch and generate the
four ordered CNOT layers in `cx_layers()` below.

```python
import math
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np

import fatqat as fq
import fatqat.operations as ops

DATA = tuple(range(9))
ANCILLAS = tuple(range(9, 17))
ALL_QUBITS = DATA + ANCILLAS
Z_CHECKS = {9: (0, 1, 3, 4), 12: (4, 5, 7, 8), 13: (1, 2), 15: (6, 7)}
X_CHECKS = {10: (1, 2, 4, 5), 11: (3, 4, 6, 7), 14: (5, 8), 16: (0, 3)}
CHECKS = {**Z_CHECKS, **X_CHECKS}
LOGICAL_Z = (0, 3, 6)
LOGICAL_X = (0, 1, 2)
COUPLINGS = tuple(
    (data, ancilla)
    for ancilla, support in CHECKS.items()
    for data in support
)
POSITIONS = {q: (q % 3, q // 3) for q in DATA}
POSITIONS.update({
    9: (0.5, 0.5), 10: (1.5, 0.5), 11: (0.5, 1.5), 12: (1.5, 1.5),
    13: (1.5, -0.5), 14: (2.5, 1.5), 15: (0.5, 2.5), 16: (-0.5, 0.5),
})


def check_matrix(checks):
    return np.array(
        [[int(q in support) for q in DATA] for support in checks.values()],
        dtype=np.uint8,
    )


HX, HZ = check_matrix(X_CHECKS), check_matrix(Z_CHECKS)
assert not np.any((HX @ HZ.T) % 2)
assert not np.any(HX[:, LOGICAL_Z].sum(axis=1) % 2)
assert not np.any(HZ[:, LOGICAL_X].sum(axis=1) % 2)
assert len(set(LOGICAL_X) & set(LOGICAL_Z)) % 2 == 1
print("Data / ancillas / couplings:", len(DATA), len(ANCILLAS), len(COUPLINGS))
print("Logical Z support:", LOGICAL_Z)
```

```python
figure, axis = plt.subplots(figsize=(7, 6))
for first, second in COUPLINGS:
    xs, ys = zip(POSITIONS[first], POSITIONS[second])
    axis.plot(xs, ys, color="#cbd5e1", linewidth=2, zorder=1)
for sites, color, marker, label in (
    (DATA, "#334155", "o", "Data"),
    (X_CHECKS, "#b45309", "s", "X check"),
    (Z_CHECKS, "#1d4ed8", "s", "Z check"),
):
    coordinates = np.array([POSITIONS[q] for q in sites])
    axis.scatter(*coordinates.T, s=650, color=color, marker=marker,
                 label=label, zorder=3)
    for q in sites:
        axis.text(*POSITIONS[q], str(q), color="white", ha="center",
                  va="center", fontsize=11, fontweight="bold", zorder=4)
logical_positions = np.array([POSITIONS[q] for q in LOGICAL_Z])
axis.plot(*logical_positions.T, color="#16a34a", linewidth=6,
          alpha=0.65, label="Logical Z", zorder=2)
axis.set(aspect="equal", xlim=(-0.9, 2.9), ylim=(2.9, -0.9),
         title="Distance-three surface code: 9 data + 8 ancillas")
axis.axis("off")
axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4)
figure.tight_layout()
plt.show()
```

The simulator starts every shot with all 17 qubits in the computational
zero state. The data therefore start in $|0\rangle^{\otimes9}$ and all
ancillas in $|0\rangle$. The initial Z-check parities and logical Z parity
are zero. No separate encoding circuit or conditional Z correction precedes
the QEC cycles.

In the ideal circuit X checks commute with the Z checks and $Z_L$, so
they preserve the initial logical Z value. This experiment tracks that
value without requiring all X-check signs to be $+1$.

The following small exhaustive check verifies that each check matrix has
rank four and that a Pauli error with no syndrome but a flipped logical
observable has minimum weight three.

```python
DATA_BITS = ((np.arange(512)[:, None] >> np.arange(8, -1, -1)) & 1).astype(np.uint8)
for checks, logical_support in ((HZ, LOGICAL_Z), (HX, LOGICAL_X)):
    syndromes = (DATA_BITS @ checks.T) % 2
    assert len(np.unique(syndromes, axis=0)) == 16
    flips = DATA_BITS[:, logical_support].sum(axis=1) % 2
    undetected = ~syndromes.any(axis=1) & (flips == 1)
    assert DATA_BITS[undetected].sum(axis=1).min() == 3
print("Four independent checks of each type; code distance 3")
print("Initial data: |0>^9; initial Z checks and logical Z have eigenvalue +1")
```

## 2. Construct the backend and noise model

The calibration snapshot contains separate values for 17 qubits and
24 coupled pairs. Here we use the arithmetic mean over qubits for the
single-qubit fidelity, $T_1$, echo $T_2$, $F_{00}$, and $F_{11}$, and the arithmetic mean over
coupled pairs for CZ fidelity. Every qubit therefore shares the same
single-qubit, idle, and readout parameters, and every CZ shares the same fidelity.


Here we use a $d$-dimensional depolarizing channel

$$
\mathcal E(\rho)=(1-p)\rho+pI/d,
$$

given gate fidelity $F_{\rm avg}$, we obtain

$$
p=\frac{d}{d-1}(1-F_{\rm avg}).
$$

Each X or SX receives one single-qubit channel, and each CZ receives one
joint two-qubit channel. RZ is virtual and noiseless. During each gate layer, idle qubits undergo amplitude damping and pure dephasing for the duration of that layer, which is 20 ns for a single-qubit gate layer and 40 ns for a CZ layer. These idle intervals are represented by identity operations with the corresponding noise channels.
All nine data qubits also undergo 600 ns of idle decoherence during each
ancilla measurement. We represent this interval by 30 consecutive 20 ns
idle channels on each data qubit. Composing these Markovian channels gives
the same relaxation and dephasing as one 600 ns interval.


```python
from fatqat.implementation import default_matrix_implementation_map

CALIBRATION_DATE = "2026-09-12"
ONE_QUBIT_FIDELITY = 0.9992652941176471
CZ_FIDELITY = 0.9945941666666667
T1_SECONDS = 38.37705882352941e-6
T2_ECHO_SECONDS = 9.894117647058823e-6
F00 = 0.995764705882353
F11 = 0.9794705882352941
IDLE_SECONDS = 20e-9
CZ_IDLE_TICKS = 2
MEASUREMENT_SECONDS = 600e-9
MEASUREMENT_IDLE_TICKS = round(MEASUREMENT_SECONDS / IDLE_SECONDS)
assert math.isclose(MEASUREMENT_IDLE_TICKS * IDLE_SECONDS, MEASUREMENT_SECONDS)


def make_qec17_noise_model():
    """Return a fresh model for active gates, explicit idles, and readout."""
    noise = fq.NoiseModel()
    # For E(rho) = (1-p) rho + p I/d, F_avg = 1-p(d-1)/d.
    one_qubit_p = 2.0 * (1.0 - ONE_QUBIT_FIDELITY)
    cz_p = (4.0 / 3.0) * (1.0 - CZ_FIDELITY)
    for gate in (ops.X, ops.SX):
        noise.add(fq.noise.Depolarizing(p=one_qubit_p), operation=gate)
    noise.add(fq.noise.Depolarizing(p=cz_p), operation=ops.CZ)

    # Convert rates explicitly: gate-level simulators accept finite channels.
    # The residual phase rate subtracts T1's contribution to T2echo.
    relaxation = fq.noise.ThermalRelaxation(t1=T1_SECONDS, t2=T2_ECHO_SECONDS)
    amplitude = fq.noise.AmplitudeDamping(rate=relaxation.amplitude_rate)
    phase = fq.noise.PhaseDamping(rate=relaxation.pure_dephasing_rate)
    noise.add(
        fq.noise.AmplitudeDamping(p=amplitude.as_probability(IDLE_SECONDS)),
        operation=ops.I,
    )
    noise.add(
        fq.noise.PhaseDamping(p=phase.as_probability(IDLE_SECONDS)),
        operation=ops.I,
    )
    # Columns are true digits and rows are reported digits.
    noise.add(fq.noise.ReadoutConfusion([[F00, 1.0 - F11], [1.0 - F00, F11]]))
    # RZ remains virtual and noise-free.
    return noise


def make_qec17_backend(*, noisy=True):
    """Build the tutorial's constrained gate map plus explicit idle support.

    Programs in the tutorial use exactly 17 qubits. The generic Simulator
    retains the map's restricted CZ edges, but is not a new capacity-limited
    public hardware profile. Smaller checks can use an explicit ResourceLayout
    to select a physical edge, for example device labels 0 and 9.
    """
    profile = fq.simulator.SCQubitSimulator(
        num_qubits=17,
        couplings=COUPLINGS,
    )
    implementation_map = profile.implementation_map
    identity_rule = default_matrix_implementation_map().implementation_for(ops.I)
    if identity_rule is None:
        raise RuntimeError("The default matrix map must implement Identity.")
    implementation_map.add(ops.I, identity_rule)
    return fq.simulator.Simulator(
        method="statevector",
        runtime="numba",
        implementation_map=implementation_map,
        noise=make_qec17_noise_model() if noisy else None,
    )
```

The model uses device-averaged calibration parameters, applied uniformly across qubits and coupled pairs. The X and SX gates are assigned the same gate fidelity, while virtual RZ gates are noiseless. Idle decoherence is modeled as Markovian amplitude damping and pure dephasing, with rates derived from $T_1$ and $T_2^{\mathrm{echo}}$.
Ancilla measurement takes 600 ns and retains the calibrated readout errors.
Reset prepares every ancilla in $|0\rangle$ with zero duration and no error,
independently of its reported measurement bit. The measurement window adds
decoherence only to the data qubits. Additional ancilla measurement dynamics,
leakage, crosstalk, and dynamical decoupling are not included.
Initialization is ideal, so no additional state-preparation error is applied
to the default all-zero state.

## 3. Build the syndrome extraction circuit

Z-type stabilizers are measured using CNOT gates with the data qubits as controls and the ancilla as the target. For X-type stabilizers, the ancilla acts as the control and the data qubits as targets. A Hadamard gate is applied to each X-check ancilla before and after its CNOT sequence. All ancillas are then measured in the Z basis.


The CNOT gates are scheduled in four layers. Each Z-check ancilla interacts with its neighbouring data qubits in the order northwest, northeast, southwest, southeast. Each X-check ancilla uses the order northwest, southwest, northeast, southeast. These directions refer to the positions of data qubits relative to the ancilla in the patch diagram. Missing neighbours at the boundaries are skipped. Each layer contains six CNOT gates acting on disjoint qubit pairs. The ordering controls how ancilla faults propagate into correlated data errors. Any alternative schedule should be checked for correct stabilizer extraction and hook-error propagation.
See [Tomita and Svore](https://arxiv.org/abs/1404.3747).

```python
def cx_layers():
    """Four disjoint layers with hook-safe X/Z corner orders."""
    x_offsets = ((-0.5, -0.5), (-0.5, 0.5), (0.5, -0.5), (0.5, 0.5))
    z_offsets = ((-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 0.5))
    layers = [[] for _ in range(4)]
    data_at = {POSITIONS[q]: q for q in DATA}
    for ancilla in ANCILLAS:
        x, y = POSITIONS[ancilla]
        for layer, (dx, dy) in enumerate(x_offsets if ancilla in X_CHECKS else z_offsets):
            data = data_at.get((x + dx, y + dy))
            if data is not None:
                assert data in CHECKS[ancilla]
                pair = (ancilla, data) if ancilla in X_CHECKS else (data, ancilla)
                layers[layer].append(pair)
    for layer in layers:
        operands = [q for pair in layer for q in pair]
        assert len(operands) == len(set(operands))
    assert sum(map(len, layers)) == len(COUPLINGS)
    return tuple(tuple(layer) for layer in layers)


def h_layer(program, targets):
    """One 20 ns native-H layer, including spectator idles."""
    targets = tuple(targets)
    for q in targets:
        program.add(ops.RZ(np.pi / 2), q)
        program.add(ops.SX, q)
        program.add(ops.RZ(np.pi / 2), q)
    for q in ALL_QUBITS:
        if q not in targets:
            program.add(ops.I, q)

CX_LAYERS = cx_layers()

def cx_layer(program, pairs):
    """Implement one disjoint H(target)-CZ-H(target) layer."""
    pairs = tuple(pairs)
    h_layer(program, (target for _, target in pairs))
    busy = {qubit for pair in pairs for qubit in pair}
    for pair in pairs:
        program.add(ops.CZ, pair)
    for qubit in ALL_QUBITS:
        if qubit not in busy:
            for _ in range(CZ_IDLE_TICKS):
                program.add(ops.I, qubit)
    h_layer(program, (target for _, target in pairs))
```

Each QEC cycle measures all eight ancillas in parallel over a 600 ns window,
then resets them to $|0\rangle$. The circuit places the ancilla projection
before the data-only idle channels, which commute with that projection.
The reset is unconditional and leaves the recorded measurement bits intact.
The last cycle includes the same measurement window and reset before the
final Z-basis data readout. No further storage interval is added after that
terminal readout.

```python
def append_memory_round(program, outputs):
    """Extract checks, idle data for the 600 ns readout, then reset ancillas."""
    h_layer(program, X_CHECKS)
    for layer in CX_LAYERS:
        cx_layer(program, layer)
    h_layer(program, X_CHECKS)
    program.measure(ANCILLAS, tuple(outputs))
    for _ in range(MEASUREMENT_IDLE_TICKS):
        for qubit in DATA:
            program.add(ops.I, qubit)
    program.add(ops.Reset, ANCILLAS)


def build_memory_z(rounds):
    """Start from physical zeros, run R QEC cycles, then read all data in Z.

    Records contain 8*R check bits followed by nine data bits.
    The simulator initializes data and ancillas to zero for every shot.
    Each cycle includes 600 ns of data idling and an ideal ancilla reset.
    """
    if type(rounds) is not int or rounds < 0:
        raise ValueError("rounds must be a nonnegative integer")
    program = fq.Program(17, 8 * rounds + 9)
    for cycle in range(rounds):
        first = 8 * cycle
        append_memory_round(program, range(first, first + 8))
    final_offset = 8 * rounds
    program.measure(DATA, tuple(range(final_offset, final_offset + 9)))
    return program

for index, layer in enumerate(CX_LAYERS):
    print(f"CX layer {index}: {layer}")
print("Classical record: eight check bits per QEC cycle, then nine data bits")
print("Each cycle: 600 ns ancilla readout with data qubit idling, then ideal reset")
```

## 4. Construct detection events including the final readout

With every ancilla initialized or reset to $|0\rangle$ before extraction,
an ideal measurement directly returns the stabilizer bit $s_t$.
A reported bit $m_t=s_t\oplus e_t$ may contain a classical readout error.
For the R storage cycles, indexed $t=0,\ldots,R-1$, define

$$
d_0=m_0,\qquad d_t=m_t\oplus m_{t-1}\quad(1\le t<R).
$$

The initial data have all-positive Z-check signs, so their reference bits
are zero. A single classical ancilla-readout flip produces events at two
adjacent times, t and t+1. For the last ancilla readout, the second event
lies on the final data boundary. Reset prevents a previous ancilla state
from carrying over into the next extraction cycle.

The final reported data bits $b$ provide four additional Z-check parities,
$s_{\mathrm{data}}=H_Zb\pmod2$. Close the detector history with

$$
d_R=s_{\mathrm{data}}\oplus m_{R-1}.
$$

The resulting Z-detector array has shape `(R + 1, 4)`. For R=0, it consists
only of $H_Zb$. These checks come from the actual noisy data readout, not
from an ideal terminal measurement or access to the quantum state. A final
Z-basis readout does not supply terminal X-check values.
Data errors during the last 600 ns measurement window can therefore appear
on this terminal boundary even when the last ancilla outcomes were correct.

```python
def binary_array(values, *, name):
    """Validate before conversion so fractional values cannot become bits."""
    values = np.asarray(values)
    if not np.all((values == 0) | (values == 1)):
        raise ValueError(f"{name} must contain only binary outcomes")
    return values.astype(np.uint8)


def detection_events(raw, final_data):
    """Return (R+1, 4) Z detectors, including the measured data boundary.

    Raw has shape (R, 4), including (0, 4) when there are no storage rounds.
    All ancillas start each extraction in |0>. The physical-zero data have
    Z-check signs +1. Only the Z-check history is used here.
    With reset, raw measurements directly report stabilizer bits.
    """
    raw = binary_array(raw, name="raw")
    final_data = binary_array(final_data, name="final_data")
    if raw.ndim != 2 or raw.shape[1] != 4:
        raise ValueError("raw must have shape (rounds, 4), with rounds >= 0")
    if final_data.shape != (9,):
        raise ValueError("final_data must contain nine data-qubit outcomes")
    final_checks = (HZ @ final_data) % 2
    if len(raw) == 0:
        return final_checks[None, :]
    detectors = np.empty((len(raw) + 1, 4), dtype=np.uint8)
    detectors[0] = raw[0]
    detectors[1:-1] = raw[1:] ^ raw[:-1]
    detectors[-1] = final_checks ^ raw[-1]
    return detectors


def unpack_record(record, rounds):
    """Return terminally closed Z detectors and the raw logical-Z bit."""
    bits = binary_array(record, name="record")
    if bits.shape != (8 * rounds + 9,):
        raise ValueError("wrong check/data record length")
    final_offset = 8 * rounds
    raw = bits[:final_offset].reshape(rounds, 8)
    final_data = bits[final_offset:]
    z_columns = [ANCILLAS.index(q) for q in Z_CHECKS]
    detectors = detection_events(raw[:, z_columns], final_data)
    measured_logical = int(final_data[list(LOGICAL_Z)].sum() % 2)
    return detectors, measured_logical
```

## 5. Predict the logical correction from the syndrome

Use Z-check detection events to infer data X errors, which can flip the
logical Z readout. Every data edge carries a nine-bit X mask. XORing masks
along selected edges and paths gives a recovery mask; its overlap parity
with `LOGICAL_Z` is the predicted logical flip. X-check measurements remain
in the circuit, but are not used by this Z-observable decoder.

The graph approximates noise as data X faults between cycles and classical
ancilla-readout flips. There are R+1 data-fault layers, including the gap
after the last QEC cycle. A final data-readout flip has the same detector
and logical effect as a data X fault in that last gap, so they share one
effective graph edge. It is not counted twice. An ancilla-readout flip at
time t joins detectors at t and t+1 on the same check. Data errors during
an ancilla measurement window belong to the gap after that extraction,
including the final gap before data readout.

All single-detector edges terminate at spatial code boundaries. There is
no open final time boundary: final reported data close the time direction.
Shortest paths give the costs and masks between pairs of events and to
spatial boundaries. A small subset search compares pairing events against
terminating them independently at a boundary.

Use $w=\log[(1-p)/p]$ with equal proxy probabilities
`data_error=readout_error=0.01`. These weights form a minimum-fault-count
heuristic, not calibrated circuit-level fault probabilities. The decoder
does not model every correlated gate fault; its subset
search is intended for short distance-three histories.

```python
from dataclasses import dataclass
from scipy.sparse.csgraph import shortest_path


@dataclass(frozen=True)
class Fault:
    """One graph edge and its nine-data-qubit X-correction mask."""

    label: tuple
    detectors: tuple[int, ...]
    correction_mask: int
    weight: float


def fault_graph(rounds, data_error=0.01, readout_error=0.01):
    """Build spatial data edges and adjacent-time ancilla-readout edges.

    Data X faults occur at R+1 gaps, from before the first storage round to
    after the last one. Final data-readout flips share the last spatial layer
    with late data X faults; they must not be counted again as separate edges.
    The last layer represents an effective proxy error rate for both causes.
    A raw-readout fault connects t and t+1, with no data correction.
    All one-detector edges terminate at spatial boundaries; time is closed
    by the actual terminal data measurement. R=0 is a purely spatial graph.
    """
    if type(rounds) is not int or rounds < 0:
        raise ValueError("rounds must be a nonnegative integer")
    if not (0 < data_error < 0.5 and 0 < readout_error < 0.5):
        raise ValueError("edge error probabilities must lie between 0 and 0.5")
    data_weight = float(np.log((1 - data_error) / data_error))
    readout_weight = float(np.log((1 - readout_error) / readout_error))
    faults = []
    for time in range(rounds + 1):
        for qubit in range(9):
            endpoints = tuple(
                int(4 * time + check) for check in np.flatnonzero(HZ[:, qubit])
            )
            faults.append(Fault(
                ("data", time, qubit), endpoints, 1 << qubit, data_weight
            ))
    for time in range(rounds):
        for check in range(4):
            endpoints = (4 * time + check, 4 * (time + 1) + check)
            faults.append(Fault(("readout", time, check), endpoints, 0, readout_weight))
    return tuple(faults)


class SpaceTimeDecoder:
    """Minimum-weight X recovery using Z detectors and spatial boundaries.

    The return value is a nine-bit X mask. Only its overlap parity with
    LOGICAL_Z is needed to correct the final logical-Z readout. No observed
    logical bit or statevector is used to select that mask. Equal-cost choices
    use fixed index order, with boundary matching before an equal-cost pair.
    """

    def __init__(self, rounds, data_error=0.01, readout_error=0.01):
        self.rounds = rounds
        self.faults = fault_graph(rounds, data_error, readout_error)
        self.num_detectors = 4 * (rounds + 1)
        size = self.num_detectors
        graph = np.full((size, size), np.inf)
        np.fill_diagonal(graph, 0.0)
        edge_mask = np.zeros((size, size), dtype=np.uint16)
        boundary_cost = np.full(size, np.inf)
        boundary_mask = np.zeros(size, dtype=np.uint16)
        for fault in self.faults:
            first = fault.detectors[0]
            if len(fault.detectors) == 1:
                proposed = (fault.weight, fault.correction_mask)
                current = (boundary_cost[first], boundary_mask[first])
                if proposed < current:
                    boundary_cost[first] = fault.weight
                    boundary_mask[first] = fault.correction_mask
            else:
                second = fault.detectors[1]
                proposed = (fault.weight, fault.correction_mask)
                current = (graph[first, second], edge_mask[first, second])
                if proposed < current:
                    graph[first, second] = graph[second, first] = fault.weight
                    edge_mask[first, second] = edge_mask[second, first] = fault.correction_mask

        # Do not let detector-pair paths take shortcuts through a shared
        # boundary vertex. Each unmatched event terminates independently.
        self.pair_cost, previous = shortest_path(
            graph, directed=False, return_predecessors=True
        )
        self.pair_mask = np.zeros((size, size), dtype=np.uint16)
        for source in range(size):
            for target in range(size):
                node = target
                while node != source and previous[source, node] >= 0:
                    predecessor = int(previous[source, node])
                    self.pair_mask[source, target] ^= edge_mask[predecessor, node]
                    node = predecessor
        destinations = np.argmin(self.pair_cost + boundary_cost[None, :], axis=1)
        self.boundary_cost = (
            self.pair_cost[np.arange(size), destinations] + boundary_cost[destinations]
        )
        self.boundary_mask = (
            self.pair_mask[np.arange(size), destinations] ^ boundary_mask[destinations]
        )
        if not np.all(np.isfinite(self.boundary_cost)):
            raise ValueError("every detector must be connected to a boundary")

    @lru_cache(maxsize=None)
    def _match(self, active):
        if active == 0:
            return 0.0, 0
        first_bit = active & -active
        first = first_bit.bit_length() - 1
        rest = active ^ first_bit
        cost, mask = self._match(rest)
        best = (cost + self.boundary_cost[first], mask ^ int(self.boundary_mask[first]))
        candidates = rest
        while candidates:
            second_bit = candidates & -candidates
            second = second_bit.bit_length() - 1
            cost, mask = self._match(rest ^ second_bit)
            candidate = (
                cost + self.pair_cost[first, second],
                mask ^ int(self.pair_mask[first, second]),
            )
            if candidate[0] < best[0]:
                best = candidate
            candidates ^= second_bit
        return best

    def _active(self, detectors):
        detectors = binary_array(detectors, name="detectors")
        if detectors.shape != (self.rounds + 1, 4):
            raise ValueError("detectors must have shape (rounds + 1, 4)")
        return sum(1 << int(node) for node in np.flatnonzero(detectors))

    def decode(self, detectors):
        """Return a nine-bit X mask, using only measured detection events."""
        return self._match(self._active(detectors))[1]

    def minimum_cost(self, detectors):
        """Return the selected graph cost, useful for independent checks."""
        return self._match(self._active(detectors))[0]
```

The decoder sees only detection events. In particular, the measured logical
bit is not an input to `decode()`. Different final data strings can have
the same Z-check parities but opposite logical parities; the decoder must
make the same prediction for such identical syndrome histories.

## 6. Count raw and corrected logical errors

Each shot ends with nine reported data bits $b_0,\ldots,b_8$. The measured
logical observable is $Z_L=(-1)^\ell$, where

$$
\ell=b_0\oplus b_3\oplus b_6.
$$

The expected logical bit is zero because the initial physical state
$|0\rangle^{\otimes9}$ has $Z_L=+1$.
The decoder receives only Z-check detection events and predicts an X mask
$x(d)$. Its predicted logical flip is
$\hat\ell(d)=x_0(d)\oplus x_3(d)\oplus x_6(d)$. It does not use the measured
logical bit to choose this prediction. Define the two rates, stored as
`logical_error_raw` and `logical_error_corrected`, respectively:

$$
\begin{aligned}
p_{\mathrm{raw}}(R)
&=\frac1N\sum_{k=1}^N\mathbf1[\ell_k\ne0],\\
p_{\mathrm{corrected}}(R)
&=\frac1N\sum_{k=1}^N\mathbf1[\ell_k\oplus\hat\ell(d_k)\ne0],
\qquad N=128.
\end{aligned}
$$

The corrected indicator is one exactly when the decoder's predicted flip
disagrees with the observed logical error. We apply this correction to the
classical interpretation of the measurement; no extra physical recovery
pulses are simulated. Both curves use the same shots and the same circuit.
"Raw" means before history-based decoding. Both curves include the same
stabilizer measurements during storage; neither applies physical correction
gates. The corrected result updates only the classical interpretation.

```python
def logical_flip(correction_mask):
    """Return the logical-Z sign flip predicted by a data-X correction mask."""
    return sum((int(correction_mask) >> q) & 1 for q in LOGICAL_Z) % 2


RUN_CONFIG = {"shot_parallelism": "serial", "kernel_parallelism": "serial"}
SHOTS = 128
QEC_CYCLES = (2, 3, 5)
noisy_backend = make_qec17_backend()
summaries = []
for cycles in QEC_CYCLES:
    decoder = SpaceTimeDecoder(cycles)
    seed = int(np.random.SeedSequence([20260912, cycles]).generate_state(1)[0])
    result = noisy_backend.run(
        build_memory_z(cycles), shots=SHOTS,
        simulation_config={**RUN_CONFIG, "seed": seed},
        result_config={"counts": True, "final_state": False},
    ).result()
    counts = result.get_counts_as_tuples()
    assert sum(counts.values()) == SHOTS
    raw_errors = corrected_errors = 0
    event_totals = np.zeros((cycles + 1, 4))
    for record, frequency in counts.items():
        detectors, measured_logical = unpack_record(record, cycles)
        # Prediction uses syndrome information, never measured_logical.
        predicted_flip = logical_flip(decoder.decode(detectors))
        raw_errors += frequency * measured_logical
        corrected_errors += frequency * (measured_logical ^ predicted_flip)
        event_totals += frequency * detectors
    summaries.append({
        "cycles": cycles,
        "raw_errors": raw_errors,
        "corrected_errors": corrected_errors,
        "logical_error_raw": raw_errors / SHOTS,
        "logical_error_corrected": corrected_errors / SHOTS,
        "detector_rates": event_totals / SHOTS,
    })
    print(f"QEC cycles={cycles}, shots={SHOTS}: "
          f"logical_error_raw={raw_errors}/{SHOTS} ({raw_errors / SHOTS:.5f}), "
          f"logical_error_corrected={corrected_errors}/{SHOTS} "
          f"({corrected_errors / SHOTS:.5f})")
```

Each cycle count uses 128 independent noisy shots. All occurrences of a
record are counted using its histogram frequency; distinct records are not
given equal weight. The simulation requests counts only and does not inspect
the final statevector. Fixed seeds and explicit serial settings make the
sampling configuration reproducible. Expect several minutes of CPU time;
the runtime depends on the machine.

Every point starts from the same ideal physical-zero state and includes
noisy QEC cycles and final readout. No shot is postselected on stabilizer
outcomes or detector activity.

## 7. Plot logical error rate versus QEC cycles

The Z-check detector map includes a terminal column reconstructed from the
same noisy data measurements used for logical readout. This terminal column
is not a perfect syndrome measurement. Both X and Z checks are measured
during every QEC cycle, but this logical-Z decoder uses only the Z-check
history; phase corrections do not change a Z-basis logical readout.

```python
last = summaries[-1]
rates = last["detector_rates"].T
figure, axis = plt.subplots(figsize=(7, 4.5))
maximum = max(0.05, float(rates.max()))
display = axis.imshow(rates, vmin=0, vmax=maximum, cmap="Blues", aspect="auto")
axis.set(xticks=range(last["cycles"] + 1),
         xticklabels=[f"Cycle {t + 1}" for t in range(last["cycles"])] + ["Final data"],
         yticks=range(4), yticklabels=[f"Z check {q}" for q in Z_CHECKS],
         xlabel="Detection-event time", title="Z-check detection events")
for row in range(4):
    for column in range(last["cycles"] + 1):
        value = rates[row, column]
        axis.text(column, row, f"{value:.3f}", ha="center", va="center",
                  color="white" if value > maximum * 0.55 else "black")
figure.colorbar(display, ax=axis, label="Detection probability")
figure.tight_layout()
plt.show()
```

The final plot reports the fraction of shots with a wrong logical result
after R QEC cycles. It is a cumulative error probability for that circuit
length, not a fitted logical error probability per cycle. Since each shot
has a binary success/failure outcome, we show pointwise 95% Wilson intervals
for binomial sampling uncertainty. Zero observed failures still has a
nonzero upper bound. The intervals do not represent calibration uncertainty.

```python
def wilson_interval(errors, shots, z=1.959963984540054):
    fraction = errors / shots
    denominator = 1 + z * z / shots
    centre = (fraction + z * z / (2 * shots)) / denominator
    half = z * math.sqrt(
        fraction * (1 - fraction) / shots + z * z / (4 * shots * shots)
    ) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


figure, axis = plt.subplots(figsize=(8, 4.8))
for rate_key, count_key, color, marker in (
    ("logical_error_raw", "raw_errors", "#64748b", "o"),
    ("logical_error_corrected", "corrected_errors", "#1d4ed8", "s"),
):
    rates = np.array([summary[rate_key] for summary in summaries])
    intervals = np.array([
        wilson_interval(summary[count_key], SHOTS) for summary in summaries
    ])
    errors = np.maximum(0.0, np.vstack((rates - intervals[:, 0], intervals[:, 1] - rates)))
    axis.errorbar(QEC_CYCLES, rates, yerr=errors, fmt=f"{marker}-",
                  capsize=4, color=color, label=rate_key)
axis.set(xticks=QEC_CYCLES, xlabel="QEC cycle", ylabel="Logical error rate",
         title=f"Surface-code memory-Z ({SHOTS} shots per point)",
         ylim=(0, None))
axis.grid(axis="y", alpha=0.25)
axis.legend()
figure.tight_layout()
plt.show()
```

A low detection-event rate and a low logical error rate answer different
questions: readout errors can trigger detectors, and a logical error can
leave every detector unchanged. Decoding can also misidentify a fault,
so the corrected rate need not improve for every finite sample. This small
decoder approximates gate-level and correlated faults with independent
data-X and ancilla-readout errors.

This memory-Z experiment measures logical Z readout errors. Phase errors
that preserve this observable are not counted, so this rate is not a
full encoded-state infidelity or a benchmark of arbitrary logical inputs.

For a quantitative memory study, increase the shot count and cycle range,
refine the circuit-level decoder and noise model, and compare with an
unencoded memory at matched elapsed times. Multiple code distances are
needed to study suppression with distance.
