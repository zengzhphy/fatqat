---
title: "Run a surface-code memory-Z experiment"
description: "Build a distance-three surface-code memory on a noisy 17-qubit superconducting simulator, recover detection events, and decode the stored logical Z value."
icon: material-shield-check-outline
figure_alts:
  - "Seventeen-qubit surface-code layout with nine data qubits, four X checks, four Z checks, and a vertical logical Z operator"
  - "Z-check detection-event frequencies across repeated syndrome measurements and the final data boundary"
  - "Raw and decoded logical Z failure probabilities versus syndrome rounds, with 95 percent Wilson intervals"
---

# Run a surface-code memory-Z experiment

A quantum memory experiment asks whether an encoded observable survives a
sequence of noisy operations. Here we prepare logical $|0_L\rangle$ in a
distance-three rotated surface code, repeatedly measure its stabilizers, and
finally read the data qubits in the $Z$ basis. The measured syndrome history
lets a classical decoder decide whether to flip the reported logical bit.
No correction gates are applied to the quantum state during this experiment.

The circuit uses nine data qubits and eight measurement ancillas on a
superconducting coupling graph. We construct its local simulator from
FatQat's public matrix and noise interfaces. The topology and numerical noise
snapshot are adapted from the contributor-supplied QEC17 simulator dated
2026-09-12; the complete configuration appears below. The circuit is a
self-contained CSS surface-code construction, with an explicit four-layer
interaction order and no ancilla reset between rounds.

We will check the ideal experiment before introducing gate noise, idle
relaxation, and readout errors. The final plots distinguish a detection event,
an uncorrected logical-bit error, and a decoder failure. A small distance and
shot count make this a worked example, not a threshold or break-even study.

## 1. Define the patch and its logical observable

Data labels `0` through `8` form a row-major $3\times3$ square. Each ancilla
measures the product of either $X$ or $Z$ over its neighbouring data qubits.
For example, ancilla 9 measures $Z_0Z_1Z_3Z_4$. Weight-two checks terminate
the patch at its boundaries.

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

The binary matrices `HX` and `HZ` record the supports of the checks.
$H_X H_Z^T=0\pmod2$ verifies that every $X$ check commutes with every $Z$
check. The chosen logical operators are

$$
Z_L=Z_0Z_3Z_6,\qquad X_L=X_0X_1X_2.
$$

They commute with the stabilizers and anticommute with each other. Starting
all data in $|0\rangle$ fixes the $Z$ checks and $Z_L$ to $+1$. The first
$X$-check measurements project the state into a code sector; their signs can
be random and are retained as a reference, rather than flagged as errors.
This prepares the logical zero up to a known stabilizer-sign frame, which
does not change $Z_L$.

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

## 2. Build a local noisy superconducting target

The reference snapshot supplies mean one- and two-qubit gate fidelities,
$T_1$, echo $T_2$, and asymmetric readout probabilities. We interpret its
reported fidelities as average gate fidelities and use FatQat's depolarizing
convention

$$
\mathcal E(\rho)=(1-p)\rho+pI/d,
\qquad p=\frac{d}{d-1}(1-F_{\rm avg}).
$$

This gives $p\approx0.0014694$ for a one-qubit gate and $p\approx0.0072078$
for a two-qubit gate. These are channel parameters, not the total probabilities
of a nonidentity Pauli fault. Each driven `X` or `SX` receives one one-qubit
channel, and each `CZ` receives one joint two-qubit channel. We do not add
gate-time relaxation again, because it may already contribute to the measured
gate infidelity. Assigning the same mean fidelity to `X` and `SX` is a
simplifying assumption. `RZ` is virtual and noiseless.

The circuit below explicitly inserts `I` operations on spectators: one tick
during a 20 ns single-qubit layer, and two during a 40 ns CZ layer. These
identities carry relaxation and dephasing derived from the reference
$T_1=38.3771\,\mu s$ and $T_2=9.8941\,\mu s$.

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
ONE_QUBIT_SECONDS = IDLE_SECONDS
CZ_SECONDS = 2 * IDLE_SECONDS


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

We reuse the coupling-constrained implementation map from
[`SCQubitSimulator`][fatqat.simulator.SCQubitSimulator], add a matrix rule for
the explicit idle operation, and pass that map to
[`Simulator`][fatqat.simulator.Simulator]. The example always declares all
17 qubits. This construction does not add a new device profile to the package.

Readout confusion changes the reported bit after the quantum measurement;
it does not flip the post-measurement ancilla state. Measurement is treated as
instantaneous here. Additional measurement-window relaxation, dynamical
decoupling, leakage, crosstalk, and per-qubit calibration variations are outside
this model. The explicit layer durations account for spectator idles only;
the matrix simulator does not infer a physical schedule.

## 3. Order the stabilizer interactions

For a $Z$ check, the data qubits control CNOTs targeting its ancilla. For an
$X$ check, prepare the ancilla in the $X$ basis, use it as the control, and
rotate it back before measurement. Each CNOT is decomposed into
$H_t\,\mathrm{CZ}_{c,t}\,H_t$ using only allowed coupling edges.

In time order, `RZ(pi/2), SX, RZ(pi/2)` implements a Hadamard up to a global
phase. A group of these Hadamards occupies one driven-gate tick because its
two `RZ` operations are virtual.

The corner order matters. In the drawing's coordinates, $X$ checks visit
northwest, southwest, northeast, southeast, while $Z$ checks visit northwest,
northeast, southwest, southeast. Every layer contains six disjoint edges.
The orders preserve the stabilizer measurements and orient two-data-qubit
hook errors away from the corresponding minimum-length logical strings.
See [Tomita and Svore](https://arxiv.org/abs/1404.3747) for why changing this
order can reduce the circuit's effective distance.

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

CX_LAYERS = cx_layers()

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

def build_memory(rounds, *, fault=None):
    """Build the memory; optional fault=(boundary, data_qubit) inserts X."""
    if type(rounds) is not int or rounds < 1:
        raise ValueError("rounds must be a positive integer")
    program = fq.Program(17, 8 * rounds + 9)
    for cycle in range(rounds):
        if fault is not None and fault[0] == cycle:
            program.add(ops.X, fault[1])
        h_layer(program, X_CHECKS)
        for layer in CX_LAYERS:
            h_layer(program, (target for _, target in layer))
            busy = {q for pair in layer for q in pair}
            for pair in layer:
                program.add(ops.CZ, pair)
            for q in ALL_QUBITS:
                if q not in busy:
                    program.add(ops.I, q)
                    program.add(ops.I, q)
            h_layer(program, (target for _, target in layer))
        h_layer(program, X_CHECKS)
        program.measure(ANCILLAS, tuple(range(8 * cycle, 8 * (cycle + 1))))
    if fault is not None and fault[0] == rounds:
        program.add(ops.X, fault[1])
    program.measure(DATA, tuple(range(8 * rounds, 8 * rounds + 9)))
    return program


for index, layer in enumerate(CX_LAYERS):
    print(f"CX layer {index}: {layer}")
```

There is no `Reset` after an ancilla measurement. An ancilla remains in its
measured state and participates in the next round. Each round writes eight
new classical slots; the final nine slots contain the data readout. Reusing
the same classical slots would erase the history needed by the decoder.

## 4. Convert measurement records into detection events

Let $m_t$ denote an ancilla's raw measurement bit in round $t$, with
$m_{-1}=0$. Without reset, the measured parity accumulates, so the stabilizer
bit is

$$
s_t=m_t\oplus m_{t-1}.
$$

A detection event compares consecutive stabilizer values:

$$
d_t=s_t\oplus s_{t-1}.
$$

For interior rounds this is $m_t\oplus m_{t-2}$, not a comparison of adjacent
raw measurements. A single readout error can therefore produce two detection
events separated by two rounds. This distinction is discussed in
[Gehér et al.](https://www.nature.com/articles/s41534-025-00998-y).

For memory-Z, only the $Z$ checks have a known initial sign. Their first-round
bits are detection events. At the end, reconstruct each $Z$ check from the
final data bits and compare it with the last measured stabilizer value.
There is no corresponding final $X$ boundary because the data are measured
in $Z$. We retain interior $X$ detection events as a consistency check, but
decode only the $Z$ history needed to protect $Z_L$ against bit flips.

```python
def detection_events(raw, final_data):
    """Return (R+1, 4) Z-check detectors, including both time boundaries.

    Raw outcomes accumulate check parity because the ancilla is not reset.
    Infer s[t] = raw[t] XOR raw[t-1], taking raw[-1] = 0 only at startup.
    Internal detectors are s[t] XOR s[t-1], i.e. raw[t] XOR raw[t-2].
    """
    raw = np.asarray(raw, dtype=np.uint8)
    final_data = np.asarray(final_data, dtype=np.uint8)
    if raw.ndim != 2 or raw.shape[0] < 1 or raw.shape[1] != 4:
        raise ValueError("raw must have shape (rounds, 4), with rounds >= 1")
    if final_data.shape != (9,):
        raise ValueError("final_data must contain nine data-qubit outcomes")
    if np.any(raw > 1) or np.any(final_data > 1):
        raise ValueError("measurement outcomes must be binary")
    syndrome = raw.copy()
    syndrome[1:] ^= raw[:-1]
    detectors = np.empty((len(raw) + 1, 4), dtype=np.uint8)
    detectors[0] = syndrome[0]  # Z checks are initially known to be +1.
    detectors[1:-1] = syndrome[1:] ^ syndrome[:-1]
    detectors[-1] = ((HZ @ final_data) % 2) ^ syndrome[-1]
    return detectors



def unpack_record(record, rounds):
    """Return Z detectors, raw logical bit, and interior X/Z detectors."""
    bits = np.asarray(record, dtype=np.uint8)
    raw = bits[:8 * rounds].reshape(rounds, 8)
    final_data = bits[8 * rounds:]
    z_indices = [ANCILLAS.index(q) for q in Z_CHECKS]
    z_detectors = detection_events(raw[:, z_indices], final_data)
    syndrome = raw.copy()
    syndrome[1:] ^= raw[:-1]
    interior = syndrome[1:] ^ syndrome[:-1]
    logical = int(final_data[list(LOGICAL_Z)].sum() % 2)
    return z_detectors, logical, interior
```

The uncorrected logical bit is the parity of final data bits 0, 3, and 6.
Its expected value is zero. A nonzero detector says a parity changed; it is
not itself a logical failure.

## 5. Decode the short space-time history

The decoder uses a deliberately simpler noise model than the simulator:
independent data bit flips between rounds and independent flips of raw
ancilla readout bits. A data fault joins the one or two checks that see it;
a one-check fault terminates at a spatial boundary. A readout fault joins
two times, including the final data boundary when necessary.

Each edge has weight $w=\log[(1-p)/p]$. We use equal proxy probabilities
`data_error=readout_error=0.01`, so the decoder favours a minimum number of
faults. These values are decoding assumptions, not a conversion of the
simulator's circuit calibration. In particular, final data readout faults
share the last spatial layer with late data faults.

Shortest paths give costs between detection events and boundaries. For this
small patch, a memoized search compares pairing an event with another event
against terminating it at a boundary. XORing the logical labels along the
selected paths produces the correction bit. The decoder sees only the
detectors, never the observed logical bit when choosing its correction.

This is minimum-weight matching on a phenomenological graph. It omits
circuit-specific correlated faults and asymmetric noise, and does not claim
maximum-likelihood decoding of the simulated channel.

```python
from dataclasses import dataclass
from scipy.sparse.csgraph import shortest_path

@dataclass(frozen=True)
class Fault:
    """A single graph edge and the logical-observable flip it predicts."""

    label: tuple
    detectors: tuple[int, ...]
    logical_flip: int
    weight: float


def fault_graph(rounds, data_error=0.01, readout_error=0.01):
    """Build spatial data-fault and no-reset readout-fault edges.

    A one-detector edge terminates at a spatial boundary. Data faults occur
    at each of the R+1 gaps, including before round 0 and before data readout.
    The last gap also represents classical errors in the final data readout.
    It must not be counted twice when assigning its effective probability.
    """
    if type(rounds) is not int or rounds < 1:
        raise ValueError("rounds must be a positive integer")
    if not (0 < data_error < 0.5 and 0 < readout_error < 0.5):
        raise ValueError("edge error probabilities must lie between 0 and 0.5")
    data_weight = float(np.log((1 - data_error) / data_error))
    readout_weight = float(np.log((1 - readout_error) / readout_error))
    faults = []
    for time in range(rounds + 1):
        for qubit in range(9):
            endpoints = tuple(
                int(4 * time + a) for a in np.flatnonzero(HZ[:, qubit])
            )
            faults.append(
                Fault(
                    ("data", time, qubit), endpoints, int(qubit in LOGICAL_Z), data_weight
                )
            )
    for time in range(rounds):
        for check in range(4):
            raw = np.zeros((rounds, 4), dtype=np.uint8)
            raw[time, check] = 1
            endpoints = tuple(
                int(v) for v in np.flatnonzero(detection_events(raw, np.zeros(9)))
            )
            faults.append(Fault(("readout", time, check), endpoints, 0, readout_weight))
    return tuple(faults)


class SpaceTimeDecoder:
    """Minimum-weight matching with spatial-boundary termination.

    Shortest paths turn the sparse fault graph into pairwise defect costs.
    The small dynamic program chooses a minimum-cost pairing; any number of
    defects may instead terminate at a boundary. The returned bit predicts
    whether the raw logical Z outcome must be flipped. It does not use the
    measured logical outcome to choose the correction.
    """

    def __init__(self, rounds, data_error=0.01, readout_error=0.01):
        self.rounds = rounds
        self.faults = fault_graph(rounds, data_error, readout_error)
        self.num_detectors = 4 * (rounds + 1)
        size = self.num_detectors
        graph = np.full((size, size), np.inf)
        np.fill_diagonal(graph, 0.0)
        edge_flip = np.zeros((size, size), dtype=np.uint8)
        boundary_cost = np.full(size, np.inf)
        boundary_flip = np.zeros(size, dtype=np.uint8)
        for fault in self.faults:
            first = fault.detectors[0]
            if len(fault.detectors) == 1:
                if fault.weight < boundary_cost[first]:
                    boundary_cost[first] = fault.weight
                    boundary_flip[first] = fault.logical_flip
            else:
                second = fault.detectors[1]
                if fault.weight < graph[first, second]:
                    graph[first, second] = graph[second, first] = fault.weight
                    edge_flip[first, second] = fault.logical_flip
                    edge_flip[second, first] = fault.logical_flip

        # Do not include a shared boundary vertex here: detector-pair paths
        # must stay within the graph. Boundary matches are handled separately.
        self.pair_cost, previous = shortest_path(
            graph, directed=False, return_predecessors=True
        )
        self.pair_flip = np.zeros((size, size), dtype=np.uint8)
        for source in range(size):
            for target in range(size):
                node = target
                while node != source and previous[source, node] >= 0:
                    predecessor = int(previous[source, node])
                    self.pair_flip[source, target] ^= edge_flip[predecessor, node]
                    node = predecessor
        destinations = np.argmin(self.pair_cost + boundary_cost[None, :], axis=1)
        self.boundary_cost = (
            self.pair_cost[np.arange(size), destinations] + boundary_cost[destinations]
        )
        self.boundary_flip = (
            self.pair_flip[np.arange(size), destinations] ^ boundary_flip[destinations]
        )

    @lru_cache(maxsize=None)
    def _match(self, active):
        if active == 0:
            return 0.0, 0
        first_bit = active & -active
        first = first_bit.bit_length() - 1
        rest = active ^ first_bit
        cost, flip = self._match(rest)
        best = (cost + self.boundary_cost[first], flip ^ int(self.boundary_flip[first]))
        candidates = rest
        while candidates:
            second_bit = candidates & -candidates
            second = second_bit.bit_length() - 1
            cost, flip = self._match(rest ^ second_bit)
            candidate = (
                cost + self.pair_cost[first, second],
                flip ^ int(self.pair_flip[first, second]),
            )
            if candidate[0] < best[0]:
                best = candidate
            candidates ^= second_bit
        return best

    def decode(self, detectors):
        """Return the predicted logical flip for one detector sample."""
        detectors = np.asarray(detectors, dtype=np.uint8)
        if detectors.size != self.num_detectors or np.any(detectors > 1):
            raise ValueError("wrong detector count or nonbinary detector outcome")
        active = sum(1 << int(node) for node in np.flatnonzero(detectors))
        return self._match(active)[1]
```

## 6. Validate and sample the memory

The following checks execute with the tutorial. We enumerate all single
faults in the decoder's own model, verify the stabilizer ranks and distance,
run ideal one- and three-round native-gate circuits, and inject a physical
single-qubit $X$ error on each data site. These checks protect the record
ordering and both time boundaries; they do not assert that the approximate
decoder corrects every possible circuit-level fault.

```python
def validate_decoder():
    """Check all single faults at every time boundary for R=1,2,3."""
    for rounds in (1, 2, 3):
        decoder = SpaceTimeDecoder(rounds)
        zero_raw = np.zeros((rounds, 4), dtype=np.uint8)
        zero_data = np.zeros(9, dtype=np.uint8)
        assert not detection_events(zero_raw, zero_data).any()
        assert decoder.decode(detection_events(zero_raw, zero_data)) == 0
        for fault in decoder.faults:
            kind, time, position = fault.label
            raw = zero_raw.copy()
            data = zero_data.copy()
            if kind == "data":
                data[position] = 1
                check_values = np.zeros_like(raw)
                check_values[time:] = HZ[:, position]
                raw = np.bitwise_xor.accumulate(check_values, axis=0)
                expected = tuple(
                    4 * time + a for a in np.flatnonzero(HZ[:, position])
                )
            else:
                raw[time, position] = 1
                # A raw readout error reappears two rounds later, except
                # when the final data boundary closes that edge earlier.
                expected = (4 * time + position, 4 * min(time + 2, rounds) + position)
            detectors = detection_events(raw, data)
            assert tuple(np.flatnonzero(detectors)) == expected == fault.detectors
            logical = int(np.bitwise_xor.reduce(data[list(LOGICAL_Z)]))
            assert decoder.decode(detectors) == logical
        print(
            f"R={rounds}: noiseless and {len(decoder.faults)} single-fault columns passed"
        )



validate_decoder()

# Verify code distance and independence using all 512 binary data patterns.
patterns = ((np.arange(512)[:, None] >> np.arange(9)) & 1).astype(np.uint8)
for checks, logical_support in ((HZ, LOGICAL_Z), (HX, LOGICAL_X)):
    syndromes = (patterns @ checks.T) % 2
    assert len(np.unique(syndromes, axis=0)) == 16
    logical_flips = patterns[:, logical_support].sum(axis=1) % 2
    undetected = ~syndromes.any(axis=1) & (logical_flips == 1)
    assert patterns[undetected].sum(axis=1).min() == 3

RUN_CONFIG = {"seed": 7, "shot_parallelism": "serial", "kernel_parallelism": "serial"}
ideal_backend = make_qec17_backend(noisy=False)
for rounds in (1, 3):
    counts = ideal_backend.run(
        build_memory(rounds), shots=8, simulation_config=RUN_CONFIG
    ).result().get_counts_as_tuples()
    assert sum(counts.values()) == 8
    for record in counts:
        detectors, logical, interior = unpack_record(record, rounds)
        assert not detectors.any() and logical == 0 and not interior.any()
    print(f"R={rounds}: ideal native-gate memory passed")

# A physical data X fault must be seen at the expected spatial boundary.
decoder = SpaceTimeDecoder(2)
for qubit in DATA:
    counts = ideal_backend.run(
        build_memory(2, fault=(1, qubit)), shots=1, simulation_config=RUN_CONFIG
    ).result().get_counts_as_tuples()
    record = next(iter(counts))
    detectors, logical, _ = unpack_record(record, 2)
    expected = np.zeros((3, 4), dtype=np.uint8)
    expected[1] = HZ[:, qubit]
    assert np.array_equal(detectors, expected)
    assert logical == int(qubit in LOGICAL_Z)
    assert logical ^ decoder.decode(detectors) == 0
print("All nine single-data-X injections decoded correctly")
```

We now run 128 independent noisy trajectories at each of one, two, and three
rounds. The explicit serial settings make the sampling configuration stable;
Numba compiles its kernels on first use. Expect this cell to take minutes on
a CPU. Increase `SHOTS` for a statistical study, or reduce it for a quick
execution check. Fixing a seed reproduces the same configuration; changing
the runtime or parallelism can change the sampled counts.

```python
SHOTS = 128
ROUND_COUNTS = (1, 2, 3)
noisy_backend = make_qec17_backend()
summaries = []
for rounds in ROUND_COUNTS:
    decoder = SpaceTimeDecoder(rounds)
    result = noisy_backend.run(
        build_memory(rounds), shots=SHOTS,
        simulation_config={**RUN_CONFIG, "seed": 7 + rounds},
        result_config={"counts": True, "final_state": False},
    ).result()
    counts = result.get_counts_as_tuples()
    assert sum(counts.values()) == SHOTS
    detector_totals = np.zeros((rounds + 1, 4))
    raw_failures = decoded_failures = 0
    for record, frequency in counts.items():
        detectors, logical, _ = unpack_record(record, rounds)
        raw_failures += frequency * logical
        decoded_failures += frequency * (logical ^ decoder.decode(detectors))
        detector_totals += frequency * detectors
    summaries.append({
        "rounds": rounds, "raw": raw_failures, "decoded": decoded_failures,
        "detector_rates": detector_totals / SHOTS,
    })
    print(f"R={rounds}, shots={SHOTS}: raw={raw_failures}, decoded={decoded_failures}")
```

## 7. Interpret the results

```python
last = summaries[-1]
rates = last["detector_rates"]
figure, axis = plt.subplots(figsize=(7, 4.5))
display = axis.imshow(rates.T, vmin=0, vmax=max(0.05, float(rates.max())),
                      cmap="Blues", aspect="auto")
axis.set(xticks=range(len(rates)),
         xticklabels=[f"Round {t + 1}" for t in range(last["rounds"])] + ["Final data"],
         yticks=range(4), yticklabels=[f"Z check {q}" for q in Z_CHECKS],
         xlabel="Detection-event time", title=f"Memory-Z detection events ({SHOTS} shots)")
for row in range(4):
    for column in range(len(rates)):
        value = rates[column, row]
        axis.text(column, row, f"{value:.3f}", ha="center", va="center",
                  color="white" if value > display.norm.vmax * 0.55 else "black")
figure.colorbar(display, ax=axis, label="Detection probability")
figure.tight_layout()
plt.show()
```

```python
def wilson_interval(failures, shots, z=1.96):
    fraction = failures / shots
    denominator = 1 + z * z / shots
    centre = (fraction + z * z / (2 * shots)) / denominator
    half = z * math.sqrt(fraction * (1 - fraction) / shots + z * z / (4 * shots * shots)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


figure, axis = plt.subplots(figsize=(7, 4.5))
for key, offset, color, label in (
    ("raw", -0.04, "#64748b", "Raw logical parity"),
    ("decoded", 0.04, "#1d4ed8", "After decoding"),
):
    failures = np.array([summary[key] for summary in summaries])
    probability = failures / SHOTS
    intervals = np.array([wilson_interval(int(count), SHOTS) for count in failures])
    errors = np.vstack((probability - intervals[:, 0], intervals[:, 1] - probability))
    axis.errorbar(np.array(ROUND_COUNTS) + offset, probability, yerr=errors,
                  fmt="o-", capsize=4, color=color, label=label)
axis.set(xticks=ROUND_COUNTS, xlabel="Syndrome rounds", ylabel="Logical Z failure probability",
         title=f"Distance-three memory-Z ({SHOTS} shots per point)", ylim=(0, None))
axis.grid(axis="y", alpha=0.25)
axis.legend()
figure.tight_layout()
plt.show()
```

The detector heatmap includes both time boundaries. Readout errors can raise
detector rates even when no data error has occurred, while a logical error
can leave no detector at all. That is why a detector fraction and a logical
failure probability answer different questions.

The raw logical failure probability counts odd final logical parity. The
decoded probability instead compares that parity with the decoder's inferred
logical flip. The 95% Wilson intervals quantify binomial sampling uncertainty,
not uncertainty in the calibration or decoder model. With this small sample,
zero observed failures still has a nonzero upper confidence bound; the curves
need not be monotone or show a statistically resolved improvement.

Increase `SHOTS` for more precise estimates before drawing comparisons.
More rounds increase both the circuit cost and the number of possible faults.
The 17-qubit statevector has $2^{17}$ amplitudes; a density matrix would require
$4^{17}$ entries, so this example uses sampled statevector trajectories.
The decoder's subset search is intended for short distance-three histories;
larger codes require a scalable matching implementation.

This memory-Z experiment tests storage of one logical observable. To study a
full quantum memory, also prepare and measure a memory-X experiment. To claim
error suppression with code distance, compare multiple distances under the
same circuit, noise, and decoding assumptions.
