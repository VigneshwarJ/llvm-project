# tryCriticalResourceDependency Heuristic Deep Dive

## Overview

The `tryCriticalResourceDependency` heuristic schedules instructions that **enable** critical resources rather than instructions that **use** them. While `tryCriticalResource` prefers the instruction consuming the most critical hardware unit, `tryCriticalResourceDependency` prefers the instruction that will unblock the highest-priority pending instruction on the most prioritized critical resource.

This heuristic is essential for regions with long latency in-region dependencies, like DS Load -> WMMA dependencies in GEMMs: scheduling the right DS load first ensures the WMMA that needs it earliest becomes ready as soon as possible, reducing register pressure and creating good ordering between them.

## Key Concepts

### The Target SU

Each `HardwareUnitInfo` maintains a **TargetSU**: the highest-priority unscheduled instruction that uses this resource. The heuristic asks: *"Which candidate enables the TargetSU?"*

```cpp
SUnit *HardwareUnitInfo::getNextTargetSU(bool LookDeep) const {
  // First, check PrioritySUs (sorted by depth)
  for (SUnit *PrioritySU : PrioritySUs) {
    if (!PrioritySU->isTopReady())
      return PrioritySU;
  }

  // If LookDeep is set, search all unscheduled SUs
  if (!LookDeep)
    return nullptr;

  unsigned MinDepth = std::numeric_limits<unsigned int>::max();
  SUnit *TargetSU = nullptr;
  for (auto *SU : AllSUs) {
    if (SU->isScheduled || SU->isTopReady())
      continue;
    if (SU->getDepth() < MinDepth) {
      MinDepth = SU->getDepth();
      TargetSU = SU;
    }
  }
  return TargetSU;
}
```

**PrioritySUs**: The set of SUs at the minimum depth for this HWUI. During population, each SU is added with depth tracking:

```cpp
void HardwareUnitInfo::insert(SUnit *SU, unsigned BlockingCycles) {
  AllSUs.insert(SU);
  TotalCycles += BlockingCycles;

  if (PrioritySUs.empty()) {
    PrioritySUs.insert(SU);
    return;
  }
  
  unsigned SUDepth = SU->getDepth();
  unsigned CurrDepth = (*PrioritySUs.begin())->getDepth();
  
  if (SUDepth > CurrDepth)
    return;  // Higher depth, don't add to priority list

  if (SUDepth == CurrDepth) {
    PrioritySUs.insert(SU);  // Same depth, add to existing set
    return;
  }

  // Lower depth = higher priority, replace the set
  PrioritySUs.clear();
  PrioritySUs.insert(SU);
}
```

**LookDeep**: When comparing DS instructions against which enables a WMMA, we use `LookDeep=true` to search beyond `PrioritySUs` into all unscheduled SUs. This is because DS loads that feed a WMMA should be prioritized even if the WMMA isn't in the priority set yet.

### Reachability: "Enables" a Resource

A candidate **enables** the TargetSU if it is reachable from the candidate in the DAG -- that is, there exists a dependency path between this instruction and the TargetSU: 

```cpp
bool CandEnables = TargetSU != Cand.SU && DAG->IsReachable(TargetSU, Cand.SU);
```

If `Cand.SU` is a predecessor of `TargetSU` (directly or transitively), then scheduling `Cand.SU` eith directly reduces `TargetSU`'s `NumPredsLeft`, or reduces the `NumPredsLeft` in one of its recurseive predecessors.

---

## sortHWUIResources: Unified Sorting for Both Heuristics

Both `tryCriticalResource` and `tryCriticalResourceDependency` now use the same sorting with `UseDependencySort=true`. The HWUI list is sorted to prioritize:

1. **Exposed cycles first**: Resources with `RemainingExposed > 0` are always prioritized - we want to be able to schedule resources with exposed cycles over those without
2. **Unready cycles**: Among resources with the same exposed status, prefer those with more unready cycles—cycles of work that cannot be scheduled yet because their instructions are blocked by dependencies 

### RegionMixInfo

The `RegionMixInfo` structure tracks the current state of ready instructions:

```cpp
struct RegionMixInfo {
  // Per-flavor counts of instructions in Available + Pending queues
  unsigned ReadyCount[NUM_FLAVORS];
  // Per-flavor sum of cycles for ready instructions
  unsigned ReadyCycles[NUM_FLAVORS];
  // Per-flavor count of already-scheduled instructions
  unsigned ScheduledCount[NUM_FLAVORS];
};
```

`RegionMixInfo` is refreshed from the scheduler boundary before sorting:

```cpp
void RegionMixInfo::refreshFromBoundary(SchedBoundary &Zone, ...) {
  // Snapshot Available + Pending: both have NumPredsLeft == 0 (DAG-ready).
  for (SUnit *SU : Zone.Available.elements())
    Bump(SU);
  for (SUnit *SU : Zone.Pending.elements())
    Bump(SU);
}
```

### The Sort Logic

```cpp
void CandidateHeuristics::sortHWUIResources(SchedBoundary *Zone,
                                            bool UseDependencySort) {
  if (UseDependencySort)
    MixInfo.refreshFromBoundary(*Zone, *SII, this);

  llvm::sort(HWUInfo, [UseExposed, UseDependencySort, this](...) {
    // 1. CoexecWindow producers first (WMMA, TRANS, MultiCycleVALU)
    if (A.producesCoexecWindow() != B.producesCoexecWindow())
      return A.producesCoexecWindow();
    
    // 2. Prefer resources with non-zero RemainingExposed
    if (UseExposed) {
      bool AExp = A.getRemainingExposed() > 0;
      bool BExp = B.getRemainingExposed() > 0;
      if (AExp != BExp)
        return AExp;  // Has exposed cycles = higher priority
      if (A.getRemainingExposed() != B.getRemainingExposed())
        return A.getRemainingExposed() > B.getRemainingExposed();
    }
    
    // 3. Prefer resources with more unready cycles
    if (UseDependencySort) {
      unsigned AReadyCycles = MixInfo.getReadyCycles(A.getType());
      unsigned ARemainingCycles = A.getRemainingCycles();
      unsigned AUnreadyCycles = ARemainingCycles > AReadyCycles
                                    ? (ARemainingCycles - AReadyCycles)
                                    : 0;

      unsigned BReadyCycles = MixInfo.getReadyCycles(B.getType());
      unsigned BRemainingCycles = B.getRemainingCycles();
      unsigned BUnreadyCycles = BRemainingCycles > BReadyCycles
                                    ? (BRemainingCycles - BReadyCycles)
                                    : 0;

      if (AUnreadyCycles != BUnreadyCycles)
        return AUnreadyCycles > BUnreadyCycles;
    }
    // ... fallthrough to TotalCycles, size, flavor order
  });
}
```

**Key insight**: Exposed cycles take precedence over unready cycles. We always want to be able to schedule resources with exposed cycles over those without, because exposed cycles represent work that will contribute to the critical path and cause stalls. Within each group (exposed vs non-exposed), we then prefer resources with more unready cycles because enabling blocked work is valuable.

---

## The tryCriticalResourceDependency Heuristic

```cpp
bool CandidateHeuristics::tryCriticalResourceDependency(
    SchedCandidate &TryCand, SchedCandidate &Cand, SchedBoundary *Zone) const {

  auto HasPrioritySU = [this, &Cand, &TryCand](unsigned ResourceIdx) {
    const HardwareUnitInfo &HWUI = HWUInfo[ResourceIdx];
    auto CandFlavor = classifyFlavor(*Cand.SU->getInstr(), *SII);
    auto TryCandFlavor = classifyFlavor(*TryCand.SU->getInstr(), *SII);
    
    // Use LookDeep when comparing DS against WMMA
    bool LookDeep = (CandFlavor == InstructionFlavor::DS ||
                     TryCandFlavor == InstructionFlavor::DS) &&
                    HWUI.getType() == InstructionFlavor::WMMA;
    auto *TargetSU = HWUI.getNextTargetSU(LookDeep);
    return TargetSU != nullptr;
  };

  auto TryEnablesResource = [&](unsigned ResourceIdx) {
    const HardwareUnitInfo &HWUI = HWUInfo[ResourceIdx];
    auto CandFlavor = classifyFlavor(*Cand.SU->getInstr(), *SII);
    bool LookDeep = CandFlavor == InstructionFlavor::DS &&
                    HWUI.getType() == InstructionFlavor::WMMA;
    auto *TargetSU = HWUI.getNextTargetSU(LookDeep);

    // Check DAG reachability: does scheduling this candidate reduce
    // TargetSU's predecessor count?
    bool CandEnables = TargetSU != Cand.SU && 
                       DAG->IsReachable(TargetSU, Cand.SU);
    bool TryCandEnables = TargetSU != TryCand.SU && 
                          DAG->IsReachable(TargetSU, TryCand.SU);

    if (!CandEnables && !TryCandEnables)
      return false;

    // Prefer the one that enables the critical target
    if (CandEnables && !TryCandEnables) {
      Cand.Reason = GenericSchedulerBase::RegCritical;
      return true;
    }
    if (!CandEnables && TryCandEnables) {
      TryCand.Reason = GenericSchedulerBase::RegCritical;
      return true;
    }

    // Both enable: prefer higher critical path (height)
    unsigned CandHeight = Cand.SU->getHeight();
    unsigned TryCandHeight = TryCand.SU->getHeight();

    if (CandHeight > TryCandHeight) {
      Cand.Reason = GenericSchedulerBase::RegCritical;
      return true;
    }
    if (CandHeight < TryCandHeight) {
      TryCand.Reason = GenericSchedulerBase::RegCritical;
      return true;
    }

    // Same height: prefer original candidate
    Cand.Reason = GenericSchedulerBase::RegCritical;
    return true;
  };

  // Iterate through HWUIs in priority order (sorted by exposed then unready)
  for (unsigned I = 0; I < HWUInfo.size(); I++) {
    if (!HasPrioritySU(I))
      continue;
    if (TryEnablesResource(I))
      return true;
  }

  // Fallback: loop-carried DS_READ ordering
  if (tryLoopCarriedDSReadOrder(TryCand, Cand))
    return true;

  return false;
}
```

### Decision Flow

1. **Sort HWUIs by exposed cycles, then unready cycles** (via `UseDependencySort=true`)
2. **For each HWUI in priority order**:
   - Get the TargetSU (highest-priority unscheduled instruction)
   - Skip if no TargetSU exists
   - Check if either candidate enables the TargetSU
   - If only one enables: prefer that one
   - If both enable: prefer higher critical path height
3. **Fallback to loop-carried ordering** if no decision was made

**Note**: Tiebreaking is done on enablement of lower priority critical resources, with depth tiebreaking handled by a later heuristic (`tryCriticalResourcePrio`).

---

## tryLoopCarriedDSReadOrder: Cross-Iteration Fallback

When neither candidate enables an in-iteration WMMA, the heuristic falls back to ordering DS loads by when their results are consumed in the **next iteration**. This handles the software pipelining pattern where DS loads in iteration N feed WMMAs in iteration N+1.

### Loop-Carried DS_READ Detection

During region initialization before scheudling, `seedLoopCarriedDSReadDefs()` identifies DS loads whose results cross the loop back-edge:

```cpp
void CandidateHeuristics::seedLoopCarriedDSReadDefs() {
  // Two-pass approach with lane precision:
  // Pass 1: Track FirstUse, LastUse, LastDSReadDef per lane slice
  // Pass 2: Mark slices where LastUse < LastDSReadDef as loop-carried
  
  DenseMap<Register, SmallVector<SeedLaneSlice, 4>> Info;
  
  // First pass: walk all instructions, track lane-level usage
  for (const MachineInstr &MI : *MBB) {
    for (const MachineOperand &MO : MI.operands()) {
      // Track uses and DS_READ defs per lane slice
    }
  }
  
  // Second pass: per-slice loop-carried test
  for (auto &[Reg, Slices] : Info) {
    for (const SeedLaneSlice &S : Slices) {
      // Loop-carried if: all uses precede the DS_READ def
      if (*S.LastUsePos > *S.LastDSReadDefPos)
        continue;  // Has in-iter consumer
      if (*S.FirstUsePos >= *S.LastDSReadDefPos)
        continue;  // Every use is after def
      // This slice is loop-carried
      LCLanes.push_back(S.Mask);
    }
  }
}
```

**Lane Precision**: The detection works at lane / subreg granularity. A single vreg may have some lanes that are loop-carried and others that are not, tracked by `LaneBitmask`.

### Recording Next-Iteration Consumers

As instructions are scheduled, `recordLoopTopConsumption()` records when each loop-carried lane slice is first consumed:

```cpp
void CandidateHeuristics::recordLoopTopConsumption(SUnit *SU) {
  // For each use operand of SU
  for (const MachineOperand &MO : MI->uses()) {
    Register Reg = MO.getReg();
    // Check if this reg has loop-carried slices
    auto LCIt = LoopCarriedDSReadDefLanes.find(Reg);
    if (LCIt == LoopCarriedDSReadDefLanes.end())
      continue;
      
    // Record the SlotIndex of this consumer for each matching slice
    for (LaneBitmask LCMask : LCIt->second) {
      if ((LCMask & UseMask).none())
        continue;
      Recorded.emplace_back(LCMask, ConsumerSlot);
    }
  }
}
```

The `SlotIndex` reflects the scheduled position: since `ScheduleDAGMI` keeps `LiveIntervals` in sync as instructions move, the recorded slot orders consumers by their actual scheduled position.

### The tryLoopCarriedDSReadOrder Decision

```cpp
bool CandidateHeuristics::tryLoopCarriedDSReadOrder(
    SchedCandidate &TryCand, SchedCandidate &Cand) const {
  
  // Both must be DS loads
  if (!SII->isDS(*TryMI) || !TryMI->mayLoad())
    return false;
  if (!SII->isDS(*CandMI) || !CandMI->mayLoad())
    return false;

  // Look up the earliest next-iter consumer for each DS_READ
  std::optional<SlotIndex> TryPos = lookupDSReadConsumerPos(TryMI, ...);
  std::optional<SlotIndex> CandPos = lookupDSReadConsumerPos(CandMI, ...);
  
  // Both must have recorded consumers
  if (!TryPos || !CandPos)
    return false;
  if (*TryPos == *CandPos)
    return false;  // Same consumer, defer to other tie-breakers

  // Prefer the DS_READ whose result is consumed EARLIER in the next iteration
  if (SlotIndex::isEarlierInstr(*TryPos, *CandPos)) {
    TryCand.Reason = GenericSchedulerBase::RegCritical;
    return true;
  }
  Cand.Reason = GenericSchedulerBase::RegCritical;
  return true;
}
```

**Intuition**: The DS load whose result is needed soonest in the next iteration should be scheduled first. This minimizes the live range of the loaded value across the back-edge.

---

## Step-by-Step Examples

### Example 1: DS Loads Feeding Different WMMAs

**Scenario**: A GEMM inner loop with 4 DS loads feeding 2 WMMAs

```
DAG Structure:
  ds_load_0 ──┐
              ├──> wmma_0 (depth=2)
  ds_load_1 ──┘
  
  ds_load_2 ──┐
              ├──> wmma_1 (depth=2)
  ds_load_3 ──┘
```

**HardwareUnitInfo State**:
```
HWUI[WMMA]:
  PrioritySUs = {wmma_0, wmma_1}  (both depth=2)
  AllSUs = {wmma_0, wmma_1}

HWUI[DS]:
  AllSUs = {ds_load_0, ds_load_1, ds_load_2, ds_load_3}
```

**Candidate Comparison**: `ds_load_0` vs `ds_load_2`

1. `sortHWUIResources(Zone, true)`:
   - Assume both WMMAs unready → WMMA has high unready cycles
   - WMMA sorted first

2. Check HWUI[WMMA]:
   - `TargetSU = wmma_0` (first in PrioritySUs that isn't ready)
   - `LookDeep=true` (comparing DS against WMMA)
   - `DAG->IsReachable(wmma_0, ds_load_0)` → **true**
   - `DAG->IsReachable(wmma_0, ds_load_2)` → **false**

3. **Decision**: `ds_load_0` wins because it enables `wmma_0`

**After scheduling ds_load_0**:

Now compare `ds_load_1` vs `ds_load_2`:
- `TargetSU` still `wmma_0` (not ready yet, needs `ds_load_1` too)
- `ds_load_1` enables `wmma_0`, `ds_load_2` does not
- **Decision**: `ds_load_1` wins

---

### Example 2: Multiple DS Loads Enable Same WMMA

**Scenario**: Two DS loads both feed the same WMMA, need tie-breaker

```
DAG Structure:
  ds_load_A (height=5) ──┐
                         ├──> wmma_0
  ds_load_B (height=3) ──┘
```

**Candidate Comparison**: `ds_load_A` vs `ds_load_B`

1. Check HWUI[WMMA]:
   - `TargetSU = wmma_0`
   - Both enable: `IsReachable(wmma_0, ds_load_A)` = true
   - Both enable: `IsReachable(wmma_0, ds_load_B)` = true

2. Both enable → tie-break on height:
   - `ds_load_A.Height = 5`
   - `ds_load_B.Height = 3`
   - Prefer higher height (longer critical path)

3. **Decision**: `ds_load_A` wins (height=5 > height=3)

---

### Example 3: Loop-Carried DS_READ Ordering

**Scenario**: Software-pipelined GEMM where DS loads in iteration N feed WMMAs in iteration N+1

```
MBB (loop body):
  ; Iteration N consumers (of loads from iteration N-1)
  wmma_0 %v0       ; uses lanes loaded in previous iter
  wmma_1 %v1       ; uses lanes loaded in previous iter
  
  ; Iteration N producers (for iteration N+1)
  ds_read %v0      ; defines lanes for next iter
  ds_read %v1      ; defines lanes for next iter
  
  branch loop_top
```

**Detection (seedLoopCarriedDSReadDefs)**:

For `%v0`:
- `FirstUsePos = 0` (wmma_0)
- `LastUsePos = 0` (wmma_0)
- `LastDSReadDefPos = 2` (ds_read %v0)
- `LastUsePos (0) < LastDSReadDefPos (2)` → **loop-carried**

For `%v1`:
- `FirstUsePos = 1` (wmma_1)
- `LastUsePos = 1` (wmma_1)
- `LastDSReadDefPos = 3` (ds_read %v1)
- **loop-carried**

**Recording Consumption**:

As scheduling progresses (top-down), when `wmma_0` is scheduled:
```
recordLoopTopConsumption(wmma_0):
  - wmma_0 uses %v0
  - LoopTopVGPRConsumerPos[%v0] = SlotIndex(wmma_0)
```

When `wmma_1` is scheduled:
```
recordLoopTopConsumption(wmma_1):
  - wmma_1 uses %v1
  - LoopTopVGPRConsumerPos[%v1] = SlotIndex(wmma_1)
```

**Candidate Comparison**: `ds_read %v0` vs `ds_read %v1`

Neither DS load enables an in-iteration WMMA (all WMMAs already scheduled), so fall through to `tryLoopCarriedDSReadOrder`:

1. `lookupDSReadConsumerPos(ds_read %v0)` → `SlotIndex(wmma_0)`
2. `lookupDSReadConsumerPos(ds_read %v1)` → `SlotIndex(wmma_1)`
3. `wmma_0` was scheduled before `wmma_1`
4. **Decision**: `ds_read %v0` wins (its result is consumed earlier in next iter)

---

### Example 4: Dependency Sort with Ready vs Unready Cycles

**Scenario**: Comparing resources with different ready/unready profiles

```
Current State:
  HWUI[WMMA]:
    RemainingCycles = 64
    ReadyCycles = 16    (2 WMMAs ready)
    UnreadyCycles = 48  (6 WMMAs blocked)

  HWUI[DS]:
    RemainingCycles = 32
    ReadyCycles = 24    (8 DS ready)
    UnreadyCycles = 8   (2 DS blocked)

  HWUI[SALU]:
    RemainingCycles = 10
    ReadyCycles = 10    (all SALU ready)
    UnreadyCycles = 0
```

**Dependency Sort Order**:
1. **WMMA** (UnreadyCycles=48) — most blocked work
2. **DS** (UnreadyCycles=8)
3. **SALU** (UnreadyCycles=0) — nothing blocked

**Decision Impact**: `tryCriticalResourceDependency` first checks if candidates enable blocked WMMAs, then blocked DS loads, then SALU. This ensures we prioritize enabling the most constrained resources.

---

## Interaction with tryCriticalResource and tryCriticalResourcePrio

The three heuristics work together in sequence:

| Heuristic | Question | Sorting |
|-----------|----------|---------|
| `tryCriticalResource` | "Which candidate uses a higher-priority resource?" | `UseDependencySort=true` |
| `tryCriticalResourceDependency` | "Which candidate enables a higher-priority resource?" | `UseDependencySort=true` |
| `tryCriticalResourcePrio` | "Which candidate has higher HWUI priority?" | (uses existing sort) |

```cpp
// From tryCandidateCoexec()
Heurs.sortHWUIResources(Zone, true);    // Unified sort with dependency awareness
if (Heurs.tryCriticalResource(...))     // Check resource usage
  return ...;

Heurs.sortHWUIResources(Zone, true);    // Same sort
if (Heurs.tryCriticalResourceDependency(...))  // Check enablement
  return ...;

if (Heurs.tryCriticalResourcePrio(...)) // HWUI priority tiebreaking
  return ...;
```

**Key design change**: `tryCriticalResource` now only uses HWUI priority tiebreaking for WMMA instructions. For all other flavors, when both candidates use the same resource, it returns `false` and defers to `tryCriticalResourceDependency` (enablement-based) and then `tryCriticalResourcePrio` (depth-based priority).

This separation ensures:
1. **WMMA**: Agreement between `tryCriticalResource` and `tryCriticalResourceDependency` on target SU (important due to in-iteration DS→WMMA dependencies)
2. **Other flavors**: Enablement-based decisions take precedence, with depth tiebreaking handled by `tryCriticalResourcePrio`

---

## Summary

The `tryCriticalResourceDependency` heuristic implements a **dependency-aware scheduling strategy**:

1. **Target Selection**: Each HWUI tracks its highest-priority unscheduled instruction - this allows agreement between `tryCriticalResourceDependency` and `tryCriticalResource` (especially important for WMMA)
2. **Unified Sorting**: Both `tryCriticalResource` and `tryCriticalResourceDependency` use `UseDependencySort=true`, prioritizing exposed cycles over unready cycles
3. **Enablement Check**: Prefer candidates that enable critical targets via DAG reachability
4. **Loop-Carried Fallback**: For software-pipelined loops, order DS loads by next-iteration consumption
5. **Critical Path Tie-Break**: When both candidates enable the same target, prefer higher height
6. **Deferred Depth Tiebreaking**: Explicit depth-based tiebreaking is handled by `tryCriticalResourcePrio`, called after this heuristic

The key insight is that by unblocking critical resources early, the scheduler creates more scheduling flexibility and reduces register pressure from long live ranges.
