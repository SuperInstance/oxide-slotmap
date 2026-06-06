# oxide-slotmap

> Ternary-state GPU resource allocation with generational safety and defragmentation.

## Background Theory

GPU resource allocation is usually framed as a binary problem: a region of memory is either allocated or free. But in a heterogeneous, multi-tenant GPU fleet, that binary model hides important transitional states. A kernel may reserve memory before it is fully committed. A slot may be held for a preflight check. A region may be allocated to one owner but pending migration to another.

`oxide-slotmap` introduces a **ternary state model**:

- `Allocated = +1` — The slot is owned and in use.
- `Reserved = 0` — The slot is held but not yet active.
- `Free = -1` — The slot is available for allocation.

This ternary model maps naturally onto the SuperInstance worldview, where `{-1, 0, +1}` represents not just numeric values but qualitative states: absent, potential, actual.

A second theoretical commitment is **generational safety**. SlotMap keys carry a generation counter. When a slot is deallocated, its generation increments, invalidating any stale keys held elsewhere. This prevents the classic use-after-free bug in resource allocators without requiring centralized reference counting.

## How It Works

`oxide-slotmap` centers on `OxideSlotMap`, a fixed-capacity allocator backed by:

- A `Vec<Slot>` storing state, generation, and owner for each slot.
- A `free_list` stack tracking indices available for allocation.

### Allocation Path

1. Pop an index from `free_list`.
2. Set the slot's state to `Allocated` (or `Reserved` for `reserve()`).
3. Record the owner.
4. Return a `SlotKey { index, generation }`.

### Deallocation Path

1. Validate that the key's generation matches the slot.
2. Validate that the slot is not already `Free`.
3. Increment generation, set state to `Free`, clear owner.
4. Push index back onto `free_list`.

### Defragmentation

Over time, allocate/free cycles create holes. `defragment()` compacts live slots to the front of the backing vector and rebuilds the free list from the remaining capacity. It returns the number of slots moved, giving callers a measurable fragmentation cost.

### Bulk Operations

`bulk_allocate(owner, count)` efficiently reserves many slots for workloads that know their resource needs upfront — a common case when reserving GPU tensor regions for a batched inference job.

## Experiments

The test suite encodes the following claims:

```rust
#[test]
fn test_generation_mismatch() {
    // Stale keys return None, proving generational safety.
}

#[test]
fn test_defragment() {
    // After fragmentation, compaction moves live slots to the front
    // and restores the free list.
}

#[test]
fn test_exhaustion() {
    // Allocation returns None when capacity is saturated.
}
```

A larger experiment: simulate a 10,000-slot allocator under a random walk of allocate/free operations. Measure:

- Average allocation latency (expected O(1)).
- Fragmentation ratio over time.
- Defragmentation cost vs. live slot count.
- Generation collision probability over 1 billion operations.

## Applications

- **GPU tensor region tracking**: Map logical tensor slots to GPU memory without leaking regions across kernel boundaries.
- **Multi-tenant GPU sharing**: Allocate/reserve slots for different workloads before they are scheduled.
- **Construct caching**: `oxide-constructs` can use slot keys to track which compiled kernels are resident on which GPUs.
- **Batch inference reservations**: Reserve slots for a batch, activate them (`Reserved → Allocated`) when inputs arrive, and free atomically when outputs are returned.
- **Fleet resource accounting**: Aggregate slot counts across agents in `oxide-fleet` to produce fleet-wide availability metrics.

## Open Questions

1. **Capacity resizing**: Should `OxideSlotMap` support dynamic capacity growth, or is fixed capacity a feature that forces explicit capacity planning?
2. **Reservation timeouts**: Reserved slots can be held indefinitely. Should the map enforce reservation TTLs to prevent silent resource starvation?
3. **NUMA-aware allocation**: On multi-GPU nodes, should slot indices encode locality hints, or should locality be handled by a higher layer?
4. **Generational width**: A 32-bit generation is effectively unbounded, but do embedded GPU controllers need smaller key encodings?

## Cross-Links

- [SuperInstance agent-knowledge / TERNARY-NUMBERS.md](https://github.com/SuperInstance/agent-knowledge/blob/main/TERNARY-NUMBERS.md) — The ternary philosophy underlying `Allocated/Reserved/Free`.
- [SuperInstance agent-knowledge / CONSERVATION-LAWS.md](https://github.com/SuperInstance/agent-knowledge/blob/main/CONSERVATION-LAWS.md) — Conservation of allocated + reserved + free = capacity.
- [SuperInstance agent-knowledge / GPU-AS-MOTOR-CORTEX.md](https://github.com/SuperInstance/agent-knowledge/blob/main/GPU-AS-MOTOR-CORTEX.md) — Why fine-grained GPU resource tracking matters.
- `oxide-fleet` — Uses slot counts in workload metrics and resource discovery.
- `oxide-constructs` — Tracks compiled construct PTX in GPU memory slots.

## Quick Start

```rust
use oxide_slotmap::{OxideSlotMap, SlotState};

let mut sm = OxideSlotMap::new(1024);

let key = sm.allocate("attention-kernel-42").unwrap();
assert_eq!(sm.get_state(key), Some(SlotState::Allocated));
assert_eq!(sm.get_owner(key), Some("attention-kernel-42"));

let reserved = sm.reserve("preflight-check").unwrap();
assert_eq!(sm.get_state(reserved), Some(SlotState::Reserved));

// Generational safety: deallocating invalidates the key.
sm.deallocate(key);
assert_eq!(sm.get_state(key), None);

// Compact live slots after fragmentation.
let moved = sm.defragment();
println!("Defragmentation moved {} slots", moved);
```
