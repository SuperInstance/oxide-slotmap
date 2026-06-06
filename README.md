# oxide-slotmap

> A narrative, ternary-state slot allocator for GPU resources — with generational safety and online defragmentation.

---

## Why another allocator?

Every runtime that touches a GPU eventually faces the same dilemma: memory and compute slots are finite, shared, and reshuffled constantly across kernels, batches, and tenants. Traditional allocators force the world into a binary story — a region is either allocated or free. That model is clean on paper, but it erases the messy, important middle of real GPU workloads.

A slot might be **promised** to an upcoming kernel before its input tensors have arrived. It might be **held** for a preflight safety check. It might be **pinned** while migrating from one owner to another. Binary allocators cannot express these states, so programmers hide them in out-of-band booleans, ad-hoc reservation tables, and fragile state machines scattered across the codebase.

`oxide-slotmap` tells a richer story. It gives you three states instead of two, generational keys instead of raw pointers, and a single compact data structure that keeps the entire allocation narrative in one place. It is a foundational building block of the [SuperInstance](https://github.com/SuperInstance/SuperInstance) agentic runtime, where fine-grained resource tracking is not an optimization — it is the core abstraction that lets agents share GPUs safely.

---

## The ternary worldview

The design starts with a simple observation: many natural processes do not jump directly from *absent* to *actual*. They pass through *potential* first.

`oxide-slotmap` encodes this in three explicit states:

| State      | Value | Meaning                                    |
|------------|-------|--------------------------------------------|
| `Allocated`| `+1`  | The slot is owned and actively in use.     |
| `Reserved` | `0`   | The slot is held but not yet activated.    |
| `Free`     | `-1`  | The slot is available for future use.      |

This `{-1, 0, +1}` vocabulary maps directly onto the [SuperInstance](https://github.com/SuperInstance/SuperInstance) worldview: negative is absence, zero is potential, and positive is actuality. A reserved slot is not a bug to be papered over; it is a first-class citizen of the allocator. You can reserve capacity for a batch inference job, run preflight validation, and then promote the slot to allocated when the kernels are ready — all without risking a double-booking or a use-after-free.

---

## Generational safety

The most dangerous bug in a slot allocator is not running out of memory. It is using a stale key.

Imagine thread A receives `SlotKey { index: 7, ... }` for a tensor region. Thread A does some work, context-switches, and meanwhile thread B frees slot 7 and reallocates it to a different kernel. When thread A wakes up and touches index 7, it is now corrupting someone else's memory. This is the classic ABA problem, and it is especially painful on GPUs where traditional locking is expensive.

`oxide-slotmap` solves this with generational counters. Each `SlotKey` carries both an `index` and a `generation`. When a slot is deallocated, its generation increments. Any existing key now carries an outdated generation and is rejected on the next lookup. The result is a cheap, lock-free safety guarantee: **stale keys simply stop working**.

```rust
use oxide_slotmap::OxideSlotMap;

let mut sm = OxideSlotMap::new(1024);
let key = sm.allocate("attention-kernel-42").unwrap();

sm.deallocate(key);

// The old key is now a ghost. It cannot resurrect the slot.
assert_eq!(sm.get_state(key), None);
```

No reference counters. No centralized locks. Just a `u32` that silently invalidates the past.

---

## How it works

`OxideSlotMap` is a fixed-capacity allocator backed by two contiguous structures:

1. A `Vec<Slot>` stores, for every index, the current state, generation, and optional owner label.
2. A `free_list` stack tracks which indices are available for the next allocation.

### Allocation

1. Pop an index from `free_list`.
2. Mark the slot as `Allocated` (or `Reserved` if you called `reserve()`).
3. Record the owner label.
4. Return a `SlotKey { index, generation }`.

Allocation is amortized *O(1)* because it is just a `Vec::pop` and a few field writes.

### Deallocation

1. Verify the key's generation matches the live slot.
2. Verify the slot is not already `Free`.
3. Increment the generation, clear the owner, and mark the slot `Free`.
4. Push the index back onto `free_list`.

If the generation does not match, or the slot is already free, the operation returns `false` instead of corrupting state.

### Defragmentation

Over time, allocate-free cycles punch holes in the backing vector. Holes hurt cache locality and complicate bulk transfers. The `defragment()` method compacts live slots to the front of the vector, rebuilds the free list at the back, and returns the number of slots that moved.

```rust
let mut sm = OxideSlotMap::new(1024);
sm.bulk_allocate("prefill-batch", 256);
sm.bulk_allocate("decode-batch", 128);

// ... later, after freeing and reallocating many times ...

let moved = sm.defragment();
println!("Compacted {} slots to the front", moved);
```

Compaction is not free — it is linear in capacity — but it gives you a measurable, tunable cost that you can schedule between kernel launches.

### Bulk operations

`bulk_allocate(owner, count)` gives you a `Vec<SlotKey>` in one call. This is the natural primitive for batched GPU work: reserve all the slots you need for an inference job up front, then hand the keys to the kernel scheduler.

---

## From theory to practice

The ideas in this crate are intentionally small and composable. You can use `oxide-slotmap` as a standalone allocator, or you can embed it inside larger systems:

- **GPU tensor region tracking.** Map logical tensor slots to GPU memory regions without leaking across kernel boundaries.
- **Multi-tenant sharing.** Allocate and reserve slots for different workloads before they are scheduled onto the device.
- **Construct caching.** Track which compiled kernels are resident on which GPUs using stable, generational keys.
- **Batch inference reservations.** Reserve slots for a batch, promote them when input tensors arrive, and free them atomically when outputs are returned.
- **Fleet resource accounting.** Aggregate slot counts across many agents to produce fleet-wide availability metrics.

For the broader philosophy that connects ternary states, conservation laws, and agentic GPU runtimes, see the [SuperInstance](https://github.com/SuperInstance/SuperInstance) project.

---

## Quick start

Add `oxide-slotmap` to your `Cargo.toml`:

```toml
[dependencies]
oxide-slotmap = "0.1"
```

Then allocate, reserve, and compact:

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

Run the test suite to see the invariants in action:

```bash
cargo test
```

---

## Design decisions and open questions

- **Fixed capacity.** `OxideSlotMap` is created with a fixed capacity. This is either a bug or a feature depending on your worldview. It forces explicit capacity planning and keeps the implementation simple and allocation-free on the hot path.
- **Reservation timeouts.** Reserved slots can be held indefinitely. A production system may want to layer reservation TTLs on top to prevent silent starvation.
- **Generational width.** A 32-bit generation is effectively unbounded for most workloads, but embedded GPU controllers may prefer more compact encodings.
- **Thread safety.** The current implementation is single-threaded. A multi-threaded workload should wrap `OxideSlotMap` in a mutex or split ownership across shards.

---

## Cross-links

- [SuperInstance](https://github.com/SuperInstance/SuperInstance) — The agentic runtime that uses `oxide-slotmap` as a GPU resource primitive.
- [SuperInstance agent-knowledge / TERNARY-NUMBERS.md](https://github.com/SuperInstance/agent-knowledge/blob/main/TERNARY-NUMBERS.md) — The ternary philosophy behind `Allocated/Reserved/Free`.
- [SuperInstance agent-knowledge / CONSERVATION-LAWS.md](https://github.com/SuperInstance/agent-knowledge/blob/main/CONSERVATION-LAWS.md) — Conservation of allocated + reserved + free = capacity.
- [SuperInstance agent-knowledge / GPU-AS-MOTOR-CORTEX.md](https://github.com/SuperInstance/agent-knowledge/blob/main/GPU-AS-MOTOR-CORTEX.md) — Why fine-grained GPU resource tracking matters.

---

## License

Licensed under the Apache License, Version 2.0. See `Cargo.toml` or the source headers for details.
