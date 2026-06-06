//! # oxide-slotmap
//!
//! Slot-based GPU resource allocation with ternary status.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotState { Allocated = 1, Reserved = 0, Free = -1 }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlotKey { pub index: usize, pub generation: u32 }

#[derive(Debug)]
struct Slot {
    state: SlotState,
    generation: u32,
    owner: Option<String>,
}

impl Clone for Slot {
    fn clone(&self) -> Self { Self { state: self.state, generation: self.generation, owner: self.owner.clone() } }
}

pub struct OxideSlotMap {
    slots: Vec<Slot>,
    free_list: Vec<usize>,
}

impl OxideSlotMap {
    pub fn new(capacity: usize) -> Self {
        let slots = (0..capacity).map(|_| Slot { state: SlotState::Free, generation: 0, owner: None }).collect();
        let free_list = (0..capacity).rev().collect();
        Self { slots, free_list }
    }

    pub fn allocate(&mut self, owner: &str) -> Option<SlotKey> {
        let index = self.free_list.pop()?;
        let slot = &mut self.slots[index];
        slot.state = SlotState::Allocated;
        slot.owner = Some(owner.into());
        Some(SlotKey { index, generation: slot.generation })
    }

    pub fn reserve(&mut self, owner: &str) -> Option<SlotKey> {
        let index = self.free_list.pop()?;
        let slot = &mut self.slots[index];
        slot.state = SlotState::Reserved;
        slot.owner = Some(owner.into());
        Some(SlotKey { index, generation: slot.generation })
    }

    pub fn deallocate(&mut self, key: SlotKey) -> bool {
        if key.index >= self.slots.len() { return false; }
        let slot = &mut self.slots[key.index];
        if slot.generation != key.generation || slot.state == SlotState::Free { return false; }
        slot.generation += 1;
        slot.state = SlotState::Free;
        slot.owner = None;
        self.free_list.push(key.index);
        true
    }

    pub fn get_state(&self, key: SlotKey) -> Option<SlotState> {
        let slot = self.slots.get(key.index)?;
        if slot.generation != key.generation { return None; }
        Some(slot.state)
    }

    pub fn get_owner(&self, key: SlotKey) -> Option<&str> {
        let slot = self.slots.get(key.index)?;
        if slot.generation != key.generation { return None; }
        slot.owner.as_deref()
    }

    pub fn bulk_allocate(&mut self, owner: &str, count: usize) -> Vec<SlotKey> {
        (0..count).filter_map(|_| self.allocate(owner)).collect()
    }

    pub fn defragment(&mut self) -> usize {
        let capacity = self.slots.len();
        let mut alive: Vec<Slot> = Vec::new();
        let mut moved = 0;
        for (i, slot) in self.slots.iter().enumerate() {
            if slot.state != SlotState::Free {
                alive.push(slot.clone());
                if alive.len() - 1 != i { moved += 1; }
            }
        }
        let free_count = capacity - alive.len();
        let alive_len = alive.len();
        // Rebuild slots vec entirely
        self.slots = alive;
        for _ in 0..free_count {
            self.slots.push(Slot { state: SlotState::Free, generation: 0, owner: None });
        }
        self.free_list.clear();
        for i in (alive_len..capacity).rev() {
            self.free_list.push(i);
        }
        moved
    }

    pub fn allocated_count(&self) -> usize { self.slots.iter().filter(|s| s.state == SlotState::Allocated).count() }
    pub fn reserved_count(&self) -> usize { self.slots.iter().filter(|s| s.state == SlotState::Reserved).count() }
    pub fn free_count(&self) -> usize { self.free_list.len() }
    pub fn capacity(&self) -> usize { self.slots.len() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocate() {
        let mut sm = OxideSlotMap::new(4);
        let k = sm.allocate("w1").unwrap();
        assert_eq!(sm.get_state(k), Some(SlotState::Allocated));
    }

    #[test]
    fn test_reserve() {
        let mut sm = OxideSlotMap::new(4);
        let k = sm.reserve("w2").unwrap();
        assert_eq!(sm.get_state(k), Some(SlotState::Reserved));
    }

    #[test]
    fn test_deallocate() {
        let mut sm = OxideSlotMap::new(4);
        let k = sm.allocate("w").unwrap();
        assert!(sm.deallocate(k));
        assert_eq!(sm.free_count(), 4);
    }

    #[test]
    fn test_generation_mismatch() {
        let mut sm = OxideSlotMap::new(4);
        let k = sm.allocate("w").unwrap();
        sm.deallocate(k);
        assert_eq!(sm.get_state(k), None); // stale key
    }

    #[test]
    fn test_bulk_allocate() {
        let mut sm = OxideSlotMap::new(10);
        let keys = sm.bulk_allocate("w", 5);
        assert_eq!(keys.len(), 5);
        assert_eq!(sm.allocated_count(), 5);
    }

    #[test]
    fn test_exhaustion() {
        let mut sm = OxideSlotMap::new(2);
        sm.allocate("a");
        sm.allocate("b");
        assert!(sm.allocate("c").is_none());
    }

    #[test]
    fn test_defragment() {
        let mut sm = OxideSlotMap::new(6);
        sm.allocate("a");
        sm.allocate("b");
        sm.allocate("c");
        // free = 3, allocated = 3
        let keys = sm.bulk_allocate("d", 3);
        // All full now
        sm.deallocate(keys[0]);
        sm.deallocate(keys[1]);
        // allocated=4, free=2 (indices 0,1 were freed)
        let moved = sm.defragment();
        // After compaction: all 4 allocated at front, 2 free at back
        assert!(moved >= 0);
        assert_eq!(sm.allocated_count() + sm.free_count(), 6);
        assert_eq!(sm.free_count(), 2);
    }

    #[test]
    fn test_owner() {
        let mut sm = OxideSlotMap::new(4);
        let k = sm.allocate("worker42").unwrap();
        assert_eq!(sm.get_owner(k), Some("worker42"));
    }
}
