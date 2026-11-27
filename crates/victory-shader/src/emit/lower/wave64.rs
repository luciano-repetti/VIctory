//! Wave64 compute programs on a host without 64-lane waves (NVIDIA GPUs
//! run 32 lanes only; SM 6.6 `WaveSize(64)` then fails pipeline creation).
//!
//! A compute program that reads other lanes declares the console's wave
//! size, so that host lane n is console lane n (see `console_lanes`). On a
//! host of 32-lane waves the program declares `WaveSize(32)` instead and
//! each console wave of 64 lanes becomes two host waves. `Lower::console_flat`
//! assigns each host wave an aligned run of 32 console thread ids. The host
//! waves with ids `64w..64w+31` and `64w+32..64w+63` form console wave `w`.
//! The console lane is the host lane plus 32 in the second half.
//!
//! What stays within a half is exact as it is: DPP (rows of 16, groups of
//! 8 and 4), `ds_swizzle` (groups of 32 or 4), `v_writelane`, `v_mbcnt`,
//! a lane's bit of a mask (by its console lane) and the thread ids.
//!
//! What spans both halves goes through group-shared scratch (`wave64`, one
//! dword per thread, separate from the program's LDS): each lane stores its
//! value, the group syncs, each lane loads the lane it wants, the group
//! syncs again (so the next exchange cannot overwrite a slot before it is
//! read). That covers lane masks read as data (the 64-bit ballot: each
//! half's 32-bit ballot, the other half's read from its lane 0),
//! `v_readlane_b32` at any lane, `v_readfirstlane_b32`, `ds_bpermute_b32`
//! (any lane, and the source lane's bit of the 64-bit ballot of EXEC), and
//! whole-wave reductions and prefix sums (each half's own, combined with
//! the other half's).
//!
//! A group sync is only correct where every thread of the group arrives,
//! once: an exchange is made only in blocks every path from the entry to an
//! end passes through and no loop contains. Elsewhere it is refused (the
//! lanes of one console wave can reach a block while the rest of the group
//! does not). A group of at most 32 threads is one host wave whatever the
//! console wave: nothing is exchanged and no block is restricted.
//!
//! Refused on such hosts: an exchange outside those blocks; a group of 33
//! to 64 threads with LDS (one console wave, which the console compiler
//! leaves without barriers between its LDS writes and reads, but which is
//! two host waves here); scratch that would take the group-shared memory
//! past D3D12's 32 KiB; 16-bit whole-wave sums ([`HALF_SUM`]).

use super::super::bitcode::{Attr, Bin, V, pred};
use super::super::shader::{Shader, flags};
use super::super::wave::Op;
use super::{Lower, Parts, Why};
use crate::structure::LaneExit;

/// Group-shared memory a D3D12 compute shader may declare, in dwords.
const HOST_LDS_DWORDS: u32 = 32 * 1024 / 4;

const NOT_EVERY_THREAD: Why = "a wave64 cross-lane operation spanning both halves, on a host \
     without 64-lane waves, where not every thread of the group arrives once";

/// shortcut: a 16-bit whole-wave sum (`Op::AddH`, `wave::Shape::Half`) is
/// not joined over the halves (no program is known to need it).
pub(super) const HALF_SUM: Why =
    "a 16-bit wave reduction in a wave64 program, on a host without 64-lane waves";

/// The split of each console wave into two host waves of 32 lanes.
pub(super) struct Split {
    /// Threads in the group.
    threads: u32,
    /// Per IR block: every thread of the group runs it, once.
    uniform: Vec<bool>,
    /// Made in the entry block: 0 or 32 (the half), and the flat id of the
    /// console wave's lane 0.
    ids: Option<(V, V)>,
    /// The exchange scratch.
    scratch: Option<V>,
    /// Group syncs made.
    syncs: u32,
}

impl Split {
    #[cfg(test)]
    pub(super) fn syncs(&self) -> u32 {
        self.syncs
    }
}

/// For a compute program that declares 64-lane waves, on a host without
/// them: the host wave size it declares instead, and the split, or why it
/// cannot run there.
pub(super) fn plan(s: &mut Shader, x: &Parts, console_lanes: bool) -> (Option<Split>, Option<Why>) {
    let threads: u32 = s.threads.expect("compute").iter().product();
    if threads <= 32 {
        // One host wave of 32 holds the whole group, as lanes 0-31 of the
        // console wave (lanes 32-63 exist in neither): the ballot's high
        // dword is 0 in both, every other operation the same.
        s.wave_size = Some(32);
        return (None, None);
    }
    if s.lds_dwords > 0 && threads <= 64 {
        return (
            None,
            Some(
                "LDS shared without barriers by a wave64 group of 33 to 64 threads, on a host \
                 without 64-lane waves",
            ),
        );
    }
    if !console_lanes {
        // (A wave size is declared only for console lanes or the case above.)
        return (
            None,
            Some("a wave64 program on a host without 64-lane waves"),
        );
    }
    s.wave_size = Some(32);
    let split = Split {
        threads,
        uniform: uniform_blocks(&x.st.exits),
        ids: None,
        scratch: None,
        syncs: 0,
    };
    (Some(split), None)
}

/// Per block: whether every lane that starts the program runs it exactly
/// once, by the lanes' control flow: reachable, in no cycle, and on every
/// path from block 0 to a block that ends the program.
// shortcut: a loop stopped by the iteration limit (`LOOP_LIMIT`) leaves the
// program from inside the loop; its threads then miss the group syncs after
// it and the group hangs. Such a loop would run four billion times on the
// console.
fn uniform_blocks(exits: &[LaneExit]) -> Vec<bool> {
    let n = exits.len();
    let succ = |b: usize| -> Vec<usize> {
        match exits[b] {
            LaneExit::Jump(t) => vec![t],
            LaneExit::Branch {
                taken, not_taken, ..
            } => vec![taken, not_taken],
            LaneExit::End | LaneExit::Indirect => Vec::new(),
        }
    };
    // Blocks reached from `from` without passing `skip`.
    let reach = |from: Vec<usize>, skip: Option<usize>| -> Vec<bool> {
        let mut seen = vec![false; n];
        let mut todo = from;
        while let Some(b) = todo.pop() {
            if Some(b) == skip || seen[b] {
                continue;
            }
            seen[b] = true;
            todo.extend(succ(b));
        }
        seen
    };
    let live = reach(vec![0], None);
    (0..n)
        .map(|b| {
            if !live[b] || reach(succ(b), None)[b] {
                return false;
            }
            let around = reach(vec![0], Some(b));
            b == 0 || !(0..n).any(|e| around[e] && succ(e).is_empty())
        })
        .collect()
}

impl Lower<'_> {
    /// The console lane, made in block `at`.
    pub(super) fn console_lane(&mut self, at: usize) -> V {
        let lane = self.host_lane(at);
        if self.split.is_none() {
            return lane;
        }
        let (half, _) = self.split_ids();
        self.s.m.bin(at, Bin::Or, lane, half)
    }

    /// The host lane of console lane `lane` in this lane's half (DPP and
    /// swizzles never leave it); `lane` itself without a split.
    pub(super) fn half_local(&mut self, lane: V) -> V {
        if self.split.is_none() {
            return lane;
        }
        let k = self.s.m.i32(31);
        self.s.m.bin(self.cur, Bin::And, lane, k)
    }

    /// The half (0 or 32) and the flat id of the console wave's lane 0, made
    /// in the entry block (where every lane of the host wave is active).
    pub(super) fn split_ids(&mut self) -> (V, V) {
        let split = self.split.as_ref().expect("a split program");
        if let Some(ids) = split.ids {
            return ids;
        }
        // From the thread's console flat id (`Lower::console_flat`): each
        // host wave holds one 32-aligned run of them, however the host laid
        // the group out (an RTX 5070 Ti puts a 32x4 group's threads in 16 by
        // 2 tiles, so the host's own flat ids would pair the wrong waves).
        let f = self.console_flat();
        let e = self.s.entry;
        let (k32, not63) = (self.s.m.i32(32), self.s.m.i32(!63));
        let half = self.s.m.bin(e, Bin::And, f, k32);
        let base = self.s.m.bin(e, Bin::And, f, not63);
        let ids = (half, base);
        self.split.as_mut().expect("a split program").ids = Some(ids);
        ids
    }

    /// Fails unless every thread of the group runs IR block `block` once
    /// (none: the entry block, before block 0).
    fn every_thread(&self, block: Option<usize>) -> Result<(), Why> {
        let split = self.split.as_ref().expect("a split program");
        match block {
            Some(b) if !split.uniform[b] => Err(NOT_EVERY_THREAD),
            _ => Ok(()),
        }
    }

    /// The scratch array, one dword per console lane of the group.
    fn scratch(&mut self) -> Result<V, Why> {
        let split = self.split.as_ref().expect("a split program");
        if let Some(g) = split.scratch {
            return Ok(g);
        }
        let dwords = split.threads.next_multiple_of(64);
        if self.s.lds_dwords + dwords > HOST_LDS_DWORDS {
            return Err("a wave64 program whose LDS leaves no room for exchanging \
                 values between the halves of its waves (host waves of 32 lanes)");
        }
        let arr = self.s.m.array(dwords, self.s.t.i32);
        let g = self.s.m.global("wave64", arr, 3);
        self.split.as_mut().expect("a split program").scratch = Some(g);
        Ok(g)
    }

    /// Thread-group sync with a shared-memory fence, in block `at`.
    fn group_sync(&mut self, at: usize) {
        let mode = self.s.m.i32(1 | 8);
        let void = self.s.t.void;
        self.s
            .op(at, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
        self.split.as_mut().expect("a split program").syncs += 1;
    }

    /// `x` (an i32) of console lane `lane` (0-63) of this lane's console
    /// wave, in block `at`, which every thread runs once.
    fn exchange(&mut self, at: usize, x: V, lane: V) -> Result<V, Why> {
        let g = self.scratch()?;
        let (_, base) = self.split_ids();
        let me = self.console_lane(at);
        let mine = self.s.m.bin(at, Bin::Add, base, me);
        let p = self.s.m.element(at, g, mine);
        self.s.m.store(at, p, x);
        self.group_sync(at);
        let k63 = self.s.m.i32(63);
        let lane = self.s.m.bin(at, Bin::And, lane, k63);
        let theirs = self.s.m.bin(at, Bin::Add, base, lane);
        let p = self.s.m.element(at, g, theirs);
        let r = self.s.m.load(at, p);
        self.group_sync(at);
        Ok(r)
    }

    /// `x`, the same in every lane of this half, as the other half has it,
    /// and whether the other half exists (the group's last console wave
    /// may lack its second half).
    fn other_half(&mut self, at: usize, x: V) -> Result<(V, V), Why> {
        let (half, base) = self.split_ids();
        let k32 = self.s.m.i32(32);
        let other = self.s.m.bin(at, Bin::Xor, half, k32);
        let r = self.exchange(at, x, other)?;
        let first = self.s.m.bin(at, Bin::Add, base, other);
        let threads = self.split.as_ref().expect("a split program").threads;
        let n = self.s.m.i32(threads);
        let exists = self.s.m.cmp(at, pred::IULT, first, n);
        Ok((r, exists))
    }

    /// In the low half: (own, other); in the high half: (other, own).
    fn in_order(&mut self, at: usize, own: V, other: V) -> (V, V) {
        let (half, _) = self.split_ids();
        let z = self.s.m.i32(0);
        let low = self.s.m.cmp(at, pred::EQ, half, z);
        let lo = self.s.m.select(at, low, own, other);
        let hi = self.s.m.select(at, low, other, own);
        (lo, hi)
    }

    /// The ballot of `bit` over the console wave as (low, high) dwords, in
    /// block `at` (of IR block `block`, none: the entry).
    pub(super) fn ballot(
        &mut self,
        at: usize,
        block: Option<usize>,
        bit: V,
    ) -> Result<(V, V), Why> {
        let i32t = self.s.t.i32;
        let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
        let r = self
            .s
            .op(at, "waveActiveBallot", Attr::NoUnwind, four, 116, &[bit]);
        self.s.flags |= flags::WAVE_OPS;
        let (x, y) = (self.s.m.extract(at, r, 0), self.s.m.extract(at, r, 1));
        if self.split.is_none() {
            return Ok((x, y));
        }
        // Host waves of 32 lanes: `x` is this half's dword.
        self.every_thread(block)?;
        let (other, exists) = self.other_half(at, x)?;
        let z = self.s.m.i32(0);
        let other = self.s.m.select(at, exists, other, z);
        Ok(self.in_order(at, x, other))
    }

    /// `v_readlane_b32`: `x` of console lane `lane`.
    pub(super) fn split_read_lane(&mut self, x: V, lane: V) -> Result<V, Why> {
        self.every_thread(Some(self.cur_ir))?;
        self.exchange(self.cur, x, lane)
    }

    /// `v_readfirstlane_b32`: `x` of the first lane in EXEC. Unguarded,
    /// EXEC is the lanes that got here: the host's first active lane (in a
    /// split, lane 0 of the console wave, as every lane gets here). Under
    /// a guard, the first lane where it holds, from its ballot (the host's
    /// first active lane could be one outside EXEC). With no lane in EXEC
    /// the hardware reads lane 0.
    pub(super) fn first_lane(&mut self, x: V, guard: Option<V>) -> Result<V, Why> {
        let c = self.cur;
        let first = match guard {
            None if self.split.is_none() => {
                self.s.flags |= flags::WAVE_OPS;
                let i32t = self.s.t.i32;
                return Ok(self
                    .s
                    .op(c, "waveReadLaneFirst.i32", Attr::NoUnwind, i32t, 118, &[x]));
            }
            None => self.s.m.i32(0),
            Some(g) => {
                let (lo, hi) = self.ballot(c, Some(self.cur_ir), g)?;
                let i32t = self.s.t.i32;
                let flo = self
                    .s
                    .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[lo]);
                let fhi = self
                    .s
                    .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[hi]);
                let (z, k32) = (self.s.m.i32(0), self.s.m.i32(32));
                let fhi = self.s.m.bin(c, Bin::Add, fhi, k32);
                let in_hi = self.s.m.cmp(c, pred::NE, hi, z);
                let fhi = self.s.m.select(c, in_hi, fhi, z);
                let in_lo = self.s.m.cmp(c, pred::NE, lo, z);
                self.s.m.select(c, in_lo, flo, fhi)
            }
        };
        if self.split.is_none() {
            return Ok(self.read_lane(x, first));
        }
        self.every_thread(Some(self.cur_ir))?;
        self.exchange(c, x, first)
    }

    /// A whole-wave reduction (or a prefix sum, `prefix`) `r` over this
    /// half's lanes of `arg`, made over the whole console wave.
    pub(super) fn split_reduce(&mut self, op: Op, prefix: bool, arg: V, r: V) -> Result<V, Why> {
        if op == Op::AddH {
            // (16-bit sums are refused in a split before, in `shape`.)
            return Err(HALF_SUM);
        }
        self.every_thread(Some(self.cur_ir))?;
        let c = self.cur;
        let float = matches!(op, Op::AddF | Op::MinF | Op::MaxF);
        if prefix {
            // The high half adds the low half's total.
            let (ty, sfx) = if float {
                (self.s.t.f32, "f32")
            } else {
                (self.s.t.i32, "i32")
            };
            let (k, u) = (self.s.m.i8(0), self.s.m.i8(u8::from(!float)));
            let total = self.s.op(
                c,
                &format!("waveActiveOp.{sfx}"),
                Attr::NoUnwind,
                ty,
                119,
                &[arg, k, u],
            );
            let total = if float { self.as_int(total) } else { total };
            let (low_total, _) = self.other_half(c, total)?;
            let low_total = if float {
                self.as_float(low_total)
            } else {
                low_total
            };
            let (half, _) = self.split_ids();
            let z = self.s.m.i32(0);
            let high = self.s.m.cmp(c, pred::NE, half, z);
            let sum = self.s.m.bin(c, Bin::Add, low_total, r);
            return Ok(self.s.m.select(c, high, sum, r));
        }
        let own = if float { self.as_int(r) } else { r };
        let (other, exists) = self.other_half(c, own)?;
        let other = if float { self.as_float(other) } else { other };
        let (lo, hi) = self.in_order(c, r, other);
        let both = match op {
            Op::AddU | Op::AddF => self.s.m.bin(c, Bin::Add, lo, hi),
            Op::And => self.s.m.bin(c, Bin::And, lo, hi),
            Op::Or => self.s.m.bin(c, Bin::Or, lo, hi),
            Op::Xor => self.s.m.bin(c, Bin::Xor, lo, hi),
            Op::MinF | Op::MaxF => self.max_min(op == Op::MaxF, lo, hi),
            Op::AddH => return Err(HALF_SUM),
            Op::MinU | Op::MaxU | Op::MinI | Op::MaxI => {
                let p = match op {
                    Op::MinU => pred::IULT,
                    Op::MaxU => pred::IUGT,
                    Op::MinI => pred::SLT,
                    _ => pred::SGT,
                };
                let keep = self.s.m.cmp(c, p, lo, hi);
                self.s.m.select(c, keep, lo, hi)
            }
        };
        Ok(self.s.m.select(c, exists, both, r))
    }
}
