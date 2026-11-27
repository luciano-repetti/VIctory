//! The console flat ids of compute threads relabelled so that each host lane
//! is its console lane (`Lower::thread_id`).

use super::Lower;
use crate::emit::bitcode::{Attr, Bin, V, pred};
use crate::emit::shader::flags;

/// The host's group-shared memory for a compute program, in dwords (32 KiB;
/// the validator refuses more).
const GROUPSHARED_DWORDS: u32 = 32 * 1024 / 4;

impl Lower<'_> {
    /// The thread's console flat id, so that its host lane is its console
    /// lane (`Lower::thread_id`); made once, in the entry block.
    ///
    /// A 1-D group: the wave's first flat id rounded down to the wave size,
    /// plus the lane index, a bijection because hosts put a 1-D group's
    /// threads into waves in order. Other groups: a host may put threads
    /// of several rows into one wave (an RTX 5070 Ti does for 2-D groups
    /// 24, 32 or 64 wide or not a power of two wide: 16 by 2 tiles; XCL's
    /// notes report 2 by 2 quads on an RX 9070), where rounding gives two
    /// waves one base (`lane_order_tests`). There each full wave takes the
    /// next wave-sized run of ids from a group-shared ticket, and the one
    /// partial wave (the group's last threads; hosts fill its lanes from 0)
    /// the last run; where the ticket does not fit (`Lower::ticket`), the
    /// rounding as for 1-D groups.
    pub(super) fn console_flat(&mut self) -> V {
        if let Some(f) = self.console_flat {
            return f;
        }
        let [x, y, z] = self.s.threads.expect("compute");
        let e = self.s.entry;
        let i32t = self.s.t.i32;
        self.s.flags |= flags::WAVE_OPS;
        let wave = self.s.wave_size.expect("set with console lanes");
        let lane = self
            .s
            .op(e, "waveGetLaneIndex", Attr::ReadOnly, i32t, 111, &[]);
        let base = if !self.ticket() {
            let flat = self.s.op(
                e,
                "flattenedThreadIdInGroup.i32",
                Attr::ReadNone,
                i32t,
                96,
                &[],
            );
            let first = self.s.op(
                e,
                "waveReadLaneFirst.i32",
                Attr::NoUnwind,
                i32t,
                118,
                &[flat],
            );
            let mask = self.s.m.i32(!(wave - 1));
            self.s.m.bin(e, Bin::And, first, mask)
        } else {
            let one = self.s.t.i1;
            let arr = self.s.m.array(1, i32t);
            let ticket = self.s.m.global("ticket", arr, 3);
            let zero = self.s.m.i32(0);
            let p = self.s.m.element(e, ticket, zero);
            // Every thread zeroes it before any wave takes from it.
            self.s.m.store(e, p, zero);
            let mode = self.s.m.i32(1 | 8);
            let void = self.s.t.void;
            self.s
                .op(e, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
            // The wave's live lanes, from its ballot.
            let on = self.s.m.bool(true);
            let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
            let ballot = self
                .s
                .op(e, "waveActiveBallot", Attr::NoUnwind, four, 116, &[on]);
            let mut live = zero;
            for word in 0..wave.div_ceil(32) {
                let w = self.s.m.extract(e, ballot, word);
                let n = self
                    .s
                    .op(e, "unaryBits.i32", Attr::ReadNone, i32t, 31, &[w]);
                live = self.s.m.bin(e, Bin::Add, live, n);
            }
            let wk = self.s.m.i32(wave);
            let full = self.s.m.cmp(e, pred::EQ, live, wk);
            let first = self
                .s
                .op(e, "waveIsFirstLane", Attr::NoUnwind, one, 110, &[]);
            let takes = self.s.m.bin(e, Bin::And, first, full);
            let add = self.s.m.select(e, takes, wk, zero);
            let old = self.s.m.atomic(e, crate::emit::bitcode::Rmw::Add, p, add);
            let taken = self.s.op(
                e,
                "waveReadLaneFirst.i32",
                Attr::NoUnwind,
                i32t,
                118,
                &[old],
            );
            let last = self.s.m.i32((x * y * z) / wave * wave);
            self.s.m.select(e, full, taken, last)
        };
        let f = self.s.m.bin(e, Bin::Add, base, lane);
        self.console_flat = Some(f);
        f
    }

    /// Whether relabelled threads take their ids by ticket
    /// (`Lower::console_flat`): in groups that are not 1-D, unless the
    /// program's LDS already takes all the host's group-shared memory, where
    /// the ticket's dword would make it fail validation.
    // shortcut: such programs keep the rounding, and with it the ids two
    // waves share on hosts that put several rows into one wave.
    pub(super) fn ticket(&self) -> bool {
        let [_, y, z] = self.s.threads.expect("compute");
        (y, z) != (1, 1) && self.s.lds_dwords < GROUPSHARED_DWORDS
    }
}
