//! How the host GPU puts a group's threads into waves.
//!
//! A translated compute program that reads other lanes relabels each
//! thread so that its host lane is its console lane (`Lower::thread_id`):
//! its flat id becomes its wave's first flat id, rounded down to the wave
//! size, plus its lane index. That is a bijection only while each host wave
//! holds one wave-aligned run of flat ids, as the console's waves do. XCL's
//! notes report hosts where that fails (3-D groups on NVIDIA, 2-D and 3-D
//! groups in 2x2 quads on an RX 9070). This probe records, per thread of
//! one group, its flat id, lane, wave size and its wave's first flat id,
//! for group shapes compute programs use, and checks both properties.
//!
//! The probe is plain HLSL, compiled by the Windows SDK's `dxc.exe`
//! (`VICTORY_PC_DXC`, else the newest SDK's); skipped without it or a
//! device.

use std::collections::BTreeMap;
use std::path::PathBuf;

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, container};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};

fn dxc() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("VICTORY_PC_DXC") {
        return Some(p.into());
    }
    let kits = PathBuf::from(r"C:\Program Files (x86)\Windows Kits\10\bin");
    let mut found: Vec<PathBuf> = std::fs::read_dir(&kits)
        .ok()?
        .filter_map(Result::ok)
        .map(|e| e.path().join(r"x64\dxc.exe"))
        .filter(|p| p.exists())
        .collect();
    found.sort();
    found.pop()
}

/// The probe for a group of `[x, y, z]` threads, its container signed.
fn probe(
    dxc: &std::path::Path,
    dir: &std::path::Path,
    [x, y, z]: [u32; 3],
    wave: Option<u32>,
) -> Vec<u8> {
    let size = wave.map_or(String::new(), |w| format!("[WaveSize({w})]"));
    let src = format!(
        r#"
{size}
[numthreads({x}, {y}, {z})]
void main(uint flat : SV_GroupIndex) {{
    RWByteAddressBuffer o = ResourceDescriptorHeap[0];
    o.Store4(flat * 16, uint4(flat, WaveGetLaneIndex(), WaveGetLaneCount(), WaveReadLaneFirst(flat)));
}}
"#
    );
    let (input, output) = (dir.join("probe.hlsl"), dir.join("probe.bin"));
    std::fs::write(&input, src).unwrap();
    let out = std::process::Command::new(dxc)
        .args(["-T", "cs_6_6", "-E", "main", "-Fo"])
        .arg(&output)
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let mut dxil = std::fs::read(output).unwrap();
    // Without the SDK's validator the container is unsigned: sign it.
    let digest = container::hash(&dxil[20..]);
    dxil[4..20].copy_from_slice(&digest);
    dxil
}

/// What one group's threads recorded: per flat id, (lane, wave size, the
/// wave's first flat id).
fn run(host: &Host, dxil: Vec<u8>, threads: u32) -> Result<Vec<(u32, u32, u32)>, String> {
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil,
        user_data: Vec::new(),
        samplers: Vec::new(),
        static_samplers: Vec::new(),
        root_cbvs: Vec::new(),
        root_srvs: Vec::new(),
        root_uavs: Vec::new(),
        memory: false,
        memory_writes: false,
        view_addresses: false,
        heap: false,
        heap_writes: false,
        view_marks: false,
    };
    let (root, pso) = host.compute_pipeline(&shader)?;
    let out = host.dispatch(
        &root,
        &pso,
        &Dispatch {
            roots: vec![(0, Root::Constants(vec![0]))],
            views: &[View::RawUav(threads * 4)],
            samplers: &[],
            groups: [1, 1, 1],
        },
    )?;
    Ok((0..threads as usize)
        .map(|t| {
            let r = &out[0][t * 4..t * 4 + 4];
            assert_eq!(r[0], t as u32, "thread {t} wrote its own record");
            (r[1], r[2], r[3])
        })
        .collect())
}

/// Per shape: whether every host wave is one wave-aligned run of flat ids,
/// whether rounding each wave's first flat id down to the wave size (the
/// relabelling of 1-D groups) gives every thread its own id, and the
/// waves that are not full and the lanes they use.
#[derive(Debug, PartialEq)]
struct Finding {
    shape: [u32; 3],
    wave: u32,
    aligned_runs: bool,
    relabel_bijective: bool,
    /// Flat ids of the first wave that is not an aligned run, if any.
    example: Option<Vec<u32>>,
    /// Waves that are not full: (members, whether their lanes run from 0).
    partial: Vec<(u32, bool)>,
}

fn examine(shape: [u32; 3], records: &[(u32, u32, u32)]) -> Finding {
    let wave = records[0].1;
    let mut waves: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    for (flat, &(_, _, first)) in records.iter().enumerate() {
        waves.entry(first).or_default().push(flat as u32);
    }
    let mut example = None;
    for members in waves.values() {
        let base = members[0] & !(wave - 1);
        let run = members
            .iter()
            .copied()
            .eq(base..base + members.len() as u32);
        if !run && example.is_none() {
            example = Some(members.clone());
        }
    }
    // The relabelled id: the wave's first flat id rounded down, plus the lane.
    let mut ids: Vec<u32> = records
        .iter()
        .map(|&(lane, _, first)| (first & !(wave - 1)) + lane)
        .collect();
    ids.sort_unstable();
    let relabel_bijective = ids.iter().copied().eq(0..records.len() as u32);
    let mut lanes: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    for &(lane, _, first) in records {
        lanes.entry(first).or_default().push(lane);
    }
    let partial = lanes
        .values_mut()
        .filter(|l| l.len() < wave as usize)
        .map(|l| {
            l.sort_unstable();
            (l.len() as u32, l.iter().copied().eq(0..l.len() as u32))
        })
        .collect();
    Finding {
        shape,
        wave,
        aligned_runs: example.is_none(),
        relabel_bijective,
        example,
        partial,
    }
}

/// Group shapes compute programs use: 1-D, 2-D tiles, 3-D blocks, and
/// sizes that are not powers of two.
const SHAPES: [[u32; 3]; 24] = [
    [32, 4, 1],
    [32, 8, 1],
    [64, 2, 1],
    [64, 4, 1],
    [16, 4, 1],
    [24, 8, 1],
    [48, 1, 1],
    [96, 1, 1],
    [64, 1, 1],
    [128, 1, 1],
    [256, 1, 1],
    [32, 2, 1],
    [8, 8, 1],
    [16, 16, 1],
    [32, 32, 1],
    [8, 4, 1],
    [16, 8, 1],
    [4, 4, 4],
    [8, 8, 4],
    [16, 8, 2],
    [2, 2, 16],
    [1, 64, 1],
    [12, 12, 1],
    [10, 10, 1],
];

/// What the relabelling relies on (`Lower::console_flat`): 1-D groups are
/// put into waves as aligned runs (rounding works there), and every group
/// has at most one partial wave, its lanes from 0 (it takes the last run).
/// Multi-dimensional groups need not be aligned runs: an RTX 5070 Ti puts
/// 2-D groups 24, 32 or 64 wide, or not a power of two wide, into 16 by 2
/// tiles, which this prints.
#[test]
fn host_waves_keep_what_the_relabelling_relies_on() {
    let Some(dxc) = dxc() else {
        eprintln!("no PC dxc.exe; skipped");
        return;
    };
    let dir = std::env::temp_dir().join(format!("victory-lane-order-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut broken = Vec::new();
    // As translated programs that read other lanes declare it (the
    // console's wave size), and without, as the host chooses.
    for (label, warp, wave) in [
        ("hardware WaveSize(32)", false, Some(32)),
        ("hardware", false, None),
        ("warp WaveSize(32)", true, Some(32)),
    ] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        for shape in SHAPES {
            let threads = shape.iter().product();
            let records = match run(&host, probe(&dxc, &dir, shape, wave), threads) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("{label} {shape:?}: refused ({e}); skipped");
                    continue;
                }
            };
            // Each wave: its member count and lowest and highest lane.
            let mut waves: BTreeMap<u32, (u32, u32, u32)> = BTreeMap::new();
            for &(lane, _, first) in &records {
                let w = waves.entry(first).or_insert((0, u32::MAX, 0));
                *w = (w.0 + 1, w.1.min(lane), w.2.max(lane));
            }
            eprintln!(
                "{label} {shape:?}: waves (members, lanes) {:?}",
                waves
                    .values()
                    .map(|&(n, a, b)| format!("{n}:{a}-{b}"))
                    .collect::<Vec<_>>()
            );
            let f = examine(shape, &records);
            eprintln!(
                "{label} {:?}: wave {}, aligned runs {}, relabel bijective {}{}",
                f.shape,
                f.wave,
                f.aligned_runs,
                f.relabel_bijective,
                f.example
                    .as_ref()
                    .map_or(String::new(), |e| format!(", first odd wave {e:?}"))
            );
            let one_d = shape[1] == 1 && shape[2] == 1;
            if one_d && !f.relabel_bijective {
                broken.push(format!(
                    "{label} {shape:?}: a 1-D group's waves are not aligned runs"
                ));
            }
            if f.partial.len() > 1 || f.partial.iter().any(|&(_, from_0)| !from_0) {
                broken.push(format!("{label} {shape:?}: partial waves {:?}", f.partial));
            }
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        broken.is_empty(),
        "the relabelling's assumptions fail: {broken:?}"
    );
}

/// A console wave32 program that reads lanes (`v_mbcnt`, so it is
/// relabelled) and stores 1 at its thread's flat id, `y * x_size + x`, in a
/// group of `[x, y, 1]`: every slot of the group must hold 1, where ids
/// repeated by the relabelling leave some at 0. With `lds_bytes`, it
/// allocates that much LDS and writes the 1 there too.
fn marks_its_thread_native(shape: [u32; 2], lds_bytes: u32) -> NativeProgram {
    let [x, y] = shape;
    assert!(x <= 64, "an inline multiplier");
    let mbcnt = [0xD765_0005_u32, 0x0001_00C1]; // v_mbcnt_lo_u32_b32 v5, -1, 0
    let mut code: Vec<u32> = vec![
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0A02,
        0xFA00_0000, // s_load_dwordx4 s[40:43], s[4:5], 0x0
        mbcnt[0],
        mbcnt[1],
        0x0B << 25 | 3 << 17 | 1 << 9 | (128 + x), // v_mul_u32_u24 v3, x, v1
        0x4A06_0700,                               // v_add_nc_u32 v3, v0, v3
        0x3408_0682,                               // v_lshlrev_b32 v4, 2, v3
        0x7E0C_0281,                               // v_mov_b32 v6, 1
        0xBF8C_C07F,                               // s_waitcnt lgkmcnt(0)
        0xE070_1000,
        0x800A_0604, // buffer_store_dword v6, v4, s[40:43], 0 offen
        0xBF81_0000, // s_endpgm
    ];
    if lds_bytes > 0 {
        // ds_write_b32 v4, v6, before the store.
        code.splice(11..11, [0xD834_0000, 0x0000_0604]);
    }
    NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 0,
            slot: 0,
        }],
        // One user SGPR; thread ids x and y; LDS in 512-byte blocks.
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 1 | 1 << 11 | (lds_bytes / 512) << 15,
            threads: [x, y, 1],
        }),
        code: code.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// [`marks_its_thread_native`] without LDS, translated.
fn marks_its_thread(shape: [u32; 2]) -> HostShader {
    let l = super::synthetic_tests::lowered(&marks_its_thread_native(shape, 0)).expect("lower");
    HostShader {
        stage: HwStage::Cs,
        dxil: l.dxil,
        user_data: l.user_data,
        samplers: l.samplers,
        static_samplers: Vec::new(),
        root_cbvs: l.root_cbvs,
        root_srvs: l.root_srvs,
        root_uavs: l.root_uavs,
        memory: l.memory,
        memory_writes: l.memory_writes,
        view_addresses: l.view_addresses,
        heap: l.heap,
        heap_writes: l.heap_writes,
        view_marks: l.view_marks,
    }
}

/// Every thread of a relabelled program gets its own id, in group shapes
/// whose host waves are aligned runs and in shapes whose waves are not
/// (16 by 2 tiles and uneven splits on an RTX 5070 Ti).
#[test]
fn relabelled_threads_each_get_their_own_id() {
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; skipped");
        return;
    };
    // Translated programs that read lanes declare the console's wave size,
    // 32; a host that runs no waves of 32 refuses them.
    if let Ok((min, max)) = host.wave_lanes()
        && !(min..=max).contains(&32)
    {
        eprintln!("the host runs waves of {min} to {max} lanes, not 32; skipped");
        return;
    }
    let mut wrong = Vec::new();
    for shape in [
        [64, 1],
        [8, 8],
        [16, 16],
        [32, 2],
        [32, 8],
        [64, 4],
        [24, 8],
        [12, 12],
        [10, 10],
    ] {
        let cs = marks_its_thread(shape);
        let (root, pso) = match host.compute_pipeline(&cs) {
            Ok(p) => p,
            // The 1-D group takes no ticket: a host refusing its program
            // refuses the wave size, whatever range it reports (an Intel UHD,
            // Xe-LP, refused it with "The parameter is incorrect").
            Err(e) if shape == [64, 1] => {
                eprintln!("the host refuses a program with waves of 32 ({e}); skipped");
                return;
            }
            Err(e) => panic!("{shape:?}: {e}"),
        };
        let n = shape[0] * shape[1];
        let ud = vec![0u32; cs.user_data_dwords().max(1) as usize];
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(n)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{shape:?}: {e}"));
        let marked = out[0].iter().filter(|&&v| v == 1).count();
        eprintln!("{shape:?}: {marked} of {n} ids marked");
        if marked != n as usize {
            wrong.push(format!("{shape:?}: {marked} of {n}"));
        }
    }
    assert!(wrong.is_empty(), "ids repeated: {wrong:?}");
}

/// A program that reads lanes in a 2-D group and already allocates all the
/// host's group-shared memory (32 KiB) has no room for the ticket's dword
/// (`Lower::ticket`): it keeps the rounding of 1-D groups and translates,
/// where with the ticket it would fail validation. With a block of LDS
/// free, it takes the ticket.
#[test]
fn programs_with_all_the_lds_keep_the_rounding() {
    let lower = |lds| super::synthetic_tests::lowered(&marks_its_thread_native([8, 8], lds));
    let full = lower(32 * 1024).expect("lower");
    let room = lower(32 * 1024 - 512).expect("lower");
    assert!(!full.ticket, "all the LDS: the rounding");
    assert!(room.ticket, "a block free: the ticket");
    let path = super::validator::Validator::default_path();
    let Ok(mut v) = super::validator::Validator::load(path.as_deref()) else {
        eprintln!("no validator; lowering checked only");
        return;
    };
    for (label, mut dxil) in [("all the LDS", full.dxil), ("a block free", room.dxil)] {
        v.validate(&mut dxil)
            .unwrap_or_else(|e| panic!("{label}: {e}"));
    }
}

/// A wave64 program split into 32-lane host waves keeps the console's
/// waves in any group layout: each logical thread stores its console lane
/// and its wave's active lane count at its own flat id. A 32x4 group can
/// occupy 16x2 host tiles (RTX 5070 Ti), where the host's own flat ids
/// paired the wrong host waves into a console wave. Regression shader by
/// @0xkn0wh3r3, ported by @stem23456-coder (#1960, #3147).
#[test]
fn a_wave64_program_runs_as_the_console_runs_it_on_a_32_lane_host() {
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let Some((name, host)) = Host::hardware()
        .into_iter()
        .find(|(_, h)| h.wave_lanes().is_ok_and(|(min, max)| (min..=max).contains(&32)))
    else {
        eprintln!("no SM 6.6 hardware supporting 32-lane waves; skipped");
        return;
    };
    let want: Vec<u32> = (0..128).flat_map(|id| [id % 64, 64]).collect();
    for [x, y] in [[128u32, 1], [16, 8], [32, 4]] {
        // In the 128x1 group v1 (thread id y) is zero, so an inline
        // multiplier of 64 also gives the right row offset.
        let code = [
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0A02,
            0xFA00_0000, // s_load_dwordx4 s[40:43], s[4:5], 0x0
            0xD765_0005,
            0x0001_00C1, // v_mbcnt_lo_u32_b32 v5, -1, 0
            0xD766_0005,
            0x0002_0AC1, // v_mbcnt_hi_u32_b32 v5, -1, v5
            0xBE8A_107E, // s_bcnt1_i32_b64 s10, exec
            0x7E0C_020A, // v_mov_b32 v6, s10
            0x0B << 25 | 3 << 17 | 1 << 9 | (128 + x.min(64)), // v_mul_u32_u24 v3, x, v1
            0x4A06_0700, // v_add_nc_u32 v3, v0, v3
            0x3408_0683, // v_lshlrev_b32 v4, 3, v3
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0xE070_1000,
            0x800A_0504, // buffer_store_dword v5, v4, s[40:43], 0 offen
            0xE070_1004,
            0x800A_0604, // buffer_store_dword v6, v4, s[40:43], 0 offen offset:4
            0xBF81_0000, // s_endpgm
        ];
        let mut native = marks_its_thread_native([x.min(64), y], 0);
        native.wave64 = true;
        native.io = StageIo::Cs(ComputeIo {
            rsrc2: 1 << 1 | 1 << 11,
            threads: [x, y, 1],
        });
        native.code = code.iter().flat_map(|w: &u32| w.to_le_bytes()).collect();
        let l = super::synthetic_tests::lowered_for(&native, false)
            .unwrap_or_else(|e| panic!("{name} {x}x{y}: lower for 32-lane waves: {e}"));
        let cs = HostShader {
            stage: HwStage::Cs,
            dxil: l.dxil,
            user_data: l.user_data,
            samplers: l.samplers,
            static_samplers: Vec::new(),
            root_cbvs: l.root_cbvs,
            root_srvs: l.root_srvs,
            root_uavs: l.root_uavs,
            memory: l.memory,
            memory_writes: l.memory_writes,
            view_addresses: l.view_addresses,
            heap: l.heap,
            heap_writes: l.heap_writes,
            view_marks: l.view_marks,
        };
        let (root, pso) = host
            .compute_pipeline(&cs)
            .unwrap_or_else(|e| panic!("{name} {x}x{y}: {e}"));
        let ud = vec![0u32; cs.user_data_dwords().max(1) as usize];
        let mut out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(2 * x * y)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{name} {x}x{y}: {e}"));
        let out = out.pop().expect("one UAV");
        assert_eq!(out, want, "{name} {x}x{y}: console lane and wave lane count");
    }
}
