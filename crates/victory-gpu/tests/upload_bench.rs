//! Synthetic benchmark: getting the console's "unified memory" contents onto
//! the host GPU each frame, by four routes, on every hardware adapter.
//!
//! The title's writes reach the host through host copies of title buffers
//! (`crate::shadow`): per submission the 64 KiB pages the title wrote are
//! copied into a persistently mapped host buffer. That buffer is today a
//! CUSTOM heap, L0, write-back (`Host::shared_buffer`): system memory the CPU
//! writes and shaders read over the bus. This benchmark models that cost with
//! N dirty ranges per frame (~600 ranges, ~17 MB by default) and compares:
//!
//! * `l0-writeback`: the current route. memcpy into a system-memory buffer
//!   the GPU reads in place.
//! * `upload+copy`: memcpy into an UPLOAD buffer, `CopyBufferRegion` of each
//!   range into a DEFAULT buffer on the direct queue, shaders read that.
//! * `upload+copyq`: the same copies on a copy queue, the direct queue waits.
//! * `gpu-upload`: memcpy into a persistently mapped `D3D12_HEAP_TYPE_GPU_UPLOAD`
//!   buffer (CPU-visible video memory, ReBAR on a discrete card), shaders read
//!   it in place. Skipped where `GPUUploadHeapSupported` is false.
//!
//! Every variant ends with the same compute pass, which sums each range's
//! dwords into a table that is read back and checked against the CPU's sums
//! every frame, so a route that drops or tears a copy fails the run.
//!
//! The frames run one after the other (each ends with a fence wait), so CPU
//! and GPU never overlap: the numbers are the cost of the route, not a
//! pipelined frame time.
//!
//! Run (ignored in normal runs; needs a GPU):
//!
//! ```text
//! cargo test --release -p victory-gpu --test upload_bench -- --ignored --nocapture
//! ```
//!
//! Environment: `UPLOAD_BENCH_RANGES` (600), `UPLOAD_BENCH_MB` (17, total
//! dirty megabytes per frame), `UPLOAD_BENCH_FRAMES` (300 measured),
//! `UPLOAD_BENCH_WARMUP` (30), `UPLOAD_BENCH_ADAPTER` (a name substring to
//! run only on matching adapters).
#![cfg(windows)]

use std::time::Instant;

use windows::Win32::Graphics::Direct3D::{D3D_FEATURE_LEVEL_12_0, Fxc::D3DCompile, ID3DBlob};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, IDXGIAdapter1, IDXGIDevice, IDXGIFactory4,
};
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::core::{Interface, PCSTR, s};

/// How many differently placed copies of the title's memory the frames rotate
/// through, so a frame's source is not still in the CPU's cache.
const SOURCE_SETS: usize = 4;
/// The console's page: ranges are separated by gaps below it.
const PAGE: u64 = 64 << 10;

const SHADER: &str = r#"
ByteAddressBuffer data : register(t0);
ByteAddressBuffer ranges : register(t1);
RWByteAddressBuffer sums : register(u0);
groupshared uint part[256];
[RootSignature("RootFlags(0), SRV(t0), SRV(t1), UAV(u0)")]
[numthreads(256, 1, 1)]
void main(uint3 g : SV_GroupID, uint t : SV_GroupIndex) {
    uint2 r = ranges.Load2(g.x * 8);
    uint acc = 0;
    for (uint i = r.x + t * 4; i < r.x + r.y; i += 1024) {
        acc += data.Load(i);
    }
    part[t] = acc;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = 128; s > 0; s >>= 1) {
        if (t < s) { part[t] += part[t + s]; }
        GroupMemoryBarrierWithGroupSync();
    }
    if (t == 0) { sums.Store(g.x * 4, part[0]); }
}
"#;

fn env<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

struct Config {
    ranges: usize,
    total: u64,
    frames: usize,
    warmup: usize,
}

struct Layout {
    /// (offset, length) in bytes, ascending, disjoint, 256-byte multiples.
    ranges: Vec<(u32, u32)>,
    /// Bytes of the buffers (the end of the last range, rounded to a page).
    size: u64,
    /// Bytes in all the ranges.
    bytes: u64,
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// `n` ranges whose sizes spread over two orders of magnitude and total about
/// `total` bytes, apart from each other like the title's scattered pages.
fn make_layout(n: usize, total: u64, rng: &mut Rng) -> Layout {
    let weights: Vec<f64> = (0..n).map(|_| (5.0 * rng.unit()).exp()).collect();
    let scale = total as f64 / weights.iter().sum::<f64>();
    let mut ranges = Vec::with_capacity(n);
    let (mut cursor, mut bytes) = (0u64, 0u64);
    for w in weights {
        let len = (((w * scale) as u64 + 128) / 256 * 256).max(256);
        cursor += rng.next() % (PAGE / 256 + 1) * 256;
        ranges.push((cursor as u32, len as u32));
        cursor += len;
        bytes += len;
    }
    Layout { ranges, size: cursor.div_ceil(PAGE) * PAGE, bytes }
}

#[derive(Clone, Copy, PartialEq)]
enum Heap {
    Default,
    Upload,
    Readback,
    GpuUpload,
    /// CUSTOM, L0, write-back: `Host::shared_buffer`'s heap.
    L0,
}

fn buffer(
    device: &ID3D12Device,
    heap: Heap,
    size: u64,
    uav: bool,
) -> windows::core::Result<(ID3D12Resource, *mut u8)> {
    let props = match heap {
        Heap::Default => D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
        Heap::Upload => D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() },
        Heap::Readback => D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() },
        Heap::GpuUpload => D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_GPU_UPLOAD, ..Default::default() },
        Heap::L0 => D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            ..Default::default()
        },
    };
    let state = match heap {
        Heap::Upload => D3D12_RESOURCE_STATE_GENERIC_READ,
        Heap::Readback => D3D12_RESOURCE_STATE_COPY_DEST,
        Heap::Default if uav => D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        _ => D3D12_RESOURCE_STATE_COMMON,
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: if uav { D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS } else { D3D12_RESOURCE_FLAG_NONE },
        ..Default::default()
    };
    let mut resource: Option<ID3D12Resource> = None;
    let mut ptr = std::ptr::null_mut();
    // SAFETY: plain creation and persistent mapping of a buffer; the out
    // pointers are live.
    unsafe {
        device.CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut resource)?;
        let resource = resource.ok_or_else(windows::core::Error::empty)?;
        if heap != Heap::Default {
            resource.Map(0, None, Some(&mut ptr))?;
        }
        Ok((resource, ptr.cast()))
    }
}

fn transition(
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // SAFETY: a borrowed pointer; the barrier never releases it.
                pResource: unsafe { std::mem::transmute_copy(resource) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Variant {
    L0WriteBack,
    UploadCopy,
    UploadCopyQueue,
    GpuUpload,
}

impl Variant {
    fn name(self) -> &'static str {
        match self {
            Variant::L0WriteBack => "l0-writeback (current)",
            Variant::UploadCopy => "upload+copy (direct)",
            Variant::UploadCopyQueue => "upload+copy (copy queue)",
            Variant::GpuUpload => "gpu-upload",
        }
    }
}

/// What every variant shares: the compute pass and its small buffers.
struct Context {
    device: ID3D12Device,
    direct: ID3D12CommandQueue,
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    query_heap: ID3D12QueryHeap,
    timestamps: (ID3D12Resource, *mut u8),
    ranges_va: u64,
    sums: ID3D12Resource,
    sums_readback: (ID3D12Resource, *mut u8),
    freq: f64,
    // Kept alive for `ranges_va`.
    _ranges: ID3D12Resource,
}

fn context(device: &ID3D12Device, layout: &Layout) -> Result<Context, String> {
    let e = |what: &'static str| move |err: windows::core::Error| format!("{what}: {err}");
    let n = layout.ranges.len() as u64;
    let (mut code, mut errors): (Option<ID3DBlob>, Option<ID3DBlob>) = (None, None);
    // SAFETY: the source and names outlive the call; the blobs are the outputs.
    let compiled = unsafe {
        D3DCompile(
            SHADER.as_ptr().cast(),
            SHADER.len(),
            PCSTR::null(),
            None,
            None,
            s!("main"),
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(err) = compiled {
        let text = errors
            // SAFETY: the blob's own bytes.
            .map(|b| unsafe { std::slice::from_raw_parts(b.GetBufferPointer().cast::<u8>(), b.GetBufferSize()) }.to_vec())
            .map(|b| String::from_utf8_lossy(&b).into_owned())
            .unwrap_or_default();
        return Err(format!("shader compile: {err} {text}"));
    }
    let code = code.ok_or("no shader blob")?;
    // SAFETY: the blob's own bytes, a shader with its root signature inside.
    let bytes = unsafe { std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize()) };
    // SAFETY: plain object creation from the compiled shader.
    let (root, pso, direct, query_heap) = unsafe {
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes).map_err(e("root signature"))?;
        let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE { pShaderBytecode: bytes.as_ptr().cast(), BytecodeLength: bytes.len() },
            ..Default::default()
        };
        let pso = device.CreateComputePipelineState(&desc).map_err(e("pipeline"))?;
        let direct: ID3D12CommandQueue = device
            .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() })
            .map_err(e("direct queue"))?;
        let mut heap: Option<ID3D12QueryHeap> = None;
        device
            .CreateQueryHeap(
                &D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: 2, NodeMask: 0 },
                &mut heap,
            )
            .map_err(e("query heap"))?;
        (root, pso, direct, heap.ok_or("no query heap")?)
    };
    let (ranges, ranges_ptr) = buffer(device, Heap::Upload, n * 8, false).map_err(e("ranges buffer"))?;
    for (i, &(off, len)) in layout.ranges.iter().enumerate() {
        // SAFETY: inside the mapping of `n * 8` bytes.
        unsafe { ranges_ptr.cast::<u32>().add(2 * i).write(off) };
        // SAFETY: as above.
        unsafe { ranges_ptr.cast::<u32>().add(2 * i + 1).write(len) };
    }
    // SAFETY: a plain query.
    let ranges_va = unsafe { ranges.GetGPUVirtualAddress() };
    let (sums, _) = buffer(device, Heap::Default, n * 4, true).map_err(e("sums buffer"))?;
    let sums_readback = buffer(device, Heap::Readback, n * 4, false).map_err(e("sums readback"))?;
    let timestamps = buffer(device, Heap::Readback, 16, false).map_err(e("timestamp readback"))?;
    // SAFETY: a plain query.
    let freq = unsafe { direct.GetTimestampFrequency() }.map_err(e("timestamp frequency"))? as f64;
    Ok(Context {
        device: device.clone(),
        direct,
        root,
        pso,
        query_heap,
        timestamps,
        ranges_va,
        sums,
        sums_readback,
        freq,
        _ranges: ranges,
    })
}

#[derive(Default)]
struct Samples {
    write: Vec<f64>,
    cpu: Vec<f64>,
    gpu: Vec<f64>,
    wall: Vec<f64>,
}

fn pct(v: &[f64], q: f64) -> f64 {
    let mut v = v.to_vec();
    v.sort_by(f64::total_cmp);
    v[((v.len() - 1) as f64 * q).round() as usize]
}

fn cell(v: &[f64]) -> String {
    format!("{:7.3} /{:7.3}", pct(v, 0.5), pct(v, 0.95))
}

/// Runs `variant` for the configured frames; the report line, or why it failed.
fn run(
    ctx: &Context,
    layout: &Layout,
    sets: &mut [Vec<u32>],
    base: &[u32],
    variant: Variant,
    cfg: &Config,
) -> Result<String, String> {
    let e = |what: &'static str| move |err: windows::core::Error| format!("{what}: {err}");
    let device = &ctx.device;
    let n = layout.ranges.len();
    // The buffer the CPU writes, and the one the shaders read.
    let (stage_heap, stage_name) = match variant {
        Variant::L0WriteBack => (Heap::L0, "L0 buffer"),
        Variant::GpuUpload => (Heap::GpuUpload, "GPU_UPLOAD buffer"),
        _ => (Heap::Upload, "UPLOAD buffer"),
    };
    let (stage, stage_ptr) = buffer(device, stage_heap, layout.size, false)
        .map_err(|err| format!("{stage_name} of {} MiB: {err}", layout.size >> 20))?;
    let copies = matches!(variant, Variant::UploadCopy | Variant::UploadCopyQueue);
    let dst = if copies {
        Some(buffer(device, Heap::Default, layout.size, false).map_err(e("DEFAULT buffer"))?.0)
    } else {
        None
    };
    // SAFETY: plain queries.
    let data_va = unsafe { dst.as_ref().unwrap_or(&stage).GetGPUVirtualAddress() };
    // SAFETY: plain object creation, and recording and submission of lists
    // whose buffers outlive the frames (each ends with a fence wait).
    unsafe {
        let make_list = |kind| -> Result<(ID3D12CommandAllocator, ID3D12GraphicsCommandList), String> {
            let alloc: ID3D12CommandAllocator = device.CreateCommandAllocator(kind).map_err(e("allocator"))?;
            let list: ID3D12GraphicsCommandList =
                device.CreateCommandList(0, kind, &alloc, None::<&ID3D12PipelineState>).map_err(e("list"))?;
            list.Close().map_err(e("close"))?;
            Ok((alloc, list))
        };
        let (alloc, list) = make_list(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
        let exec: ID3D12CommandList = list.cast().map_err(e("cast"))?;
        let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(e("fence"))?;
        let event = CreateEventW(None, false, false, None).map_err(e("event"))?;
        let mut value = 0u64;
        // The copy queue, for that variant only.
        let copy = if variant == Variant::UploadCopyQueue {
            let queue: ID3D12CommandQueue = device
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_COPY, ..Default::default() })
                .map_err(e("copy queue"))?;
            let (alloc, list) = make_list(D3D12_COMMAND_LIST_TYPE_COPY)?;
            let exec: ID3D12CommandList = list.cast().map_err(e("cast"))?;
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(e("copy fence"))?;
            Some((queue, alloc, list, exec, fence))
        } else {
            None
        };

        let record_copies = |list: &ID3D12GraphicsCommandList, dst: &ID3D12Resource| {
            for &(off, len) in &layout.ranges {
                list.CopyBufferRegion(dst, off as u64, &stage, off as u64, len as u64);
            }
        };

        let mut samples = Samples::default();
        let mut bad = 0usize;
        let mut first_bad = String::new();
        for frame in 0..cfg.warmup + cfg.frames {
            // The title writes (not timed): a tag in each range's first dword.
            let tag = frame as u32 + 1;
            let set = &mut sets[frame % SOURCE_SETS];
            for &(off, _) in &layout.ranges {
                set[off as usize / 4] = tag;
            }
            let source = set.as_ptr().cast::<u8>();

            let started = Instant::now();
            for &(off, len) in &layout.ranges {
                std::ptr::copy_nonoverlapping(source.add(off as usize), stage_ptr.add(off as usize), len as usize);
            }
            let wrote = started.elapsed();

            if let Some((queue, alloc, list, exec, fence)) = &copy {
                alloc.Reset().map_err(e("copy allocator reset"))?;
                list.Reset(alloc, None::<&ID3D12PipelineState>).map_err(e("copy list reset"))?;
                record_copies(list, dst.as_ref().ok_or("no DEFAULT buffer")?);
                list.Close().map_err(e("copy close"))?;
                queue.ExecuteCommandLists(&[Some(exec.clone())]);
                queue.Signal(fence, frame as u64 + 1).map_err(e("copy signal"))?;
                ctx.direct.Wait(fence, frame as u64 + 1).map_err(e("queue wait"))?;
            }

            alloc.Reset().map_err(e("allocator reset"))?;
            list.Reset(&alloc, None::<&ID3D12PipelineState>).map_err(e("list reset"))?;
            list.EndQuery(&ctx.query_heap, D3D12_QUERY_TYPE_TIMESTAMP, 0);
            if variant == Variant::UploadCopy {
                let dst = dst.as_ref().ok_or("no DEFAULT buffer")?;
                list.ResourceBarrier(&[transition(dst, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST)]);
                record_copies(&list, dst);
                list.ResourceBarrier(&[transition(
                    dst,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                )]);
            }
            list.SetComputeRootSignature(&ctx.root);
            list.SetPipelineState(&ctx.pso);
            list.SetComputeRootShaderResourceView(0, data_va);
            list.SetComputeRootShaderResourceView(1, ctx.ranges_va);
            list.SetComputeRootUnorderedAccessView(2, ctx.sums.GetGPUVirtualAddress());
            list.Dispatch(n as u32, 1, 1);
            if variant == Variant::UploadCopy {
                let dst = dst.as_ref().ok_or("no DEFAULT buffer")?;
                list.ResourceBarrier(&[transition(
                    dst,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    D3D12_RESOURCE_STATE_COMMON,
                )]);
            }
            list.EndQuery(&ctx.query_heap, D3D12_QUERY_TYPE_TIMESTAMP, 1);
            list.ResourceBarrier(&[transition(
                &ctx.sums,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            list.CopyBufferRegion(&ctx.sums_readback.0, 0, &ctx.sums, 0, n as u64 * 4);
            list.ResolveQueryData(&ctx.query_heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &ctx.timestamps.0, 0);
            list.ResourceBarrier(&[transition(
                &ctx.sums,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )]);
            list.Close().map_err(e("close"))?;
            ctx.direct.ExecuteCommandLists(&[Some(exec.clone())]);
            value += 1;
            ctx.direct.Signal(&fence, value).map_err(e("signal"))?;
            let submitted = started.elapsed();

            fence.SetEventOnCompletion(value, event).map_err(e("fence event"))?;
            WaitForSingleObject(event, INFINITE);
            let wall = started.elapsed();
            device.GetDeviceRemovedReason().map_err(|err| format!("device removed in frame {frame}: {err}"))?;

            let stamps = std::slice::from_raw_parts(ctx.timestamps.1.cast::<u64>(), 2);
            let gpu = stamps[1].saturating_sub(stamps[0]) as f64 / ctx.freq * 1e3;
            let sums = std::slice::from_raw_parts(ctx.sums_readback.1.cast::<u32>(), n);
            for (i, &got) in sums.iter().enumerate() {
                let want = base[i].wrapping_add(tag);
                if got != want {
                    bad += 1;
                    if first_bad.is_empty() {
                        first_bad = format!("frame {frame} range {i}: GPU sum {got:#x}, expected {want:#x}");
                    }
                }
            }
            if frame >= cfg.warmup {
                samples.write.push(wrote.as_secs_f64() * 1e3);
                samples.cpu.push(submitted.as_secs_f64() * 1e3);
                samples.gpu.push(gpu);
                samples.wall.push(wall.as_secs_f64() * 1e3);
            }
        }
        if bad > 0 {
            return Err(format!("CHECKSUM MISMATCH ({bad} ranges): {first_bad}"));
        }
        let gbs = layout.bytes as f64 / (pct(&samples.write, 0.5) * 1e-3) / 1e9;
        Ok(format!(
            "{:26} | {} | {} | {} | {} | {gbs:6.2}",
            variant.name(),
            cell(&samples.write),
            cell(&samples.cpu),
            cell(&samples.gpu),
            cell(&samples.wall),
        ))
    }
}

struct Adapter {
    name: String,
    device: ID3D12Device,
}

fn adapters() -> Vec<Adapter> {
    let mut found = Vec::new();
    // SAFETY: plain DXGI and device creation calls with live out pointers.
    unsafe {
        let Ok(factory) = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0)) else {
            return found;
        };
        let mut index = 0;
        let mut seen = std::collections::HashSet::new();
        while let Ok(adapter) = factory.EnumAdapters1(index) {
            index += 1;
            let Ok(desc) = adapter.GetDesc1() else { continue };
            let name = String::from_utf16_lossy(&desc.Description).trim_end_matches('\0').to_string();
            // 0x1414: Microsoft's software adapters (WARP, Basic Render).
            if desc.VendorId == 0x1414 {
                eprintln!("skipping software adapter {name}");
                continue;
            }
            // DXGI can list one adapter more than once.
            let luid = (desc.AdapterLuid.HighPart, desc.AdapterLuid.LowPart);
            if !seen.insert(luid) {
                continue;
            }
            let mut device: Option<ID3D12Device> = None;
            if D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device).is_err() {
                eprintln!("skipping {name}: no D3D12 device at feature level 12_0");
                continue;
            }
            let Some(device) = device else { continue };
            describe(&name, &adapter, &desc, &device);
            found.push(Adapter { name, device });
        }
    }
    found
}

/// What the adapter reports about the memory routes.
fn describe(
    name: &str,
    adapter: &IDXGIAdapter1,
    desc: &windows::Win32::Graphics::Dxgi::DXGI_ADAPTER_DESC1,
    device: &ID3D12Device,
) {
    let driver = {
        // SAFETY: a plain query.
        let v = unsafe { adapter.CheckInterfaceSupport(&IDXGIDevice::IID) }.unwrap_or(0) as u64;
        format!("{}.{}.{}.{}", v >> 48, (v >> 32) & 0xffff, (v >> 16) & 0xffff, v & 0xffff)
    };
    let mut arch = D3D12_FEATURE_DATA_ARCHITECTURE1::default();
    let mut o16 = D3D12_FEATURE_DATA_D3D12_OPTIONS16::default();
    // SAFETY: each query writes its own structure of the given size.
    let (arch_ok, o16_ok) = unsafe {
        (
            device
                .CheckFeatureSupport(D3D12_FEATURE_ARCHITECTURE1, (&raw mut arch).cast(), size_of_val(&arch) as u32)
                .is_ok(),
            device
                .CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS16, (&raw mut o16).cast(), size_of_val(&o16) as u32)
                .is_ok(),
        )
    };
    println!("== {name}");
    println!(
        "   vendor {:#06x} device {:#06x} driver {driver}, dedicated VRAM {} MiB, shared system {} MiB",
        desc.VendorId,
        desc.DeviceId,
        desc.DedicatedVideoMemory >> 20,
        desc.SharedSystemMemory >> 20
    );
    println!(
        "   UMA {} cache-coherent UMA {} GPUUploadHeapSupported {}",
        arch_ok.then(|| arch.UMA.as_bool().to_string()).unwrap_or_else(|| "?".into()),
        arch_ok.then(|| arch.CacheCoherentUMA.as_bool().to_string()).unwrap_or_else(|| "?".into()),
        o16_ok.then(|| o16.GPUUploadHeapSupported.as_bool().to_string()).unwrap_or_else(|| "query failed".into()),
    );
}

fn gpu_upload_supported(device: &ID3D12Device) -> bool {
    let mut o16 = D3D12_FEATURE_DATA_D3D12_OPTIONS16::default();
    // SAFETY: the query writes its own structure of the given size.
    unsafe {
        device
            .CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS16, (&raw mut o16).cast(), size_of_val(&o16) as u32)
            .is_ok()
            && o16.GPUUploadHeapSupported.as_bool()
    }
}

#[test]
#[ignore = "benchmark, needs a GPU: cargo test --release -p victory-gpu --test upload_bench -- --ignored --nocapture"]
fn upload_bench() {
    let cfg = Config {
        ranges: env("UPLOAD_BENCH_RANGES", 600),
        total: env::<u64>("UPLOAD_BENCH_MB", 17) << 20,
        frames: env("UPLOAD_BENCH_FRAMES", 300),
        warmup: env("UPLOAD_BENCH_WARMUP", 30),
    };
    let filter = std::env::var("UPLOAD_BENCH_ADAPTER").ok().map(|s| s.to_lowercase());
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let layout = make_layout(cfg.ranges, cfg.total, &mut rng);
    println!(
        "{} ranges/frame, {:.2} MB dirty ({:.1} KB mean), buffers {} MiB, {} frames after {} warmup, ms median / p95",
        layout.ranges.len(),
        layout.bytes as f64 / 1e6,
        layout.bytes as f64 / 1e3 / layout.ranges.len() as f64,
        layout.size >> 20,
        cfg.frames,
        cfg.warmup
    );

    // The title's memory: the same bytes in every set, but each range's first
    // dword (the frame tag), which is zero in the sums the GPU is checked by.
    let mut sets: Vec<Vec<u32>> = Vec::new();
    let mut first = vec![0u32; layout.size as usize / 4];
    first.iter_mut().for_each(|d| *d = rng.next() as u32);
    for &(off, _) in &layout.ranges {
        first[off as usize / 4] = 0;
    }
    let base: Vec<u32> = layout
        .ranges
        .iter()
        .map(|&(off, len)| {
            first[off as usize / 4..(off + len) as usize / 4].iter().fold(0u32, |a, &d| a.wrapping_add(d))
        })
        .collect();
    for _ in 0..SOURCE_SETS {
        sets.push(first.clone());
    }
    drop(first);

    let found = adapters();
    if found.is_empty() {
        eprintln!("skipped upload benchmark: no hardware D3D12 adapter");
        return;
    }
    let mut failed = Vec::new();
    for adapter in &found {
        if let Some(f) = &filter
            && !adapter.name.to_lowercase().contains(f)
        {
            continue;
        }
        println!("\n== running on {}", adapter.name);
        let ctx = match context(&adapter.device, &layout) {
            Ok(c) => c,
            Err(err) => {
                println!("   skipped: {err}");
                continue;
            }
        };
        println!(
            "{:26} | {:15} | {:15} | {:15} | {:15} | write",
            "variant", "write (memcpy)", "cpu total", "gpu (direct q)", "wall"
        );
        let supported = gpu_upload_supported(&adapter.device);
        for variant in [Variant::L0WriteBack, Variant::UploadCopy, Variant::UploadCopyQueue, Variant::GpuUpload] {
            if variant == Variant::GpuUpload && !supported {
                println!(
                    "{:26} | SKIPPED: this adapter reports GPUUploadHeapSupported = false (no ReBAR, or no driver support)",
                    variant.name()
                );
                continue;
            }
            match run(&ctx, &layout, &mut sets, &base, variant, &cfg) {
                Ok(line) => println!("{line}"),
                Err(err) => {
                    println!("{:26} | FAILED: {err}", variant.name());
                    failed.push(format!("{} on {}: {err}", variant.name(), adapter.name));
                }
            }
        }
        println!("   (gpu is the direct queue's timestamps; for copy-queue it leaves out the copies)");
    }
    assert!(failed.is_empty(), "failed variants: {failed:#?}");
}
