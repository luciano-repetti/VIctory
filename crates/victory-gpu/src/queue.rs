//! Command queues. Submitted lists run on the host GPU and are waited for,
//! so fence signals complete at once.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};

use crate::object::{self, E_INVALIDARG, HRESULT, Iid, Object, QUEUE, S_OK, Table, This, guid};
use crate::resource::PAGEABLE_IIDS;

const QUEUE_IIDS: [Iid; 4] = [
    guid(
        0x0ec870a6,
        0x5d7e,
        0x4c22,
        [0x8c, 0xfc, 0x5b, 0xaa, 0xe0, 0x76, 0x16, 0xed],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct QueueDesc {
    kind: u32,
    priority: i32,
    flags: u32,
    nodes: u32,
}

/// The title reads the list type right after the header.
#[repr(C)]
struct Queue {
    object: Object,
    kind: u32,
    desc: QueueDesc,
}
const _: () = assert!(std::mem::offset_of!(Queue, kind) == 0x10);

fn queue<'a>(this: This) -> &'a Queue {
    // SAFETY: called from queue methods only.
    unsafe { &*this.cast::<Queue>() }
}

pub(crate) extern "system" fn create(
    _: This,
    desc: *const QueueDesc,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    tracing::debug!(?desc, "command queue");
    let queue = Box::new(Queue {
        object: if crate::device::newer() {
            Object::new(&NEWER_FUNCTIONS)
        } else {
            Object::new(&FUNCTIONS)
        },
        kind: desc.kind,
        desc,
    });
    object::hand_out(QUEUE, queue, iid, &QUEUE_IIDS, out)
}

/// The console's queue description: an engine or pipe and a queue on it
/// in place of PC's priority and node mask.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct QueueDescX {
    kind: u32,
    flags: u32,
    engine_or_pipe: u32,
    queue: u32,
}

/// Device method `CreateCommandQueueX`. The host picks engines itself, so
/// the queue is created as on PC.
pub(crate) extern "system" fn create_x(
    this: This,
    desc: *const QueueDescX,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&x) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    tracing::debug!(desc = ?x, "console command queue");
    let desc = QueueDesc {
        kind: x.kind,
        priority: 0,
        flags: x.flags,
        nodes: 1,
    };
    create(this, &desc, iid, out)
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(
        QUEUE,
        iid,
        object::is(iid, &QUEUE_IIDS).then_some(this),
        out,
    )
}
extern "system" fn release(this: This) -> u32 {
    object::release_boxed::<Queue>(this)
}

/// Queues the lists' host lists behind what the queue holds, and runs what
/// may run (`crate::hold`; decision 0007).
extern "system" fn execute(this: This, count: u32, lists: *const This) {
    if lists.is_null() {
        return;
    }
    #[cfg(windows)]
    if crate::host::host().is_some() {
        // SAFETY: the title passes `count` lists.
        let lists = unsafe { std::slice::from_raw_parts(lists, count as usize) };
        let mut held = held();
        for &l in lists {
            for segment in crate::list::segments(l) {
                let waits = segment.waits();
                held.push(this as usize, Held::Segment(segment, waits));
            }
        }
        account().submissions += 1;
        let async_submit = behind();
        if async_submit {
            drop(held);
            wake();
        } else {
            pump(&mut held, Why::Submission);
        }
        return;
    }
    static ONCE: AtomicBool = AtomicBool::new(false);
    if !ONCE.swap(true, Ordering::Relaxed) {
        tracing::warn!(count, "command lists submitted; no host GPU runs them");
    }
}

/// What ran held work: the title's submission, a queue's Signal or Wait, a
/// CPU fence call, PresentX, or our timer.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Why {
    Submission,
    Signal,
    Wait,
    Fence,
    Present,
    Timer,
}

#[cfg(windows)]
use crate::hold::Item as Held;

/// Every queue's held work.
#[cfg(windows)]
type Holding = crate::hold::Held<std::sync::Arc<crate::host::HostList>>;

#[cfg(windows)]
static HELD: std::sync::Mutex<Holding> = std::sync::Mutex::new(crate::hold::Held::new());

#[cfg(windows)]
fn held() -> std::sync::MutexGuard<'static, Holding> {
    HELD.lock().unwrap_or_else(|e| e.into_inner())
}

#[cfg(windows)]
pub(crate) fn behind() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_ASYNC_SUBMIT").is_some_and(|v| v == "1") && crate::host::late();
        on && worker_started(std::thread::Builder::new().name("victory-submit".into()).spawn(worker))
    })
}

#[cfg(windows)]
fn worker_started(result: std::io::Result<std::thread::JoinHandle<()>>) -> bool {
    match result {
        Ok(_) => {
            tracing::info!("submissions are prepared and sent by the submission worker (VICTORY_ASYNC_SUBMIT=1)");
            true
        }
        Err(error) => {
            tracing::warn!(%error, "could not start the submission worker; submissions remain synchronous");
            false
        }
    }
}

#[cfg(windows)]
static WORK: (Mutex<bool>, Condvar) = (Mutex::new(false), Condvar::new());

#[cfg(windows)]
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

#[cfg(windows)]
fn wake() {
    *lock(&WORK.0) = true;
    WORK.1.notify_one();
}

#[cfg(windows)]
fn drain(work: &mut bool, f: impl FnOnce()) -> bool {
    if !*work {
        return false;
    }
    *work = false;
    f();
    true
}

#[cfg(windows)]
fn take_work(work: &mut bool) -> bool {
    std::mem::replace(work, false)
}

#[cfg(windows)]
fn worker() {
    loop {
        let mut work = lock(&WORK.0);
        while !*work {
            work = WORK.1.wait(work).unwrap_or_else(|e| e.into_inner());
        }
        // Clear the flag before pumping so a wake from the work we run is
        // kept for the next iteration, and an idle worker waits again.
        let mut due = take_work(&mut work);
        drop(work);
        drain(&mut due, || pump_held(Why::Timer));
    }
}

#[cfg(windows)]
pub(crate) fn frames_ahead() -> usize {
    static AHEAD: OnceLock<usize> = OnceLock::new();
    *AHEAD.get_or_init(|| {
        std::env::var("VICTORY_FRAMES_AHEAD")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .filter(|&n| n >= 1)
            .unwrap_or(1)
    })
}

#[cfg(windows)]
fn may_send_behind(held: u64, ahead: usize) -> bool {
    held < ahead as u64
}

#[cfg(windows)]
static PRESENTS_HELD: AtomicU64 = AtomicU64::new(0);

#[cfg(windows)]
fn present_behind() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_ASYNC_PRESENT").is_some_and(|v| v == "1")
            && behind()
            && crate::host::late_free_present();
        if on {
            tracing::info!(frames_ahead = frames_ahead(), "presents run behind the queue's line (VICTORY_ASYNC_PRESENT=1)");
        }
        on
    })
}

/// Queues a PresentX behind the work its queue holds; its frame number, or
/// `None` where it must run now.
#[cfg(windows)]
fn send_behind(this: This, planes: u32, plane_parameters: *const Plane) -> Option<u64> {
    if this.is_null() || planes == 0 || plane_parameters.is_null() || !may_send_behind(PRESENTS_HELD.load(Ordering::Relaxed), frames_ahead()) {
        return None;
    }
    #[cfg(feature = "diagnostics")]
    if crate::capture::wanted(FRAMES.load(Ordering::Relaxed) + 1) || crate::capture::listed(FRAMES.load(Ordering::Relaxed) + 1) {
        return None;
    }
    // SAFETY: the title passes `planes` plane descriptions.
    let slice = unsafe { std::slice::from_raw_parts(plane_parameters, planes.min(8) as usize) };
    // The host textures, not the title's resources: the title may release
    // them before the queue gets here, and what the host needs of them (the
    // texture, its state) is theirs.
    let mut sources = Vec::new();
    for plane in slice {
        if plane.count == 0 || plane.resources.is_null() {
            continue;
        }
        // SAFETY: the plane lists `count` resources.
        let resource = unsafe { *plane.resources };
        if let Some(copy) = crate::resource::host_copy(resource) {
            sources.push(copy);
        }
    }
    if sources.is_empty() {
        return None;
    }
    // One number per present, taken when it is queued: the next PresentX
    // before this one is shown is the frame after it.
    let frame = FRAMES.fetch_add(1, Ordering::Relaxed) + 1;
    held().push(this as usize, Held::Present(sources, frame));
    PRESENTS_HELD.fetch_add(1, Ordering::Relaxed);
    wake();
    Some(frame)
}

/// Shows a queued PresentX's planes. Their states are the ones the lists
/// before it left (`HostList::prelude` moves them as a submission is
/// prepared), so they are read here, as `present` reads them.
#[cfg(windows)]
fn show_present(planes: Vec<Arc<crate::resource::HostTexture>>, frame: u64) {
    const COPY_SOURCE: u32 = 0x800;
    if let Some(host) = crate::host::host() {
        // Distinct title resources may name one host texture: it is moved
        // once, and every plane of it has the state it had before.
        let mut moved: Vec<(&Arc<crate::resource::HostTexture>, u32)> = Vec::new();
        let mut refs = Vec::with_capacity(planes.len());
        for plane in &planes {
            let state = match moved.iter().find(|(earlier, _)| Arc::ptr_eq(earlier, plane)) {
                Some(&(_, state)) => state,
                None => {
                    let before = plane.state.swap(COPY_SOURCE, Ordering::Relaxed);
                    moved.push((plane, before));
                    before
                }
            };
            refs.push((&plane.texture, state));
        }
        let _ = host.present(&refs, frame);
        for (plane, before) in moved {
            plane.state.store(before, Ordering::Relaxed);
        }
    }
    PRESENTS_HELD.fetch_sub(1, Ordering::Relaxed);
    #[cfg(feature = "diagnostics")]
    crate::frame::presented();
}

/// Runs held work that may run now (`why` for the frame line).
pub(crate) fn pump_held(why: Why) {
    #[cfg(windows)]
    pump(&mut held(), why);
    #[cfg(not(windows))]
    let _ = why;
}

/// The same unless another thread is at it (calls that poll: a fence's
/// value, our timer).
pub(crate) fn pump_held_if_free(why: Why) {
    #[cfg(windows)]
    if let Ok(mut held) = HELD.try_lock() {
        pump(&mut held, why);
    }
    #[cfg(not(windows))]
    let _ = why;
}

/// How long a queue's head may wait before it runs anyway.
// shortcut: a safety valve against a wait nothing satisfies (a write we do
// not make), which would hold the title's fences until its 30 s watchdog
// ends the run; counted and logged, remove once runs show it never fires.
#[cfg(windows)]
const VALVE: std::time::Duration = std::time::Duration::from_secs(10);

#[cfg(windows)]
fn pump(held: &mut Holding, why: Why) {
    if held.is_empty() {
        return;
    }
    let Some(host) = crate::host::host() else { return };
    let read = read_value;
    for (queue, waits) in held.force_older_than(VALVE) {
        account().forced += 1;
        tracing::warn!(queue = kind(queue), ?waits, "held work runs without its waits after {VALVE:?} (safety valve)");
    }
    let fence = |f: usize| crate::fence::value(f as This);
    let started = std::time::Instant::now();
    let mut ran = std::time::Duration::ZERO;
    let mut run = |_: usize, segments: Vec<Arc<crate::host::HostList>>| {
        let lists: Vec<&crate::host::HostList> = segments.iter().map(|s| &**s).collect();
        if crate::host::late_blocked(&lists) || crate::host::stale_under_flights(&lists) {
            crate::host::settle_now();
        }
        let at = std::time::Instant::now();
        if crate::host::late() {
            host.execute_ahead(segments);
        } else {
            host.execute(&lists);
            drop(lists);
            segments.into_iter().for_each(crate::list::recycle);
        }
        ran += at.elapsed();
    };
    let mut signal = |f: usize, value: u64| {
        if !crate::host::late_signal(f, value) {
            crate::fence::set(f as This, value);
            crate::fence::let_go(f as This);
        }
    };
    let mut present = show_present;
    let pumped = held.pump(&mut crate::hold::World {
        read: &read,
        fence: &fence,
        run: &mut run,
        signal: &mut signal,
        present: &mut present,
    });
    if pumped.segments > 0 || pumped.unknown > 0 {
        let mut a = account();
        a.pump += started.elapsed().saturating_sub(ran);
        a.pieces += pumped.pieces;
        a.segments += pumped.segments;
        a.unknown += pumped.unknown;
        for (queue, waited) in pumped.waited {
            a.released.push((kind(queue), waited, why));
        }
    }
    // Once work stays held, a timer tests it every 2 ms: memory the title
    // writes from the CPU, or the valve, may let it go with no call of ours.
    if held.blocked(&read).is_empty() {
        return;
    }
    static TIMER: std::sync::Once = std::sync::Once::new();
    TIMER.call_once(|| {
        let _ = std::thread::Builder::new().name("victory held work".into()).spawn(|| loop {
            std::thread::sleep(std::time::Duration::from_millis(2));
            pump_held_if_free(Why::Timer);
        });
    });
}

/// The value a wait tests (4 or 8 bytes), if the memory can be read.
#[cfg(windows)]
pub(crate) fn read_value(address: u64, bytes: u8) -> Option<u64> {
    let mut value = [0u8; 8];
    crate::host::read_into(address as *const u8, &mut value[..usize::from(bytes.min(8))]).then(|| u64::from_le_bytes(value))
}

/// A queue's type, for logs (0 graphics, 2 compute, 3 copy). Titles do not
/// release queues that hold work.
#[cfg(windows)]
fn kind(queue: usize) -> u32 {
    self::queue(queue as This).kind
}

/// What queues did with held work since the last presented frame.
#[cfg(windows)]
#[derive(Default)]
struct Account {
    submissions: u32,
    pieces: u32,
    segments: u32,
    /// Segments that waited: queue type, how long, what ran them.
    released: Vec<(u32, std::time::Duration, Why)>,
    unknown: u32,
    forced: u32,
    mappings: u32,
    /// Time in the pump outside host submissions.
    pump: std::time::Duration,
}

#[cfg(windows)]
fn account() -> std::sync::MutexGuard<'static, Account> {
    static ACCOUNT: std::sync::OnceLock<std::sync::Mutex<Account>> = std::sync::OnceLock::new();
    ACCOUNT.get_or_init(Default::default).lock().unwrap_or_else(|e| e.into_inner())
}

/// The frame line of held work: what waited, how long and what released it,
/// and what is still held at the present, with the waits that do not hold.
#[cfg(windows)]
fn account_frame(frame: u64) {
    let (cuts, early) = crate::list::take_cuts();
    let (fence_waits, fence_waited) = crate::fence::take_waited();
    let a = std::mem::take(&mut *account());
    let blocked = held().blocked(&read_value);
    if a.released.is_empty() && blocked.is_empty() && a.forced == 0 && a.unknown == 0 && a.mappings == 0 {
        return;
    }
    let longest = a.released.iter().map(|r| r.1).max().unwrap_or_default();
    let mut by = [0u32; 6];
    for r in &a.released {
        by[r.2 as usize] += 1;
    }
    tracing::info!(
        frame,
        submissions = a.submissions,
        pieces = a.pieces,
        segments = a.segments,
        cuts,
        early_resets = early,
        waited = a.released.len(),
        longest_ms = longest.as_secs_f64() * 1e3,
        pump_ms = a.pump.as_secs_f64() * 1e3,
        fence_waits,
        fence_wait_ms = fence_waited.as_secs_f64() * 1e3,
        "held work this frame (submissions; host submissions, segments; lists cut at waits; lists reset or released \
         while held, each keeping its own host list; segments that waited, released by \
         submission/signal/wait/fence/present/timer {by:?}; time in the pump outside host submissions; events the          title waited on for fence values and their summed waits); held at the present: {}",
        blocked.len()
    );
    tracing::debug!(
        frame,
        "held work released this frame (queue type, ms, by): {:?}",
        a.released.iter().map(|(k, d, w)| (k, d.as_secs_f64() * 1e3, w)).collect::<Vec<_>>()
    );
    for (queue, items, waited, unmet) in &blocked {
        let level_warn = blocked.len() > 1;
        let unmet: Vec<String> = unmet
            .iter()
            .map(|w| format!("{:#x} cmp {} {:#x} (is {:?})", w.address, w.comparison, w.reference, read_value(w.address, w.bytes)))
            .collect();
        if level_warn {
            tracing::warn!(frame, queue = kind(*queue), items, waited_ms = waited.as_millis(), ?unmet, "two or more queues hold work at the present");
        } else {
            tracing::info!(frame, queue = kind(*queue), items, waited_ms = waited.as_millis(), ?unmet, "a queue holds work at the present");
        }
    }
    if a.forced > 0 || a.unknown > 0 || a.mappings > 0 {
        tracing::warn!(
            frame,
            forced = a.forced,
            unknown = a.unknown,
            mappings = a.mappings,
            "held work: run without its waits (valve), waits with an unknown comparison (taken as holding), \
             page mappings done ahead of held work"
        );
    }
}

/// One plane of a `PresentX` call (0x48-byte stride): frame token,
/// resource count, and pointer to the plane's resources.
#[repr(C)]
pub(crate) struct Plane {
    token: u64,
    count: u32,
    resources: *const This,
    _rest: [u64; 6],
}
const _: () = assert!(std::mem::size_of::<Plane>() == 0x48);
/// Submission tagged with the frame it belongs to `(count, lists, frame
/// token, flags)`; frames are paced elsewhere, so it runs as a plain one.
extern "system" fn execute_x(this: This, count: u32, lists: *const This, _token: u64, _flags: u32) {
    execute(this, count, lists);
}
/// Signal with sync settings for the top and bottom of the pipe; our
/// submissions are complete when they run, so it signals as Signal does.
extern "system" fn signal2_x(
    this: This,
    fence: This,
    value: u64,
    _bottom: *const u8,
    _top: *const u8,
) -> HRESULT {
    signal(this, fence, value)
}
/// Markers and events only matter to profilers.
extern "system" fn marker(_: This) {}
/// Signals now, or leaves a `Late::Signal` for the host's line. The fence
/// is kept from the moment it is pushed until [`crate::host::Host::settle`]
/// lets it go; a signal that lands now balances the keep here.
fn signal_deferred(fence: This, value: u64, defer: impl FnOnce(usize, u64) -> bool) {
    crate::fence::keep(fence);
    if !defer(fence as usize, value) {
        crate::fence::set(fence, value);
        crate::fence::let_go(fence);
    }
}

/// At once, or behind the work the queue holds (its fence, and the memory
/// the title reads it at, move only once that work has run).
extern "system" fn signal(this: This, fence: This, value: u64) -> HRESULT {
    if fence.is_null() {
        return E_INVALIDARG;
    }
    #[cfg(windows)]
    {
        let mut held = held();
        if held.holds_work(this as usize) {
            crate::fence::keep(fence);
            held.push(this as usize, Held::Signal(fence as usize, value));
        } else {
            signal_deferred(fence, value, crate::host::late_signal);
        }
        // Another queue may wait for this fence.
        pump(&mut held, Why::Signal);
    }
    #[cfg(not(windows))]
    {
        let _ = this;
        crate::fence::set(fence, value);
    }
    S_OK
}
/// The queue's later work waits until the fence reaches `value`.
///
/// shortcut: the fence is not kept alive while the queue waits for it;
/// Grand Theft Auto VI makes no queue wait for a fence.
extern "system" fn wait(this: This, fence: This, value: u64) -> HRESULT {
    if fence.is_null() {
        return E_INVALIDARG;
    }
    #[cfg(windows)]
    if crate::host::host().is_some() && crate::fence::value(fence) < value {
        let mut held = held();
        held.push(this as usize, Held::FenceWait(fence as usize, value));
        pump(&mut held, Why::Wait);
    }
    #[cfg(not(windows))]
    let _ = (this, value);
    S_OK
}
/// Ticks per second of GPU timestamps (see [`Written::GpuTimestamp`]).
extern "system" fn timestamp_frequency(_: This, out: *mut u64) -> HRESULT {
    if out.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: checked non-null.
    unsafe { out.write(TIMESTAMP_FREQUENCY) };
    S_OK
}
const TIMESTAMP_FREQUENCY: u64 = 100_000_000;

/// What a list's bottom-of-pipe write stores when the list has run.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Written {
    Value(u64),
    /// The GPU clock: [`TIMESTAMP_FREQUENCY`] ticks per second since the
    /// process started (our GPU work is done on the CPU's timeline).
    GpuTimestamp,
    /// The CPU's performance counter.
    SystemTimestamp,
}

impl Written {
    /// Stores the low `bytes` (4 or 8) of the value at `at`.
    pub(crate) fn write(self, at: *mut u8, bytes: usize) {
        let value = match self {
            Written::Value(v) => v,
            Written::GpuTimestamp => {
                static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
                let elapsed = START.get_or_init(std::time::Instant::now).elapsed();
                (elapsed.as_nanos() / (1_000_000_000 / u128::from(TIMESTAMP_FREQUENCY))) as u64
            }
            Written::SystemTimestamp => system_timestamp(),
        };
        // SAFETY: the title names writable GPU memory (unified with ours).
        unsafe {
            if bytes == 4 {
                at.cast::<u32>().write_unaligned(value as u32);
            } else {
                at.cast::<u64>().write_unaligned(value);
            }
        }
    }
}

fn system_timestamp() -> u64 {
    #[cfg(windows)]
    {
        let mut count = 0;
        // SAFETY: a valid out pointer.
        unsafe { windows_sys::Win32::System::Performance::QueryPerformanceCounter(&mut count) };
        count as u64
    }
    #[cfg(not(windows))]
    0
}

/// Frames presented so far.
static FRAMES: AtomicU64 = AtomicU64::new(0);

pub(crate) fn presented() -> u64 {
    FRAMES.load(Ordering::Relaxed)
}

/// Shows the first resource of each presented plane in our window.
extern "system" fn present(
    this: This,
    planes: u32,
    plane_parameters: *const Plane,
    _parameters: *const u8,
) -> HRESULT {
    #[cfg(windows)]
    let sent = if present_behind() { send_behind(this, planes, plane_parameters) } else { None };
    #[cfg(not(windows))]
    let sent: Option<u64> = None;
    let sent_behind = sent.is_some();
    if !sent_behind {
        if crate::host::late() && !crate::host::late_free_present() {
            crate::host::settle_for_present();
        }
        pump_held(Why::Present);
    }
    let frame = sent.unwrap_or_else(|| FRAMES.fetch_add(1, Ordering::Relaxed) + 1);
    #[cfg(windows)]
    if !sent_behind {
        account_frame(frame);
    }
    #[cfg(feature = "diagnostics")]
    if !sent_behind {
        // A held present counts in `show_present`, where it actually runs.
        crate::frame::presented();
    }
    if frame.is_power_of_two() {
        let (draws, dispatches) = crate::list::take_counts();
        #[cfg(windows)]
        let (drawn, dispatched) = {
            static DRAWN: AtomicU64 = AtomicU64::new(0);
            static DISPATCHED: AtomicU64 = AtomicU64::new(0);
            let since = |last: &AtomicU64, now: u64| now - last.swap(now, Ordering::Relaxed);
            (
                since(&DRAWN, crate::draw::drawn()),
                since(&DISPATCHED, crate::draw::dispatched()),
            )
        };
        #[cfg(not(windows))]
        let (drawn, dispatched) = (0, 0);
        tracing::info!(
            frame,
            draws,
            drawn,
            dispatches,
            dispatched,
            "frame presented (since the last line: draws and dispatches, and those run on the host)"
        );
        #[cfg(windows)]
        if frame >= 256 {
            let (hits, misses) = crate::shader_cache::counts();
            tracing::info!(hits, misses, "shaders from the cache and translated so far");
        }
    }
    #[cfg(all(windows, feature = "diagnostics"))]
    {
        crate::capture::presented(frame);
        if crate::capture::listed(frame) || crate::capture::has_targets() {
            crate::capture::dump_targets(frame);
        }
    }
    #[cfg(windows)]
    if !sent_behind
        && planes > 0
        && !plane_parameters.is_null()
        && let Some(host) = crate::host::host()
    {
        use windows::Win32::Graphics::Direct3D12::ID3D12Resource;
        use windows::core::Interface;

        const COPY_SOURCE: u32 = 0x800;
        // SAFETY: the title passes the plane descriptions.
    // SAFETY: the title passes `planes` plane descriptions.
    let slice = unsafe { std::slice::from_raw_parts(plane_parameters, planes.min(8) as usize) };
        let mut resources: Vec<(This, ID3D12Resource, u32, bool)> =
            Vec::with_capacity(slice.len());
        for plane in slice {
            if plane.count == 0 || plane.resources.is_null() {
                continue;
            }
            // SAFETY: the plane lists `count` resources.
            let resource = unsafe { *plane.resources };
            if let Some(texture) = crate::resource::host_resource(resource) {
                // Distinct title resources may refer to one host texture.
                // Keep its original state for every plane and restore it once.
                let prior = resources.iter().find_map(|(_, earlier, state, _)| {
                    (earlier.as_raw() == texture.as_raw()).then_some(*state)
                });
                let state = prior.unwrap_or_else(|| {
                    crate::resource::host_transition(resource, COPY_SOURCE)
                        .map_or(COPY_SOURCE, |(_, before)| before)
                });
                resources.push((resource, texture, state, prior.is_none()));
            }
        }
        if !resources.is_empty() {
            let sources: Vec<(&_, u32)> = resources
                .iter()
                .map(|(_, tex, state, _)| (tex, *state))
                .collect();
            let shown = host.present(&sources, frame);
            for &(resource, _, state, first) in &resources {
                if first {
                    crate::resource::host_transition(resource, state);
                }
            }
            #[cfg(feature = "diagnostics")]
            blinks(frame, resources[0].0, resources[0].2, shown);
            #[cfg(not(feature = "diagnostics"))]
            let _ = shown;
        }
    }
    #[cfg(not(windows))]
    let _ = (planes, plane_parameters);
    S_OK
}
/// Diagnostics: logs a presented frame that is black after one that was
/// not, with what was presented and how many draws ran since.
#[cfg(all(windows, feature = "diagnostics"))]
fn blinks(frame: u64, resource: This, state: u32, shown: Option<bool>) {
    static LAST: std::sync::Mutex<(bool, u64, u64)> = std::sync::Mutex::new((false, 0, 0));
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    let drawn = crate::draw::drawn();
    let black = shown.unwrap_or(false);
    let draws = crate::draw::take_frame();
    if black && !last.0 {
        last.2 += 1;
        tracing::warn!(
            frame,
            resource = ?resource,
            state = format_args!("{state:#x}"),
            draws = drawn - last.1,
            copied = shown.is_some(),
            black_frames = last.2,
            levels = ?crate::capture::recent_levels(),
            "a black frame after a frame that was not"
        );
        for d in draws {
            tracing::warn!("black frame {frame}: {d}");
        }
        AFTER.store(frame + 6, Ordering::Relaxed);
    }
    // Six frames on: the levels around it (a fade or a single drop).
    static AFTER: AtomicU64 = AtomicU64::new(0);
    if AFTER.load(Ordering::Relaxed) == frame {
        tracing::warn!(frame, levels = ?crate::capture::recent_levels(), "levels after the black frame");
    }
    *last = (black, drawn, last.2);
}

/// A run of destination pages: how they are mapped (0 pool pages from
/// `start` on, 1 pool page `start` each, 2 unmapped, 3 left as they are)
/// and how many.
#[repr(C)]
pub(crate) struct PageRange {
    kind: u32,
    start: u32,
    count: u32,
}

/// The ranges mapped from a destination address on.
#[repr(C)]
pub(crate) struct PageBatch {
    destination: u64,
    count: u32,
    ranges: *const PageRange,
}

/// Bytes per mapped page.
const PAGE: u64 = 64 << 10;

/// `CopyPageMappingsBatchX(count, batches, pool, pool pages, flags)`: maps
/// pages of a pool (memory the title mapped physical pages into) at the
/// batches' addresses. Grand Theft Auto VI streams textures into pool pages
/// and maps them into its tiled heap this way (title+0xf2c470, ranges of
/// one constant page or of unmapped pages). Done at once.
///
/// shortcut: done ahead of work the queue still holds (counted in the frame
/// line); queue it as an item like Signal if that count is not 0.
extern "system" fn copy_page_mappings(
    this: This,
    count: u32,
    batches: *const PageBatch,
    pool: u64,
    pool_pages: u32,
    _flags: u32,
) -> HRESULT {
    let Some(map) = crate::resource::page_mapper() else {
        return E_INVALIDARG;
    };
    if batches.is_null() && count > 0 {
        return E_INVALIDARG;
    }
    #[cfg(windows)]
    if held().holds_work(this as usize) {
        account().mappings += 1;
    }
    #[cfg(not(windows))]
    let _ = this;
    let batches = (count > 0).then(|| {
        // SAFETY: the title passes `count` batches.
        unsafe { std::slice::from_raw_parts(batches, count as usize) }
    });
    let mut ok = true;
    for b in batches.into_iter().flatten() {
        if b.ranges.is_null() {
            continue;
        }
        // SAFETY: each batch holds `count` ranges.
        let ranges = unsafe { std::slice::from_raw_parts(b.ranges, b.count as usize) };
        let mut at = b.destination;
        for r in ranges {
            let len = u64::from(r.count) * PAGE;
            let inside = |pages: u32| r.start.checked_add(pages).is_some_and(|e| e <= pool_pages);
            let source = pool + u64::from(r.start) * PAGE;
            ok &= match r.kind {
                0 if inside(r.count) => map(at, source, len),
                1 if inside(1) => (0..u64::from(r.count)).all(|k| map(at + k * PAGE, source, PAGE)),
                2 => map(at, 0, len),
                3 => true,
                _ => false,
            };
            at += len;
        }
    }
    if !ok {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!(pool = format_args!("{pool:#x}"), "a page mapping failed"));
    }
    S_OK
}

extern "system" fn desc(this: This, out: *mut QueueDesc) {
    if !out.is_null() {
        // SAFETY: checked non-null; the title passes room for a description.
        unsafe { out.write(queue(this).desc) };
    }
}

/// The queue methods by their slot in older titles' queues (the 2020-era
/// layout): IUnknown 3, object 4, device child 1, queue 26 (12 as on PC,
/// then console ones).
const METHODS: [(usize, *const ()); 15] = [
    (0, query as *const ()),
    (2, release as *const ()),
    (7, object::parent_device as *const ()), // 0x38
    (10, execute as *const ()),              // 0x50
    (11, marker as *const ()),               // 0x58 set marker
    (12, marker as *const ()),               // 0x60 begin event
    (13, marker as *const ()),               // 0x68 end event
    (14, signal as *const ()),               // 0x70
    (15, wait as *const ()),                 // 0x78
    (16, timestamp_frequency as *const ()),  // 0x80
    (26, desc as *const ()),                 // 0xd0
    (29, copy_page_mappings as *const ()),   // 0xe8 page mapping batch
    (30, present as *const ()),              // 0xf0
    (33, execute_x as *const ()),            // 0x108
    (32, signal2_x as *const ()),            // 0x100
];

/// Newer titles' queues have two more methods before Signal, which moved
/// from 0x70 to 0x80: every slot from 0x70 on is two later (taken here as
/// starting exactly at 0x70, the methods before it kept in place).
/// Inferred from Grand Theft Auto VI's calls: Signal 0x80 `(fence,
/// value)`, timestamp frequency 0x90 (title+0x24841a6), suspend and resume
/// 0xb8/0xc0, PresentX 0x100, the frame-token submission 0x118 `(count,
/// lists, token, flags)` (title+0x24821dd).
const fn newer_slot(slot: usize) -> usize {
    if slot >= 14 { slot + 2 } else { slot }
}

const fn queue_table(newer: bool) -> [*const (); 36] {
    let mut f = object::d3d12_object(object::missing_table::<QUEUE, 36>());
    let mut i = 0;
    while i < METHODS.len() {
        let (slot, method) = METHODS[i];
        f[if newer { newer_slot(slot) } else { slot }] = method;
        i += 1;
    }
    f
}

static FUNCTIONS: Table<36> = Table(queue_table(false));
static NEWER_FUNCTIONS: Table<36> = Table(queue_table(true));

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::slot;

    /// Distinct title resources can name the same host texture. Presenting
    /// both planes must preserve that texture's original rendering state.
    #[cfg(windows)]
    #[test]
    fn aliased_present_planes_preserve_the_host_texture_state() {
        use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATE_RENDER_TARGET;

        let Some(host) = crate::host::host() else {
            return;
        };
        const TARGET: u32 = D3D12_RESOURCE_STATE_RENDER_TARGET.0 as u32;
        let address = crate::watch::tests::pool(1);
        let null = std::ptr::null();
        let iid = crate::object::guid(
            0x696442be,
            0xa72e,
            0x4059,
            [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
        );
        #[repr(align(8))]
        struct TextureWords([u32; 14]);
        let desc = TextureWords([3, 0, 0, 0, 32, 0, 32, 1 | 1 << 16, 87, 1, 0, 1, 1, 0]);
        let resources = std::array::from_fn::<_, 2, _>(|_| {
            let mut resource = null;
            assert_eq!(
                crate::resource::create_at_address(
                    null,
                    address as u64,
                    desc.0.as_ptr().cast(),
                    0,
                    null.cast(),
                    &iid,
                    &mut resource,
                ),
                S_OK
            );
            resource
        });
        assert_ne!(resources[0], resources[1]);
        let primary = crate::resource::host_copy(resources[0]).unwrap();
        let alias = crate::resource::host_copy(resources[1]).unwrap();
        assert!(std::sync::Arc::ptr_eq(&primary, &alias));
        let list = host.new_list().unwrap();
        crate::list::ensure_state(&list, resources[0], TARGET);
        // SAFETY: the texture is a live render target, on an open list.
        unsafe {
            list.list.ClearRenderTargetView(
                host.rtv(&primary.texture, None),
                &[0.25, 0.5, 0.75, 1.0],
                None,
            );
        }
        list.close();
        host.execute(&[&list]);
        assert_eq!(primary.state.load(Ordering::Relaxed), TARGET);

        let planes = resources.each_ref().map(|resource| Plane {
            token: 0,
            count: 1,
            resources: std::ptr::from_ref(resource),
            _rest: [0; 6],
        });
        assert_eq!(present(null, 2, planes.as_ptr(), null.cast()), S_OK);
        let restored = primary.state.load(Ordering::Relaxed);
        let release = |resource| slot::<extern "system" fn(This) -> u32>(resource, 0x10)(resource);
        for resource in resources {
            assert_eq!(release(resource), 0);
        }
        drop((list, primary, alias));
        crate::watch::tests::free(address);
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
        assert_eq!(restored, TARGET, "the two planes share one rendering state");
    }

    /// Where a queue's signal is: newer layouts have it two slots later.
    /// Another test may switch layouts while this one runs, so the queue's
    /// own table decides.
    fn signal_slot(q: This) -> usize {
        // SAFETY: one of our queues; its first field is the table.
        let table = unsafe { *q.cast::<*const *const ()>() };
        if table == NEWER_FUNCTIONS.0.as_ptr() {
            0x80
        } else {
            0x70
        }
    }

    #[test]
    fn queue_signals_complete_at_once() {
        let null = std::ptr::null();
        let desc = QueueDesc {
            kind: 0,
            priority: 0,
            flags: 0,
            nodes: 0,
        };
        let mut q = null;
        assert_eq!(create(null, &desc, &QUEUE_IIDS[0], &mut q), S_OK);
        // SAFETY: the title reads the type at +0x10.
        assert_eq!(unsafe { q.cast::<u8>().add(0x10).cast::<u32>().read() }, 0);
        let mut fence = null;
        let fence_iid = &crate::fence::FENCE_IIDS[0];
        assert_eq!(
            crate::fence::create(null, 0, 0, fence_iid, &mut fence),
            S_OK
        );
        let signal = slot::<extern "system" fn(This, This, u64) -> HRESULT>(q, signal_slot(q));
        assert_eq!(signal(q, fence, 7), S_OK);
        assert_eq!(
            slot::<extern "system" fn(This) -> u64>(fence, 0x40)(fence),
            7
        );
        assert_eq!(slot::<extern "system" fn(This) -> u32>(q, 0x10)(q), 0);
    }

    /// A title texture with a host copy, a queue, and a fence a PresentX can
    /// be held behind. Only with `VICTORY_ASYNC_PRESENT=1` (and the switches
    /// it needs); otherwise nothing is sent behind.
    #[cfg(windows)]
    struct HeldPresent {
        address: usize,
        resource: This,
        queue: This,
        fence: This,
    }

    #[cfg(windows)]
    impl HeldPresent {
        fn new() -> Option<Self> {
            crate::host::host()?;
            if !present_behind() {
                return None;
            }
            let address = crate::watch::tests::pool(1);
            let null = std::ptr::null();
            let iid = crate::object::guid(0x696442be, 0xa72e, 0x4059, [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad]);
            #[repr(align(8))]
            struct TextureWords([u32; 14]);
            let desc = TextureWords([3, 0, 0, 0, 32, 0, 32, 1 | 1 << 16, 87, 1, 0, 1, 1, 0]);
            let mut resource = null;
            assert_eq!(
                crate::resource::create_at_address(null, address as u64, desc.0.as_ptr().cast(), 0, null.cast(), &iid, &mut resource),
                S_OK
            );
            let mut queue = null;
            let queue_desc = QueueDesc { kind: 0, priority: 0, flags: 0, nodes: 0 };
            assert_eq!(create(null, &queue_desc, &QUEUE_IIDS[0], &mut queue), S_OK);
            let mut fence = null;
            assert_eq!(crate::fence::create(null, 0, 0, &crate::fence::FENCE_IIDS[0], &mut fence), S_OK);
            // The queue's head is a wait nothing satisfies yet, so no worker
            // shows a PresentX queued behind it before the test says so.
            held().push(queue as usize, Held::FenceWait(fence as usize, 1));
            Some(HeldPresent { address, resource, queue, fence })
        }

        fn present(&self) -> Option<u64> {
            let plane = Plane { token: 0, count: 1, resources: &self.resource, _rest: [0; 6] };
            send_behind(self.queue, 1, &plane)
        }

        /// Reaches the fence and waits until every queued PresentX was shown.
        fn finish(self) {
            crate::fence::set(self.fence, 1);
            pump_held(Why::Timer);
            let until = std::time::Instant::now() + std::time::Duration::from_secs(10);
            while PRESENTS_HELD.load(Ordering::Relaxed) > 0 && std::time::Instant::now() < until {
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
            assert_eq!(PRESENTS_HELD.load(Ordering::Relaxed), 0, "the held presents were shown");
            let release = |object| slot::<extern "system" fn(This) -> u32>(object, 0x10)(object);
            release(self.resource);
            release(self.queue);
            release(self.fence);
            crate::watch::tests::free(self.address);
        }
    }

    /// A PresentX sent behind the queue's work is shown in the state the
    /// lists before it leave the texture in (a submission moves the host
    /// texture's state as it is prepared, after PresentX was called), and
    /// the state it leaves is the one the next list starts from.
    #[cfg(windows)]
    #[test]
    fn a_held_present_shows_the_state_the_earlier_lists_left() {
        use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATE_RENDER_TARGET;
        const TARGET: u32 = D3D12_RESOURCE_STATE_RENDER_TARGET.0 as u32;
        let Some(held_present) = HeldPresent::new() else { return };
        let host = crate::host::host().unwrap();
        let copy = crate::resource::host_copy(held_present.resource).unwrap();
        assert!(held_present.present().is_some(), "queued behind the wait");
        // The frame's list runs after PresentX was called: the worker prepares
        // it later. It leaves the texture a render target.
        let list = host.new_list().unwrap();
        crate::list::ensure_state(&list, held_present.resource, TARGET);
        // SAFETY: the texture is a live render target, on an open list.
        unsafe {
            list.list.ClearRenderTargetView(host.rtv(&copy.texture, None), &[0.25, 0.5, 0.75, 1.0], None);
        }
        list.close();
        host.execute(&[&list]);
        assert_eq!(copy.state.load(Ordering::Relaxed), TARGET);
        held_present.finish();
        assert_eq!(
            copy.state.load(Ordering::Relaxed),
            TARGET,
            "the held present put back the state the texture had when PresentX was called"
        );
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// Presents sent behind the queue take consecutive frame numbers when
    /// they are queued, not the same one until the first is shown.
    #[cfg(windows)]
    #[test]
    fn held_presents_take_consecutive_frame_numbers() {
        if frames_ahead() < 2 {
            return;
        }
        let Some(held_present) = HeldPresent::new() else { return };
        let before = presented();
        let first = held_present.present().expect("queued");
        let second = held_present.present().expect("queued");
        assert_eq!(second, first + 1, "{first} then {second}");
        assert_eq!(presented(), before + 2, "both frames are counted once queued");
        held_present.finish();
    }

    /// Grand Theft Auto VI's compute queue: `{COMPUTE, 0, pipe 1, queue 0}`.
    #[test]
    fn console_queues_are_created_as_on_pc() {
        let null = std::ptr::null();
        let desc = QueueDescX {
            kind: 2,
            flags: 0,
            engine_or_pipe: 1,
            queue: 0,
        };
        let mut q = null;
        assert_eq!(create_x(null, &desc, &QUEUE_IIDS[0], &mut q), S_OK);
        assert_eq!(queue(q).desc.kind, 2);
        let none = std::ptr::null();
        assert_eq!(create_x(null, none, &QUEUE_IIDS[0], &mut q), E_INVALIDARG);
    }

    #[test]
    fn newer_queues_have_their_methods_two_slots_later() {
        let (old, new) = (queue_table(false), queue_table(true));
        assert_eq!(old[10], new[10]); // submission stays at 0x50
        assert_eq!(old[14], new[16]); // signal 0x70 → 0x80
        assert_eq!(old[30], new[32]); // present 0xf0 → 0x100
        assert_eq!(new[35], execute_x as *const ()); // 0x118
    }

    #[test]
    fn timestamps_and_values_are_written() {
        let mut at = [0u64; 2];
        Written::Value(0x1_0000_0002).write(at.as_mut_ptr().cast(), 4);
        assert_eq!(at[0], 2);
        Written::GpuTimestamp.write(at[1..].as_mut_ptr().cast(), 8);
        let first = at[1];
        std::thread::sleep(std::time::Duration::from_millis(2));
        Written::GpuTimestamp.write(at[1..].as_mut_ptr().cast(), 8);
        assert!(at[1] > first);
    }

    #[cfg(windows)]
    #[test]
    fn async_submit_worker_drains_pending_work() {
        let mut work = true;
        let mut drained = 0;
        assert!(drain(&mut work, || drained += 1));
        assert_eq!(drained, 1);
        assert!(!work);
        assert!(!drain(&mut work, || drained += 1));
        assert_eq!(drained, 1);
    }

    #[cfg(windows)]
    #[test]
    fn frames_ahead_bounds_held_presents() {
        assert!(!may_send_behind(1, 1));
        assert!(may_send_behind(0, 1));
        assert!(may_send_behind(1, 2));
        assert!(!may_send_behind(2, 2));
        assert_eq!(frames_ahead(), frames_ahead().max(1));
    }

    #[cfg(windows)]
    #[test]
    fn failed_worker_creation_keeps_synchronous_submission() {
        assert!(!worker_started(Err(std::io::Error::other("no worker"))));
    }

    #[cfg(windows)]
    #[test]
    fn async_submit_worker_clears_its_flag_when_idle() {
        let mut work = true;
        let mut due = take_work(&mut work);
        assert!(due, "a wake is taken once");
        assert!(!work, "no stale flag left to spin on");
        assert!(drain(&mut due, || {}), "the taken wake is drained");
        assert!(!due && !work, "idle after draining");
    }

    #[test]
    fn late_signal_keeps_the_fence_until_settled() {
        let mut fence = std::ptr::null();
        assert_eq!(
            crate::fence::create(
                std::ptr::null(),
                1,
                0,
                &crate::fence::FENCE_IIDS[0],
                &mut fence,
            ),
            S_OK
        );
        let before = crate::object::refs(fence as This);
        let mut deferred = false;
        signal_deferred(fence as This, 7, |_, value| {
            deferred = value == 7;
            true
        });
        assert!(deferred);
        assert_eq!(crate::object::refs(fence as This), before + 1, "held by the line");
        assert_eq!(crate::fence::value(fence as This), 1, "not signalled early");
        // What `Host::settle` does with the `Late::Signal`.
        crate::fence::set(fence as This, 7);
        crate::fence::let_go(fence as This);
        assert_eq!(crate::object::refs(fence as This), before);
        assert_eq!(crate::fence::value(fence as This), 7);

        let mut immediate = std::ptr::null();
        assert_eq!(
            crate::fence::create(
                std::ptr::null(),
                0,
                0,
                &crate::fence::FENCE_IIDS[0],
                &mut immediate,
            ),
            S_OK
        );
        let before = crate::object::refs(immediate as This);
        signal_deferred(immediate as This, 9, |_, _| false);
        assert_eq!(crate::object::refs(immediate as This), before, "keep released");
        assert_eq!(crate::fence::value(immediate as This), 9);
    }
}
