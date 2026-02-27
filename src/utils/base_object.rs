//! Base object providing identification and event handling for all Pipecat components.
//!
//! This is the Rust port of Python's `pipecat.utils.base_object.BaseObject`.
//! Every major component in the framework embeds a [`BaseObject`] to obtain a
//! unique ID, a human-readable name, and a lightweight event-handler system.
//!
//! # Event system
//!
//! Events are registered with [`BaseObject::register_event_handler`] and fired
//! with [`BaseObject::call_event_handler`].  Each event can be either
//! *synchronous* (the handler future is awaited inline) or *asynchronous*
//! (the handler future is spawned as a background Tokio task).  Multiple
//! handlers can be attached to the same event name.

use std::collections::HashMap;
use std::fmt;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use tokio::sync::Mutex;
use tokio::task::JoinHandle;

// ---------------------------------------------------------------------------
// Global counters
// ---------------------------------------------------------------------------

/// Global monotonically-increasing object ID counter.
static OBJECT_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Generate a globally unique object identifier.
///
/// Each call returns a value one greater than the previous call, starting
/// from 0 (matching the Python `itertools.count()` behaviour).
pub fn obj_id() -> u64 {
    OBJECT_ID_COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Per-type instance counters.
///
/// In Python the equivalent uses `collections.defaultdict(itertools.count)`.
/// Here we use a `HashMap<String, AtomicU64>` behind a `std::sync::Mutex` so
/// the function is safe to call from any thread.
static OBJ_COUNTS: std::sync::OnceLock<std::sync::Mutex<HashMap<String, u64>>> =
    std::sync::OnceLock::new();

/// Return a per-type instance count for the given type name, then increment.
///
/// The first call for a given `type_name` returns 0, the second returns 1, etc.
pub fn obj_count(type_name: &str) -> u64 {
    let mut map = OBJ_COUNTS
        .get_or_init(|| std::sync::Mutex::new(HashMap::new()))
        .lock()
        .expect("obj_count lock poisoned");
    let entry = map.entry(type_name.to_string()).or_insert(0);
    let val = *entry;
    *entry += 1;
    val
}

// ---------------------------------------------------------------------------
// Event handler types
// ---------------------------------------------------------------------------

/// Type alias for an async event handler callback.
///
/// Handlers are trait objects that, when called, return a pinned future.
/// They receive no arguments; the caller is expected to capture any required
/// context via `Arc` / `Clone` before registering the handler.
pub type EventHandler = Arc<
    dyn Fn() -> Pin<Box<dyn std::future::Future<Output = ()> + Send>> + Send + Sync,
>;

/// A named event with its registered handlers and execution mode.
pub struct EventHandlerEntry {
    /// The event name this entry corresponds to.
    pub name: String,
    /// The list of handler callbacks registered for this event.
    pub handlers: Vec<EventHandler>,
    /// When `true` each handler future is awaited inline (synchronous
    /// execution).  When `false` each handler is spawned as a background
    /// Tokio task.
    pub is_sync: bool,
}

// ---------------------------------------------------------------------------
// BaseObject
// ---------------------------------------------------------------------------

/// Foundational object providing identification and event handling.
///
/// Every significant component in the Pipecat pipeline embeds or wraps a
/// `BaseObject` to obtain:
///
/// * A unique numeric [`id`](BaseObject::id).
/// * A human-readable [`name`](BaseObject::name) (auto-generated or custom).
/// * A lightweight [event handler system](BaseObject::call_event_handler).
/// * Async [cleanup](BaseObject::cleanup) that waits for in-flight event tasks.
pub struct BaseObject {
    /// Unique numeric identifier.
    id: u64,
    /// Human-readable name.
    name: String,
    /// Registered event handlers keyed by event name.
    event_handlers: HashMap<String, EventHandlerEntry>,
    /// Background (non-sync) event tasks that are still running.  Each entry
    /// stores the event name together with its `JoinHandle` so that
    /// [`cleanup`](BaseObject::cleanup) can report which events it is waiting on.
    event_tasks: Arc<Mutex<Vec<(String, JoinHandle<()>)>>>,
}

impl BaseObject {
    /// Create a new `BaseObject` with an optional custom name.
    ///
    /// When `name` is `None` the name defaults to `"BaseObject#<count>"` where
    /// the count is per-type (i.e. the first `BaseObject` is `#0`, the second
    /// is `#1`, etc.).
    pub fn new(name: Option<String>) -> Self {
        let id = obj_id();
        let name = name.unwrap_or_else(|| format!("BaseObject#{}", obj_count("BaseObject")));
        Self {
            id,
            name,
            event_handlers: HashMap::new(),
            event_tasks: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Create a new `BaseObject` using a given *type name* for auto-naming.
    ///
    /// This is the primary constructor used by derived types.  When `name` is
    /// `None` the generated name follows the pattern `"<type_name>#<count>"`.
    pub fn with_type_name(type_name: &str, name: Option<String>) -> Self {
        let id = obj_id();
        let name = name.unwrap_or_else(|| format!("{}#{}", type_name, obj_count(type_name)));
        Self {
            id,
            name,
            event_handlers: HashMap::new(),
            event_tasks: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Return the unique numeric identifier for this object.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Return the human-readable name of this object.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Register a new event type.
    ///
    /// This must be called before any handlers can be added with
    /// [`add_event_handler`](BaseObject::add_event_handler).
    ///
    /// * `event_name` -- a unique string identifying the event.
    /// * `is_sync` -- when `true`, handlers are awaited inline; when `false`
    ///   they are spawned as background tasks.
    ///
    /// Registering the same event name twice logs a warning and is a no-op.
    pub fn register_event_handler(&mut self, event_name: &str, is_sync: bool) {
        if self.event_handlers.contains_key(event_name) {
            tracing::warn!("{}: event handler {} already registered", self.name, event_name);
            return;
        }
        self.event_handlers.insert(
            event_name.to_string(),
            EventHandlerEntry {
                name: event_name.to_string(),
                handlers: Vec::new(),
                is_sync,
            },
        );
    }

    /// Attach a handler to a previously registered event.
    ///
    /// If `event_name` was never registered via
    /// [`register_event_handler`](BaseObject::register_event_handler) a
    /// warning is logged and the handler is **not** stored.
    pub fn add_event_handler(&mut self, event_name: &str, handler: EventHandler) {
        if let Some(entry) = self.event_handlers.get_mut(event_name) {
            entry.handlers.push(handler);
        } else {
            tracing::warn!("{}: event handler {} not registered", self.name, event_name);
        }
    }

    /// Fire all handlers registered for `event_name`.
    ///
    /// * **Synchronous events** (`is_sync == true`): each handler future is
    ///   awaited sequentially in the caller's task.
    /// * **Asynchronous events** (`is_sync == false`): each handler is spawned
    ///   via `tokio::spawn` and the resulting `JoinHandle` is tracked so that
    ///   [`cleanup`](BaseObject::cleanup) can wait for completion.
    ///
    /// If `event_name` is not registered the call is a silent no-op (matching
    /// the Python behaviour).
    pub async fn call_event_handler(&self, event_name: &str) {
        let entry = match self.event_handlers.get(event_name) {
            Some(e) => e,
            None => return,
        };

        for handler in &entry.handlers {
            if entry.is_sync {
                // Await inline -- any panic will propagate to the caller.
                (handler)().await;
            } else {
                let h = handler.clone();
                let event_tasks = self.event_tasks.clone();
                let ev_name = event_name.to_string();

                let handle = tokio::spawn(async move {
                    (h)().await;
                });

                // Track the spawned task so cleanup can wait on it.
                event_tasks.lock().await.push((ev_name, handle));
            }
        }

        // Prune completed tasks while we are here to avoid unbounded growth.
        self.prune_finished_tasks().await;
    }

    /// Wait for all in-flight background event handler tasks to complete.
    ///
    /// This should be called when the object is being torn down so that no
    /// background work is silently dropped.
    pub async fn cleanup(&self) {
        let tasks = {
            let mut guard = self.event_tasks.lock().await;
            std::mem::take(&mut *guard)
        };

        if tasks.is_empty() {
            return;
        }

        let event_names: Vec<&str> = tasks.iter().map(|(n, _)| n.as_str()).collect();
        tracing::debug!(
            "{}: waiting on event handlers to finish {:?}...",
            self.name,
            event_names,
        );

        for (_name, handle) in tasks {
            // We intentionally ignore join errors (e.g. if a task panicked)
            // to mirror the Python behaviour where exceptions in background
            // event tasks are logged but do not crash the cleanup.
            let _ = handle.await;
        }
    }

    /// Remove tasks from the tracked list that have already completed.
    async fn prune_finished_tasks(&self) {
        let mut guard = self.event_tasks.lock().await;
        guard.retain(|(_name, handle)| !handle.is_finished());
    }
}

// ---------------------------------------------------------------------------
// Trait implementations
// ---------------------------------------------------------------------------

impl fmt::Display for BaseObject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

impl fmt::Debug for BaseObject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BaseObject")
            .field("id", &self.id)
            .field("name", &self.name)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;

    #[test]
    fn obj_id_increments() {
        let a = obj_id();
        let b = obj_id();
        assert_eq!(b, a + 1);
    }

    #[test]
    fn obj_count_per_type() {
        // Use a unique type name so other tests don't interfere.
        let a = obj_count("TestTypeAlpha");
        let b = obj_count("TestTypeAlpha");
        let c = obj_count("TestTypeBeta");
        assert_eq!(b, a + 1);
        // A different type starts at 0.
        assert_eq!(c, 0);
    }

    #[test]
    fn default_name_uses_type_and_count() {
        let obj = BaseObject::with_type_name("MyProcessor", None);
        assert!(obj.name().starts_with("MyProcessor#"));
    }

    #[test]
    fn custom_name_is_used() {
        let obj = BaseObject::new(Some("custom".into()));
        assert_eq!(obj.name(), "custom");
    }

    #[test]
    fn display_and_debug() {
        let obj = BaseObject::new(Some("display-test".into()));
        assert_eq!(format!("{}", obj), "display-test");
        let dbg = format!("{:?}", obj);
        assert!(dbg.contains("display-test"));
    }

    #[test]
    fn register_and_add_handler() {
        let mut obj = BaseObject::new(Some("evtest".into()));
        obj.register_event_handler("on_start", false);
        let handler: EventHandler = Arc::new(|| Box::pin(async {}));
        obj.add_event_handler("on_start", handler);
        assert_eq!(obj.event_handlers["on_start"].handlers.len(), 1);
    }

    #[test]
    fn add_handler_to_unregistered_event_is_noop() {
        let mut obj = BaseObject::new(Some("evtest2".into()));
        let handler: EventHandler = Arc::new(|| Box::pin(async {}));
        obj.add_event_handler("bogus", handler);
        assert!(!obj.event_handlers.contains_key("bogus"));
    }

    #[tokio::test]
    async fn sync_event_handler_runs_inline() {
        let mut obj = BaseObject::new(Some("sync-ev".into()));
        obj.register_event_handler("on_ready", true);

        let flag = Arc::new(AtomicBool::new(false));
        let flag_clone = flag.clone();
        let handler: EventHandler = Arc::new(move || {
            let f = flag_clone.clone();
            Box::pin(async move {
                f.store(true, Ordering::SeqCst);
            })
        });
        obj.add_event_handler("on_ready", handler);

        obj.call_event_handler("on_ready").await;
        // Because the handler is sync, the flag must be set by the time
        // call_event_handler returns.
        assert!(flag.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn async_event_handler_runs_in_background() {
        let mut obj = BaseObject::new(Some("async-ev".into()));
        obj.register_event_handler("on_done", false);

        let flag = Arc::new(AtomicBool::new(false));
        let flag_clone = flag.clone();
        let handler: EventHandler = Arc::new(move || {
            let f = flag_clone.clone();
            Box::pin(async move {
                f.store(true, Ordering::SeqCst);
            })
        });
        obj.add_event_handler("on_done", handler);

        obj.call_event_handler("on_done").await;
        // Wait for the spawned task via cleanup.
        obj.cleanup().await;
        assert!(flag.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn cleanup_is_safe_when_no_tasks() {
        let obj = BaseObject::new(Some("empty-cleanup".into()));
        obj.cleanup().await; // should not panic
    }

    #[tokio::test]
    async fn calling_unregistered_event_is_noop() {
        let obj = BaseObject::new(Some("noop-ev".into()));
        obj.call_event_handler("nonexistent").await; // should not panic
    }
}
