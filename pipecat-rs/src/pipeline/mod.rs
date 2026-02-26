// Copyright (c) 2024-2026, Daily
// SPDX-License-Identifier: BSD-2-Clause

//! Pipeline orchestration for connecting and managing frame processors.
//!
//! This module provides Pipeline, PipelineTask, and PipelineRunner for
//! composing and executing frame processing pipelines.

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::{Mutex, Notify};

use crate::frames::{CancelFrame, EndFrame, Frame, StartFrame};
use crate::observers::Observer;
use crate::processors::{
    BaseProcessor, FrameDirection, FrameProcessor, FrameProcessorSetup, drive_processor,
};

/// Source processor at the beginning of a pipeline chain.
/// Simply passes frames through in both directions.
struct PipelineSource {
    base: BaseProcessor,
}

impl PipelineSource {
    fn new(name: String) -> Self {
        Self {
            base: BaseProcessor::new(Some(name), true),
        }
    }
}

impl fmt::Debug for PipelineSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PipelineSource({})", self.base.name())
    }
}

impl fmt::Display for PipelineSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.base.name())
    }
}

#[async_trait]
impl FrameProcessor for PipelineSource {
    fn id(&self) -> u64 { self.base.id() }
    fn name(&self) -> &str { self.base.name() }
    fn is_direct_mode(&self) -> bool { true }

    async fn setup(&mut self, setup: &FrameProcessorSetup) {
        self.base.observer = setup.observer.clone();
    }

    async fn process_frame(&mut self, frame: Arc<dyn Frame>, direction: FrameDirection) {
        self.push_frame(frame, direction).await;
    }

    fn link(&mut self, next: Arc<Mutex<dyn FrameProcessor>>) { self.base.next = Some(next); }
    fn set_prev(&mut self, prev: Arc<Mutex<dyn FrameProcessor>>) { self.base.prev = Some(prev); }
    fn next_processor(&self) -> Option<Arc<Mutex<dyn FrameProcessor>>> { self.base.next.clone() }
    fn prev_processor(&self) -> Option<Arc<Mutex<dyn FrameProcessor>>> { self.base.prev.clone() }
    fn pending_frames_mut(&mut self) -> &mut Vec<(Arc<dyn Frame>, FrameDirection)> {
        &mut self.base.pending_frames
    }
}

/// Sink processor at the end of a pipeline chain.
/// Simply passes frames through in both directions.
struct PipelineSink {
    base: BaseProcessor,
}

impl PipelineSink {
    fn new(name: String) -> Self {
        Self {
            base: BaseProcessor::new(Some(name), true),
        }
    }
}

impl fmt::Debug for PipelineSink {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PipelineSink({})", self.base.name())
    }
}

impl fmt::Display for PipelineSink {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.base.name())
    }
}

#[async_trait]
impl FrameProcessor for PipelineSink {
    fn id(&self) -> u64 { self.base.id() }
    fn name(&self) -> &str { self.base.name() }
    fn is_direct_mode(&self) -> bool { true }

    async fn setup(&mut self, setup: &FrameProcessorSetup) {
        self.base.observer = setup.observer.clone();
    }

    async fn process_frame(&mut self, frame: Arc<dyn Frame>, direction: FrameDirection) {
        self.push_frame(frame, direction).await;
    }

    fn link(&mut self, next: Arc<Mutex<dyn FrameProcessor>>) { self.base.next = Some(next); }
    fn set_prev(&mut self, prev: Arc<Mutex<dyn FrameProcessor>>) { self.base.prev = Some(prev); }
    fn next_processor(&self) -> Option<Arc<Mutex<dyn FrameProcessor>>> { self.base.next.clone() }
    fn prev_processor(&self) -> Option<Arc<Mutex<dyn FrameProcessor>>> { self.base.prev.clone() }
    fn pending_frames_mut(&mut self) -> &mut Vec<(Arc<dyn Frame>, FrameDirection)> {
        &mut self.base.pending_frames
    }
}

/// Main pipeline that connects frame processors in sequence.
///
/// Creates a linear chain: Source -> [Processor1] -> [Processor2] -> ... -> Sink.
pub struct Pipeline {
    base: BaseProcessor,
    processors: Vec<Arc<Mutex<dyn FrameProcessor>>>,
    source_idx: usize,
    sink_idx: usize,
}

impl Pipeline {
    /// Create a new pipeline from a list of processors.
    pub fn new(processors: Vec<Arc<Mutex<dyn FrameProcessor>>>) -> Self {
        let name = format!("Pipeline#{}", crate::utils::base_object::obj_id());

        let source = Arc::new(Mutex::new(PipelineSource::new(format!("{}::Source", name))))
            as Arc<Mutex<dyn FrameProcessor>>;
        let sink = Arc::new(Mutex::new(PipelineSink::new(format!("{}::Sink", name))))
            as Arc<Mutex<dyn FrameProcessor>>;

        let mut all_processors = Vec::with_capacity(processors.len() + 2);
        all_processors.push(source);
        all_processors.extend(processors);
        all_processors.push(sink);

        let source_idx = 0;
        let sink_idx = all_processors.len() - 1;

        Self {
            base: BaseProcessor::new(Some(name), true),
            processors: all_processors,
            source_idx,
            sink_idx,
        }
    }

    /// Link all processors in sequence. Must be called from an async context.
    pub async fn link_processors(&self) {
        for i in 0..self.processors.len() - 1 {
            let next = self.processors[i + 1].clone();
            let current = self.processors[i].clone();

            {
                let mut curr = current.lock().await;
                curr.link(next.clone());
            }
            {
                let mut nxt = next.lock().await;
                nxt.set_prev(current.clone());
            }
        }
    }

    /// Get the source processor.
    pub fn source(&self) -> &Arc<Mutex<dyn FrameProcessor>> {
        &self.processors[self.source_idx]
    }

    /// Get the sink processor.
    pub fn sink(&self) -> &Arc<Mutex<dyn FrameProcessor>> {
        &self.processors[self.sink_idx]
    }

    /// Get all processors including source and sink.
    pub fn all_processors(&self) -> &[Arc<Mutex<dyn FrameProcessor>>] {
        &self.processors
    }
}

impl fmt::Debug for Pipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Pipeline({})", self.base.name())
    }
}

impl fmt::Display for Pipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.base.name())
    }
}

#[async_trait]
impl FrameProcessor for Pipeline {
    fn id(&self) -> u64 { self.base.id() }
    fn name(&self) -> &str { self.base.name() }
    fn is_direct_mode(&self) -> bool { true }

    fn processors(&self) -> Vec<Arc<Mutex<dyn FrameProcessor>>> {
        self.processors.clone()
    }

    fn entry_processors(&self) -> Vec<Arc<Mutex<dyn FrameProcessor>>> {
        vec![self.processors[self.source_idx].clone()]
    }

    async fn setup(&mut self, setup: &FrameProcessorSetup) {
        self.base.observer = setup.observer.clone();
        // Link internal processors
        self.link_processors().await;

        // Connect inner endpoints to outer chain for nested pipeline support.
        // When this pipeline is used as a processor inside another pipeline,
        // self.base.next/prev are set by the outer pipeline's link_processors.
        // We connect the inner sink's next to the outer next, and the inner
        // source's prev to the outer prev, so frames flow through seamlessly.
        if let Some(outer_next) = &self.base.next {
            let mut sink = self.processors[self.sink_idx].lock().await;
            sink.link(outer_next.clone());
        }
        if let Some(outer_prev) = &self.base.prev {
            let mut source = self.processors[self.source_idx].lock().await;
            source.set_prev(outer_prev.clone());
        }

        // Setup all internal processors
        for p in &self.processors {
            let mut processor = p.lock().await;
            processor.setup(setup).await;
        }
    }

    async fn cleanup(&mut self) {
        for p in &self.processors {
            let mut processor = p.lock().await;
            processor.cleanup().await;
        }
    }

    async fn process_frame(&mut self, frame: Arc<dyn Frame>, direction: FrameDirection) {
        // Route frames to the appropriate internal entry point and drive
        // the entire internal chain. This is called while Pipeline's lock is
        // held, but drive_processor only transiently locks internal processors
        // (lock, process, unlock, forward), so there's no deadlock.
        let entry = match direction {
            FrameDirection::Downstream => self.processors[self.source_idx].clone(),
            FrameDirection::Upstream => self.processors[self.sink_idx].clone(),
        };
        drive_processor(entry, frame, direction).await;
    }

    fn link(&mut self, next: Arc<Mutex<dyn FrameProcessor>>) { self.base.next = Some(next); }
    fn set_prev(&mut self, prev: Arc<Mutex<dyn FrameProcessor>>) { self.base.prev = Some(prev); }
    fn next_processor(&self) -> Option<Arc<Mutex<dyn FrameProcessor>>> { self.base.next.clone() }
    fn prev_processor(&self) -> Option<Arc<Mutex<dyn FrameProcessor>>> { self.base.prev.clone() }
    fn pending_frames_mut(&mut self) -> &mut Vec<(Arc<dyn Frame>, FrameDirection)> {
        &mut self.base.pending_frames
    }
}

/// Parameters for pipeline task execution.
#[derive(Debug, Clone)]
pub struct PipelineParams {
    pub allow_interruptions: bool,
    pub enable_metrics: bool,
    pub enable_usage_metrics: bool,
    pub heartbeat_interval_secs: f64,
}

impl Default for PipelineParams {
    fn default() -> Self {
        Self {
            allow_interruptions: false,
            enable_metrics: false,
            enable_usage_metrics: false,
            heartbeat_interval_secs: 5.0,
        }
    }
}

/// Pipeline task that manages a pipeline's execution lifecycle.
///
/// Wraps a pipeline with source/sink processors for frame injection and
/// collection, manages startup (StartFrame), shutdown (EndFrame/CancelFrame),
/// and optional heartbeat monitoring.
pub struct PipelineTask {
    name: String,
    pipeline: Arc<Mutex<Pipeline>>,
    params: PipelineParams,
    observers: Vec<Arc<dyn Observer>>,
    #[allow(dead_code)]
    cancel_on_idle_timeout: bool,
    /// Channel to queue frames into the pipeline task
    frame_tx: tokio::sync::mpsc::Sender<Arc<dyn Frame>>,
    frame_rx: Arc<Mutex<tokio::sync::mpsc::Receiver<Arc<dyn Frame>>>>,
    /// Notifies when the task should stop
    stop_notify: Arc<Notify>,
}

impl PipelineTask {
    pub fn new(
        pipeline: Pipeline,
        params: PipelineParams,
        observers: Vec<Arc<dyn Observer>>,
        cancel_on_idle_timeout: bool,
    ) -> Self {
        let name = format!("PipelineTask#{}", crate::utils::base_object::obj_id());
        let (frame_tx, frame_rx) = tokio::sync::mpsc::channel(1024);

        Self {
            name,
            pipeline: Arc::new(Mutex::new(pipeline)),
            params,
            observers,
            cancel_on_idle_timeout,
            frame_tx,
            frame_rx: Arc::new(Mutex::new(frame_rx)),
            stop_notify: Arc::new(Notify::new()),
        }
    }

    /// Get the task name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Queue a frame to be sent into the pipeline.
    pub async fn queue_frame(&self, frame: Arc<dyn Frame>) {
        let _ = self.frame_tx.send(frame).await;
    }

    /// Run the pipeline task to completion.
    pub async fn run(&self) {
        // Setup the pipeline
        let observer: Option<Arc<dyn Observer>> = if self.observers.len() == 1 {
            Some(self.observers[0].clone())
        } else {
            None
        };

        let setup = FrameProcessorSetup { observer };

        {
            let mut pipeline = self.pipeline.lock().await;
            pipeline.setup(&setup).await;
        }

        // Send StartFrame via drive_processor (no lock held during chain processing)
        let start_frame = Arc::new(StartFrame::new(
            16000,
            24000,
            self.params.allow_interruptions,
            self.params.enable_metrics,
        ));

        {
            let pipeline_dyn = self.pipeline.clone() as Arc<Mutex<dyn FrameProcessor>>;
            drive_processor(pipeline_dyn, start_frame, FrameDirection::Downstream).await;
        }

        // Process queued frames
        let frame_rx = self.frame_rx.clone();
        let pipeline = self.pipeline.clone();
        let stop_notify = self.stop_notify.clone();

        loop {
            let mut rx = frame_rx.lock().await;
            tokio::select! {
                Some(frame) = rx.recv() => {
                    drop(rx);
                    let frame_ref: &dyn Frame = frame.as_ref();
                    let is_end = frame_ref.as_any().downcast_ref::<EndFrame>().is_some();
                    let is_cancel = frame_ref.as_any().downcast_ref::<CancelFrame>().is_some();

                    let pipeline_dyn = pipeline.clone() as Arc<Mutex<dyn FrameProcessor>>;
                    drive_processor(pipeline_dyn, frame, FrameDirection::Downstream).await;

                    if is_end || is_cancel {
                        break;
                    }
                }
                _ = stop_notify.notified() => {
                    break;
                }
            }
        }

        // Cleanup
        {
            let mut pipeline = self.pipeline.lock().await;
            pipeline.cleanup().await;
        }
    }

    /// Stop the pipeline task when current processing is complete.
    pub async fn stop_when_done(&self) {
        let _ = self.frame_tx.send(Arc::new(EndFrame::new())).await;
    }

    /// Cancel the pipeline task immediately.
    pub async fn cancel(&self) {
        let _ = self.frame_tx.send(Arc::new(CancelFrame::new(None))).await;
    }
}

impl fmt::Display for PipelineTask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

/// Pipeline runner that manages task execution with lifecycle handling.
pub struct PipelineRunner {
    name: String,
}

impl PipelineRunner {
    pub fn new() -> Self {
        Self {
            name: format!("PipelineRunner#{}", crate::utils::base_object::obj_id()),
        }
    }

    /// Run a pipeline task to completion.
    pub async fn run(&self, task: &PipelineTask) {
        tracing::debug!("Runner {} started running {}", self.name, task);
        task.run().await;
        tracing::debug!("Runner {} finished running {}", self.name, task);
    }
}

impl Default for PipelineRunner {
    fn default() -> Self {
        Self::new()
    }
}
