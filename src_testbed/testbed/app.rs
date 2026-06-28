//! TestbedApp - the main application runner.

use crate::Camera;
use crate::debug_render::{DebugRenderPipelineResource, debug_render_scene};
use crate::graphics::GraphicsManager;
use crate::harness::Harness;
use crate::mouse::SceneMouse;
use crate::save::SerializableTestbedState;
use crate::testbed::hover::highlight_hovered_body;
use crate::ui;
use kiss3d::color::Color;
use kiss3d::event::{Action, Key, WindowEvent};
use kiss3d::window::Window;
use rapier::dynamics::RigidBodyActivation;
use std::mem;

use super::Plugins;
use super::graphics_context::TestbedGraphics;
use super::keys::KeysState;
use super::state::{RAPIER_BACKEND, RunMode, TestbedActionFlags, TestbedState, TestbedStateFlags};
use super::testbed::{SimulationBuilders, Testbed};

#[cfg(feature = "other-backends")]
use super::OtherBackends;

#[cfg(all(feature = "dim3", feature = "other-backends"))]
use super::state::{PHYSX_BACKEND_PATCH_FRICTION, PHYSX_BACKEND_TWO_FRICTION_DIR};

/// The main testbed application
pub struct TestbedApp {
    builders: SimulationBuilders,
    graphics: GraphicsManager,
    state: TestbedState,
    harness: Harness,
    #[cfg(feature = "other-backends")]
    other_backends: OtherBackends,
    plugins: Plugins,
}

impl TestbedApp {
    pub fn save_file_path() -> String {
        format!("testbed_state_{}.autosave.json", env!("CARGO_CRATE_NAME"))
    }

    pub fn new_empty() -> Self {
        let graphics = GraphicsManager::new();
        let state = TestbedState::default();
        let harness = Harness::new_empty();
        #[cfg(feature = "other-backends")]
        let other_backends = OtherBackends {
            #[cfg(feature = "dim3")]
            physx: None,
        };

        TestbedApp {
            builders: Vec::new(),
            plugins: Plugins(Vec::new()),
            graphics,
            state,
            harness,
            #[cfg(feature = "other-backends")]
            other_backends,
        }
    }

    pub fn from_builders(builders: SimulationBuilders) -> Self {
        let mut res = TestbedApp::new_empty();
        res.set_builders(builders);
        res
    }

    pub fn set_builders(&mut self, builders: SimulationBuilders) {
        use super::state::ExampleEntry;
        use indexmap::IndexSet;

        // Collect unique groups in order of first appearance
        let mut groups: IndexSet<&'static str> = IndexSet::new();
        for example in &builders {
            groups.insert(example.group);
        }

        // Build the display order: group by group, preserving original order within each group
        let mut examples = Vec::new();
        for group in &groups {
            for (builder_index, example) in builders.iter().enumerate() {
                if example.group == *group {
                    examples.push(ExampleEntry {
                        name: example.name,
                        group: example.group,
                        builder_index,
                    });
                }
            }
        }

        self.state.example_groups = groups.into_iter().collect();
        self.state.examples = examples;
        self.builders = builders;
    }

    pub async fn run(self) {
        self.run_with_init(|_| {}).await
    }

    pub async fn run_with_init(mut self, init: impl FnMut(&mut Testbed)) {
        #[cfg(feature = "profiler_ui")]
        profiling::puffin::set_scopes_on(true);

        // Check for benchmark mode
        let args: Vec<String> = std::env::args().collect();
        if args.iter().any(|a| a == "--bench") {
            self.run_benchmark();
            return;
        }

        self.run_async(init).await
    }

    fn run_benchmark(&mut self) {
        use std::fs::File;
        use std::io::{BufWriter, Write};

        let num_bench_iters = 1000u32;
        let builders = mem::take(&mut self.builders);
        let backend_names = self.state.backend_names.clone();

        let collisions_last_name = "rapier (collisions last)";
        let mut summary_lines: Vec<String> = Vec::new();
        summary_lines
            .push("Benchmark Summary: rapier step() vs step_collisions_last()".to_string());
        summary_lines.push(format!(
            "Iterations per benchmark: {} (first skipped as warmup)\n",
            num_bench_iters
        ));
        summary_lines.push(String::new());

        for builder in &builders {
            let mut results = Vec::new();
            println!("Running benchmark for {}", builder.name);

            for (backend_id, backend) in backend_names.iter().enumerate() {
                println!("|_ using backend {backend}");
                self.state.selected_backend = backend_id;
                self.harness = Harness::new_empty();

                let mut testbed = Testbed {
                    graphics: None,
                    state: &mut self.state,
                    harness: &mut self.harness,
                    #[cfg(feature = "other-backends")]
                    other_backends: &mut self.other_backends,
                    plugins: &mut self.plugins,
                };
                (builder.builder)(&mut testbed);

                let mut timings = Vec::new();
                for k in 0..num_bench_iters {
                    if self.state.selected_backend == RAPIER_BACKEND {
                        self.harness.step();
                    }

                    #[cfg(all(feature = "dim3", feature = "other-backends"))]
                    {
                        if self.state.selected_backend == PHYSX_BACKEND_PATCH_FRICTION
                            || self.state.selected_backend == PHYSX_BACKEND_TWO_FRICTION_DIR
                        {
                            self.other_backends.physx.as_mut().unwrap().step(
                                &mut self.harness.physics.pipeline.counters,
                                &self.harness.physics.integration_parameters,
                            );
                            self.other_backends.physx.as_mut().unwrap().sync(
                                &mut self.harness.physics.bodies,
                                &mut self.harness.physics.colliders,
                            );
                        }
                    }

                    if k > 0 {
                        timings.push(self.harness.physics.pipeline.counters.step_time.time_ms());
                    }
                }
                results.push(timings);
            }

            // Run with step_collisions_last.
            {
                println!("|_ using backend {collisions_last_name}");
                self.state.selected_backend = RAPIER_BACKEND;
                self.harness = Harness::new_empty();
                self.harness.use_step_collisions_last = true;

                let mut testbed = Testbed {
                    graphics: None,
                    state: &mut self.state,
                    harness: &mut self.harness,
                    #[cfg(feature = "other-backends")]
                    other_backends: &mut self.other_backends,
                    plugins: &mut self.plugins,
                };
                (builder.builder)(&mut testbed);

                let mut timings = Vec::new();
                for k in 0..num_bench_iters {
                    self.harness.step();
                    if k > 0 {
                        timings.push(self.harness.physics.pipeline.counters.step_time.time_ms());
                    }
                }
                results.push(timings);
                self.harness.use_step_collisions_last = false;
            }

            use inflector::Inflector;
            let filename = format!("{}.csv", builder.name.to_camel_case());
            let mut file = BufWriter::new(File::create(filename).unwrap());

            write!(file, "{}", backend_names[0]).unwrap();
            for backend in &backend_names[1..] {
                write!(file, ",{backend}").unwrap();
            }
            write!(file, ",{collisions_last_name}").unwrap();
            writeln!(file).unwrap();

            for i in 0..results[0].len() {
                write!(file, "{}", results[0][i]).unwrap();
                for result in &results[1..] {
                    write!(file, ",{}", result[i]).unwrap();
                }
                writeln!(file).unwrap();
            }

            // Compute summary statistics for `step()` vs `step_collisions_last()`.
            let rapier_idx = RAPIER_BACKEND;
            let collisions_last_idx = results.len() - 1;
            let rapier_timings = &results[rapier_idx];
            let cl_timings = &results[collisions_last_idx];

            if !rapier_timings.is_empty() && !cl_timings.is_empty() {
                let (r_mean, r_std, r_med, r_q1, r_q3) = compute_stats(rapier_timings);
                let (c_mean, c_std, c_med, c_q1, c_q3) = compute_stats(cl_timings);
                let (t_stat, df, p_value) =
                    welch_t_test(rapier_timings, cl_timings, r_mean, c_mean, r_std, c_std);
                let n = rapier_timings.len();
                let pct_diff = if r_mean.abs() > 1e-12 {
                    (c_mean - r_mean) / r_mean * 100.0
                } else {
                    0.0
                };

                summary_lines.push(format!("=== {} ===", builder.name));
                summary_lines.push(format!("  N = {n}"));
                summary_lines.push(String::new());
                summary_lines.push(format!(
                    "  {:30} {:>12} {:>12}",
                    "", "step()", "step_collisions_last()"
                ));
                summary_lines.push(format!(
                    "  {:30} {:>12.4} {:>12.4}",
                    "Mean (ms):", r_mean, c_mean
                ));
                summary_lines.push(format!(
                    "  {:30} {:>12.4} {:>12.4}",
                    "Std Dev (ms):", r_std, c_std
                ));
                summary_lines.push(format!(
                    "  {:30} {:>12.4} {:>12.4}",
                    "Median (ms):", r_med, c_med
                ));
                summary_lines.push(format!("  {:30} {:>12.4} {:>12.4}", "Q1 (ms):", r_q1, c_q1));
                summary_lines.push(format!("  {:30} {:>12.4} {:>12.4}", "Q3 (ms):", r_q3, c_q3));
                summary_lines.push(format!(
                    "  {:30} {:>12.4} {:>12.4}",
                    "IQR (ms):",
                    r_q3 - r_q1,
                    c_q3 - c_q1
                ));
                summary_lines.push(String::new());
                summary_lines
                    .push(format!("  Difference: {pct_diff:+.2}% (collisions_last vs step)"));
                summary_lines.push(format!(
                    "  Welch's t = {t_stat:.4}, df = {df:.1}, p = {p_value:.6}"
                ));
                let sig = if p_value < 0.01 {
                    "YES (p < 0.01)"
                } else if p_value < 0.05 {
                    "YES (p < 0.05)"
                } else {
                    "NO (p >= 0.05)"
                };
                summary_lines.push(format!("  Statistically significant: {sig}"));
                summary_lines.push(String::new());
            }
        }

        // Write the summary file.
        {
            let summary_path = "benchmark_summary.txt";
            let mut file = BufWriter::new(File::create(summary_path).unwrap());
            for line in &summary_lines {
                writeln!(file, "{line}").unwrap();
            }
            println!("Summary written to {summary_path}");
        }
    }

    async fn run_async(mut self, mut init: impl FnMut(&mut Testbed)) {
        let title = if cfg!(feature = "dim2") {
            "Rapier: 2D demos"
        } else {
            "Rapier: 3D demos"
        };

        let mut window = Window::new_with_size(title, 1280, 720).await;
        window.set_background_color(Color::new(245.0 / 255.0, 245.0 / 255.0, 236.0 / 255.0, 1.0));

        let mut debug_render = DebugRenderPipelineResource::default();
        let mut camera = Camera::default();
        let mut scene_mouse = SceneMouse::new();
        let mut keys = KeysState::default();

        // User init
        let testbed_gfx = TestbedGraphics {
            graphics: &mut self.graphics,
            window: &mut window,
            camera: &mut camera,
            mouse: &mut scene_mouse,
            keys: &mut keys,
            settings: None,
        };

        let mut testbed = Testbed {
            graphics: Some(testbed_gfx),
            state: &mut self.state,
            harness: &mut self.harness,
            #[cfg(feature = "other-backends")]
            other_backends: &mut self.other_backends,
            plugins: &mut self.plugins,
        };

        init(&mut testbed);

        // Main render loop
        #[cfg(feature = "dim3")]
        while window
            .render_3d(self.graphics.scene_mut(), &mut camera)
            .await
        {
            self.run_frame(
                &mut window,
                &mut debug_render,
                &mut camera,
                &mut scene_mouse,
                &mut keys,
            );
        }

        #[cfg(feature = "dim2")]
        while window
            .render_2d(self.graphics.scene_mut(), &mut camera)
            .await
        {
            self.run_frame(
                &mut window,
                &mut debug_render,
                &mut camera,
                &mut scene_mouse,
                &mut keys,
            );
        }
    }

    fn run_frame(
        &mut self,
        window: &mut Window,
        debug_render: &mut DebugRenderPipelineResource,
        camera: &mut Camera,
        scene_mouse: &mut SceneMouse,
        keys: &mut KeysState,
    ) {
        profiling::finish_frame!();

        // Handle input events
        self.handle_events(window, keys);

        // Handle the vehicle controller if there is one.
        #[cfg(feature = "dim3")]
        {
            self.update_vehicle_controller(keys);
        }

        // Update mouse state
        let cursor_pos = window.cursor_pos();
        scene_mouse.update_from_window(cursor_pos, window.size().into(), camera);

        // Handle action flags
        self.handle_action_flags(window, camera, scene_mouse, keys);

        // Handle sleep settings
        self.handle_sleep_settings();

        // Run simulation
        if self.state.running != RunMode::Stop {
            for _ in 0..self.state.nsteps {
                if self.state.selected_backend == RAPIER_BACKEND {
                    let mut testbed_gfx = TestbedGraphics {
                        graphics: &mut self.graphics,
                        window,
                        camera,
                        mouse: scene_mouse,
                        keys,
                        settings: Some(&mut self.state.example_settings),
                    };
                    self.harness.step_with_graphics(Some(&mut testbed_gfx));

                    for plugin in &mut self.plugins.0 {
                        plugin.step(&mut self.harness.physics);
                    }
                }

                #[cfg(all(feature = "dim3", feature = "other-backends"))]
                {
                    if self.state.selected_backend == PHYSX_BACKEND_PATCH_FRICTION
                        || self.state.selected_backend == PHYSX_BACKEND_TWO_FRICTION_DIR
                    {
                        self.other_backends.physx.as_mut().unwrap().step(
                            &mut self.harness.physics.pipeline.counters,
                            &self.harness.physics.integration_parameters,
                        );
                        self.other_backends.physx.as_mut().unwrap().sync(
                            &mut self.harness.physics.bodies,
                            &mut self.harness.physics.colliders,
                        );
                    }
                }

                for plugin in &mut self.plugins.0 {
                    plugin.run_callbacks(&mut self.harness);
                }
            }

            if self.state.running == RunMode::Step {
                self.state.running = RunMode::Stop;
            }
        }

        // Autosave state.
        #[cfg(not(target_arch = "wasm32"))]
        {
            let new_save_data = self.state.save_data(*camera);
            if self.state.prev_save_data != new_save_data {
                // Save the data in a file.
                let data = serde_json::to_string_pretty(&new_save_data).unwrap();
                if let Err(e) = std::fs::write(Self::save_file_path(), &data) {
                    eprintln!("Failed to write autosave file: {}", e);
                }
                self.state.prev_save_data = new_save_data;
            }
        }

        highlight_hovered_body(&mut self.graphics, scene_mouse, &self.harness.physics);

        // Update graphics
        self.graphics.draw(
            self.state.flags,
            &self.harness.physics.bodies,
            &self.harness.physics.colliders,
        );

        // Draw debug render
        debug_render_scene(window, debug_render, &self.harness);

        // Draw UI
        window.draw_ui(|ctx| {
            ui::update_ui(ctx, &mut self.state, &mut self.harness, debug_render);
        });

        self.state.prev_flags = self.state.flags;
    }

    fn handle_events(&mut self, window: &mut Window, keys: &mut KeysState) {
        for event in window.events().iter() {
            match event.value {
                WindowEvent::Key(key, Action::Press, _) => {
                    // Track pressed keys
                    if !keys.pressed_keys.contains(&key) {
                        keys.pressed_keys.push(key);
                    }
                    // Update modifier states
                    match key {
                        Key::LShift | Key::RShift => keys.shift = true,
                        Key::LControl | Key::RControl => keys.ctrl = true,
                        Key::LAlt | Key::RAlt => keys.alt = true,
                        _ => {}
                    }
                }
                WindowEvent::Key(key, Action::Release, _) => {
                    // Remove from pressed keys
                    keys.pressed_keys.retain(|k| *k != key);
                    // Handle special keys
                    match key {
                        Key::T => {
                            if self.state.running == RunMode::Stop {
                                self.state.running = RunMode::Running;
                            } else {
                                self.state.running = RunMode::Stop;
                            }
                        }
                        Key::S => {
                            self.state.running = RunMode::Step;
                        }
                        Key::R => {
                            self.state
                                .action_flags
                                .set(TestbedActionFlags::EXAMPLE_CHANGED, true);
                        }
                        Key::LShift | Key::RShift => keys.shift = false,
                        Key::LControl | Key::RControl => keys.ctrl = false,
                        Key::LAlt | Key::RAlt => keys.alt = false,
                        _ => {}
                    }
                }
                _ => {}
            }
        }
    }

    #[cfg(feature = "dim3")]
    fn update_vehicle_controller(&mut self, keys: &mut KeysState) {
        use rapier::prelude::QueryFilter;

        if self.state.running == RunMode::Stop {
            return;
        }

        if let Some(vehicle) = &mut self.state.vehicle_controller {
            let mut engine_force = 0.0;
            let mut steering_angle = 0.0;

            println!("Pressed: {:?}", keys);
            if keys.pressed(Key::Right) {
                steering_angle += -0.7;
            }
            if keys.pressed(Key::Left) {
                steering_angle += 0.7;
            }
            if keys.pressed(Key::Up) {
                engine_force += 30.0;
            }
            if keys.pressed(Key::Down) {
                engine_force += -30.0;
            }

            let wheels = vehicle.wheels_mut();
            wheels[0].engine_force = engine_force;
            wheels[0].steering = steering_angle;
            wheels[1].engine_force = engine_force;
            wheels[1].steering = steering_angle;

            let query_pipeline = self.harness.physics.broad_phase.as_query_pipeline_mut(
                self.harness.physics.narrow_phase.query_dispatcher(),
                &mut self.harness.physics.bodies,
                &mut self.harness.physics.colliders,
                QueryFilter::exclude_dynamic().exclude_rigid_body(vehicle.chassis),
            );

            vehicle.update_vehicle(
                self.harness.physics.integration_parameters.dt,
                query_pipeline,
            );
        }
    }

    fn handle_action_flags(
        &mut self,
        window: &mut Window,
        camera: &mut Camera,
        scene_mouse: &mut SceneMouse,
        keys: &mut KeysState,
    ) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let app_started = self
                .state
                .action_flags
                .contains(TestbedActionFlags::APP_STARTED);

            if app_started {
                self.state
                    .action_flags
                    .set(TestbedActionFlags::APP_STARTED, false);
                if let Some(saved_state) = std::fs::read(Self::save_file_path())
                    .ok()
                    .and_then(|data| serde_json::from_slice::<SerializableTestbedState>(&data).ok())
                {
                    self.state.apply_saved_data(saved_state, camera);
                    self.state.camera_locked = true;
                }
            }
        }

        let backend_changed = self
            .state
            .action_flags
            .contains(TestbedActionFlags::BACKEND_CHANGED);
        if backend_changed {
            self.state
                .action_flags
                .set(TestbedActionFlags::BACKEND_CHANGED, false);
            self.state
                .action_flags
                .set(TestbedActionFlags::EXAMPLE_CHANGED, true);
            self.state.camera_locked = true;
        }

        let restarted = self
            .state
            .action_flags
            .contains(TestbedActionFlags::RESTART);
        if restarted {
            self.state
                .action_flags
                .set(TestbedActionFlags::RESTART, false);
            self.state.camera_locked = true;
            self.state
                .action_flags
                .set(TestbedActionFlags::EXAMPLE_CHANGED, true);
        }

        let example_changed = self
            .state
            .action_flags
            .contains(TestbedActionFlags::EXAMPLE_CHANGED);
        if example_changed {
            self.state
                .action_flags
                .set(TestbedActionFlags::EXAMPLE_CHANGED, false);
            self.clear(window);
            self.harness.clear_callbacks();

            if !self.state.camera_locked {
                *camera = Camera::default();
            }

            if !restarted {
                self.state.example_settings.clear();
            }

            // Clamp selected_display_index to valid range
            let max_index = self.state.examples.len().saturating_sub(1);
            self.state.selected_display_index = self.state.selected_display_index.min(max_index);

            if !self.builders.is_empty() {
                let builder_index = self.state.selected_builder_index();
                let builder = self.builders[builder_index].builder;
                let testbed_gfx = TestbedGraphics {
                    graphics: &mut self.graphics,
                    window,
                    camera,
                    mouse: scene_mouse,
                    keys,
                    settings: None,
                };

                let mut testbed = Testbed {
                    graphics: Some(testbed_gfx),
                    state: &mut self.state,
                    harness: &mut self.harness,
                    #[cfg(feature = "other-backends")]
                    other_backends: &mut self.other_backends,
                    plugins: &mut self.plugins,
                };
                builder(&mut testbed);
            }

            self.state.camera_locked = false;
        }

        if self
            .state
            .action_flags
            .contains(TestbedActionFlags::RESET_WORLD_GRAPHICS)
        {
            self.state
                .action_flags
                .set(TestbedActionFlags::RESET_WORLD_GRAPHICS, false);
            for (handle, _) in self.harness.physics.bodies.iter() {
                self.graphics.add_body_colliders(
                    window,
                    handle,
                    &self.harness.physics.bodies,
                    &self.harness.physics.colliders,
                );
            }

            for (handle, co) in self.harness.physics.colliders.iter() {
                if co.parent().is_none() {
                    self.graphics
                        .add_collider(window, handle, &self.harness.physics.colliders);
                }
            }
        }

        if self
            .state
            .action_flags
            .contains(TestbedActionFlags::TAKE_SNAPSHOT)
        {
            self.state
                .action_flags
                .set(TestbedActionFlags::TAKE_SNAPSHOT, false);
            self.state.snapshot = Some(self.harness.physics.snapshot());
        }

        if self
            .state
            .action_flags
            .contains(TestbedActionFlags::RESTORE_SNAPSHOT)
        {
            self.state
                .action_flags
                .set(TestbedActionFlags::RESTORE_SNAPSHOT, false);
            if let Some(snapshot) = &self.state.snapshot {
                self.harness.physics.restore_snapshot(snapshot.clone());
                self.state
                    .action_flags
                    .set(TestbedActionFlags::RESET_WORLD_GRAPHICS, true);
            }
        }

        if example_changed
            || self.state.prev_flags.contains(TestbedStateFlags::WIREFRAME)
                != self.state.flags.contains(TestbedStateFlags::WIREFRAME)
        {
            self.graphics.toggle_wireframe_mode(
                &self.harness.physics.colliders,
                self.state.flags.contains(TestbedStateFlags::WIREFRAME),
            );
        }
    }

    fn handle_sleep_settings(&mut self) {
        if self.state.prev_flags.contains(TestbedStateFlags::SLEEP)
            != self.state.flags.contains(TestbedStateFlags::SLEEP)
        {
            if self.state.flags.contains(TestbedStateFlags::SLEEP) {
                for (_, body) in self.harness.physics.bodies.iter_mut() {
                    body.activation_mut().normalized_linear_threshold =
                        RigidBodyActivation::default_normalized_linear_threshold();
                    body.activation_mut().angular_threshold =
                        RigidBodyActivation::default_angular_threshold();
                }
            } else {
                for (_, body) in self.harness.physics.bodies.iter_mut() {
                    body.wake_up(true);
                    body.activation_mut().normalized_linear_threshold = -1.0;
                }
            }
        }
    }

    fn clear(&mut self, window: &mut Window) {
        self.state.can_grab_behind_ground = false;
        self.graphics.clear();

        for mut plugin in self.plugins.0.drain(..) {
            plugin.clear_graphics(&mut self.graphics, window);
        }
    }
}

/// Returns (mean, std_dev, median, q1, q3) for the given data.
fn compute_stats(data: &[f64]) -> (f64, f64, f64, f64, f64) {
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let std_dev = variance.sqrt();

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = percentile_sorted(&sorted, 50.0);
    let q1 = percentile_sorted(&sorted, 25.0);
    let q3 = percentile_sorted(&sorted, 75.0);

    (mean, std_dev, median, q1, q3)
}

/// Linear interpolation percentile on already-sorted data.
fn percentile_sorted(sorted: &[f64], pct: f64) -> f64 {
    if sorted.len() == 1 {
        return sorted[0];
    }
    let rank = pct / 100.0 * (sorted.len() - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = lo + 1;
    let frac = rank - lo as f64;
    if hi >= sorted.len() {
        sorted[lo]
    } else {
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

/// Welch's t-test for two independent samples with unequal variances.
/// Returns (t_statistic, degrees_of_freedom, p_value).
fn welch_t_test(
    a: &[f64],
    b: &[f64],
    mean_a: f64,
    mean_b: f64,
    std_a: f64,
    std_b: f64,
) -> (f64, f64, f64) {
    let n_a = a.len() as f64;
    let n_b = b.len() as f64;
    let var_a = std_a * std_a;
    let var_b = std_b * std_b;
    let se = (var_a / n_a + var_b / n_b).sqrt();

    if se < 1e-15 {
        return (0.0, n_a + n_b - 2.0, 1.0);
    }

    let t = (mean_a - mean_b) / se;

    // Welch-Satterthwaite degrees of freedom.
    let num = (var_a / n_a + var_b / n_b).powi(2);
    let denom = (var_a / n_a).powi(2) / (n_a - 1.0) + (var_b / n_b).powi(2) / (n_b - 1.0);
    let df = num / denom;

    // Two-tailed p-value via regularized incomplete beta function.
    let p = two_tailed_t_p_value(t.abs(), df);
    (t, df, p)
}

/// Approximate two-tailed p-value for Student's t distribution.
/// p = I_{df/(df+t^2)}(df/2, 1/2)  where I is the regularized incomplete beta.
fn two_tailed_t_p_value(t_abs: f64, df: f64) -> f64 {
    let x = df / (df + t_abs * t_abs);
    regularized_incomplete_beta(x, df / 2.0, 0.5)
}

/// Regularized incomplete beta function I_x(a, b) via continued fraction (Lentz).
fn regularized_incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    // Use the symmetry relation if needed for better convergence.
    if x > (a + 1.0) / (a + b + 2.0) {
        return 1.0 - regularized_incomplete_beta(1.0 - x, b, a);
    }

    let ln_prefix = a * x.ln() + b * (1.0 - x).ln() - ln_beta(a, b) - a.ln();
    let prefix = ln_prefix.exp();

    // Lentz's continued fraction.
    let mut f = 1.0_f64;
    let mut c = 1.0_f64;
    let mut d: f64;

    let max_iter = 200;
    let eps = 1e-14;
    let tiny = 1e-30;

    d = 1.0 - (a + b) * x / (a + 1.0);
    if d.abs() < tiny {
        d = tiny;
    }
    d = 1.0 / d;
    f = d;

    for m in 1..=max_iter {
        let m_f = m as f64;

        // Even step: d_{2m}
        let num_even = m_f * (b - m_f) * x / ((a + 2.0 * m_f - 1.0) * (a + 2.0 * m_f));
        d = 1.0 + num_even * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + num_even / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        f *= c * d;

        // Odd step: d_{2m+1}
        let num_odd = -(a + m_f) * (a + b + m_f) * x / ((a + 2.0 * m_f) * (a + 2.0 * m_f + 1.0));
        d = 1.0 + num_odd * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + num_odd / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let delta = c * d;
        f *= delta;

        if (delta - 1.0).abs() < eps {
            break;
        }
    }

    prefix * f
}

/// ln(Beta(a,b)) = ln(Gamma(a)) + ln(Gamma(b)) - ln(Gamma(a+b))
fn ln_beta(a: f64, b: f64) -> f64 {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

/// Lanczos approximation for ln(Gamma(x)).
fn ln_gamma(x: f64) -> f64 {
    let coeffs = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
    ];
    let y = x;
    let tmp = x + 5.5;
    let tmp = tmp - (x + 0.5) * tmp.ln();
    let mut ser = 1.000000000190015_f64;
    for (i, &c) in coeffs.iter().enumerate() {
        ser += c / (y + 1.0 + i as f64);
    }
    -tmp + (2.5066282746310005 * ser / x).ln()
}
