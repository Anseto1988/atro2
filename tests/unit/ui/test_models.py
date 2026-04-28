"""Tests for PipelineModel and related data classes."""
from __future__ import annotations

import pytest

from astroai.ui.models import PipelineModel, PipelineStep, StepState


class TestPipelineStep:
    def test_initial_state(self) -> None:
        step = PipelineStep("cal", "Kalibrierung")
        assert step.key == "cal"
        assert step.label == "Kalibrierung"
        assert step.state is StepState.PENDING
        assert step.progress == 0.0


class TestPipelineModel:
    @pytest.fixture()
    def model(self) -> PipelineModel:
        return PipelineModel()

    def test_default_steps(self, model: PipelineModel) -> None:
        steps = model.steps
        assert len(steps) == 14
        assert steps[0].key == "calibrate"
        assert steps[1].key == "synthetic_flat"
        assert steps[4].key == "comet_stacking"
        assert steps[5].key == "drizzle"
        assert steps[6].key == "mosaic"
        assert steps[7].key == "channel_combine"
        assert steps[8].key == "stretch"
        assert steps[9].key == "color_calibration"
        assert steps[-1].key == "export"

    def test_step_by_key(self, model: PipelineModel) -> None:
        assert model.step_by_key("stack") is not None
        assert model.step_by_key("nonexistent") is None

    def test_set_step_state(self, model: PipelineModel) -> None:
        model.set_step_state("calibrate", StepState.ACTIVE)
        step = model.step_by_key("calibrate")
        assert step is not None
        assert step.state is StepState.ACTIVE

    def test_set_step_progress_clamps(self, model: PipelineModel) -> None:
        model.set_step_progress("stack", 1.5)
        step = model.step_by_key("stack")
        assert step is not None
        assert step.progress == 1.0

        model.set_step_progress("stack", -0.3)
        assert step.progress == 0.0

    def test_active_step(self, model: PipelineModel) -> None:
        assert model.active_step() is None
        model.set_step_state("register", StepState.ACTIVE)
        active = model.active_step()
        assert active is not None
        assert active.key == "register"

    def test_reset(self, model: PipelineModel) -> None:
        model.set_step_state("calibrate", StepState.DONE)
        model.set_step_progress("calibrate", 1.0)
        model.reset()
        optional_enabled = {
            "starless": model.starless_enabled,
            "deconvolution": model.deconvolution_enabled,
            "drizzle": model.drizzle_enabled,
            "mosaic": model.mosaic_enabled,
        }
        for step in model.steps:
            if step.optional and not optional_enabled.get(step.key, False):
                assert step.state is StepState.DISABLED
            else:
                assert step.state is StepState.PENDING
            assert step.progress == 0.0

    def test_advance_to(self, model: PipelineModel) -> None:
        model.advance_to("stack")
        steps = model.steps
        assert steps[0].state is StepState.DONE    # calibrate
        # synthetic_flat at index 1 is DISABLED (optional, not enabled)
        assert steps[1].state is StepState.DISABLED
        assert steps[2].state is StepState.DONE    # register
        assert steps[3].state is StepState.ACTIVE  # stack
        # comet_stacking at index 4 is DISABLED (optional, not enabled)
        assert steps[4].state is StepState.DISABLED
        # drizzle at index 5 is DISABLED (optional, not enabled)
        assert steps[5].state is StepState.DISABLED
        # mosaic at index 6 is DISABLED (optional, not enabled)
        assert steps[6].state is StepState.DISABLED
        # channel_combine at index 7 is DISABLED (optional, not enabled)
        assert steps[7].state is StepState.DISABLED
        assert steps[8].state is StepState.PENDING  # stretch
        # color_calibration at index 9 is DISABLED (optional, not enabled)
        assert steps[9].state is StepState.DISABLED

    def test_step_changed_signal(self, model: PipelineModel, qtbot) -> None:  # type: ignore[no-untyped-def]
        with qtbot.waitSignal(model.step_changed, timeout=500):
            model.set_step_state("calibrate", StepState.ACTIVE)

    def test_progress_changed_signal(self, model: PipelineModel, qtbot) -> None:  # type: ignore[no-untyped-def]
        with qtbot.waitSignal(model.progress_changed, timeout=500):
            model.set_step_progress("calibrate", 0.5)

    def test_pipeline_reset_signal(self, model: PipelineModel, qtbot) -> None:  # type: ignore[no-untyped-def]
        with qtbot.waitSignal(model.pipeline_reset, timeout=500):
            model.reset()


class TestPipelineModelStarless:
    """Tests for PipelineModel starless configuration properties and signals."""

    @pytest.fixture()
    def model(self) -> PipelineModel:
        return PipelineModel()

    # -- starless_enabled property --

    def test_starless_enabled_default(self, model: PipelineModel) -> None:
        assert model.starless_enabled is False

    def test_starless_enabled_setter(self, model: PipelineModel) -> None:
        model.starless_enabled = True
        assert model.starless_enabled is True

    def test_starless_enabled_emits_signal(self, model: PipelineModel, qtbot) -> None:  # type: ignore[no-untyped-def]
        with qtbot.waitSignal(model.starless_config_changed, timeout=500):
            model.starless_enabled = True

    def test_starless_enabled_same_value_no_signal(self, model: PipelineModel, qtbot) -> None:  # type: ignore[no-untyped-def]
        """Setting the same value should NOT emit a signal (no-op optimization)."""
        # Default is False, setting False again should be a no-op
        signals = []
        model.starless_config_changed.connect(lambda: signals.append(True))
        model.starless_enabled = False
        assert len(signals) == 0

    def test_starless_enabled_toggles_step_to_pending(self, model: PipelineModel) -> None:
        """Enabling starless should change the starless step from DISABLED to PENDING."""
        step = model.step_by_key("starless")
        assert step is not None
        assert step.state is StepState.DISABLED  # initially disabled
        model.starless_enabled = True
        assert step.state is StepState.PENDING

    def test_starless_enabled_toggles_step_to_disabled(self, model: PipelineModel) -> None:
        """Disabling starless should change the starless step from PENDING to DISABLED."""
        model.starless_enabled = True
        step = model.step_by_key("starless")
        assert step is not None
        assert step.state is StepState.PENDING
        model.starless_enabled = False
        assert step.state is StepState.DISABLED

    def test_starless_enabled_emits_step_changed(self, model: PipelineModel, qtbot) -> None:  # type: ignore[no-untyped-def]
        """Toggling starless_enabled should emit step_changed with the new state."""
        with qtbot.waitSignal(model.step_changed, timeout=500) as blocker:
            model.starless_enabled = True
        assert blocker.args == ["starless", StepState.PENDING.value]

    # -- starless_strength property --

    def test_starless_strength_default(self, model: PipelineModel) -> None:
        assert model.starless_strength == 1.0

    def test_starless_strength_setter(self, model: PipelineModel) -> None:
        model.starless_strength = 0.7
        assert model.starless_strength == pytest.approx(0.7)

    def test_starless_strength_emits_signal(self, model: PipelineModel, qtbot) -> None:  # type: ignore[no-untyped-def]
        with qtbot.waitSignal(model.starless_config_changed, timeout=500):
            model.starless_strength = 0.5

    def test_starless_strength_same_value_no_signal(self, model: PipelineModel, qtbot) -> None:  # type: ignore[no-untyped-def]
        """Setting the same value should NOT emit a signal."""
        signals = []
        model.starless_config_changed.connect(lambda: signals.append(True))
        model.starless_strength = 1.0  # default is 1.0
        assert len(signals) == 0

    def test_starless_strength_clamps_above_one(self, model: PipelineModel) -> None:
        model.starless_strength = 2.5
        assert model.starless_strength == 1.0

    def test_starless_strength_clamps_below_zero(self, model: PipelineModel) -> None:
        model.starless_strength = -0.3
        assert model.starless_strength == 0.0

    def test_starless_strength_boundary_zero(self, model: PipelineModel) -> None:
        model.starless_strength = 0.0
        assert model.starless_strength == 0.0

    def test_starless_strength_boundary_one(self, model: PipelineModel) -> None:
        model.starless_strength = 1.0
        # Default is already 1.0 so no signal, but value stays valid
        assert model.starless_strength == 1.0

    # -- starless_format property --

    def test_starless_format_default(self, model: PipelineModel) -> None:
        assert model.starless_format == "xisf"

    def test_starless_format_setter(self, model: PipelineModel) -> None:
        model.starless_format = "fits"
        assert model.starless_format == "fits"

    def test_starless_format_emits_signal(self, model: PipelineModel, qtbot) -> None:  # type: ignore[no-untyped-def]
        with qtbot.waitSignal(model.starless_config_changed, timeout=500):
            model.starless_format = "tiff"

    def test_starless_format_same_value_no_signal(self, model: PipelineModel, qtbot) -> None:  # type: ignore[no-untyped-def]
        """Setting the same value should NOT emit a signal."""
        signals = []
        model.starless_config_changed.connect(lambda: signals.append(True))
        model.starless_format = "xisf"  # default
        assert len(signals) == 0

    # -- save_star_mask property --

    def test_save_star_mask_default(self, model: PipelineModel) -> None:
        assert model.save_star_mask is True

    def test_save_star_mask_setter(self, model: PipelineModel) -> None:
        model.save_star_mask = False
        assert model.save_star_mask is False

    def test_save_star_mask_emits_signal(self, model: PipelineModel, qtbot) -> None:  # type: ignore[no-untyped-def]
        with qtbot.waitSignal(model.starless_config_changed, timeout=500):
            model.save_star_mask = False

    def test_save_star_mask_same_value_no_signal(self, model: PipelineModel, qtbot) -> None:  # type: ignore[no-untyped-def]
        """Setting the same value should NOT emit a signal."""
        signals = []
        model.starless_config_changed.connect(lambda: signals.append(True))
        model.save_star_mask = True  # default
        assert len(signals) == 0

    # -- export_config method --

    def test_export_config_default(self, model: PipelineModel) -> None:
        """Default config: starless disabled, so export_starless/star_mask are False."""
        config = model.export_config()
        assert config == {
            "fmt_value": "xisf",
            "export_starless": False,
            "export_star_mask": False,
        }

    def test_export_config_starless_enabled(self, model: PipelineModel) -> None:
        """When starless is enabled and save_star_mask=True, both flags are True."""
        model.starless_enabled = True
        config = model.export_config()
        assert config == {
            "fmt_value": "xisf",
            "export_starless": True,
            "export_star_mask": True,
        }

    def test_export_config_starless_enabled_no_mask(self, model: PipelineModel) -> None:
        """When starless is enabled but save_star_mask=False, mask export is False."""
        model.starless_enabled = True
        model.save_star_mask = False
        config = model.export_config()
        assert config == {
            "fmt_value": "xisf",
            "export_starless": True,
            "export_star_mask": False,
        }

    def test_export_config_respects_format(self, model: PipelineModel) -> None:
        """Export config should reflect the current format setting."""
        model.starless_format = "fits"
        config = model.export_config()
        assert config["fmt_value"] == "fits"

    def test_export_config_disabled_ignores_mask_setting(self, model: PipelineModel) -> None:
        """When starless is disabled, export_star_mask is False regardless of save_star_mask."""
        model.save_star_mask = True  # default, but explicit
        model.starless_enabled = False
        config = model.export_config()
        assert config["export_star_mask"] is False

    # -- reset interaction with starless --

    def test_reset_keeps_starless_disabled_when_not_enabled(self, model: PipelineModel) -> None:
        """After reset, optional starless step should stay DISABLED if starless is off."""
        model.set_step_state("starless", StepState.ERROR)
        model.reset()
        step = model.step_by_key("starless")
        assert step is not None
        assert step.state is StepState.DISABLED

    def test_reset_sets_starless_pending_when_enabled(self, model: PipelineModel) -> None:
        """After reset, starless step should be PENDING if starless is enabled."""
        model.starless_enabled = True
        model.set_step_state("starless", StepState.DONE)
        model.reset()
        step = model.step_by_key("starless")
        assert step is not None
        assert step.state is StepState.PENDING


class TestPipelineModelChannelCombine:
    """Tests for PipelineModel channel_combine configuration properties and signals."""

    @pytest.fixture()
    def model(self) -> PipelineModel:
        return PipelineModel()

    def test_channel_combine_enabled_default(self, model: PipelineModel) -> None:
        assert model.channel_combine_enabled is False

    def test_channel_combine_step_initially_disabled(self, model: PipelineModel) -> None:
        step = model.step_by_key("channel_combine")
        assert step is not None
        assert step.state is StepState.DISABLED

    def test_channel_combine_enabled_toggles_step_to_pending(self, model: PipelineModel) -> None:
        model.channel_combine_enabled = True
        step = model.step_by_key("channel_combine")
        assert step is not None
        assert step.state is StepState.PENDING

    def test_channel_combine_enabled_toggles_step_to_disabled(self, model: PipelineModel) -> None:
        model.channel_combine_enabled = True
        model.channel_combine_enabled = False
        step = model.step_by_key("channel_combine")
        assert step is not None
        assert step.state is StepState.DISABLED

    def test_channel_combine_enabled_emits_signal(self, model: PipelineModel, qtbot) -> None:  # type: ignore[no-untyped-def]
        with qtbot.waitSignal(model.channel_combine_config_changed, timeout=500):
            model.channel_combine_enabled = True

    def test_channel_combine_enabled_same_value_no_signal(self, model: PipelineModel, qtbot) -> None:  # type: ignore[no-untyped-def]
        signals: list[bool] = []
        model.channel_combine_config_changed.connect(lambda: signals.append(True))
        model.channel_combine_enabled = False  # default
        assert len(signals) == 0

    def test_channel_combine_emits_step_changed(self, model: PipelineModel, qtbot) -> None:  # type: ignore[no-untyped-def]
        with qtbot.waitSignal(model.step_changed, timeout=500) as blocker:
            model.channel_combine_enabled = True
        assert blocker.args == ["channel_combine", StepState.PENDING.value]

    def test_channel_combine_mode_default(self, model: PipelineModel) -> None:
        assert model.channel_combine_mode == "lrgb"

    def test_channel_combine_mode_setter(self, model: PipelineModel) -> None:
        model.channel_combine_mode = "narrowband"
        assert model.channel_combine_mode == "narrowband"

    def test_channel_combine_mode_emits_signal(self, model: PipelineModel, qtbot) -> None:  # type: ignore[no-untyped-def]
        with qtbot.waitSignal(model.channel_combine_config_changed, timeout=500):
            model.channel_combine_mode = "narrowband"

    def test_channel_combine_palette_default(self, model: PipelineModel) -> None:
        assert model.channel_combine_palette == "SHO"

    def test_channel_combine_palette_setter(self, model: PipelineModel) -> None:
        model.channel_combine_palette = "HOO"
        assert model.channel_combine_palette == "HOO"

    def test_reset_keeps_channel_combine_disabled_when_not_enabled(self, model: PipelineModel) -> None:
        model.set_step_state("channel_combine", StepState.ERROR)
        model.reset()
        step = model.step_by_key("channel_combine")
        assert step is not None
        assert step.state is StepState.DISABLED

    def test_reset_sets_channel_combine_pending_when_enabled(self, model: PipelineModel) -> None:
        model.channel_combine_enabled = True
        model.set_step_state("channel_combine", StepState.DONE)
        model.reset()
        step = model.step_by_key("channel_combine")
        assert step is not None
        assert step.state is StepState.PENDING


class TestPipelineModelDeconvolution:
    """Tests for PipelineModel deconvolution configuration properties and signals."""

    @pytest.fixture()
    def model(self) -> PipelineModel:
        return PipelineModel()

    # -- deconvolution_enabled --

    def test_deconvolution_enabled_default(self, model: PipelineModel) -> None:
        assert model.deconvolution_enabled is False

    def test_deconvolution_enabled_setter(self, model: PipelineModel) -> None:
        model.deconvolution_enabled = True
        assert model.deconvolution_enabled is True

    def test_deconvolution_enabled_emits_signal(self, model: PipelineModel, qtbot) -> None:  # type: ignore[no-untyped-def]
        with qtbot.waitSignal(model.deconvolution_config_changed, timeout=500):
            model.deconvolution_enabled = True

    def test_deconvolution_enabled_same_value_no_signal(self, model: PipelineModel) -> None:
        signals: list[bool] = []
        model.deconvolution_config_changed.connect(lambda: signals.append(True))
        model.deconvolution_enabled = False  # default
        assert len(signals) == 0

    def test_deconvolution_enabled_toggles_step_to_pending(self, model: PipelineModel) -> None:
        step = model.step_by_key("deconvolution")
        assert step is not None
        assert step.state is StepState.DISABLED
        model.deconvolution_enabled = True
        assert step.state is StepState.PENDING

    def test_deconvolution_enabled_toggles_step_to_disabled(self, model: PipelineModel) -> None:
        model.deconvolution_enabled = True
        step = model.step_by_key("deconvolution")
        assert step is not None
        assert step.state is StepState.PENDING
        model.deconvolution_enabled = False
        assert step.state is StepState.DISABLED

    def test_deconvolution_enabled_emits_step_changed_pending(self, model: PipelineModel, qtbot) -> None:  # type: ignore[no-untyped-def]
        with qtbot.waitSignal(model.step_changed, timeout=500) as blocker:
            model.deconvolution_enabled = True
        assert blocker.args == ["deconvolution", StepState.PENDING.value]

    def test_deconvolution_enabled_emits_step_changed_disabled(self, model: PipelineModel, qtbot) -> None:  # type: ignore[no-untyped-def]
        model.deconvolution_enabled = True
        with qtbot.waitSignal(model.step_changed, timeout=500) as blocker:
            model.deconvolution_enabled = False
        assert blocker.args == ["deconvolution", StepState.DISABLED.value]

    # -- deconvolution_iterations --

    def test_deconvolution_iterations_default(self, model: PipelineModel) -> None:
        assert model.deconvolution_iterations == 10

    def test_deconvolution_iterations_setter(self, model: PipelineModel) -> None:
        model.deconvolution_iterations = 50
        assert model.deconvolution_iterations == 50

    def test_deconvolution_iterations_emits_signal(self, model: PipelineModel, qtbot) -> None:  # type: ignore[no-untyped-def]
        with qtbot.waitSignal(model.deconvolution_config_changed, timeout=500):
            model.deconvolution_iterations = 20

    def test_deconvolution_iterations_clamps_below_one(self, model: PipelineModel) -> None:
        model.deconvolution_iterations = 0
        assert model.deconvolution_iterations == 1

    def test_deconvolution_iterations_clamps_above_100(self, model: PipelineModel) -> None:
        model.deconvolution_iterations = 200
        assert model.deconvolution_iterations == 100

    def test_deconvolution_iterations_same_value_no_signal(self, model: PipelineModel) -> None:
        signals: list[bool] = []
        model.deconvolution_config_changed.connect(lambda: signals.append(True))
        model.deconvolution_iterations = 10  # default
        assert len(signals) == 0

    # -- deconvolution_psf_sigma --

    def test_deconvolution_psf_sigma_default(self, model: PipelineModel) -> None:
        assert model.deconvolution_psf_sigma == pytest.approx(1.0)

    def test_deconvolution_psf_sigma_setter(self, model: PipelineModel) -> None:
        model.deconvolution_psf_sigma = 2.5
        assert model.deconvolution_psf_sigma == pytest.approx(2.5)

    def test_deconvolution_psf_sigma_emits_signal(self, model: PipelineModel, qtbot) -> None:  # type: ignore[no-untyped-def]
        with qtbot.waitSignal(model.deconvolution_config_changed, timeout=500):
            model.deconvolution_psf_sigma = 3.0

    def test_deconvolution_psf_sigma_clamps_below_min(self, model: PipelineModel) -> None:
        model.deconvolution_psf_sigma = 0.0
        assert model.deconvolution_psf_sigma == pytest.approx(0.1)

    def test_deconvolution_psf_sigma_clamps_above_max(self, model: PipelineModel) -> None:
        model.deconvolution_psf_sigma = 99.0
        assert model.deconvolution_psf_sigma == pytest.approx(10.0)

    def test_deconvolution_psf_sigma_same_value_no_signal(self, model: PipelineModel) -> None:
        signals: list[bool] = []
        model.deconvolution_config_changed.connect(lambda: signals.append(True))
        model.deconvolution_psf_sigma = 1.0  # default
        assert len(signals) == 0

    # -- _update_deconvolution_step_state: step is None guard --

    def test_update_deconvolution_step_state_missing_step(self, model: PipelineModel) -> None:
        """_update_deconvolution_step_state returns early when step not found."""
        model._steps = [s for s in model._steps if s.key != "deconvolution"]
        # Should not raise
        model._update_deconvolution_step_state()

    def test_reset_keeps_deconvolution_disabled_when_not_enabled(self, model: PipelineModel) -> None:
        model.set_step_state("deconvolution", StepState.ERROR)
        model.reset()
        step = model.step_by_key("deconvolution")
        assert step is not None
        assert step.state is StepState.DISABLED

    def test_reset_sets_deconvolution_pending_when_enabled(self, model: PipelineModel) -> None:
        model.deconvolution_enabled = True
        model.set_step_state("deconvolution", StepState.DONE)
        model.reset()
        step = model.step_by_key("deconvolution")
        assert step is not None
        assert step.state is StepState.PENDING


class TestPipelineModelDrizzle:
    """Tests for PipelineModel drizzle configuration properties and signals."""

    @pytest.fixture()
    def model(self) -> PipelineModel:
        return PipelineModel()

    # -- drizzle_enabled --

    def test_drizzle_enabled_default(self, model: PipelineModel) -> None:
        assert model.drizzle_enabled is False

    def test_drizzle_enabled_setter(self, model: PipelineModel) -> None:
        model.drizzle_enabled = True
        assert model.drizzle_enabled is True

    def test_drizzle_enabled_emits_signal(self, model: PipelineModel, qtbot) -> None:  # type: ignore[no-untyped-def]
        with qtbot.waitSignal(model.drizzle_config_changed, timeout=500):
            model.drizzle_enabled = True

    def test_drizzle_enabled_same_value_no_signal(self, model: PipelineModel) -> None:
        signals: list[bool] = []
        model.drizzle_config_changed.connect(lambda: signals.append(True))
        model.drizzle_enabled = False  # default
        assert len(signals) == 0

    def test_drizzle_enabled_toggles_step_to_pending(self, model: PipelineModel) -> None:
        step = model.step_by_key("drizzle")
        assert step is not None
        assert step.state is StepState.DISABLED
        model.drizzle_enabled = True
        assert step.state is StepState.PENDING

    def test_drizzle_enabled_toggles_step_to_disabled(self, model: PipelineModel) -> None:
        model.drizzle_enabled = True
        model.drizzle_enabled = False
        step = model.step_by_key("drizzle")
        assert step is not None
        assert step.state is StepState.DISABLED

    def test_drizzle_enabled_emits_step_changed_pending(self, model: PipelineModel, qtbot) -> None:  # type: ignore[no-untyped-def]
        with qtbot.waitSignal(model.step_changed, timeout=500) as blocker:
            model.drizzle_enabled = True
        assert blocker.args == ["drizzle", StepState.PENDING.value]

    def test_drizzle_enabled_emits_step_changed_disabled(self, model: PipelineModel, qtbot) -> None:  # type: ignore[no-untyped-def]
        model.drizzle_enabled = True
        with qtbot.waitSignal(model.step_changed, timeout=500) as blocker:
            model.drizzle_enabled = False
        assert blocker.args == ["drizzle", StepState.DISABLED.value]

    # -- drizzle_drop_size --

    def test_drizzle_drop_size_default(self, model: PipelineModel) -> None:
        assert model.drizzle_drop_size == pytest.approx(0.7)

    def test_drizzle_drop_size_setter(self, model: PipelineModel) -> None:
        model.drizzle_drop_size = 0.9
        assert model.drizzle_drop_size == pytest.approx(0.9)

    def test_drizzle_drop_size_emits_signal(self, model: PipelineModel, qtbot) -> None:  # type: ignore[no-untyped-def]
        with qtbot.waitSignal(model.drizzle_config_changed, timeout=500):
            model.drizzle_drop_size = 0.5

    def test_drizzle_drop_size_same_value_no_signal(self, model: PipelineModel) -> None:
        signals: list[bool] = []
        model.drizzle_config_changed.connect(lambda: signals.append(True))
        model.drizzle_drop_size = 0.7  # default
        assert len(signals) == 0

    # -- drizzle_scale --

    def test_drizzle_scale_default(self, model: PipelineModel) -> None:
        assert model.drizzle_scale == pytest.approx(1.0)

    def test_drizzle_scale_setter(self, model: PipelineModel) -> None:
        model.drizzle_scale = 2.0
        assert model.drizzle_scale == pytest.approx(2.0)

    def test_drizzle_scale_emits_signal(self, model: PipelineModel, qtbot) -> None:  # type: ignore[no-untyped-def]
        with qtbot.waitSignal(model.drizzle_config_changed, timeout=500):
            model.drizzle_scale = 1.5

    def test_drizzle_scale_clamps_below_min(self, model: PipelineModel) -> None:
        model.drizzle_scale = 0.1
        assert model.drizzle_scale == pytest.approx(0.5)

    def test_drizzle_scale_clamps_above_max(self, model: PipelineModel) -> None:
        model.drizzle_scale = 10.0
        assert model.drizzle_scale == pytest.approx(3.0)

    def test_drizzle_scale_same_value_no_signal(self, model: PipelineModel) -> None:
        signals: list[bool] = []
        model.drizzle_config_changed.connect(lambda: signals.append(True))
        model.drizzle_scale = 1.0  # default
        assert len(signals) == 0

    # -- drizzle_pixfrac --

    def test_drizzle_pixfrac_default(self, model: PipelineModel) -> None:
        assert model.drizzle_pixfrac == pytest.approx(1.0)

    def test_drizzle_pixfrac_setter(self, model: PipelineModel) -> None:
        model.drizzle_pixfrac = 0.8
        assert model.drizzle_pixfrac == pytest.approx(0.8)

    def test_drizzle_pixfrac_emits_signal(self, model: PipelineModel, qtbot) -> None:  # type: ignore[no-untyped-def]
        with qtbot.waitSignal(model.drizzle_config_changed, timeout=500):
            model.drizzle_pixfrac = 0.5

    def test_drizzle_pixfrac_clamps_below_min(self, model: PipelineModel) -> None:
        model.drizzle_pixfrac = 0.0
        assert model.drizzle_pixfrac == pytest.approx(0.1)

    def test_drizzle_pixfrac_clamps_above_max(self, model: PipelineModel) -> None:
        model.drizzle_pixfrac = 2.0
        assert model.drizzle_pixfrac == pytest.approx(1.0)

    def test_drizzle_pixfrac_same_value_no_signal(self, model: PipelineModel) -> None:
        signals: list[bool] = []
        model.drizzle_config_changed.connect(lambda: signals.append(True))
        model.drizzle_pixfrac = 1.0  # default
        assert len(signals) == 0

    # -- _update_drizzle_step_state: step is None guard --

    def test_update_drizzle_step_state_missing_step(self, model: PipelineModel) -> None:
        model._steps = [s for s in model._steps if s.key != "drizzle"]
        model._update_drizzle_step_state()  # should not raise

    def test_reset_keeps_drizzle_disabled_when_not_enabled(self, model: PipelineModel) -> None:
        model.set_step_state("drizzle", StepState.ERROR)
        model.reset()
        step = model.step_by_key("drizzle")
        assert step is not None
        assert step.state is StepState.DISABLED

    def test_reset_sets_drizzle_pending_when_enabled(self, model: PipelineModel) -> None:
        model.drizzle_enabled = True
        model.set_step_state("drizzle", StepState.DONE)
        model.reset()
        step = model.step_by_key("drizzle")
        assert step is not None
        assert step.state is StepState.PENDING


class TestPipelineModelMosaic:
    """Tests for PipelineModel mosaic configuration properties and signals."""

    @pytest.fixture()
    def model(self) -> PipelineModel:
        return PipelineModel()

    def test_mosaic_enabled_default(self, model: PipelineModel) -> None:
        assert model.mosaic_enabled is False

    def test_mosaic_enabled_setter(self, model: PipelineModel) -> None:
        model.mosaic_enabled = True
        assert model.mosaic_enabled is True

    def test_mosaic_enabled_emits_signal(self, model: PipelineModel, qtbot) -> None:  # type: ignore[no-untyped-def]
        with qtbot.waitSignal(model.mosaic_config_changed, timeout=500):
            model.mosaic_enabled = True

    def test_mosaic_enabled_same_value_no_signal(self, model: PipelineModel) -> None:
        signals: list[bool] = []
        model.mosaic_config_changed.connect(lambda: signals.append(True))
        model.mosaic_enabled = False
        assert len(signals) == 0

    def test_mosaic_enabled_toggles_step_to_pending(self, model: PipelineModel) -> None:
        step = model.step_by_key("mosaic")
        assert step is not None
        assert step.state is StepState.DISABLED
        model.mosaic_enabled = True
        assert step.state is StepState.PENDING

    def test_mosaic_enabled_toggles_step_to_disabled(self, model: PipelineModel) -> None:
        model.mosaic_enabled = True
        model.mosaic_enabled = False
        step = model.step_by_key("mosaic")
        assert step is not None
        assert step.state is StepState.DISABLED

    def test_mosaic_enabled_emits_step_changed_pending(self, model: PipelineModel, qtbot) -> None:  # type: ignore[no-untyped-def]
        with qtbot.waitSignal(model.step_changed, timeout=500) as blocker:
            model.mosaic_enabled = True
        assert blocker.args == ["mosaic", StepState.PENDING.value]

    def test_mosaic_blend_mode_default(self, model: PipelineModel) -> None:
        assert model.mosaic_blend_mode == "average"

    def test_mosaic_blend_mode_setter(self, model: PipelineModel) -> None:
        model.mosaic_blend_mode = "overlay"
        assert model.mosaic_blend_mode == "overlay"

    def test_mosaic_blend_mode_emits_signal(self, model: PipelineModel, qtbot) -> None:  # type: ignore[no-untyped-def]
        with qtbot.waitSignal(model.mosaic_config_changed, timeout=500):
            model.mosaic_blend_mode = "overlay"

    def test_mosaic_blend_mode_same_value_no_signal(self, model: PipelineModel) -> None:
        signals: list[bool] = []
        model.mosaic_config_changed.connect(lambda: signals.append(True))
        model.mosaic_blend_mode = "average"
        assert len(signals) == 0

    def test_mosaic_gradient_correct_default(self, model: PipelineModel) -> None:
        assert model.mosaic_gradient_correct is True

    def test_mosaic_gradient_correct_setter(self, model: PipelineModel) -> None:
        model.mosaic_gradient_correct = False
        assert model.mosaic_gradient_correct is False

    def test_mosaic_gradient_correct_emits_signal(self, model: PipelineModel, qtbot) -> None:  # type: ignore[no-untyped-def]
        with qtbot.waitSignal(model.mosaic_config_changed, timeout=500):
            model.mosaic_gradient_correct = False

    def test_mosaic_output_scale_default(self, model: PipelineModel) -> None:
        assert model.mosaic_output_scale == pytest.approx(1.0)

    def test_mosaic_output_scale_setter(self, model: PipelineModel) -> None:
        model.mosaic_output_scale = 2.0
        assert model.mosaic_output_scale == pytest.approx(2.0)

    def test_mosaic_output_scale_clamps_below_min(self, model: PipelineModel) -> None:
        model.mosaic_output_scale = 0.1
        assert model.mosaic_output_scale == pytest.approx(0.25)

    def test_mosaic_output_scale_clamps_above_max(self, model: PipelineModel) -> None:
        model.mosaic_output_scale = 10.0
        assert model.mosaic_output_scale == pytest.approx(4.0)

    # -- mosaic_panels --

    def test_mosaic_panels_default_empty(self, model: PipelineModel) -> None:
        assert model.mosaic_panels == []

    def test_mosaic_panels_setter(self, model: PipelineModel) -> None:
        model.mosaic_panels = ["/a/b.fit", "/c/d.fit"]
        assert model.mosaic_panels == ["/a/b.fit", "/c/d.fit"]

    def test_mosaic_panels_setter_emits_signal(self, model: PipelineModel, qtbot) -> None:  # type: ignore[no-untyped-def]
        with qtbot.waitSignal(model.mosaic_config_changed, timeout=500):
            model.mosaic_panels = ["/x.fit"]

    def test_mosaic_panels_same_value_no_signal(self, model: PipelineModel) -> None:
        signals: list[bool] = []
        model.mosaic_config_changed.connect(lambda: signals.append(True))
        model.mosaic_panels = []  # default
        assert len(signals) == 0

    def test_mosaic_panels_returns_copy(self, model: PipelineModel) -> None:
        model.mosaic_panels = ["/a.fit"]
        copy = model.mosaic_panels
        copy.append("/b.fit")
        assert model.mosaic_panels == ["/a.fit"]

    def test_add_mosaic_panel(self, model: PipelineModel) -> None:
        model.add_mosaic_panel("/a.fit")
        assert "/a.fit" in model.mosaic_panels

    def test_add_mosaic_panel_emits_signal(self, model: PipelineModel, qtbot) -> None:  # type: ignore[no-untyped-def]
        with qtbot.waitSignal(model.mosaic_config_changed, timeout=500):
            model.add_mosaic_panel("/a.fit")

    def test_add_mosaic_panel_no_duplicate(self, model: PipelineModel) -> None:
        signals: list[bool] = []
        model.add_mosaic_panel("/a.fit")
        model.mosaic_config_changed.connect(lambda: signals.append(True))
        model.add_mosaic_panel("/a.fit")
        assert len(signals) == 0
        assert model.mosaic_panels.count("/a.fit") == 1

    def test_remove_mosaic_panel(self, model: PipelineModel) -> None:
        model.add_mosaic_panel("/a.fit")
        model.remove_mosaic_panel("/a.fit")
        assert "/a.fit" not in model.mosaic_panels

    def test_remove_mosaic_panel_emits_signal(self, model: PipelineModel, qtbot) -> None:  # type: ignore[no-untyped-def]
        model.add_mosaic_panel("/b.fit")
        with qtbot.waitSignal(model.mosaic_config_changed, timeout=500):
            model.remove_mosaic_panel("/b.fit")

    def test_remove_mosaic_panel_absent_no_signal(self, model: PipelineModel) -> None:
        signals: list[bool] = []
        model.mosaic_config_changed.connect(lambda: signals.append(True))
        model.remove_mosaic_panel("/nonexistent.fit")
        assert len(signals) == 0

    # -- _update_mosaic_step_state: step is None guard --

    def test_update_mosaic_step_state_missing_step(self, model: PipelineModel) -> None:
        model._steps = [s for s in model._steps if s.key != "mosaic"]
        model._update_mosaic_step_state()  # should not raise

    def test_reset_keeps_mosaic_disabled_when_not_enabled(self, model: PipelineModel) -> None:
        model.set_step_state("mosaic", StepState.ERROR)
        model.reset()
        step = model.step_by_key("mosaic")
        assert step is not None
        assert step.state is StepState.DISABLED

    def test_reset_sets_mosaic_pending_when_enabled(self, model: PipelineModel) -> None:
        model.mosaic_enabled = True
        model.set_step_state("mosaic", StepState.DONE)
        model.reset()
        step = model.step_by_key("mosaic")
        assert step is not None
        assert step.state is StepState.PENDING


class TestPipelineModelColorCalibration:
    """Tests for PipelineModel color_calibration configuration properties and signals."""

    @pytest.fixture()
    def model(self) -> PipelineModel:
        return PipelineModel()

    # -- color_calibration_enabled --

    def test_color_calibration_enabled_default(self, model: PipelineModel) -> None:
        assert model.color_calibration_enabled is False

    def test_color_calibration_enabled_setter(self, model: PipelineModel) -> None:
        model.color_calibration_enabled = True
        assert model.color_calibration_enabled is True

    def test_color_calibration_enabled_emits_signal(self, model: PipelineModel, qtbot) -> None:  # type: ignore[no-untyped-def]
        with qtbot.waitSignal(model.color_calibration_config_changed, timeout=500):
            model.color_calibration_enabled = True

    def test_color_calibration_enabled_same_value_no_signal(self, model: PipelineModel) -> None:
        signals: list[bool] = []
        model.color_calibration_config_changed.connect(lambda: signals.append(True))
        model.color_calibration_enabled = False
        assert len(signals) == 0

    def test_color_calibration_enabled_toggles_step_to_pending(self, model: PipelineModel) -> None:
        step = model.step_by_key("color_calibration")
        assert step is not None
        assert step.state is StepState.DISABLED
        model.color_calibration_enabled = True
        assert step.state is StepState.PENDING

    def test_color_calibration_enabled_toggles_step_to_disabled(self, model: PipelineModel) -> None:
        model.color_calibration_enabled = True
        model.color_calibration_enabled = False
        step = model.step_by_key("color_calibration")
        assert step is not None
        assert step.state is StepState.DISABLED

    def test_color_calibration_enabled_emits_step_changed_pending(self, model: PipelineModel, qtbot) -> None:  # type: ignore[no-untyped-def]
        with qtbot.waitSignal(model.step_changed, timeout=500) as blocker:
            model.color_calibration_enabled = True
        assert blocker.args == ["color_calibration", StepState.PENDING.value]

    def test_color_calibration_enabled_emits_step_changed_disabled(self, model: PipelineModel, qtbot) -> None:  # type: ignore[no-untyped-def]
        model.color_calibration_enabled = True
        with qtbot.waitSignal(model.step_changed, timeout=500) as blocker:
            model.color_calibration_enabled = False
        assert blocker.args == ["color_calibration", StepState.DISABLED.value]

    # -- color_calibration_catalog --

    def test_color_calibration_catalog_default(self, model: PipelineModel) -> None:
        assert model.color_calibration_catalog == "gaia_dr3"

    def test_color_calibration_catalog_setter(self, model: PipelineModel) -> None:
        model.color_calibration_catalog = "tycho2"
        assert model.color_calibration_catalog == "tycho2"

    def test_color_calibration_catalog_emits_signal(self, model: PipelineModel, qtbot) -> None:  # type: ignore[no-untyped-def]
        with qtbot.waitSignal(model.color_calibration_config_changed, timeout=500):
            model.color_calibration_catalog = "tycho2"

    def test_color_calibration_catalog_same_value_no_signal(self, model: PipelineModel) -> None:
        signals: list[bool] = []
        model.color_calibration_config_changed.connect(lambda: signals.append(True))
        model.color_calibration_catalog = "gaia_dr3"
        assert len(signals) == 0

    # -- color_calibration_sample_radius --

    def test_color_calibration_sample_radius_default(self, model: PipelineModel) -> None:
        assert model.color_calibration_sample_radius == 8

    def test_color_calibration_sample_radius_setter(self, model: PipelineModel) -> None:
        model.color_calibration_sample_radius = 12
        assert model.color_calibration_sample_radius == 12

    def test_color_calibration_sample_radius_emits_signal(self, model: PipelineModel, qtbot) -> None:  # type: ignore[no-untyped-def]
        with qtbot.waitSignal(model.color_calibration_config_changed, timeout=500):
            model.color_calibration_sample_radius = 10

    def test_color_calibration_sample_radius_clamps_below_min(self, model: PipelineModel) -> None:
        model.color_calibration_sample_radius = 1
        assert model.color_calibration_sample_radius == 3

    def test_color_calibration_sample_radius_clamps_above_max(self, model: PipelineModel) -> None:
        model.color_calibration_sample_radius = 100
        assert model.color_calibration_sample_radius == 20

    def test_color_calibration_sample_radius_same_value_no_signal(self, model: PipelineModel) -> None:
        signals: list[bool] = []
        model.color_calibration_config_changed.connect(lambda: signals.append(True))
        model.color_calibration_sample_radius = 8
        assert len(signals) == 0

    # -- _update_color_calibration_step_state: step is None guard --

    def test_update_color_calibration_step_state_missing_step(self, model: PipelineModel) -> None:
        model._steps = [s for s in model._steps if s.key != "color_calibration"]
        model._update_color_calibration_step_state()  # should not raise

    def test_reset_keeps_color_calibration_disabled_when_not_enabled(self, model: PipelineModel) -> None:
        model.set_step_state("color_calibration", StepState.ERROR)
        model.reset()
        step = model.step_by_key("color_calibration")
        assert step is not None
        assert step.state is StepState.DISABLED

    def test_reset_sets_color_calibration_pending_when_enabled(self, model: PipelineModel) -> None:
        model.color_calibration_enabled = True
        model.set_step_state("color_calibration", StepState.DONE)
        model.reset()
        step = model.step_by_key("color_calibration")
        assert step is not None
        assert step.state is StepState.PENDING


class TestPipelineModelCometStack:
    """Tests for PipelineModel comet_stack configuration properties and signals."""

    @pytest.fixture()
    def model(self) -> PipelineModel:
        return PipelineModel()

    # -- comet_stack_enabled --

    def test_comet_stack_enabled_default(self, model: PipelineModel) -> None:
        assert model.comet_stack_enabled is False

    def test_comet_stack_enabled_setter(self, model: PipelineModel) -> None:
        model.comet_stack_enabled = True
        assert model.comet_stack_enabled is True

    def test_comet_stack_enabled_emits_signal(self, model: PipelineModel, qtbot) -> None:  # type: ignore[no-untyped-def]
        with qtbot.waitSignal(model.comet_stack_config_changed, timeout=500):
            model.comet_stack_enabled = True

    def test_comet_stack_enabled_same_value_no_signal(self, model: PipelineModel) -> None:
        signals: list[bool] = []
        model.comet_stack_config_changed.connect(lambda: signals.append(True))
        model.comet_stack_enabled = False
        assert len(signals) == 0

    def test_comet_stack_enabled_toggles_step_to_pending(self, model: PipelineModel) -> None:
        step = model.step_by_key("comet_stacking")
        assert step is not None
        assert step.state is StepState.DISABLED
        model.comet_stack_enabled = True
        assert step.state is StepState.PENDING

    def test_comet_stack_enabled_toggles_step_to_disabled(self, model: PipelineModel) -> None:
        model.comet_stack_enabled = True
        model.comet_stack_enabled = False
        step = model.step_by_key("comet_stacking")
        assert step is not None
        assert step.state is StepState.DISABLED

    def test_comet_stack_enabled_emits_step_changed_pending(self, model: PipelineModel, qtbot) -> None:  # type: ignore[no-untyped-def]
        with qtbot.waitSignal(model.step_changed, timeout=500) as blocker:
            model.comet_stack_enabled = True
        assert blocker.args == ["comet_stacking", StepState.PENDING.value]

    def test_comet_stack_enabled_emits_step_changed_disabled(self, model: PipelineModel, qtbot) -> None:  # type: ignore[no-untyped-def]
        model.comet_stack_enabled = True
        with qtbot.waitSignal(model.step_changed, timeout=500) as blocker:
            model.comet_stack_enabled = False
        assert blocker.args == ["comet_stacking", StepState.DISABLED.value]

    # -- comet_tracking_mode --

    def test_comet_tracking_mode_default(self, model: PipelineModel) -> None:
        assert model.comet_tracking_mode == "blend"

    def test_comet_tracking_mode_setter(self, model: PipelineModel) -> None:
        model.comet_tracking_mode = "stars"
        assert model.comet_tracking_mode == "stars"

    def test_comet_tracking_mode_emits_config_signal(self, model: PipelineModel, qtbot) -> None:  # type: ignore[no-untyped-def]
        with qtbot.waitSignal(model.comet_stack_config_changed, timeout=500):
            model.comet_tracking_mode = "stars"

    def test_comet_tracking_mode_same_value_no_signal(self, model: PipelineModel) -> None:
        signals: list[bool] = []
        model.comet_stack_config_changed.connect(lambda: signals.append(True))
        model.comet_tracking_mode = "blend"
        assert len(signals) == 0

    # -- comet_blend_factor --

    def test_comet_blend_factor_default(self, model: PipelineModel) -> None:
        assert model.comet_blend_factor == pytest.approx(0.5)

    def test_comet_blend_factor_setter(self, model: PipelineModel) -> None:
        model.comet_blend_factor = 0.8
        assert model.comet_blend_factor == pytest.approx(0.8)

    def test_comet_blend_factor_emits_config_signal(self, model: PipelineModel, qtbot) -> None:  # type: ignore[no-untyped-def]
        with qtbot.waitSignal(model.comet_stack_config_changed, timeout=500):
            model.comet_blend_factor = 0.3

    def test_comet_blend_factor_clamps_below_zero(self, model: PipelineModel) -> None:
        model.comet_blend_factor = -0.5
        assert model.comet_blend_factor == pytest.approx(0.0)

    def test_comet_blend_factor_clamps_above_one(self, model: PipelineModel) -> None:
        model.comet_blend_factor = 2.0
        assert model.comet_blend_factor == pytest.approx(1.0)

    def test_comet_blend_factor_same_value_no_signal(self, model: PipelineModel) -> None:
        signals: list[bool] = []
        model.comet_stack_config_changed.connect(lambda: signals.append(True))
        model.comet_blend_factor = 0.5
        assert len(signals) == 0

    # -- _update_comet_stack_step_state: step is None guard --

    def test_update_comet_stack_step_state_missing_step(self, model: PipelineModel) -> None:
        model._steps = [s for s in model._steps if s.key != "comet_stacking"]
        model._update_comet_stack_step_state()  # should not raise

    # -- set_step_state / set_step_progress with unknown key --

    def test_set_step_state_unknown_key_no_crash(self, model: PipelineModel) -> None:
        model.set_step_state("nonexistent_key", StepState.ACTIVE)  # should not raise

    def test_set_step_progress_unknown_key_no_crash(self, model: PipelineModel) -> None:
        model.set_step_progress("nonexistent_key", 0.5)  # should not raise

    # -- advance_to without finding the key --

    def test_advance_to_unknown_key_no_crash(self, model: PipelineModel) -> None:
        """advance_to with an unknown key sets nothing ACTIVE."""
        model.advance_to("nonexistent_key")
        active = model.active_step()
        assert active is None

    def test_reset_keeps_comet_stack_disabled_when_not_enabled(self, model: PipelineModel) -> None:
        model.set_step_state("comet_stacking", StepState.ERROR)
        model.reset()
        step = model.step_by_key("comet_stacking")
        assert step is not None
        assert step.state is StepState.DISABLED

    def test_reset_sets_comet_stack_pending_when_enabled(self, model: PipelineModel) -> None:
        model.comet_stack_enabled = True
        model.set_step_state("comet_stacking", StepState.DONE)
        model.reset()
        step = model.step_by_key("comet_stacking")
        assert step is not None
        assert step.state is StepState.PENDING

    # -- _update_starless_step_state: step is None guard --

    def test_update_starless_step_state_missing_step(self, model: PipelineModel) -> None:
        model._steps = [s for s in model._steps if s.key != "starless"]
        model._update_starless_step_state()  # should not raise


class TestPipelineModelChannelCombineNoOps:
    @pytest.fixture()
    def model(self) -> PipelineModel:
        return PipelineModel()

    def test_channel_combine_mode_same_value_no_signal(self, model: PipelineModel, qtbot) -> None:
        """line 233: return when mode unchanged."""
        signals: list[bool] = []
        model.channel_combine_config_changed.connect(lambda: signals.append(True))
        model.channel_combine_mode = model.channel_combine_mode
        assert len(signals) == 0

    def test_channel_combine_palette_same_value_no_signal(self, model: PipelineModel, qtbot) -> None:
        """line 244: return when palette unchanged."""
        signals: list[bool] = []
        model.channel_combine_config_changed.connect(lambda: signals.append(True))
        model.channel_combine_palette = model.channel_combine_palette
        assert len(signals) == 0

    def test_update_channel_combine_missing_step_no_raise(self, model: PipelineModel) -> None:
        """line 251: step is None guard in _update_channel_combine_step_state."""
        model._steps = [s for s in model._steps if s.key != "channel_combine"]
        model._update_channel_combine_step_state()  # should not raise
