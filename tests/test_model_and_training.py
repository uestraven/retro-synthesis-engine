"""Model, training loop and sampler.

Kept deliberately tiny — a few hundred parameters and a handful of steps — so
the suite stays fast. These check that the machinery is correct, not that it
produces good music; that is what `rse evaluate` is for.
"""

from __future__ import annotations

import dataclasses

import pytest

torch = pytest.importorskip("torch")

from rse.events.convert import composition_from_intake  # noqa: E402
from rse.generation.plan import make_plan  # noqa: E402
from rse.generation.procedural import generate  # noqa: E402
from rse.generation.sampler import (  # noqa: E402
    SamplingConfig,
    build_prompt,
    filter_logits,
    sample,
)
from rse.models.transformer import ModelConfig, RseTransformer  # noqa: E402
from rse.tokenizers.tokenizer import Conditioning, RseTokenizer  # noqa: E402
from rse.training.dataset import TRAIN, VALID, build_windows  # noqa: E402
from rse.training.train import (  # noqa: E402
    TrainConfig,
    evaluate,
    learning_rate_at,
    load_checkpoint,
    make_batch,
    resolve_device,
    train,
)


@pytest.fixture
def tokenizer():
    return RseTokenizer()


@pytest.fixture
def tiny_config(tokenizer):
    return ModelConfig(
        vocab_size=len(tokenizer), n_layers=2, n_heads=2, d_model=32, max_seq_len=256
    )


@pytest.fixture
def windows(nes_profile, tokenizer):
    entries = []
    for index in range(6):
        generated = generate(nes_profile, make_plan(seed=100 + index, n_bars=4))
        composition = composition_from_intake(generated.record)
        split = TRAIN if index < 4 else VALID
        entries.append((composition, split, f"piece_{index}"))
    built, _stats = build_windows(entries, tokenizer, window_bars=4)
    return built


# -- model --------------------------------------------------------------------


def test_config_rejects_an_indivisible_width():
    with pytest.raises(ValueError, match="must divide"):
        ModelConfig(vocab_size=10, n_heads=3, d_model=32)


def test_config_rejects_nonsense():
    with pytest.raises(ValueError, match="vocab_size"):
        ModelConfig(vocab_size=0)
    with pytest.raises(ValueError, match="max_seq_len"):
        ModelConfig(vocab_size=10, max_seq_len=1)


def test_config_round_trips(tiny_config):
    assert ModelConfig.from_dict(tiny_config.to_dict()) == tiny_config


def test_forward_shape(tiny_config):
    model = RseTransformer(tiny_config)
    ids = torch.randint(0, tiny_config.vocab_size, (3, 16))
    assert model(ids).shape == (3, 16, tiny_config.vocab_size)


def test_a_too_long_sequence_is_refused(tiny_config):
    model = RseTransformer(tiny_config)
    ids = torch.zeros((1, tiny_config.max_seq_len + 1), dtype=torch.long)
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        model(ids)


def test_the_model_cannot_see_the_future(tiny_config):
    """Changing a later token must not alter an earlier position's logits."""
    model = RseTransformer(tiny_config).eval()
    ids = torch.randint(0, tiny_config.vocab_size, (1, 12))
    with torch.no_grad():
        before = model(ids)[0, 5]
        altered = ids.clone()
        altered[0, 9] = (altered[0, 9] + 1) % tiny_config.vocab_size
        after = model(altered)[0, 5]
    assert torch.allclose(before, after, atol=1e-5)


def test_padding_mask_changes_nothing_for_real_positions(tiny_config):
    model = RseTransformer(tiny_config).eval()
    ids = torch.randint(0, tiny_config.vocab_size, (1, 10))
    mask = torch.ones((1, 10), dtype=torch.bool)
    with torch.no_grad():
        assert torch.allclose(model(ids), model(ids, mask), atol=1e-5)


def test_padding_is_excluded_from_attention(tiny_config):
    """A real position's output must not depend on what sits in the padding."""
    model = RseTransformer(tiny_config).eval()
    ids = torch.randint(1, tiny_config.vocab_size, (1, 10))
    mask = torch.ones((1, 10), dtype=torch.bool)
    mask[0, 6:] = False
    altered = ids.clone()
    altered[0, 6:] = 0
    with torch.no_grad():
        a = model(ids, mask)[0, :6]
        b = model(altered, mask)[0, :6]
    assert torch.allclose(a, b, atol=1e-5)


def test_tied_weights_share_storage(tokenizer):
    tied = RseTransformer(ModelConfig(vocab_size=len(tokenizer), d_model=32, n_heads=2))
    assert tied.head.weight is tied.token_embedding.weight
    untied = RseTransformer(
        ModelConfig(vocab_size=len(tokenizer), d_model=32, n_heads=2, tie_weights=False)
    )
    assert untied.head.weight is not untied.token_embedding.weight
    assert untied.n_parameters > tied.n_parameters


def test_parameter_count_is_reported(tiny_config):
    model = RseTransformer(tiny_config)
    assert model.n_parameters > 0
    assert model.n_parameters == model.n_trainable_parameters


# -- training mechanics -------------------------------------------------------


def test_device_resolution():
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device() is not None


def test_learning_rate_warms_up_then_decays():
    config = TrainConfig(max_steps=100, warmup_steps=10, learning_rate=1e-3, min_learning_rate=1e-5)
    assert learning_rate_at(0, config) < config.learning_rate
    assert learning_rate_at(9, config) == pytest.approx(config.learning_rate)
    assert learning_rate_at(50, config) < config.learning_rate
    assert learning_rate_at(99, config) == pytest.approx(config.min_learning_rate, abs=1e-6)


def test_batches_are_shifted_by_one(tokenizer, windows):
    inputs, targets, mask = make_batch(
        windows, [0], pad_id=tokenizer.vocabulary.pad_id, seq_len=32, device=torch.device("cpu")
    )
    assert inputs.shape == targets.shape == mask.shape
    assert inputs[0, 1].item() == targets[0, 0].item()


def test_short_windows_are_padded_and_masked(tokenizer):
    from rse.training.dataset import Window

    pad = tokenizer.vocabulary.pad_id
    short = Window(ids=[1, 2, 3], split=TRAIN, composition_id="a", start_bar=0, n_bars=1)
    longer = Window(ids=[1, 2, 3, 4, 5, 6], split=TRAIN, composition_id="b", start_bar=0, n_bars=1)
    inputs, _targets, mask = make_batch(
        [short, longer], [0, 1], pad_id=pad, seq_len=32, device=torch.device("cpu")
    )
    assert inputs.shape == (2, 5)
    assert mask[0].tolist() == [True, True, False, False, False]
    assert inputs[0, 2].item() == pad


def test_evaluate_returns_a_finite_loss(tiny_config, tokenizer, windows):
    model = RseTransformer(tiny_config)
    loss = evaluate(
        model,
        [w for w in windows if w.split == VALID],
        TrainConfig(batch_size=2, seq_len=128),
        pad_id=tokenizer.vocabulary.pad_id,
        device=torch.device("cpu"),
        max_batches=1,
    )
    assert loss > 0 and loss == loss  # finite, not NaN


def test_evaluate_on_nothing_is_nan(tiny_config, tokenizer):
    model = RseTransformer(tiny_config)
    loss = evaluate(
        model, [], TrainConfig(), pad_id=tokenizer.vocabulary.pad_id, device=torch.device("cpu")
    )
    assert loss != loss  # NaN


# -- training end to end ------------------------------------------------------


def test_training_reduces_loss(tokenizer, windows, tiny_config):
    """The whole point: a few steps on a tiny model must learn something."""
    config = TrainConfig(
        seed=0,
        max_steps=30,
        batch_size=4,
        seq_len=128,
        eval_every=10,
        eval_batches=2,
        learning_rate=3e-3,
        warmup_steps=5,
        patience=0,
        amp=False,
        device="cpu",
    )
    _model, history, report = train(windows, tokenizer, model_config=tiny_config, config=config)
    assert len(history.train_loss) == 3
    assert history.train_loss[-1] < history.train_loss[0]
    assert report["n_parameters"] == RseTransformer(tiny_config).n_parameters
    assert report["train"]["seed"] == 0


def test_training_is_reproducible(tokenizer, windows, tiny_config):
    config = TrainConfig(
        seed=7,
        max_steps=10,
        batch_size=4,
        seq_len=128,
        eval_every=10,
        eval_batches=1,
        amp=False,
        device="cpu",
        patience=0,
    )
    _m1, h1, _r1 = train(windows, tokenizer, model_config=tiny_config, config=config)
    _m2, h2, _r2 = train(windows, tokenizer, model_config=tiny_config, config=config)
    assert h1.train_loss == h2.train_loss


def test_training_without_train_windows_is_refused(tokenizer, windows, tiny_config):
    valid_only = [w for w in windows if w.split == VALID]
    with pytest.raises(ValueError, match="no training windows"):
        train(valid_only, tokenizer, model_config=tiny_config)


def test_a_mismatched_vocab_size_is_refused(tokenizer, windows):
    with pytest.raises(ValueError, match="does not match the tokenizer"):
        train(
            windows,
            tokenizer,
            model_config=ModelConfig(vocab_size=len(tokenizer) + 1, d_model=32, n_heads=2),
        )


def test_checkpoint_carries_its_vocabulary(tokenizer, windows, tiny_config, tmp_path):
    """A model without the exact vocabulary it was trained on is unusable."""
    path = tmp_path / "ckpt.pt"
    config = TrainConfig(
        seed=0,
        max_steps=10,
        batch_size=4,
        seq_len=128,
        eval_every=10,
        eval_batches=1,
        amp=False,
        device="cpu",
        patience=0,
    )
    train(windows, tokenizer, model_config=tiny_config, config=config, checkpoint_path=path)
    model, restored_tokenizer, payload = load_checkpoint(path, device="cpu")
    assert len(restored_tokenizer) == len(tokenizer)
    assert restored_tokenizer.vocabulary.tokens == tokenizer.vocabulary.tokens
    assert model.config == tiny_config
    assert payload["checkpoint_version"].startswith("rse-checkpoint/1")


def test_a_future_checkpoint_version_is_refused(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"checkpoint_version": "rse-checkpoint/9.0.0"}, path)
    with pytest.raises(ValueError, match="unsupported checkpoint version"):
        load_checkpoint(path)


# -- sampling -----------------------------------------------------------------


def test_sampling_config_validates():
    with pytest.raises(ValueError, match="temperature"):
        SamplingConfig(temperature=0)
    with pytest.raises(ValueError, match="top_k"):
        SamplingConfig(top_k=0)
    with pytest.raises(ValueError, match="top_p"):
        SamplingConfig(top_p=1.5)
    with pytest.raises(ValueError, match="max_tokens"):
        SamplingConfig(max_tokens=1)


def test_prompt_is_a_conditioning_header(tokenizer):
    prompt = build_prompt(tokenizer, Conditioning(mood="dark", energy="high", n_bars=8))
    assert prompt[0] == "<bos>"
    assert "Mood_dark" in prompt and "Energy_high" in prompt and "Length_8" in prompt
    assert prompt[-1] == "Loop_start"


def test_filter_logits_masks_to_allowed_tokens():
    logits = torch.zeros(10)
    filtered = filter_logits(logits, temperature=1.0, top_k=None, top_p=None, allowed=(2, 5))
    assert torch.isfinite(filtered[[2, 5]]).all()
    assert torch.isinf(filtered[[0, 1, 3, 4, 6, 7, 8, 9]]).all()


def test_top_k_keeps_only_the_best():
    logits = torch.tensor([5.0, 1.0, 4.0, 0.0])
    filtered = filter_logits(logits, temperature=1.0, top_k=2, top_p=None)
    assert torch.isfinite(filtered[[0, 2]]).all()
    assert torch.isinf(filtered[[1, 3]]).all()


def test_top_p_always_keeps_at_least_one_token():
    logits = torch.tensor([10.0, 0.0, 0.0])
    filtered = filter_logits(logits, temperature=1.0, top_k=None, top_p=0.01)
    assert torch.isfinite(filtered).any()


def test_masking_happens_before_top_k():
    """Otherwise top-k can select tokens that the grammar then removes."""
    logits = torch.tensor([10.0, 9.0, 1.0, 0.5])
    filtered = filter_logits(logits, temperature=1.0, top_k=2, top_p=None, allowed=(2, 3))
    assert torch.isfinite(filtered[[2, 3]]).all()
    assert torch.isinf(filtered[[0, 1]]).all()


def test_sampling_produces_a_decodable_composition(tokenizer, tiny_config):
    model = RseTransformer(tiny_config).eval()
    result = sample(
        model,
        tokenizer,
        conditioning=Conditioning(n_bars=4),
        config=SamplingConfig(seed=0, max_tokens=300),
    )
    assert result.tokens[0] == "<bos>"
    assert result.n_grammar_rejections == 0
    # An untrained model makes noise, but it must be *decodable* noise.
    assert result.composition.hardware_profile == "nes"


def test_sampling_is_reproducible_from_its_seed(tokenizer, tiny_config):
    model = RseTransformer(tiny_config).eval()
    config = SamplingConfig(seed=42, max_tokens=200)
    first = sample(model, tokenizer, config=config)
    second = sample(model, tokenizer, config=config)
    assert first.ids == second.ids


def test_different_seeds_diverge(tokenizer, tiny_config):
    model = RseTransformer(tiny_config).eval()
    a = sample(model, tokenizer, config=SamplingConfig(seed=1, max_tokens=200))
    b = sample(model, tokenizer, config=SamplingConfig(seed=2, max_tokens=200))
    assert a.ids != b.ids


def test_grammar_constrained_output_always_decodes_cleanly(tokenizer, tiny_config):
    """The reason the sampler is grammar-constrained at all."""
    model = RseTransformer(tiny_config).eval()
    for seed in range(4):
        result = sample(model, tokenizer, config=SamplingConfig(seed=seed, max_tokens=250))
        decoded = tokenizer.decode(result.ids)
        assert decoded.n_tokens_skipped == 0, result.tokens[:40]


def test_unconstrained_sampling_is_allowed_but_reports_rejections(tokenizer, tiny_config):
    model = RseTransformer(tiny_config).eval()
    result = sample(
        model,
        tokenizer,
        config=SamplingConfig(seed=0, max_tokens=120, constrain_to_grammar=False),
    )
    # An untrained model ignores the grammar, so this should find violations.
    assert result.n_grammar_rejections >= 0
    assert len(result.ids) > 6


def test_the_token_limit_is_reported(tokenizer, tiny_config):
    model = RseTransformer(tiny_config).eval()
    result = sample(model, tokenizer, config=SamplingConfig(seed=0, max_tokens=20))
    assert result.hit_token_limit
    assert len(result.ids) <= 20


# -- crash resilience ---------------------------------------------------------


def test_a_checkpoint_exists_before_the_run_finishes(tokenizer, windows, tiny_config, tmp_path):
    """A dropped connection should cost minutes, not the whole run."""
    path = tmp_path / "interim.pt"
    config = TrainConfig(
        seed=0,
        max_steps=30,
        batch_size=4,
        seq_len=128,
        eval_every=10,
        eval_batches=1,
        amp=False,
        device="cpu",
        patience=0,
        checkpoint_every_evals=1,
    )

    seen: list[bool] = []
    real_evaluate = evaluate

    import rse.training.train as train_module

    def spy(*args, **kwargs):
        # Record whether a usable checkpoint is on disk partway through.
        seen.append(path.is_file())
        return real_evaluate(*args, **kwargs)

    train_module.evaluate = spy
    try:
        train(windows, tokenizer, model_config=tiny_config, config=config, checkpoint_path=path)
    finally:
        train_module.evaluate = real_evaluate

    # The first evaluation precedes any write; later ones must find a file.
    assert seen[0] is False
    assert any(seen[1:]), "no interim checkpoint was written during the run"
    model, restored, _payload = load_checkpoint(path, device="cpu")
    assert len(restored) == len(tokenizer)
    assert model.config == tiny_config


def test_interim_checkpoints_hold_the_best_weights_not_the_latest(
    tokenizer, windows, tiny_config, tmp_path
):
    path = tmp_path / "best.pt"
    config = TrainConfig(
        seed=1,
        max_steps=20,
        batch_size=4,
        seq_len=128,
        eval_every=10,
        eval_batches=1,
        amp=False,
        device="cpu",
        patience=0,
        checkpoint_every_evals=1,
    )
    _model, history, _report = train(
        windows,
        tokenizer,
        model_config=tiny_config,
        config=config,
        checkpoint_path=path,
    )
    _restored, _tok, payload = load_checkpoint(path, device="cpu")
    assert payload["history"]["best_valid_loss"] == pytest.approx(history.best_valid_loss)


def test_periodic_checkpointing_can_be_switched_off(tokenizer, windows, tiny_config, tmp_path):
    path = tmp_path / "off.pt"
    config = TrainConfig(
        seed=0,
        max_steps=10,
        batch_size=4,
        seq_len=128,
        eval_every=10,
        eval_batches=1,
        amp=False,
        device="cpu",
        patience=0,
        checkpoint_every_evals=0,
    )
    train(windows, tokenizer, model_config=tiny_config, config=config, checkpoint_path=path)
    # Still written at the end, just not during.
    assert path.is_file()
    assert config.to_dict()["checkpoint_every_evals"] == 0


# -- per-channel temperature --------------------------------------------------


def test_channel_temperature_validates():
    with pytest.raises(ValueError, match="channel 'NO' must be positive"):
        SamplingConfig(channel_temperature={"NO": 0.0})


def test_channel_temperature_is_recorded(tokenizer, tiny_config):
    config = SamplingConfig(seed=0, max_tokens=120, channel_temperature={"NO": 1.3})
    assert config.to_dict()["channel_temperature"] == {"NO": 1.3}
    model = RseTransformer(tiny_config).eval()
    result = sample(model, tokenizer, config=config)
    assert result.config.channel_temperature == {"NO": 1.3}


def test_a_channel_override_changes_only_that_channel(tokenizer, tiny_config):
    """The noise collapse is channel-specific, so the fix must be too.

    The model is seeded explicitly: an unseeded one draws its weights from
    global torch RNG, so whether a random model happens to emit a noise note
    inside the token budget depended on which tests ran first. This passed or
    failed by luck, and a vocabulary change that shifted token ids exposed it.
    """
    torch.manual_seed(0)
    model = RseTransformer(tiny_config).eval()
    base = sample(model, tokenizer, config=SamplingConfig(seed=3, temperature=0.7, max_tokens=400))
    # The override can only bite if a noise note is actually sampled.
    assert any(tokenizer.vocabulary.token_of(i) == "Channel_NO" for i in base.ids), (
        "no noise note sampled, so this test would pass vacuously"
    )
    overridden = sample(
        model,
        tokenizer,
        config=SamplingConfig(
            seed=3, temperature=0.7, channel_temperature={"NO": 1.4}, max_tokens=400
        ),
    )
    # Same seed and same melodic temperature, so the streams agree until the
    # first noise note and diverge after it.
    assert base.ids[:8] == overridden.ids[:8]
    assert base.ids != overridden.ids


def test_the_token_budget_is_configurable(tokenizer, tiny_config):
    """2048 truncated 32-bar pieces to 27 bars; it must be raisable."""
    model = RseTransformer(tiny_config).eval()
    short = sample(model, tokenizer, config=SamplingConfig(seed=0, max_tokens=60))
    longer = sample(model, tokenizer, config=SamplingConfig(seed=0, max_tokens=300))
    assert len(short.ids) <= 60
    assert len(longer.ids) > len(short.ids)


def test_train_passes_the_prefix_mask_to_validation(monkeypatch, tokenizer, windows):
    """The defect this guards against is a call site, not a signature: an
    `evaluate` that accepts `mask_prefix` and a `train` that never passes it
    reports an unmasked validation loss while training on a masked one, so an
    arm whose conditioning token is constant scores better for predicting its
    own prefix."""
    from rse.training import train as train_module

    seen: list[int | None] = []
    real = train_module.evaluate

    def spy(*args, **kwargs):
        seen.append(kwargs.get("mask_prefix"))
        return real(*args, **kwargs)

    monkeypatch.setattr(train_module, "evaluate", spy)
    train_module.train(
        windows,
        tokenizer,
        model_config=ModelConfig(
            vocab_size=len(tokenizer), n_layers=1, n_heads=2, d_model=32, max_seq_len=256
        ),
        config=TrainConfig(max_steps=1, batch_size=2, seq_len=64, eval_every=1),
        mask_prefix=7,
    )
    assert seen and set(seen) == {7}


def test_length_buckets_keep_window_sampling_uniform(windows):
    """Bucketing is a batching change, not a distribution change. If it
    shifted the sampling distribution the comparison against any earlier run
    would be confounded by it."""
    from collections import Counter

    from rse.training.train import LengthBucketSampler

    train = [w for w in windows if w.split == TRAIN]
    sampler = LengthBucketSampler(train, n_buckets=2, seed=0)
    draws = Counter()
    for _ in range(4000):
        draws.update(sampler.batch(4))
    share = [draws[i] / sum(draws.values()) for i in range(len(train))]
    expected = 1 / len(train)
    assert all(abs(s - expected) < expected * 0.35 for s in share), share


def test_piece_sampling_equalizes_composition_influence(windows):
    """Uniform window sampling gives a long piece proportionally more weight;
    measured on the real corpus the busiest piece took 4.08% of draws against
    0.18% for the median, a factor of 23."""
    from collections import Counter

    from rse.training.train import LengthBucketSampler

    train = [w for w in windows if w.split == TRAIN]
    # Make one composition dominate the window count.
    heavy = [dataclasses.replace(w, composition_id="heavy") for w in train]
    pool = [*train, *heavy, *heavy]

    by_window = LengthBucketSampler(pool, n_buckets=2, seed=1, sample_by="window")
    by_piece = LengthBucketSampler(pool, n_buckets=2, seed=1, sample_by="piece")

    def piece_share(sampler):
        draws = Counter()
        for _ in range(3000):
            draws.update(str(pool[i].composition_id) for i in sampler.batch(4))
        total = sum(draws.values())
        return draws["heavy"] / total

    assert piece_share(by_window) > piece_share(by_piece)
    # One of several compositions, so its share should be far from dominant.
    assert piece_share(by_piece) < 0.5


def test_the_schedule_continues_rather_than_restarting_on_resume():
    """The failure this guards against would have wasted an hour of GPU. The
    100k run decayed to `min_learning_rate` at its own final step, so a naive
    resume either re-warms to full peak -- discarding what was learned -- or
    sits at 3e-5 and learns almost nothing. Extending `max_steps` and
    evaluating at the absolute step puts step 100,000 where it would have been
    had the longer budget been planned from the start."""
    from rse.training.train import learning_rate_at

    short = TrainConfig(max_steps=100_000)
    long = TrainConfig(max_steps=300_000)

    # The original run ends at the floor.
    assert learning_rate_at(99_999, short) == pytest.approx(short.min_learning_rate, rel=1e-3)
    # A schedule recomputed from zero would re-warm.
    assert learning_rate_at(0, long) < long.learning_rate / 10
    # Continuing on the longer horizon gives a useful rate, and it decays.
    at_resume = learning_rate_at(100_000, long)
    assert at_resume > short.min_learning_rate * 5
    assert at_resume < long.learning_rate
    assert learning_rate_at(200_000, long) < at_resume
    assert learning_rate_at(299_999, long) == pytest.approx(long.min_learning_rate, rel=1e-3)


def test_rewarmup_ramps_from_the_floor_after_a_resume():
    """Adam's moments are absent from any checkpoint written before
    continuation was supported, so a discontinuous rate jump would be taken
    with no second-moment estimate to scale it."""
    from rse.training.train import learning_rate_at

    config = TrainConfig(max_steps=300_000)
    scheduled = learning_rate_at(100_000, config)
    first = learning_rate_at(100_000, config, resume_step=100_000, rewarmup_steps=500)
    middle = learning_rate_at(100_250, config, resume_step=100_000, rewarmup_steps=500)
    after = learning_rate_at(100_600, config, resume_step=100_000, rewarmup_steps=500)
    assert first < middle < scheduled
    assert first == pytest.approx(config.min_learning_rate, rel=0.05)
    assert after == pytest.approx(learning_rate_at(100_600, config))


def test_checkpoints_carry_optimizer_state_for_a_clean_continuation(tmp_path, tokenizer, windows):
    from rse.training.train import load_checkpoint, train

    out = tmp_path / "ckpt.pt"
    train(
        windows,
        tokenizer,
        model_config=ModelConfig(
            vocab_size=len(tokenizer), n_layers=1, n_heads=2, d_model=32, max_seq_len=256
        ),
        config=TrainConfig(max_steps=2, batch_size=2, seq_len=64, eval_every=1),
        checkpoint_path=out,
    )
    import torch

    payload = torch.load(out, map_location="cpu", weights_only=False)
    assert payload["optimizer"] is not None
    assert payload["torch_rng_state"] is not None
    # And it still loads through the normal path.
    model, loaded_tokenizer, _p = load_checkpoint(out, device="cpu")
    assert len(loaded_tokenizer) == len(tokenizer)


def _structural_counts(composition):
    """Channels sounding per occupied tick -- the saturation statistic."""
    from collections import Counter

    per_tick = Counter(e.tick for e in composition.sounding_events())
    return (
        sum(per_tick.values()) / len(per_tick) if per_tick else 0.0,
        len(per_tick),
    )


def test_structural_temperature_defaults_to_existing_behaviour(tokenizer, tiny_config):
    """An omitted option must change nothing: the whole experiment rests on
    the baseline arm being the code that was already running."""
    from rse.generation.sampler import SamplingConfig, sample
    from rse.models.transformer import RseTransformer

    torch.manual_seed(0)
    model = RseTransformer(tiny_config)
    base = sample(
        model,
        tokenizer,
        config=SamplingConfig(seed=7, temperature=0.9, max_tokens=400),
    )
    explicit_none = sample(
        model,
        tokenizer,
        config=SamplingConfig(seed=7, temperature=0.9, max_tokens=400, structural_temperature=None),
    )
    assert base.ids == explicit_none.ids


def test_structural_temperature_of_one_is_bit_exact(tokenizer, tiny_config):
    """1.0 must reproduce the baseline token stream exactly, not merely a
    similar distribution. Category-level and token-level sampling are
    mathematically identical there, but a two-stage draw consumes the
    generator differently -- so the control arm takes the single-draw path.
    Without this the 1.0 arm differed from the baseline by construction, and
    every comparison against it was shifted."""
    from rse.generation.sampler import SamplingConfig, sample
    from rse.models.transformer import RseTransformer

    torch.manual_seed(0)
    model = RseTransformer(tiny_config)
    for seed in (1, 2, 3, 4):
        base = sample(
            model,
            tokenizer,
            config=SamplingConfig(seed=seed, temperature=0.85, max_tokens=400),
        )
        control = sample(
            model,
            tokenizer,
            config=SamplingConfig(
                seed=seed,
                temperature=0.85,
                max_tokens=400,
                structural_temperature=1.0,
            ),
        )
        assert base.ids == control.ids, f"seed {seed}"


def test_structural_temperature_below_one_changes_the_output(tokenizer, tiny_config):
    """The complement of the control test: if sharpening changed nothing the
    experiment would be measuring noise."""
    from rse.generation.sampler import SamplingConfig, sample
    from rse.models.transformer import RseTransformer

    torch.manual_seed(0)
    model = RseTransformer(tiny_config)
    changed = 0
    for seed in (1, 2, 3, 4):
        base = sample(
            model,
            tokenizer,
            config=SamplingConfig(seed=seed, temperature=0.85, max_tokens=400),
        )
        sharpened = sample(
            model,
            tokenizer,
            config=SamplingConfig(
                seed=seed,
                temperature=0.85,
                max_tokens=400,
                structural_temperature=0.25,
            ),
        )
        changed += base.ids != sharpened.ids
    assert changed >= 3, changed


def test_greedy_structural_decisions_stay_grammatical(tokenizer, tiny_config):
    """Structure-aware decoding must never emit a token the decoder cannot
    read, however the category was chosen."""
    from rse.generation.sampler import SamplingConfig, sample
    from rse.models.transformer import RseTransformer

    torch.manual_seed(0)
    model = RseTransformer(tiny_config)
    for structural in (0.0, 0.25, 0.5, 1.0):
        result = sample(
            model,
            tokenizer,
            config=SamplingConfig(
                seed=11,
                temperature=0.9,
                max_tokens=500,
                structural_temperature=structural,
            ),
        )
        assert result.n_grammar_rejections == 0
        decoded = tokenizer.decode(result.ids)
        assert decoded.n_tokens_skipped == 0, f"structural={structural}"


def test_category_temperature_one_reproduces_the_models_own_mass():
    """At 1.0 the category distribution must equal the model's aggregated
    mass, which is what makes 1.0 a faithful control arm. For flat logits
    that aggregate genuinely is 4/20 -- the model really does believe a
    Position token is four times as likely, because it is indifferent among
    twenty tokens of which sixteen are Positions."""
    from rse.generation.sampler import sample_structural
    from rse.tokenizers.tokenizer import RseTokenizer

    tok = RseTokenizer()
    vocab = tok.vocabulary
    channels = [vocab.id_of(f"Channel_{c}") for c in ("P1", "P2", "TR", "NO")]
    positions = [vocab.id_of(f"Position_{i}") for i in range(16)]
    allowed = tuple(channels + positions)

    logits = torch.zeros(len(vocab))
    generator = torch.Generator(device="cpu").manual_seed(0)
    picks = [
        sample_structural(
            logits,
            allowed=allowed,
            vocabulary=vocab,
            structural_temperature=1.0,
            music_temperature=1.0,
            top_k=None,
            top_p=None,
            generator=generator,
        )
        for _ in range(600)
    ]
    share = sum(1 for p in picks if p in channels) / len(picks)
    assert 0.13 < share < 0.27, share


def test_sharpening_follows_probability_mass_not_token_count():
    """The property the category-level design exists for. Channel_ holds two
    tokens against Position_'s sixteen but carries more total mass, so
    lowering the structural temperature must move toward Channel_. A
    token-level temperature would instead sharpen within the larger family
    and drift the other way."""
    from rse.generation.sampler import sample_structural
    from rse.tokenizers.tokenizer import RseTokenizer

    tok = RseTokenizer()
    vocab = tok.vocabulary
    channels = [vocab.id_of(f"Channel_{c}") for c in ("P1", "P2")]
    positions = [vocab.id_of(f"Position_{i}") for i in range(16)]
    allowed = tuple(channels + positions)

    # Two channel tokens carry 0.72 of the mass against sixteen position
    # tokens' 0.28, so mass and token count point in opposite directions.
    logits = torch.full((len(vocab),), float("-inf"))
    for token_id in channels:
        logits[token_id] = 4.0
    for token_id in positions:
        logits[token_id] = 0.0

    def channel_share(structural: float) -> float:
        generator = torch.Generator(device="cpu").manual_seed(3)
        picks = [
            sample_structural(
                logits,
                allowed=allowed,
                vocabulary=vocab,
                structural_temperature=structural,
                music_temperature=1.0,
                top_k=None,
                top_p=None,
                generator=generator,
            )
            for _ in range(400)
        ]
        return sum(1 for p in picks if p in channels) / len(picks)

    warm = channel_share(1.0)
    cool = channel_share(0.25)
    assert cool > warm, (warm, cool)


def test_greedy_structural_follows_the_model_not_the_token_count():
    """At temperature 0 the chosen category must be the one holding the most
    probability mass, regardless of how many tokens it contains."""
    from rse.generation.sampler import sample_structural
    from rse.tokenizers.tokenizer import RseTokenizer

    tok = RseTokenizer()
    vocab = tok.vocabulary
    channels = [vocab.id_of(f"Channel_{c}") for c in ("P1", "P2")]
    positions = [vocab.id_of(f"Position_{i}") for i in range(16)]
    logits = torch.full((len(vocab),), -10.0)
    # One channel token carries almost all the mass; 16 position tokens share
    # a little. Greedy must pick the channel.
    logits[channels[0]] = 5.0
    for i in positions:
        logits[i] = 0.0
    chosen = sample_structural(
        logits,
        allowed=tuple(channels + positions),
        vocabulary=vocab,
        structural_temperature=0.0,
        music_temperature=1.0,
        top_k=None,
        top_p=None,
        generator=None,
    )
    assert chosen in channels
