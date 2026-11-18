"""Smoke test of bcd/vlm.py + bcd/train.py on a tiny, randomly initialised Qwen2-VL built locally
(no downloads). Checks shapes, Grad-CAM patch re-ordering, the training forward pass, LoRA updates and
adapter roll-back. Skipped when torch/transformers are unavailable."""
import json
import numpy as np
import pytest

try:
    import torch
    import transformers  # noqa
    from transformers import Qwen2VLConfig, Qwen2VLForConditionalGeneration, Qwen2VLProcessor, Qwen2TokenizerFast
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
    import peft  # noqa
except Exception as e:  # pragma: no cover
    pytest.skip(f"torch/transformers unavailable: {e}", allow_module_level=True)

SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>",
            "<|image_pad|>", "<|video_pad|>"]
TEMPLATE = ("{% for message in messages %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}"
            "{{ message['content'] }}{% else %}{% for c in message['content'] %}{% if c['type'] == 'image' %}"
            "<|vision_start|><|image_pad|><|vision_end|>{% elif c['type'] == 'text' %}{{ c['text'] }}{% endif %}"
            "{% endfor %}{% endif %}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
    from bcd.vlm import SYSTEM_PROMPT, QUESTION
    d = tmp_path_factory.mktemp("tinyqwen")
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    corpus = [SYSTEM_PROMPT, QUESTION, "glioma meningioma pituitary none Plan: refer to neurosurgery",
              "system user assistant"] * 50
    tok.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=400, special_tokens=SPECIALS,
                                                        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    fast = Qwen2TokenizerFast(tokenizer_object=tok, eos_token="<|im_end|>", pad_token="<|endoftext|>",
                              unk_token=None, bos_token=None)
    ip = Qwen2VLImageProcessor(min_pixels=56 * 56, max_pixels=56 * 56)
    proc = Qwen2VLProcessor(image_processor=ip, tokenizer=fast, chat_template=TEMPLATE)
    proc.save_pretrained(d)
    ids = {t: fast.convert_tokens_to_ids(t) for t in SPECIALS}
    cfg = Qwen2VLConfig(vocab_size=len(fast), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048,
                        rope_scaling={"type": "mrope", "mrope_section": [4, 2, 2]},
                        vision_config=dict(depth=2, embed_dim=32, num_heads=2, mlp_ratio=2, hidden_size=64,
                                           patch_size=14, spatial_merge_size=2, temporal_patch_size=2, in_chans=3),
                        image_token_id=ids["<|image_pad|>"], video_token_id=ids["<|video_pad|>"],
                        vision_start_token_id=ids["<|vision_start|>"], vision_end_token_id=ids["<|vision_end|>"],
                        bos_token_id=ids["<|endoftext|>"], eos_token_id=ids["<|im_end|>"],
                        pad_token_id=ids["<|endoftext|>"], tie_word_embeddings=False)
    torch.manual_seed(0)
    Qwen2VLForConditionalGeneration(cfg).save_pretrained(d)
    return str(d)


def _img(seed):
    from PIL import Image
    rng = np.random.default_rng(seed)
    return Image.fromarray((rng.random((56, 56, 3)) * 255).astype(np.uint8))


def test_vlm_end_to_end(tiny_model_dir, tmp_path):
    from bcd.vlm import VLM
    from bcd.train import train_examples, distill_round
    from bcd.config import load_config
    classes = ["glioma", "meningioma", "pituitary", "no_tumour"]
    lora = dict(r=4, alpha=8, dropout=0.0, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    v = VLM(tiny_model_dir, classes, dtype="float32", device="cpu", image_size=56, lora=lora)
    assert len(set(v.class_token_ids.tolist())) == 4

    ims = [_img(i) for i in range(3)]
    z = v.class_logits(ims, batch_size=2)
    assert z.shape == (3, 4) and np.isfinite(z).all()

    zc, cam = v.gradcam(ims[0])
    assert cam.shape == (56, 56) and np.isfinite(cam).all() and cam.max() <= 1.0 + 1e-6
    np.testing.assert_allclose(zc, z[0], rtol=1e-4, atol=1e-4)      # same logits with or without the hook

    z2, nll = v.forward_for_training(ims[:2], ["glioma", "no_tumour"], ["refer to neurosurgery", None])
    assert z2.shape == (2, 4) and nll.shape == (2,) and nll[1].item() == 0.0 and nll[0].item() > 0
    np.testing.assert_allclose(z2.detach().numpy(), z[:2], rtol=1e-4, atol=1e-4)  # class position is the answer slot

    paths = {f"img{i}": im for i, im in enumerate(ims)}
    load = lambda p: paths[p]
    before = v.adapter_state()
    ex = [dict(path="img0", y_star=0, z_cloud=[3.0, 0, 0, 0], plan="refer to neurosurgery", weight=1.0, teacher_wrong=False),
          dict(path="img1", y_star=3, z_cloud=None, plan=None, weight=0.8, teacher_wrong=False)]
    for e in ex:
        e["z_cloud"] = e["z_cloud"] if e["z_cloud"] is not None else None
    loss = train_examples(v, ex, load, classes, epochs=3, lr=5e-3, batch_size=1, grad_accum=1, log=lambda *a: None)
    assert np.isfinite(loss)
    after = v.adapter_state()
    assert any(not torch.equal(before[k], after[k]) for k in before)       # LoRA weights moved
    v.load_adapter_state(before)
    assert all(torch.equal(before[k], v.adapter_state()[k]) for k in before)  # roll-back works

    cfg = load_config(distill__epochs=1, distill__lr=1e-3)
    res = distill_round(v, ex, [ex[1]], ["img2", "img0"], np.array([1, 0]), load, classes, cfg,
                        log=lambda *a: None)
    assert "promoted" in res
    plan = v.generate_plan(ims[0], "glioma", max_new_tokens=5)
    assert isinstance(plan, str)
    v.save_adapter(str(tmp_path / "adapter"))
    assert (tmp_path / "adapter" / "adapter_config.json").exists()


def test_cam_variants(tiny_model_dir):
    from bcd.vlm import VLM
    v = VLM(tiny_model_dir, ["glioma", "meningioma", "pituitary", "no_tumour"], dtype="float32", device="cpu",
            image_size=56, lora=None)
    z, maps = v.cam_variants(_img(1))
    assert "pixel" in maps and any(k.startswith("gradcam@") for k in maps) and any(k.startswith("gradxact@") for k in maps)
    for m in maps.values():
        assert m.shape == (56, 56) and np.isfinite(m).all() and m.max() <= 1 + 1e-6
    v.cam_variant = "pixel"
    assert np.allclose(v.gradcam(_img(1))[1], maps["pixel"], atol=1e-5)


def test_regex_lora_covers_vision_and_trains(tiny_model_dir):
    from bcd.vlm import VLM
    from bcd.config import load_config
    from bcd.train import train_examples
    cfg = load_config("configs/default.yaml")
    lora = dict(cfg.lora); lora["r"], lora["alpha"] = 4, 8
    v = VLM(tiny_model_dir, ["glioma", "meningioma", "pituitary", "no_tumour"], dtype="float32", device="cpu",
            image_size=56, lora=lora)
    names = [n for n, p in v.model.named_parameters() if p.requires_grad]
    assert any("visual.blocks" in n for n in names) and any("mlp.down_proj" in n for n in names)
    ims = {f"p{i}": _img(i) for i in range(4)}
    ex = [dict(path=f"p{i}", y_star=i % 4, z_cloud=None, plan=None, weight=1.0, teacher_wrong=False) for i in range(4)]
    loss = train_examples(v, ex, ims.__getitem__, ["glioma", "meningioma", "pituitary", "no_tumour"], epochs=1,
                          batch_size=2, grad_accum=1, log=lambda *a: None)
    assert np.isfinite(loss)


def test_generate_text(tiny_model_dir):
    from bcd.vlm import VLM
    v = VLM(tiny_model_dir, ["glioma", "meningioma", "pituitary", "no_tumour"], dtype="float32", device="cpu",
            image_size=56, lora=None)
    out = v.generate_text("Score this plan", max_new_tokens=4)
    assert isinstance(out, str)


def test_generate_with_image_and_zero_lora(tiny_model_dir):
    import torch
    from bcd.vlm import VLM
    from bcd.compare import _zero_lora
    v = VLM(tiny_model_dir, ["glioma", "meningioma", "pituitary", "no_tumour"], dtype="float32", device="cpu",
            image_size=56, lora=dict(r=4, alpha=8, dropout=0.0, target_modules=["q_proj", "v_proj"]))
    base = v.class_logits([_img(3)])
    with torch.no_grad():                                   # simulate a trained adapter
        for n, p in v.model.named_parameters():
            if "lora_B" in n:
                p.add_(0.05)
    trained = v.class_logits([_img(3)])
    assert not np.allclose(base, trained)
    v.load_adapter_state(_zero_lora(v.adapter_state()))     # 'untrained' = LoRA B set to zero
    np.testing.assert_allclose(v.class_logits([_img(3)]), base, rtol=1e-4, atol=1e-4)
    assert isinstance(v.generate_with_image(_img(3), "Describe", max_new_tokens=3), str)
    assert 0 < v.lora_parameter_count() < v.parameter_count()


def test_compare_untrained_vs_trained_runs(tiny_model_dir, tmp_path):
    from types import SimpleNamespace
    from pathlib import Path
    from bcd.vlm import VLM
    from bcd.config import load_config
    from bcd.compare import untrained_vs_trained, attribution_pictures
    classes = ["glioma", "meningioma", "pituitary", "no_tumour"]
    lora = dict(r=4, alpha=8, dropout=0.0, target_modules=["q_proj", "v_proj"])
    s = VLM(tiny_model_dir, classes, dtype="float32", device="cpu", image_size=56, lora=lora)
    t = VLM(tiny_model_dir, classes, dtype="float32", device="cpu", image_size=56, lora=lora)
    s.save_adapter(str(tmp_path / "ad"))
    ims = {f"p{i}": _img(i) for i in range(12)}
    rt = dict(student=SimpleNamespace(s=s), teacher=SimpleNamespace(t=t), load=ims.__getitem__,
              val=[(f"p{i}", i % 4) for i in range(6)], test=[(f"p{i}", i % 4) for i in range(6, 12)],
              boxes={"p6": [(5, 5, 30, 30, 0.9)]})
    cfg = load_config()
    df, Z = untrained_vs_trained(cfg, rt, Path(tmp_path / "o"), {"X": str(tmp_path / "ad")}, log=lambda *a: None)
    assert len(df) == 2 * 5 and (tmp_path / "o" / "figure_reliability.png").exists()
    attribution_pictures(cfg, rt, Path(tmp_path / "o"), str(tmp_path / "ad"), "X", log=lambda *a: None)
    assert (tmp_path / "o" / "figure_untrained_vs_trained_gradcam.png").exists()
