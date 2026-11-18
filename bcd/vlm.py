"""Student / teacher vision-language models (Qwen2-VL) — Steps 2, 4-5, 7 and training hooks.

Class probabilities are read from the next-token logits at the answer position, restricted to
the first token of each class word (Eq. 3.2). Attribution maps (Grad-CAM and variants) are computed on blocks of the
vision encoder with the predicted-class score as target (Eq. 3.6-3.7).

Tested with transformers 4.45-4.51 (Qwen2VLForConditionalGeneration). T4 GPUs: use float16.
"""
from __future__ import annotations
import contextlib
import numpy as np
import torch
import torch.nn.functional as F

SYSTEM_PROMPT = ("You are an assistant supporting a qualified clinician in reading brain MRI. "
                 "You do not make final decisions; your output will be reviewed.")
QUESTION = ("Classify this brain MRI slice. Answer with exactly one word from: "
            "glioma, meningioma, pituitary, none. Then, on a new line starting with 'Plan:', "
            "draft a concise treatment plan if a tumour is present.")
ANSWER_WORDS = {"glioma": "glioma", "meningioma": "meningioma", "pituitary": "pituitary", "no_tumour": "none"}


class VLM:
    def __init__(self, model_id: str, classes: list[str], dtype: str = "float16", load_4bit: bool = False,
                 device: str | int = 0, image_size: int = 448, lora: dict | None = None,
                 adapter_path: str | None = None):
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        self.classes, self.image_size = list(classes), image_size
        self.device = torch.device(f"cuda:{device}" if isinstance(device, int) else device)
        self.dtype = getattr(torch, dtype)
        px = image_size * image_size
        self.processor = AutoProcessor.from_pretrained(model_id, min_pixels=px, max_pixels=px)
        self.tok = self.processor.tokenizer
        self.tok.padding_side = "right"          # answer-position indexing assumes right padding
        kw = dict(torch_dtype=self.dtype, device_map={"": self.device})
        if load_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=self.dtype,
                                                           bnb_4bit_use_double_quant=True,
                                                           # keep the vision encoder in fp16: 4-bit NF4 on the ViT
                                                           # degrades the image features (v1.4)
                                                           llm_int8_skip_modules=["visual", "lm_head"])
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **kw)
        self.load_4bit = load_4bit
        if lora is not None or adapter_path is not None:
            self._attach_lora(lora, adapter_path)
        self.model.eval()
        self.class_token_ids = self._class_token_ids()
        self._prompt_cache = None

    # ------------------------------------------------------------------ setup
    def _attach_lora(self, lora, adapter_path):
        from peft import LoraConfig, get_peft_model, PeftModel
        # 4-bit: no prepare_model_for_kbit_training (it up-casts every non-quantised weight to fp32, which no
        # longer fits on a T4 once the vision encoder stays fp16); gradient checkpointing and input grads
        # are enabled in self.training() exactly as for the fp16 student.
        if adapter_path:
            self.model = PeftModel.from_pretrained(self.model, adapter_path, is_trainable=True)
        else:
            tm = lora["target_modules"]
            tm = tm if isinstance(tm, str) else list(tm)        # str = regex over full module names
            cfg = LoraConfig(r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
                             target_modules=tm, bias="none", task_type="CAUSAL_LM")
            self.model = get_peft_model(self.model, cfg)

    def _class_token_ids(self):
        ids = []
        for c in self.classes:
            t = self.tok.encode(ANSWER_WORDS[c], add_special_tokens=False)
            ids.append(t[0])
        if len(set(ids)) != len(ids):
            raise ValueError(f"Class words share a first token: {ids}. Change ANSWER_WORDS.")
        return torch.tensor(ids, device=self.device)

    # ------------------------------------------------------------------ prompts
    def prompt_text(self) -> str:
        if self._prompt_cache is None:
            msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": QUESTION}]}]
            self._prompt_cache = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return self._prompt_cache

    def _inputs(self, images, texts):
        enc = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        return {k: v.to(self.device) for k, v in enc.items()}

    @staticmethod
    def answer_text(class_name: str, plan: str | None) -> str:
        a = ANSWER_WORDS[class_name]
        if plan:
            a += "\nPlan: " + plan.strip()
        return a + "<|im_end|>"

    # ------------------------------------------------------------------ Step 2: class logits
    @torch.no_grad()
    def class_logits(self, images, batch_size: int = 8) -> np.ndarray:
        """Raw logits z over the class tokens, shape [N, C] (Eq. 3.2 before softmax)."""
        self.model.eval()
        out = []
        for s in range(0, len(images), batch_size):
            ims = images[s:s + batch_size]
            inp = self._inputs(ims, [self.prompt_text()] * len(ims))
            with torch.autocast("cuda", dtype=self.dtype):
                logits = self.model(**inp).logits
            last = inp["attention_mask"].sum(1) - 1            # right padding -> last prompt token
            z = logits[torch.arange(len(ims), device=self.device), last][:, self.class_token_ids]
            out.append(z.float().cpu().numpy())
        return np.concatenate(out, 0)

    # ------------------------------------------------------------------ Steps 4-5: Grad-CAM
    # Attribution variants computed from ONE forward/backward pass. The variant used by the gate is
    # chosen on the validation set (runtime.choose_cam_variant) and stored in self.cam_variant.
    CAM_LAYERS = (-1, -8, -16)
    cam_variant = "gradcam@-1"

    def _vision_blocks(self):
        for name, mod in self.model.named_modules():
            if name.endswith("visual.blocks"):
                return mod
        raise RuntimeError("Could not find visual.blocks in model")

    def _vision_last_block(self):
        return self._vision_blocks()[-1]

    def _to_grid(self, seq, thw):
        """Token sequence [N] in Qwen2-VL patch order -> [image_size, image_size] map in [0,1]."""
        t, gh, gw = thw
        m = seq.reshape(t, gh // 2, gw // 2, 2, 2).permute(0, 1, 3, 2, 4).reshape(t, gh, gw)[0]
        m = F.interpolate(m[None, None], size=(self.image_size, self.image_size), mode="bilinear",
                          align_corners=False)[0, 0]
        m = m - m.min() if m.min() < 0 else m
        return (m / m.max().clamp_min(1e-8)).detach().cpu().numpy()

    def cam_variants(self, image, T: float = 1.0, target: int | None = None):
        """Returns (class logits [C], {variant_name: map [H, W] in [0,1]}).
        gradcam@L : Grad-CAM (Eq. 3.6-3.7) at vision block L - channel weights = mean gradient
        gradxact@L: token-wise gradient x activation at block L (better suited to transformers)
        pixel     : |input x gradient| on the image patches"""
        blocks = self._vision_blocks()
        layers = [l for l in self.CAM_LAYERS if -l <= len(blocks)]
        store, hooks = {}, []
        for l in layers:
            def hook(_m, _i, out, l=l):
                o = out[0] if isinstance(out, tuple) else out
                if o.requires_grad:
                    o.retain_grad()
                store[l] = o
            hooks.append(blocks[l].register_forward_hook(hook))
        try:
            inp = self._inputs([image], [self.prompt_text()])
            pv = inp["pixel_values"].detach().clone().requires_grad_(True)
            inp["pixel_values"] = pv
            with torch.enable_grad(), torch.autocast("cuda", dtype=self.dtype):
                logits = self.model(**inp).logits
                z = logits[0, -1, self.class_token_ids].float()
                tgt = int(torch.argmax(z)) if target is None else target
                z[tgt].backward()                      # class score before softmax (standard Grad-CAM)
            thw = [int(v) for v in inp["image_grid_thw"][0]]
            out = {}
            for l in layers:
                a, g = store[l].detach().float(), store[l].grad
                if g is None:
                    continue
                g = g.float()
                out[f"gradcam@{l}"] = self._to_grid(F.relu((a * g.mean(0)).sum(-1)), thw)
                out[f"gradxact@{l}"] = self._to_grid(F.relu((a * g).sum(-1)), thw)
            if pv.grad is not None:
                out["pixel"] = self._to_grid((pv.detach().float() * pv.grad.float()).abs().sum(-1), thw)
        finally:
            for h in hooks:
                h.remove()
            self.model.zero_grad(set_to_none=True)
        return z.detach().cpu().numpy(), out

    def gradcam(self, image, T: float = 1.0, target: int | None = None):
        """Returns (class logits [C], attribution map [H, W] in [0,1]) using self.cam_variant."""
        z, maps = self.cam_variants(image, T, target)
        cam = maps.get(self.cam_variant)
        if cam is None:
            cam = next(iter(maps.values()))
        return z, cam

    # ------------------------------------------------------------------ plans
    @torch.no_grad()
    def generate_text(self, prompt: str, max_new_tokens: int = 60) -> str:
        """Text-only generation (used for the LLM-as-judge plan scoring). Greedy decoding."""
        msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = self.processor(text=[text], return_tensors="pt", padding=True)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.autocast("cuda", dtype=self.dtype, enabled=torch.cuda.is_available()):
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        return self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    @torch.no_grad()
    def generate_with_image(self, image, prompt: str, max_new_tokens: int = 80) -> str:
        """Free-text answer about one image (used for the models' own explanations and for the
        bounding-box mismatch prompt sent to the cloud model). Greedy decoding."""
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = self._inputs([image], [text])
        with torch.autocast("cuda", dtype=self.dtype, enabled=torch.cuda.is_available()):
            out = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
        return self.tok.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def lora_parameter_count(self) -> int:
        return int(sum(p.numel() for n, p in self.model.named_parameters() if "lora_" in n))

    def parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.model.parameters()))

    @torch.no_grad()
    def generate_plan(self, image, class_name: str, guidance: str = "", max_new_tokens: int = 384) -> str:
        if class_name == "no_tumour":
            return ""
        prefix = ANSWER_WORDS[class_name] + "\nPlan:"
        q = self.prompt_text()
        if guidance:
            q = q.replace(QUESTION, QUESTION + "\nReference guidance (for context only):\n" + guidance)
        inp = self._inputs([image], [q + prefix])
        with torch.autocast("cuda", dtype=self.dtype):
            out = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=True,
                                      temperature=0.2, top_p=0.9)
        return self.tok.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    # ------------------------------------------------------------------ training forward
    def forward_for_training(self, images, class_names, plans):
        """Return (class logits [B,C], plan NLL [B] or None) with gradients."""
        prompt = self.prompt_text()
        texts = [prompt + self.answer_text(c, p) for c, p in zip(class_names, plans)]
        inp = self._inputs(images, texts)
        p_inp = self._inputs(images[:1], [prompt])
        Lp = int(p_inp["attention_mask"].sum())                             # identical for all (fixed image size)
        with torch.autocast("cuda", dtype=self.dtype):
            logits = self.model(**inp).logits
        z = logits[:, Lp - 1, self.class_token_ids].float()
        plan_nll = None
        if any(p for p in plans):
            ids = inp["input_ids"]; am = inp["attention_mask"]
            # plan tokens start after class word + "\nPlan:" ; supervise everything after the class word
            word_len = max(len(self.tok.encode(ANSWER_WORDS[c], add_special_tokens=False)) for c in class_names)
            start = Lp + word_len
            lp = F.log_softmax(logits[:, start - 1:-1].float(), -1)
            tgt = ids[:, start:]
            nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            m = am[:, start:].float()
            has = torch.tensor([1.0 if p else 0.0 for p in plans], device=self.device)
            plan_nll = (nll * m).sum(1) / m.sum(1).clamp_min(1) * has
        return z, plan_nll

    # ------------------------------------------------------------------ adapters
    def adapter_state(self):
        from peft import get_peft_model_state_dict
        return {k: v.detach().clone().cpu() for k, v in get_peft_model_state_dict(self.model).items()}

    def load_adapter_state(self, state):
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(self.model, state)

    def save_adapter(self, path):
        self.model.save_pretrained(path)

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    @contextlib.contextmanager
    def training(self):
        self.model.train()
        try:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
        except Exception:
            pass
        try:
            yield
        finally:
            self.model.eval()
