"""
Dieses Skript diente zum Feintuning des TrOCR-Modells für die Handschrifterkennung. 
Es ist nicht mehr Teil der finalen Pipeline, da die Genauigkeit mit dem vortrainierten Modell höher ist.
"""

import argparse
import os
import inspect
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)

# Dient zum Feintuning des TrOCR-Modells
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune TrOCR for handwriting recognition")

    parser.add_argument("--base_model", type=str, default="fhswf/TrOCR_german_handwritten")
    parser.add_argument("--output_dir", type=str, default="models/trocr-finetuned-handwritten")

    # Datenquelle: entweder Hugging Face Dataset oder lokale CSV.
    parser.add_argument("--dataset_name", type=str, default="fhswf/german_handwriting")
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--eval_split", type=str, default="validation")

    parser.add_argument(
        "--annotations_csv",
        type=str,
        default=None,
        help="Optional local CSV with columns: image_path,text",
    )
    parser.add_argument(
        "--images_root",
        type=str,
        default=".",
        help="Base folder for relative image_path values from CSV",
    )

    parser.add_argument("--image_column", type=str, default=None)
    parser.add_argument("--text_column", type=str, default=None)

    parser.add_argument("--max_target_length", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--logging_steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--quick", action="store_true", help="Fast CPU-oriented run (few steps, no eval)")
    parser.add_argument("--quality", action="store_true", help="Accuracy-oriented run (more steps, eval on)")
    parser.add_argument("--disable_eval", action="store_true", help="Disable evaluation to speed up training")
    parser.add_argument("--freeze_encoder", action="store_true", help="Freeze vision encoder to speed up training")
    parser.add_argument("--train_subset", type=int, default=0, help="Use only first N train samples (0 = all)")
    parser.add_argument("--eval_subset", type=int, default=0, help="Use only first N eval samples (0 = all)")

    return parser.parse_args()


def resolve_columns(ds: Dataset, image_column: Optional[str], text_column: Optional[str]) -> Tuple[str, str]:
    # Spaltennamen je nach Dataset automatisch erkennen.
    columns = list(ds.features.keys())

    if image_column is None:
        for candidate in ["image", "img", "path", "file_name"]:
            if candidate in columns:
                image_column = candidate
                break
    if text_column is None:
        for candidate in ["text", "label", "transcription", "sentence", "target"]:
            if candidate in columns:
                text_column = candidate
                break

    if image_column is None or text_column is None:
        raise ValueError(
            "Could not infer columns automatically. "
            f"Available columns: {columns}. "
            "Set --image_column and --text_column explicitly."
        )

    return image_column, text_column


def load_local_csv_dataset(csv_path: str, images_root: str) -> DatasetDict:
    # Lokales Format: image_path,text
    ds = load_dataset("csv", data_files={"train": csv_path})["train"]
    if "image_path" not in ds.features or "text" not in ds.features:
        raise ValueError("CSV must contain columns: image_path,text")

    def absolutize(example: Dict[str, Any]) -> Dict[str, Any]:
        p = example["image_path"]
        if not os.path.isabs(p):
            p = os.path.join(images_root, p)
        return {"image_path": p, "text": example["text"]}

    ds = ds.map(absolutize)
    split = ds.train_test_split(test_size=0.15, seed=42)
    return DatasetDict(train=split["train"], validation=split["test"])


def ensure_rgb(image_like: Any, image_path: Optional[str] = None) -> Image.Image:
    # Vereinheitlicht verschiedene Eingabeformate auf PIL RGB.
    if isinstance(image_like, Image.Image):
        return image_like.convert("RGB")

    if isinstance(image_like, np.ndarray):
        return Image.fromarray(image_like).convert("RGB")

    if isinstance(image_like, str):
        return Image.open(image_like).convert("RGB")

    if isinstance(image_like, dict) and "path" in image_like and image_like["path"]:
        return Image.open(image_like["path"]).convert("RGB")

    if image_path is not None:
        return Image.open(image_path).convert("RGB")

    raise ValueError(f"Unsupported image value type: {type(image_like)}")


@dataclass
class OCRDataCollator:
    processor: TrOCRProcessor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Dataset liefert je nach Backend Listen oder Tensoren -> robust vereinheitlichen.
        pixel_tensors = []
        for f in features:
            pv = f["pixel_values"]
            if not isinstance(pv, torch.Tensor):
                pv = torch.tensor(pv, dtype=torch.float32)
            else:
                pv = pv.to(dtype=torch.float32)
            pixel_tensors.append(pv)

        pixel_values = torch.stack(pixel_tensors)
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"]
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels}


def main() -> None:
    args = parse_args()

    # Zwei Profile gleichzeitig waeren widerspruechlich.
    if args.quick and args.quality:
        raise ValueError("Bitte nur einen Modus setzen: --quick oder --quality")

    # Mischpraezision nur auf CUDA sinnvoll.
    if args.fp16 and not torch.cuda.is_available():
        print("Hinweis: --fp16 wurde gesetzt, aber keine CUDA-GPU gefunden. fp16 wird deaktiviert.")
        args.fp16 = False

    # Schnellprofil fuer CPU/Smoke-Tests: weniger Daten, weniger Steps, kein Eval.
    if args.quick:
        if args.max_steps < 0:
            args.max_steps = 80
        args.epochs = 1
        args.gradient_accumulation_steps = 1
        args.train_batch_size = 1
        args.eval_batch_size = min(args.eval_batch_size, 2)
        args.max_target_length = min(args.max_target_length, 64)
        args.disable_eval = True
        args.freeze_encoder = True
        if args.train_subset <= 0:
            args.train_subset = 512
        print(f"Quick-Modus aktiv: max_steps={args.max_steps}, epochs={args.epochs}, eval=aus")

    # Qualitaetsprofil: mehr Steps + Eval, kein Encoder-Freeze.
    if args.quality:
        if args.max_steps < 0:
            args.max_steps = 600
        args.epochs = max(args.epochs, 3)
        args.train_batch_size = min(args.train_batch_size, 2)
        args.eval_batch_size = min(args.eval_batch_size, 2)
        args.gradient_accumulation_steps = max(args.gradient_accumulation_steps, 2)
        args.disable_eval = False
        args.freeze_encoder = False
        args.train_subset = 0
        print(f"Quality-Modus aktiv: max_steps={args.max_steps}, epochs={args.epochs}, eval=an")

    # Basis-Modell laden, das dann feinjustiert wird.
    processor = TrOCRProcessor.from_pretrained(args.base_model)
    model = VisionEncoderDecoderModel.from_pretrained(args.base_model)

    # Decoder-Konfig muss explizit gesetzt sein, damit Generation stabil funktioniert.
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    if args.annotations_csv:
        # Lokale eigene Labels (empfohlen fuer deine Handschrift).
        dataset = load_local_csv_dataset(args.annotations_csv, args.images_root)
        image_column, text_column = "image_path", "text"
    else:
        # Externes Standard-Dataset.
        dataset = load_dataset(args.dataset_name, args.dataset_config)
        if args.train_split not in dataset:
            raise ValueError(f"Split '{args.train_split}' not found. Available: {list(dataset.keys())}")
        if args.eval_split not in dataset:
            split = dataset[args.train_split].train_test_split(test_size=0.15, seed=args.seed)
            dataset = DatasetDict(train=split["train"], validation=split["test"])
            eval_split = "validation"
        else:
            eval_split = args.eval_split
            dataset = DatasetDict(train=dataset[args.train_split], validation=dataset[eval_split])

        image_column, text_column = resolve_columns(dataset["train"], args.image_column, args.text_column)

    # Optionale Subsets fuer schnellere Iterationen.
    if args.train_subset > 0:
        subset_size = min(args.train_subset, len(dataset["train"]))
        dataset["train"] = dataset["train"].select(range(subset_size))
        print(f"Train-Subset aktiv: {subset_size} Beispiele")

    if args.eval_subset > 0 and "validation" in dataset:
        subset_size = min(args.eval_subset, len(dataset["validation"]))
        dataset["validation"] = dataset["validation"].select(range(subset_size))
        print(f"Eval-Subset aktiv: {subset_size} Beispiele")

    # Encoder-Freeze beschleunigt Training, kostet aber oft Genauigkeit.
    if args.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False
        print("Vision-Encoder eingefroren (schnelleres Training)")

    def preprocess(example: Dict[str, Any]) -> Dict[str, Any]:
        # Bild + Text in Modell-Inputs umwandeln.
        image_value = example.get(image_column)
        image_path = example.get("image_path") if "image_path" in example else None
        image = ensure_rgb(image_value, image_path=image_path)

        raw_text = example.get(text_column, "")
        if raw_text is None:
            raw_text = ""
        target_text = str(raw_text)

        pixel_values = processor(images=image, return_tensors="pt").pixel_values[0]
        labels = processor.tokenizer(
            text=target_text,
            max_length=args.max_target_length,
            padding="max_length",
            truncation=True,
        ).input_ids

        return {"pixel_values": pixel_values, "labels": labels}

    train_ds = dataset["train"].map(
        preprocess,
        remove_columns=dataset["train"].column_names,
        num_proc=args.num_workers if args.num_workers > 0 else None,
    )
    eval_ds = None
    # Eval-Set wird nur gebaut, wenn es aktiviert ist.
    if not args.disable_eval:
        eval_ds = dataset["validation"].map(
            preprocess,
            remove_columns=dataset["validation"].column_names,
            num_proc=args.num_workers if args.num_workers > 0 else None,
        )

    data_collator = OCRDataCollator(processor=processor)

    try:
        import evaluate

        cer_metric = evaluate.load("cer") if not args.disable_eval else None

        def compute_metrics(pred):
            # CER = Character Error Rate, Standardmetrik fuer OCR.
            if cer_metric is None:
                return {}
            pred_ids = pred.predictions
            label_ids = pred.label_ids

            pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)

            label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
            label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

            cer = cer_metric.compute(predictions=pred_str, references=label_str)
            return {"cer": cer}

    except Exception:
        compute_metrics = None

    # Kompatibel zu transformers 4.x/5.x (Parametername unterscheidet sich).
    sig = inspect.signature(Seq2SeqTrainingArguments.__init__)
    uses_eval_strategy = "eval_strategy" in sig.parameters

    do_eval = not args.disable_eval and eval_ds is not None

    training_kwargs = dict(
        output_dir=args.output_dir,
        predict_with_generate=do_eval,
        save_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16,
        dataloader_pin_memory=torch.cuda.is_available(),
        save_total_limit=2,
        load_best_model_at_end=do_eval,
        report_to="none",
        seed=args.seed,
        max_steps=args.max_steps,
    )

    if do_eval and compute_metrics is not None:
        training_kwargs["metric_for_best_model"] = "cer"
        training_kwargs["greater_is_better"] = False
    elif do_eval:
        training_kwargs["metric_for_best_model"] = "eval_loss"
        training_kwargs["greater_is_better"] = False

    if uses_eval_strategy:
        training_kwargs["eval_strategy"] = "steps" if do_eval else "no"
    else:
        training_kwargs["evaluation_strategy"] = "steps" if do_eval else "no"

    training_args = Seq2SeqTrainingArguments(**training_kwargs)

    # Kompatibel zu unterschiedlichen Processor-APIs.
    processor_for_trainer = getattr(processor, "image_processor", None)
    if processor_for_trainer is None:
        processor_for_trainer = getattr(processor, "feature_extractor", None)

    trainer_sig = inspect.signature(Seq2SeqTrainer.__init__)
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=data_collator,
    )

    if do_eval:
        trainer_kwargs["eval_dataset"] = eval_ds
        trainer_kwargs["compute_metrics"] = compute_metrics

    # Kompatibel zu transformers 4.x/5.x (processing_class vs tokenizer).
    if "processing_class" in trainer_sig.parameters:
        trainer_kwargs["processing_class"] = processor_for_trainer
    else:
        trainer_kwargs["tokenizer"] = processor_for_trainer

    trainer = Seq2SeqTrainer(**trainer_kwargs)

    # Start Fine-Tuning.
    trainer.train()

    # Lokales Speichern fuer spaeteres Laden in run_pipeline.py.
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    print(f"Model saved to: {args.output_dir}")
    print("Use this in run_pipeline.py:")
    print(f"model_name = '{args.output_dir}'")


if __name__ == "__main__":
    main()
