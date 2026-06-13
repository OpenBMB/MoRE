import argparse
import glob
import gc
import json
import os
import re


DEFAULT_DATALOADER_BATCH_SIZE = 4096 * 768


class SimpleProgress:
    def __init__(self, iterable, desc=None, unit="it", total=None):
        self.iterable = iterable
        self.desc = desc or "Progress"
        self.unit = unit
        self.total = total if total is not None else len(iterable) if hasattr(iterable, "__len__") else None
        self.count = 0
        self.postfix = {}
        self.print_every = max((self.total or 1000) // 100, 1)

    def __iter__(self):
        for item in self.iterable:
            yield item
            self.count += 1
            if self.count == self.total or self.count % self.print_every == 0:
                postfix = ""
                if self.postfix:
                    postfix = " " + " ".join(f"{key}={value}" for key, value in self.postfix.items())
                if self.total is None:
                    print(f"[INFO] {self.desc}: {self.count} {self.unit}{postfix}")
                else:
                    print(f"[INFO] {self.desc}: {self.count}/{self.total} {self.unit}{postfix}")

    def set_postfix(self, **kwargs):
        self.postfix = kwargs


def progress_bar(iterable, desc=None, unit="it", total=None):
    try:
        from tqdm import tqdm
        return tqdm(iterable, desc=desc, unit=unit, total=total)
    except ImportError:
        return SimpleProgress(iterable, desc=desc, unit=unit, total=total)


def split_text_with_window(text, title, window_size=100, step_size=50):
    words = text.split()
    if len(words) <= 5:
        return []
    if len(words) <= window_size:
        return [f"{title} {text}"]

    segments = []
    for i in range(0, len(words) - window_size + 1, step_size):
        segment = " ".join(words[i:i + window_size])
        segments.append(f"{title} {segment}")

    if len(words) % window_size != 0:
        last_segment = " ".join(words[-window_size:])
        segments.append(f"{title} {last_segment}")

    return segments


def find_wiki_files(directory):
    wiki_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith("wiki_"):
                wiki_files.append(os.path.join(root, file))
    return sorted(wiki_files)


def extract_passages(directory, output_path, max_passages=None):
    all_segments = []
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    wiki_files = find_wiki_files(directory)
    print(f"[INFO] Found {len(wiki_files)} wiki files")

    with open(output_path, "w", encoding="utf-8") as output_file:
        progress = progress_bar(wiki_files, desc="Reading wiki files", unit="file")

        for file_path in progress:
            if max_passages is not None and len(all_segments) >= max_passages:
                break

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                docs = content.split("</doc>")

                for doc in docs:
                    if max_passages is not None and len(all_segments) >= max_passages:
                        break

                    if "<doc" not in doc:
                        continue

                    try:
                        spl = doc.split(">")[0].split('"')
                        _id = spl[1]
                        _title = spl[5]
                    except Exception:
                        continue

                    title_lower = _title.lower()
                    if "(disambiguation)" in title_lower:
                        continue
                    if "(disambiguation page)" in title_lower:
                        continue
                    if re.match(r"(List of .+)|(Index of .+)|(Outline of .+)", _title):
                        continue

                    try:
                        text = doc.split(">", 1)[1].strip()
                    except Exception:
                        continue

                    if not text:
                        continue

                    word_count = len(text.split())

                    if word_count > 100:
                        segments = split_text_with_window(text, _title)

                        if max_passages is not None:
                            remaining = max_passages - len(all_segments)
                            segments = segments[:remaining]

                        all_segments.extend(segments)

                        for seg in segments:
                            json.dump({"text": seg}, output_file, ensure_ascii=False)
                            output_file.write("\n")

                    elif word_count > 7:
                        segment = f"{_title} {text}"
                        all_segments.append(segment)

                        json.dump({"text": segment}, output_file, ensure_ascii=False)
                        output_file.write("\n")

            progress.set_postfix(passages=len(all_segments))

    print(f"[INFO] wiki passages filtered: {len(all_segments)} passages")
    return all_segments


def resolve_devices(devices_arg):
    if devices_arg == "auto":
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible_devices and visible_devices.strip() not in {"", "-1"}:
            device_count = len([device for device in visible_devices.split(",") if device.strip()])
            return [f"cuda:{device_id}" for device_id in range(device_count)]
        return None

    devices = [device.strip() for device in devices_arg.split(",") if device.strip()]
    if not devices:
        return None

    return [f"cuda:{device}" if device.isdigit() else device for device in devices]


def patch_transformers_torch_fx_available():
    try:
        from transformers.utils import import_utils
    except ImportError:
        return

    if hasattr(import_utils, "is_torch_fx_available"):
        return

    def is_torch_fx_available():
        try:
            import torch.fx  # noqa: F401
            return True
        except Exception:
            return False

    import_utils.is_torch_fx_available = is_torch_fx_available


def load_bge_m3_model(model_path, use_fp16=True, devices=None):
    patch_transformers_torch_fx_available()

    from FlagEmbedding import BGEM3FlagModel

    print("[INFO] BGE-M3 devices:", devices if devices is not None else "all visible devices")

    try:
        return BGEM3FlagModel(model_path, use_fp16=use_fp16, devices=devices)
    except TypeError:
        if devices is not None:
            print("[WARN] Current FlagEmbedding does not support the devices argument; falling back to default device logic.")
        return BGEM3FlagModel(model_path, use_fp16=use_fp16)


def close_bge_m3_model(model):
    """
    Explicitly stop FlagEmbedding multiprocessing pool.

    This avoids shutdown-time errors like:
    AttributeError: 'NoneType' object has no attribute 'SIGTERM'
    resource_tracker: leaked semaphore objects
    """
    if model is None:
        return

    print("[INFO] Closing BGE-M3 model multiprocessing pool...")

    try:
        if hasattr(model, "stop_self_pool"):
            model.stop_self_pool()
    except Exception as e:
        print(f"[WARN] model.stop_self_pool() failed during cleanup: {repr(e)}")

    # Prevent AbsEmbedder.__del__ from trying to stop the same pool again
    # when Python interpreter is already shutting down.
    try:
        setattr(model, "stop_self_pool", lambda *args, **kwargs: None)
    except Exception:
        pass

    # Best-effort cleanup for common pool attributes.
    for attr in ("pool", "process_pool", "multiprocessing_pool"):
        try:
            if hasattr(model, attr):
                setattr(model, attr, None)
        except Exception:
            pass

    print("[INFO] BGE-M3 cleanup done.")


def wiki_embed(segment, embed_dir, model, batch_size, encode_batch_size, max_length, num_workers):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    os.makedirs(embed_dir, exist_ok=True)

    dataloader = DataLoader(
        segment,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    total_chunks = (len(segment) + batch_size - 1) // batch_size

    print(f"[INFO] Total passages for embedding: {len(segment)}")
    print(f"[INFO] DataLoader batch size: {batch_size}")
    print(f"[INFO] BGE-M3 encode batch size: {encode_batch_size}")
    print(f"[INFO] GPU encoding chunks: {total_chunks}")

    chunk_count = 0

    for batch in progress_bar(dataloader, total=total_chunks, desc="GPU encoding chunks", unit="chunk"):
        chunk_count += 1

        with torch.no_grad():
            output = model.encode(
                batch,
                batch_size=encode_batch_size,
                max_length=max_length,
            )["dense_vecs"]

            file_name = os.path.join(embed_dir, f"embeded_wiki_{chunk_count}.npy")
            np.save(file_name, output)

            print(f"[INFO] Saved embedding chunk {chunk_count}: {file_name}")


def _chunk_id_from_path(path):
    match = re.search(r"embeded_wiki_(\d+)\.npy$", os.path.basename(path))
    if not match:
        raise ValueError(f"Cannot parse chunk id from embedding file: {path}")
    return int(match.group(1))


def _embedding_paths(embed_dir, num_index_chunks):
    if num_index_chunks is None:
        paths = glob.glob(os.path.join(embed_dir, "embeded_wiki_*.npy"))
        if not paths:
            raise FileNotFoundError(f"No embedding files found in {embed_dir}")
        return sorted(paths, key=_chunk_id_from_path)

    return [
        os.path.join(embed_dir, f"embeded_wiki_{chunk_count}.npy")
        for chunk_count in range(1, num_index_chunks + 1)
    ]


def index(embed_dir, index_dir, num_index_chunks):
    import faiss
    import numpy as np

    os.makedirs(index_dir, exist_ok=True)

    base = 0

    for embedding_path in _embedding_paths(embed_dir, num_index_chunks):
        chunk_count = _chunk_id_from_path(embedding_path)

        embedding_list = np.load(embedding_path).astype("float32")
        print("[INFO]", chunk_count, embedding_list.shape)

        hashed_id_list = np.arange(base, base + embedding_list.shape[0], dtype="int64")
        base += embedding_list.shape[0]

        hashed_id_path = os.path.join(embed_dir, f"hashed_id_wiki_{chunk_count}.npy")
        np.save(hashed_id_path, hashed_id_list)

        assert len(hashed_id_list) == len(set(hashed_id_list)), "IDs should be unique"

        faiss.normalize_L2(embedding_list)

        d = embedding_list.shape[1]
        metric = faiss.METRIC_INNER_PRODUCT

        cpu_index = faiss.index_factory(
            d,
            "IDMap,Flat",
            metric,
        )

        print("Creating FAISS index with the following parameters:")
        print("Index type: Flat")
        print(f"Metric: {metric}")
        print(f"Dimension: {d}")

        ngpus = faiss.get_num_gpus()
        print(f"Number of GPUs used for indexing: {ngpus}")

        if ngpus > 0:
            co = faiss.GpuMultipleClonerOptions()
            co.shard = True

            index_gpu = faiss.index_cpu_to_all_gpus(cpu_index, co=co, ngpu=ngpus)
            index_gpu.add_with_ids(embedding_list, hashed_id_list)

            index_cpu = faiss.index_gpu_to_cpu(index_gpu)
        else:
            print("[WARN] No FAISS GPU detected; indexing on CPU.")
            cpu_index.add_with_ids(embedding_list, hashed_id_list)
            index_cpu = cpu_index

        index_path = os.path.join(index_dir, f"indexed_wiki_{chunk_count}.index")

        faiss.write_index(index_cpu, index_path)

        print(f"Successfully indexed {index_cpu.ntotal} documents")
        print(f"Index saved to: {index_path}")
        print(f"Hashed ids saved to: {hashed_id_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess Wikipedia passages and build BGE-M3 FAISS indexes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--wiki-extracted-dir",
        required=True,
        help="Path to the WikiExtractor output directory that contains wiki_* files.",
    )
    parser.add_argument(
        "--wiki-extract-path",
        required=True,
        help="Output JSONL path for filtered wiki passages.",
    )
    parser.add_argument(
        "--embed-dir",
        required=True,
        help="Directory for embeded_wiki_*.npy and hashed_id_wiki_*.npy files.",
    )
    parser.add_argument(
        "--index-dir",
        required=True,
        help="Directory for indexed_wiki_*.index files.",
    )
    parser.add_argument(
        "--bge-m3-path",
        default="BAAI/bge-m3",
        help="BGE-M3 model name or local path. Use this to point to a downloaded bge-m3 directory.",
    )
    parser.add_argument(
        "--dataloader-batch-size",
        type=int,
        default=DEFAULT_DATALOADER_BATCH_SIZE,
        help="Batch size used by the PyTorch DataLoader before calling BGE-M3.",
    )
    parser.add_argument(
        "--encode-batch-size",
        type=int,
        default=4096,
        help="Batch size passed to BGEM3FlagModel.encode.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum token length passed to BGEM3FlagModel.encode.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--num-index-chunks",
        type=int,
        default=None,
        help="Number of embedding chunks to index. By default all embeded_wiki_*.npy files are discovered.",
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable fp16 when loading BGE-M3.",
    )
    parser.add_argument(
        "--devices",
        default="auto",
        help=(
            "Devices for BGE-M3 embedding. Default auto uses all GPUs exposed by "
            "CUDA_VISIBLE_DEVICES. Examples: auto, cuda:0,cuda:1, 0,1, cpu."
        ),
    )
    parser.add_argument(
        "--max-passages",
        type=int,
        default=None,
        help="Maximum number of passages to extract and encode. Useful for quick tests, e.g. 10000.",
    )

    return parser.parse_args()


def print_config(args):
    print("[INFO] wiki_extracted_dir:", args.wiki_extracted_dir)
    print("[INFO] wiki_extract_path:", args.wiki_extract_path)
    print("[INFO] embed_dir:", args.embed_dir)
    print("[INFO] index_dir:", args.index_dir)
    print("[INFO] bge_m3_path:", args.bge_m3_path)
    print("[INFO] CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>"))
    print("[INFO] devices:", args.devices)
    print("[INFO] dataloader_batch_size:", args.dataloader_batch_size)
    print("[INFO] encode_batch_size:", args.encode_batch_size)
    print("[INFO] max_length:", args.max_length)
    print("[INFO] num_workers:", args.num_workers)
    print("[INFO] num_index_chunks:", args.num_index_chunks)
    print("[INFO] max_passages:", args.max_passages)
    print("[INFO] no_fp16:", args.no_fp16)


def clear_cuda_cache():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("[INFO] torch.cuda.empty_cache() done.")
    except Exception as e:
        print(f"[WARN] clear CUDA cache failed: {repr(e)}")


def main():
    args = parse_args()
    print_config(args)

    segments = extract_passages(
        args.wiki_extracted_dir,
        args.wiki_extract_path,
        args.max_passages,
    )

    devices = resolve_devices(args.devices)
    model = None

    try:
        model = load_bge_m3_model(
            args.bge_m3_path,
            use_fp16=not args.no_fp16,
            devices=devices,
        )

        wiki_embed(
            segments,
            args.embed_dir,
            model,
            args.dataloader_batch_size,
            args.encode_batch_size,
            args.max_length,
            args.num_workers,
        )

    finally:
        close_bge_m3_model(model)

        try:
            del model
        except Exception:
            pass

        gc.collect()
        clear_cuda_cache()

    index(
        args.embed_dir,
        args.index_dir,
        args.num_index_chunks,
    )

    print("[INFO] All done.")


if __name__ == "__main__":
    main()
