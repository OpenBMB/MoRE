import os
import re
import json

import numpy as np
import torch
import faiss
from tqdm import tqdm
from bge_embedding import encode
from torch.utils.data import DataLoader


def split_text_with_window(text, title, window_size=100, step_size=50):
    words = text.split()
    if len(words) <= 5:
        return []
    if len(words) <= window_size:
        return [text]

    segments = []
    for i in range(0, len(words) - window_size + 1, step_size):
        segment = " ".join(words[i:i + window_size])
        segments.append(f"{title} {segment}")

    if len(words) % window_size != 0:
        last_segment = " ".join(words[-window_size:])
        segments.append(f"{title} {last_segment}")

    return segments


def extract_passages(directory):
    all_segments = []
    if os.path.exists('/path/to/wiki_extract.jsonl'):
        with open('/path/to/wiki_extract.jsonl', 'r') as f:
            for line in f:
                all_segments.append(json.loads(line))

    with open(f'/path/to/wiki_extract.jsonl', 'w', encoding='utf-8') as output_file:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.startswith('wiki_'):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        docs = content.split('</doc>')
                        for doc in docs:
                            if '<doc' in doc:
                                spl = doc.split('>')[0].split('"')
                                _id = spl[1]
                                _title = spl[5]
                                if '(disambiguation)' in _title.lower():
                                    continue
                                if '(disambiguation page)' in _title.lower():
                                    continue
                                if re.match(r'(List of .+)|(Index of .+)|(Outline of .+)', _title):
                                    continue
                                text = doc.split('>')[1].strip()
                                if not text:
                                    continue
                                if len(text.split()) > 100:
                                    segments = split_text_with_window(text, id)
                                    all_segments.extend(segments)
                                    for seg in segments:
                                        json.dump({"text": seg}, output_file, ensure_ascii=False)
                                        output_file.write('\n')
                                elif len(text.split()) > 7:
                                    all_segments.append(f"{id} {text}")
                                    json.dump({"text": f"{id} {text}"}, output_file, ensure_ascii=False)
                                    output_file.write('\n')

    print("[INFO] wiki passages filtered")
    return all_segments

def wiki_embed(segment, batch_size):
    dataloader = DataLoader(
        segment,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True,
    )
    # encoded = np.empty((0, 1024), dtype=np.float16)
    chunk_count = 0
    for batch in tqdm(dataloader):
        with torch.no_grad():
            output = encode(batch)
            file_name = f'./wikipedia/embed/embeded_wiki_{chunk_count}.npy'
            np.save(file_name, output)
        chunk_count += 1


def index():
    base = 0
    for chunk_count in range(0, 18):
        embedding_list = np.load(f'/path/to/wikipedia/embed/embeded_wiki_{chunk_count}.npy').astype(
            'float32')
        print("[INFO] ", chunk_count, embedding_list.shape)

        hashed_id_list = np.arange(base, base + embedding_list.shape[0], dtype='int64')
        base += embedding_list.shape[0]

        # hashed_id_list = np.load('./wikipedia/hashed_id_wiki.npy')
        np.save(f'/path/to/wikipedia/embed/hashed_id_wiki_{chunk_count}.npy', hashed_id_list)
        assert len(hashed_id_list) == len(set(hashed_id_list)), "IDs should be unique"

        # Normalize the embeddings
        faiss.normalize_L2(embedding_list)

        # Dimension of the embeddings
        d = embedding_list.shape[1]

        # Create the FAISS index on the CPU
        metric = faiss.METRIC_INNER_PRODUCT
        cpu_index = faiss.index_factory(
            d,
            "IDMap,Flat",
            metric,
        )

        print("Creating FAISS index with the following parameters:")
        print(f"Index type: Flat")
        print(f"Metric: {metric}")
        print(f"Dimension: {d}")

        # Distribute the index across multiple GPUs
        ngpus = faiss.get_num_gpus()
        print(f"Number of GPUs used for indexing: {ngpus}")
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        index_gpu = faiss.index_cpu_to_all_gpus(cpu_index, co=co, ngpu=ngpus)

        # Add data to the GPU index
        index_gpu.add_with_ids(embedding_list, hashed_id_list)

        # Transfer the GPU index back to the CPU for saving
        index_cpu = faiss.index_gpu_to_cpu(index_gpu)

        index_path = f'/path/to/wikipedia/index/indexed_wiki_{chunk_count}.index'

        faiss.write_index(index_cpu, index_path)
        print(f"Successfully indexed {index_cpu.ntotal} documents")
        print(f"Index saved to: {index_path}")


if __name__ == "__main__":
    segments = extract_passages('/path/to/wiki_extracted')
    wiki_embed(segments, 4096*768)
    index()
