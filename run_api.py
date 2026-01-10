import os
import torch
from multiprocessing import freeze_support

from warp.engine.config import WARPRunConfig
from warp.engine.searcher import WARPSearcher
from warp.engine.utils.collection_indexer import index
from warp.engine.utils.index_converter import convert_index
import performance_metric
import getpass
import json
import time
import numpy as np
import argparse


class CustomWARPRunConfig(WARPRunConfig):
    def __init__(self, collection, nbits: int = 4, k: int = 10, nprobe: int = 16):
        self.collection = collection
        self.nbits = nbits
        self.k = k
        self.nprobe = nprobe

    @property
    def experiment_name(self):
        return f"{self.collection.name}"

    @property
    def index_name(self):
        return f"{self.collection.name}.nbits={self.nbits}"

    @property
    def collection_path(self):
        return self.collection.path


class CustomCollection:
    def __init__(self, name: str, path: str):
        self.name = name
        self.path = path


def construct_index(config: WARPRunConfig, embedding_path: str):
    index(config, embedding_path=embedding_path)
    convert_index(os.path.join(config.index_root, config.index_name))


def save_answer(dataset: str, method_name: str,
                build_index_suffix: str, retrieval_suffix: str,
                topk: int):
    answer_filename = os.path.expanduser(
        f'~/Dataset/billion-scale-multi-vector-retrieval/xtr/Result/answer/{dataset}-{method_name}-top{topk}-{build_index_suffix}-{retrieval_suffix}.tsv')
    with open(answer_filename, 'w') as f:
        for query_id, pID, rank, score in result_l:
            f.write(f'{query_id}\t{pID}\t{rank}\t{score}\n')


def read_query_document(dataset: str):
    query_filename = os.path.expanduser(
        f'~/Dataset/billion-scale-multi-vector-retrieval/RawData/{dataset}/document/queries.dev.tsv')
    qID_l = []
    qtxt_l = []
    with open(query_filename, 'r') as f:
        for line in f:
            if line == '':
                continue
            qID, text = line.split('\t')
            qID_l.append(qID)
            qtxt_l.append(text.strip())

    pID_l = []
    collection_filename = os.path.expanduser(
        f'~/Dataset/billion-scale-multi-vector-retrieval/RawData/{dataset}/document/collection.tsv')
    with open(collection_filename, 'r') as f:
        for line in f:
            if line == '':
                continue
            pID, text = line.split('\t')
            pID_l.append(int(pID))
    return qID_l, qtxt_l, pID_l


def print_query(searcher: WARPSearcher, query: str, collection_filename: str):
    passageID2text_m = {}
    cnt = 0
    with open(collection_filename, 'r') as f:
        for line in f:
            if line == '':
                continue
            pID, text = line.split('\t')
            passageID2text_m[cnt] = text.strip()
            cnt += 1
    print(passageID2text_m)
    print(f"Query: {query}")
    passage_ids, _, scores = searcher.search(query, k=10)
    print(passage_ids)
    for pid, score in zip(passage_ids, scores):
        print(pid, passageID2text_m[int(pid)], score)
    print("====================")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='argparse')
    parser.add_argument('--n_thread', type=int, default=32)
    parser.add_argument('--dataset', type=str, default='lotte-500-gnd')

    args = parser.parse_args()
    n_thread = args.n_thread
    dataset = args.dataset
    torch.set_num_threads(n_thread)

    method_name = 'WARP'
    n_bit = 2
    username = getpass.getuser()
    index_path = os.path.expanduser(f'~/Dataset/billion-scale-multi-vector-retrieval/xtr/Index/{dataset}')
    os.makedirs(index_path, exist_ok=True)
    os.environ["INDEX_ROOT"] = index_path

    embedding_path = os.path.expanduser(f'~/Dataset/billion-scale-multi-vector-retrieval/xtr/Embedding/{dataset}')
    os.makedirs(embedding_path, exist_ok=True)

    freeze_support()


    # Define the collection (i.e., list of passages)
    collection_filename = os.path.expanduser(
        f'~/Dataset/billion-scale-multi-vector-retrieval/RawData/{dataset}/document/collection.tsv')
    collection = CustomCollection(
        name="warp",
        path=collection_filename,
    )
    config = CustomWARPRunConfig(
        nbits=n_bit,
        collection=collection,
    )

    # Construct an index over the provided collection.
    # construct_index(config, embedding_path=embedding_path)
    searcher = WARPSearcher(config)

    for topk in [10, 100]:
        for nprobe in [1, 2, 4, 8, 16, 32, 64]:
        # for nprobe in [ 64]:
            # Prepare for searching via the constructed index.
            searcher.set_nprobe(nprobe=nprobe)

            qID_l, qtxt_l, pID_l = read_query_document(dataset=dataset)
            n_query = len(qtxt_l)

            result_l = []
            retrieval_time_l = []
            encode_time_l = []
            search_time_l = []
            for query, query_id in zip(qtxt_l, qID_l):
                start_time = time.time()
                encode_time, search_time, search_result = searcher.search(query, k=topk)
                passage_ids, _, scores = search_result
                retrieval_time_l.append((time.time() - start_time) * 1e3)
                encode_time_l.append(encode_time)
                search_time_l.append(search_time)

                for local_pID, rank_0, score in zip(passage_ids, range(len(passage_ids)), scores):
                    result_l.append((query_id, local_pID, rank_0 + 1, score))
            search_time_m = {
                'total_retrieval_time_ms': '{:.3f}'.format(sum(retrieval_time_l)),
                'average_retrieval_time_ms': '{:.3f}'.format(1.0 * np.average(retrieval_time_l)),
                'average_encode_time_ms': '{:.3f}'.format(1.0 * np.average(encode_time_l)),
                'average_search_time_ms': '{:.3f}'.format(np.average(search_time_l)),
            }

            build_index_suffix = f'n_bit_{n_bit}'
            retrieval_suffix = f'nprobe_{nprobe}-n_thread_{n_thread}'
            retrieval_config = {'nprobe': nprobe, 'n_thread': n_thread}

            save_answer(dataset=dataset, method_name=method_name,
                        build_index_suffix=build_index_suffix, retrieval_suffix=retrieval_suffix,
                        topk=topk)

            mrr_gnd, success_gnd = performance_metric.load_groundtruth(username=username, dataset=dataset,
                                                                       topk=topk)
            mrr_l, success_l, search_accuracy_m = performance_metric.count_accuracy(
                username=username, dataset=dataset, topk=topk,
                method_name=method_name, build_index_suffix=build_index_suffix, retrieval_suffix=retrieval_suffix,
                mrr_gnd=mrr_gnd, success_gnd=success_gnd)

            retrieval_info_m = {
                'n_query': len(qID_l), 'topk': topk, 'build_index': {},
                'retrieval': retrieval_config,
                'search_time': search_time_m, 'search_accuracy': search_accuracy_m
            }
            print(retrieval_info_m)

            method_performance_name = f'{dataset}-retrieval-{method_name}-top{topk}-{build_index_suffix}-{retrieval_suffix}.json'
            result_performance_path = f'/home/{username}/Dataset/billion-scale-multi-vector-retrieval/xtr/Result/performance'
            performance_filename = os.path.join(result_performance_path, method_performance_name)
            with open(performance_filename, "w") as f:
                json.dump(retrieval_info_m, f)
