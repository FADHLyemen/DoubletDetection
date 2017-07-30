#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import numpy as np
import json
import doubletdetection
# FNAME = "~/Google Drive/Computational Genomics/pbmc8k_dense.csv"
# FNAME = "~/Google Drive/Computational Genomics/pbmc_4k_dense.csv"
FNAME = "~/Google Drive/Computational Genomics/clean_5050.csv"
HITS_FNAME = "doubletHistogram-hits.txt"
RAW_FNAME = "doubletHistogram-raw_results.npy"


if __name__ == '__main__':
    start_time = {'wall': time.time(), 'CPU': time.process_time()}

    raw_counts = doubletdetection.load_csv(FNAME)
    BRs = [round(0.1 * i, 2) for i in range(1, int(1 + 0.8 / 0.1))]
    KNNs = list(range(10, 60, 10))
    # PCAs = list(range(15, 30, 5)) + list(range(30, 60, 10))
    PCAs = [15, 20, 25, 30, 40, 50, 75]
    MAX_HITS = len(BRs) * len(KNNs) * len(PCAs)
    (num_cells, num_genes) = raw_counts.shape
    raw_results = np.zeros((len(BRs), len(KNNs), len(PCAs), num_cells))
    hits = np.zeros((num_cells))

    runs_so_far = 0
    for i_br, br in enumerate(BRs):
        for j_knn, knn in enumerate(KNNs):
            for k_pca, pca in enumerate(PCAs):
                runs_so_far += 1
                clf = doubletdetection.BoostClassifier(boost_rate=br, knn=knn, n_pca=pca)
                raw_results[i_br, j_knn, k_pca, :] = clf.fit(raw_counts)
                hits = hits + raw_results[i_br, j_knn, k_pca, :]
                print("({0:d}/{1:d})".format(runs_so_far, MAX_HITS))
        with open(HITS_FNAME, 'w') as f:
            f.write(FNAME + "\n")
            f.write("BRs = {}\n".format(BRs))
            f.write("KNNs = {}\n".format(KNNs))
            f.write("PCAs = {}\n".format(PCAs))
            f.write("Max hits possible so far: {0:d}\n".format(runs_so_far))
            f.write("Most recent run ({0:d}/{1:d}) --- ".format(runs_so_far, MAX_HITS))
            f.write("Elapsed runtime(wall, CPU): ({0:.2f}, {0:.2f}) seconds\n".format(
                time.time() - start_time['wall'], time.process_time() - start_time['CPU']))
            f.write("Boost rate: {0:.2f}, KNN: {1:d}, n_pca: {2:d}\n".format(
                br, knn, pca))
            json.dump(hits.tolist(), f)
            f.write("\n")
        f.closed
        print("Wrote hits so far to {}".format(HITS_FNAME))

    np.save(RAW_FNAME, raw_results)
    print("Raw results exported to {}".format(RAW_FNAME))

    print("Total run time(wall, CPU): ({0:.2f}, {0:.2f}) seconds".format(
        time.time() - start_time['wall'], time.process_time() - start_time['CPU']))
