#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import numpy as np
import json
import doubletdetection
FNAME = "~/Google Drive/Computational Genomics/pbmc_4k_dense.csv"


if __name__ == '__main__':
    start_time = time.time()

    raw_counts = doubletdetection.load_csv(FNAME)
    BRs = [round(0.1 * i, 2) for i in range(1, int(1 + 0.8 / 0.1))]
    KNNs = list(range(10, 60, 10))
    PCAs = list(range(15, 30, 5)) + list(range(30, 60, 10))
    (num_cells, num_genes) = raw_counts.shape
    raw_results = np.zeros((len(BRs), len(KNNs), len(PCAs), num_cells))
    hits = np.zeros((num_cells))

    for i_br, br in enumerate(BRs):
        for j_knn, knn in enumerate(KNNs):
            for k_pca, pca in enumerate(PCAs):
                clf = doubletdetection.BoostClassifier(boost_rate=br, knn=knn, n_pca=pca)
                raw_results[i_br, j_knn, k_pca, :] = clf.fit(raw_counts)
                hits = hits + raw_results[i_br, j_knn, k_pca, :]
        with open('doubletHistogram-hits.txt', 'w') as f:
            f.write(FNAME + "\n")
            f.write("Most recent run --- Elapsed runtime: {0:.2f} seconds\n".format(
                time.time() - start_time))
            f.write("Boost rate: {0:.2f}, KNN: {1:d}, n_pca: {2:d}\n".format(
                br, knn, pca))
            json.dump(hits.tolist(), f)
            f.write("\n")
        f.closed

    np.save('doubletHistogram-raw_results.npy', raw_results)

    print("Total run time: {0:.2f} seconds".format(time.time() - start_time))
