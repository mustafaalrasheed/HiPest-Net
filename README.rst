HiPest-Net: a controlled study on IP102
=======================================

Code for the paper *Do attention, multi-scale fusion and taxonomy-aware heads improve insect pest
recognition? A controlled study on IP102* (Almustafa Abed, under review).

HiPest-Net adds identity-gated staged attention (coordinate attention, a channel/spatial
re-attention block and large-kernel attention), a two-layer BiFPN and a crop-then-species head to
ConvNeXt-Tiny. The repository trains it and plain baselines with one recipe on the official IP102
split and on the Citrus Pest Benchmark (CPB), selects every checkpoint on the validation split and
reads the test split once per run.

Main result (IP102 test top-1, mean +/- s.d. over seeds)
--------------------------------------------------------

====================  ==================  ==================
Model                 ImageNet-1k (3)     ImageNet-22k (2)
====================  ==================  ==================
ConvNeXt-Tiny         76.08 +/- 0.06      76.76 +/- 0.04
HiPest-Net            75.34 +/- 0.22      76.19 +/- 0.02
====================  ==================  ==================

The added modules do not improve on the plain backbone; see the paper for the component study,
paired bootstrap tests, CPB results and post-hoc evaluations.

Files
-----

- ``train.py`` - training and single test evaluation (writes ``log.jsonl``, ``results.json``,
  ``final.pt``, ``test_pred.npy``, ``test_labels.npy`` to the run folder)
- ``model.py``, ``attention.py``, ``bifpn.py`` - HiPest-Net and the baselines (torchvision ConvNeXt-T,
  ResNeXt-50, any ``timm:<name>`` model)
- ``losses.py`` - cross-entropy with label smoothing, class-balanced focal loss, taxonomy head loss
- ``data.py``, ``config.py`` - IP102 and CPB loaders, crop taxonomy
- ``engine.py`` - training loop, metrics (top-1/5, macro-F1, many/medium/few-shot terciles)
- ``weights22k.py`` - loads ImageNet-22k ConvNeXt-Tiny weights from timm into torchvision
- ``evalx.py`` - post-hoc evaluation: test-time size and flip chosen on validation, uniform and
  greedy weight soups, saved log-probabilities
- ``analysis/paired_bootstrap.py`` - paired bootstrap of the accuracy difference between two runs
- ``scripts/run_experiments.sh`` - the commands for every run reported in the paper

Setup
-----

Python 3.10 or later and a CUDA GPU (all runs in the paper used one RTX 2060 with 6 GB)::

    pip install -r requirements.txt

Data
----

- IP102: https://github.com/xpwu95/IP102 (classification part, official ``train.txt``,
  ``val.txt``, ``test.txt``). Pass the folder that holds these files and ``images/`` as
  ``--data-root``. ``config.py`` assumes the class indices are contiguous per crop in the order
  rice, corn, wheat, beet, alfalfa, vitis, citrus, mango; check this against your copy.
- CPB: https://github.com/edsonbollis/Citrus-Pest-Benchmark (official split, binary mite label).
  Set ``HIPEST_DATASET=cpb`` and pass the folder with the images and the
  ``pests_{train,validation,test}_original.csv`` files.

Usage
-----

Final recipe, baseline and HiPest-Net, one seed::

    export HIPEST_DATASET=ip102
    python train.py --data-root /data/IP102 --out runs/v2_convnext_s0 --seed 0 --arch convnext_t \
        --batch-size 32 --epochs1 30 --warmup-epochs 2 --cutmix 0.5 --smoothing 0.1
    python train.py --data-root /data/IP102 --out runs/v2_hipest_s0 --seed 0 --arch hipest \
        --fusion skip --loss ce --skip-stage2 --batch-size 32 --epochs1 30 --warmup-epochs 2 \
        --cutmix 0.5 --smoothing 0.1
    python analysis/paired_bootstrap.py runs/v2_hipest_s0 runs/v2_convnext_s0

Interrupted runs continue from the last completed epoch (use ``--no-resume`` to start over).

License
-------

MIT, see ``LICENSE``.
