# Project 1 Dimensionality Reduction

- AutoEncoder

  ~~~bash
  python3 AutoEncoder.py --data-root=./data/raw --gpu-id=0 --batch-size=128 --initial-lr=-3 --final-lr=-4 --num-epochs=200 --num-workers=4 --n-components=64
  ~~~

- LLE

  ~~~bash
  python3 LLE.py --n-components=128 --n-neighbors=64
  ~~~

- tSNE

  ~~~bash
  python3 tSNE.py --n-components=2 --init=pca --method=barnes_hut
  ~~~

- LDA+tSNE

  ~~~bash
  python3 LDA-tSNE.py --n-components=2 --init=pca --method=barnes_hut --preprocess=49
  ~~~

- MDS

  ~~~
  python3 MDS.py --n-components=32
  ~~~
