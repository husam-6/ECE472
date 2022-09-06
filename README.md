## Readme instructions obtained from Chris Curro in ECE472 Deep Learning

### Setup

1. Install docker. https://docs.docker.com/get-docker/
2. Reboot your computer.

### Example setup for developing homeworks

```
.
├── Dockerfile
├── hw1
│   └── main.py
├── linear-example
│   └── main.py
├── README.md
├── requirements.txt
└── tf.sh
```

### Run linear example

```
$ ./tf.sh python linear-example/main.py --num_features 2 --num_samples 100 --batch_size 32 --num_iters 300 --random_seed 398729765279 --debug
$ ./tf.sh python linear-example/main.py --num_features 1 --num_samples 100 --batch_size 32 --num_iters 300 --random_seed 398729765279 --debug
```

After running the second command, open `fit.pdf`

### Generate single PDF for submission

```
$ ./tf.sh gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -sOutputFile=linear-example.pdf linear-example/main.py.pdf linear-example/fit.pdf
```

### Clean up directory

```
$ ./clean.sh hw1
```
