# FeGen

1. install python antlr tools
```bash
$ pip install -r requirements.txt
```

2. generate files

```bash
$ cd grammar
$ chmod +x ./generate.sh
$ ./generate.sh
```

to clean the generated, use the following cmds:

```bash
$ ./generate.sh clean
```

3. run Driver.py

```bash
$ cd ..
$ python ./Driver.py
```

`file_path` in `Driver.py` can be one of:
* `./example/for.fegen`
* `./example/test.fegen`