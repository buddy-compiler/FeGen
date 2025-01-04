# Prepare

1. clone submodules
```bash
$ git submodule update --init
```

2. build mlir
```bash
$ cd thirdparty/llvm
$ cmake -G Ninja -Sllvm -Bbuild \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_RTTI=ON \
    -DCMAKE_BUILD_TYPE=RELEASE
$ cd build && ninja
```

# Build and Install FeGen

1. install python antlr tools
```bash
$ conda create -n fegen python=3.10.14
$ conda activate fegen
$ pip install -r requirements.txt
```

2. build C++ files
```bash
$ cmake -G Ninja -Sllvm -Bbuild \
    -DLLVM_DIR=$(pwd)/thirdparty/llvm/build/lib/cmake/llvm \
    -DMLIR_DIR=$(pwd)/thirdparty/llvm/build/lib/cmake/mlir \
    -DPython3_EXECUTABLE=$(which python) \
    -DLLVM_ENABLE_ASSERTIONS=ON 
```

3. generate files

```bash
$ bash script/gen_grammar.sh
```

4. build dist

```bash
$ pip python setup.py bdist_wheel
```
# How to use : TODO

<!-- 
3. run Driver.py

```bash
$ cd ..
$ python ./Driver.py
```

`file_path` in `Driver.py` can be one of:
* `./example/for.fegen`
* `./example/test.fegen` -->