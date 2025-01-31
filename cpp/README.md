To compile the C++ example you will need the libraries

- Eigen3 (https://eigen.tuxfamily.org/index.php?title=Main_Page)
- Catch2 (https://github.com/catchorg/Catch2)

To build and run
```
mkdir build
cd build
cmake ..
make
./unit_tests
./example
```

If you don't want to install Catch2 you can comment out the unit tests inside CmakeLists.txt.
