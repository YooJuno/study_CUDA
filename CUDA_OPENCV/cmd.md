bash```
nvcc -c kernel.cu -o kernel.o
g++ -std=c++17 main.cpp kernel.o `pkg-config --cflags --libs opencv4` -lcudart -o main
```