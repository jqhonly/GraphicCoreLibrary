# GraphicCoreLibrary

GraphicCoreLibrary, a.k.a **GCL**, is a small graphic and image processing library with interface based on CUDA for Microsoft Windows 10 platform.

##Pre-request
- MSVC v140
- CUDA 8.0

All the environment and some CUDA local dependencies relied of projects in this repository which is able to find in [here](https://github.com/CompileSense/caffe_windows_binary). Please check before you start to config or use.

##GraphicCoreLibrary
This project is the core engine of **GCL** which will be compiled into static library. The core part is written by CUDA to support different real-time image/ graphic processing tasks.

##GCL
This project is the .Net interface of GraphicCoreLibrary which is meanly written by C++/CLI. In order to compile this project, you need set *Common language Runtime Support* in your Visual Studio.

##GCLC
This project is the C++ interface of GraphicCoreLibrary which is meanly written by native C++. In order to compile this project, you need C++11 support compiler. 

##CSharpTest
This is a C# test project of GraphicCoreLibrary which is written by C#. In order to compile this project, you need .Net Framework 4.5.2. 


##Contributors
- inlmosuse
- Yifu Zhang