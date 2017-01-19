// GCL.cpp: 主项目文件。

#include "stdafx.h"
#include "GCL.clr.h"
#include <Windows.h>
using namespace System;
//using namespace GCL;
using namespace GCLSharp;

int main(array<System::String ^> ^args)
{
    Console::WriteLine(L"Hello World");
	ColorTransSharp^ ctrans = gcnew ColorTransSharp(1280, 720,0);//INIT
	array<Byte>^ yv12 = gcnew array<Byte>(1280 * 720 * 3 / 2);
	array<Byte>^ rgba32= gcnew array<Byte>(1280 * 720 * 4);
	for (size_t i = 0; i < yv12->Length; i++)
	{
		yv12[i] = i*10 % 255;
	}
	
	int a = ctrans->Managed_ColorTrans_YV12toARGB32_RetineX(yv12, rgba32);
	for (size_t i = 0; i < 10; i++)
	{
		Console::WriteLine(rgba32[i]);
	}
    return 0;
}
